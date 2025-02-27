# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
from jax import export
from transformers import FlaxAutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import time
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map


def create_2d_mesh():
    devices = jax.devices()
    n_devices = len(devices)
    mesh_x = int(n_devices**0.5)
    mesh_y = n_devices // mesh_x
    return mesh_utils.create_device_mesh((mesh_x, mesh_y))


def get_param_sharding_rule(param_name, param_tensor, mesh):
    if isinstance(param_tensor, dict):
        leaves = jax.tree_util.tree_leaves(param_tensor)
        ndim = len(leaves[0].shape)
    else:
        ndim = len(param_tensor.shape)

    if ndim == 1:
        return NamedSharding(mesh, P())
    elif "kernel" in param_name.lower():
        return NamedSharding(mesh, P(None, "model"))
    elif "embedding" in param_name.lower():
        return NamedSharding(mesh, P(None, "model"))
    else:
        return NamedSharding(mesh, P("data", None))


def shard_params_across_mesh(params, mesh):
    sharded_params = {}
    devices = jax.devices()
    for k, v in params.items():
        if isinstance(v, dict):
            sharded_params[k] = shard_params_across_mesh(v, mesh)
        else:
            sharding = get_param_sharding_rule(k, v, mesh)
            print(f"Sharding {k} with shape {v.shape} using {sharding.spec}")
            sharded_params[k] = jax.device_put(v, sharding)

            for i, local_slice in enumerate(
                jax.device_get(sharded_params[k].addressable_data(i)) for i in range(len(devices))
            ):
                print(f"Device {i} slice shape: {local_slice.shape}")
    return sharded_params


def load_model_and_tokenizer(model_name="bert-base-uncased", mesh=None):
    config = AutoConfig.from_pretrained(model_name)
    model = FlaxAutoModelForSequenceClassification.from_config(config)

    pretrained = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)
    model.params = shard_params_across_mesh(pretrained.params, mesh)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_batch_input(texts, tokenizer, mesh, max_length=128):
    n_devices_x = mesh.devices.shape[0]
    batch_size = len(texts)
    target_size = ((batch_size + n_devices_x - 1) // n_devices_x) * n_devices_x
    if batch_size < target_size:
        texts = texts + [texts[0]] * (target_size - batch_size)

    encoded = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="jax")

    batch_sharding = NamedSharding(mesh, P("data", None))
    input_ids = jax.device_put(encoded["input_ids"], batch_sharding)
    attention_mask = jax.device_put(encoded["attention_mask"], batch_sharding)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def create_inference_fn(model, mesh):
    def forward(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, params=model.params, train=False)
        return jax.nn.softmax(outputs.logits, axis=-1)

    sharded_forward = forward
    sharded_comp = shard_map(
        sharded_forward, mesh=mesh, in_specs=(P("data", None), P("data", None)), out_specs=P("data", None)
    )

    return sharded_comp


def run_profiled_inference(model, forward_fn, inputs, mesh, num_warmup=1, num_runs=5):
    run_times = []

    with mesh:
        for _ in range(num_warmup):
            _ = forward_fn(inputs["input_ids"], inputs["attention_mask"])

        fwd_jit = jax.jit(forward_fn)
        jit_lowered = fwd_jit.lower(inputs["input_ids"], inputs["attention_mask"])
        jitted_stablehlo = jit_lowered.compiler_ir(dialect="stablehlo")
        print(jitted_stablehlo.dump())

        for i in range(num_runs):
            start_time = time.time()
            predictions = forward_fn(inputs["input_ids"], inputs["attention_mask"])
            run_time = time.time() - start_time
            run_times.append(run_time)

    avg_time = sum(run_times) / len(run_times)
    print(f"\nAverage batch time: {avg_time:.4f}s")

    return predictions


def main():
    jax.experimental.compilation_cache.compilation_cache.reset_cache()
    device_mesh = create_2d_mesh()
    mesh = Mesh(device_mesh, ("data", "model"))
    print(f"Created {mesh.devices.shape} device mesh")

    base_texts = [
        "This is a great movie!",
        "I really didn't like this film.",
        "An absolute masterpiece!",
        "Terrible waste of time.",
    ]

    min_samples_per_device = 4
    target_batch_size = mesh.devices.shape[0] * min_samples_per_device
    repeats = (target_batch_size + len(base_texts) - 1) // len(base_texts)
    texts = base_texts * repeats

    model, tokenizer = load_model_and_tokenizer("bert-base-uncased", mesh)
    inputs = prepare_batch_input(texts, tokenizer, mesh)
    forward_fn = create_inference_fn(model, mesh)

    predictions = run_profiled_inference(model, forward_fn, inputs, mesh)
    predictions = jax.device_get(predictions)

    for text, probs in list(zip(texts, predictions))[:3]:
        predicted_class = int(jnp.argmax(probs))
        confidence = float(jnp.max(probs))
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
