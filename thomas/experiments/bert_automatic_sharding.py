# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForSequenceClassification, AutoTokenizer
import time
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import jax.export
import pathlib


def create_2d_mesh():
    devices = jax.devices()
    n_devices = len(devices)
    mesh_x = int(n_devices**0.5)
    mesh_y = n_devices // mesh_x
    device_mesh = mesh_utils.create_device_mesh((mesh_x, mesh_y))
    print(f"\nDevice mesh shape: {device_mesh.shape}")
    print(f"Available devices: {jax.devices()}")
    return device_mesh


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
    for k, v in params.items():
        if isinstance(v, dict):
            sharded_params[k] = shard_params_across_mesh(v, mesh)
        else:
            sharding = get_param_sharding_rule(k, v, mesh)
            print(f"Sharding {k} with shape {v.shape} using {sharding.spec}")
            sharded_params[k] = jax.device_put(v, sharding)
    return sharded_params


def load_model_and_tokenizer(model_name="bert-base-uncased", mesh=None):
    model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\nSharding model parameters across devices:")
    params = shard_params_across_mesh(model.params, mesh)
    return model, tokenizer, params


def prepare_batch_input(texts, tokenizer, mesh, max_length=128):
    n_devices_x = mesh.devices.shape[0]
    batch_size = len(texts)
    target_size = ((batch_size + n_devices_x - 1) // n_devices_x) * n_devices_x
    if batch_size < target_size:
        texts = texts + [texts[0]] * (target_size - batch_size)

    encoded = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="jax")

    batch_sharding = NamedSharding(mesh, P("data", None))
    print(f"\nInput sharding spec: {batch_sharding.spec}")
    print(f"Input shape before sharding: {encoded['input_ids'].shape}")

    input_ids = jax.device_put(encoded["input_ids"], batch_sharding)
    attention_mask = jax.device_put(encoded["attention_mask"], batch_sharding)

    print(f"Input device assignment: {input_ids.sharding.mesh.devices}")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def create_inference_fn(model):
    def forward(params, input_ids, attention_mask):
        print(f"\nInput tensor shape on device: {input_ids.shape}")
        # print(f"Current sharding spec: {input_ids.sharding.spec}")
        # print(f"Device assignment: {input_ids.sharding.mesh.devices}")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, params=params, train=False)
        return jax.nn.softmax(outputs.logits, axis=-1)

    return forward


def export_to_stablehlo(forward_fn, params, sample_inputs):
    fwd_jit = jax.jit(forward_fn)
    fwd_lowered = fwd_jit.lower(params, **sample_inputs)

    fwd_stablehlo = fwd_lowered.compile()
    print(fwd_lowered.as_text())
    # fwd_stablehlo = fwd_lowered.compiler_ir(dialect='stablehlo')
    # print(fwd_stablehlo.dump())


def run_profiled_inference(forward_fn, params, inputs, mesh, num_warmup=1, num_runs=5):
    run_times = []

    with mesh:
        for _ in range(num_warmup):
            _ = forward_fn(params, inputs["input_ids"], inputs["attention_mask"])

        for i in range(num_runs):
            start_time = time.time()
            predictions = forward_fn(params, inputs["input_ids"], inputs["attention_mask"])
            run_time = time.time() - start_time
            run_times.append(run_time)

    avg_time = sum(run_times) / len(run_times)
    print(f"Average batch time: {avg_time:.4f}s")

    return predictions


def main():
    device_mesh = create_2d_mesh()
    mesh = Mesh(device_mesh, ("data", "model"))

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

    model, tokenizer, params = load_model_and_tokenizer("bert-base-uncased", mesh)
    inputs = prepare_batch_input(texts, tokenizer, mesh)
    forward_fn = create_inference_fn(model)

    # jax.config.update("jax_use_shardy_partitioner", True)
    export_to_stablehlo(forward_fn, params, inputs)

    predictions = run_profiled_inference(forward_fn, params, inputs, mesh)
    predictions = jax.device_get(predictions)

    print("\nSample Results:")
    for text, probs in list(zip(texts, predictions))[:3]:
        predicted_class = int(jnp.argmax(probs))
        confidence = float(jnp.max(probs))
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
