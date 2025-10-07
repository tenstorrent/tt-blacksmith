# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForSequenceClassification

# Generic model initialization function.
def init_model(model_name, num_labels=2, seed=None, device="tt"):
    # Initialize parameters on CPU (https://github.com/tenstorrent/tt-mlir/issues/979).
    with jax.default_device(jax.devices("cpu")[0]):
        kwargs = {
            "num_labels": num_labels,
            "dtype": jnp.bfloat16,
        }
        if seed is not None:
            kwargs["seed"] = seed

        model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)

    # Put parameters back to tt device.
    params = jax.device_put(model.params, jax.devices(device)[0])
    return model, params
