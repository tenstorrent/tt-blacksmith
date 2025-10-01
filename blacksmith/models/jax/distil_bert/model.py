# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from transformers import (
    FlaxBertForSequenceClassification,
    FlaxDistilBertForSequenceClassification,
)


def init_teacher(model_name="textattack/bert-base-uncased-SST-2", num_labels=2, device="tt"):
    # Initialize parameters on CPU (https://github.com/tenstorrent/tt-mlir/issues/979).
    with jax.default_device(jax.devices("cpu")[0]):
        teacher = FlaxBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            dtype=jnp.bfloat16,
        )
    # Put parameters back to tt device.
    teacher_params = jax.device_put(teacher.params, jax.devices(device)[0])
    return teacher, teacher_params


def init_student(model_name="distilbert-base-uncased", num_labels=2, seed=42, device="tt"):
    # Initialize parameters on CPU (https://github.com/tenstorrent/tt-mlir/issues/979).
    with jax.default_device(jax.devices("cpu")[0]):
        student = FlaxDistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            dtype=jnp.bfloat16,
            seed=seed,
        )
    # Put parameters back to tt device.
    student_params = jax.device_put(student.params, jax.devices(device)[0])
    return student, student_params
