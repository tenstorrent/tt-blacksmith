# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import optax
from jax import random
from flax import linen as nn
from flax.training import train_state
from jax import export
import tensorflow_datasets as tfds
from functools import partial

# Load MNIST data
def load_mnist():
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_images, train_labels = train_ds["image"], train_ds["label"]
    test_images, test_labels = test_ds["image"], test_ds["label"]
    train_images = train_images[..., None] / 255.0
    test_images = test_images[..., None] / 255.0
    return train_images, train_labels, test_images, test_labels


train_images, train_labels, test_images, test_labels = load_mnist()

# Define the CNN model with batch normalization and fully connected layers
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.Dense(features=256)(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


# Define the forward pass
@partial(jax.jit, static_argnums=(3,))
def forward_pass(params, batch_stats, x, train: bool):
    def apply_fn(x):
        return CNN().apply({"params": params, "batch_stats": batch_stats}, x, train=train, mutable=["batch_stats"])

    return apply_fn(x)


# Define the loss function
def compute_loss(params, batch_stats, x, y, train: bool):
    logits, new_model_state = forward_pass(params, batch_stats, x, train)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    return loss, new_model_state


# Define the backward pass
@jax.jit
def backward_pass(params, batch_stats, x, y):
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, new_model_state), grads = grad_fn(params, batch_stats, x, y, True)
    return grads, loss, new_model_state


# Define the training step
@jax.jit
def train_step(state, batch_stats, x, y):
    grads, loss, new_model_state = backward_pass(state.params, batch_stats, x, y)
    state = state.apply_gradients(grads=grads)
    return state, new_model_state, loss


# Define the evaluation step
@jax.jit
def eval_step(params, batch_stats, x, y):
    logits, _ = forward_pass(params, batch_stats, x, False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
    return loss, accuracy


# Initialize the model and optimizer
rng = random.PRNGKey(0)
input_shape = (1, 28, 28, 1)
model = CNN()
params = model.init(rng, jnp.ones(input_shape), train=True)["params"]
batch_stats = model.init(rng, jnp.ones(input_shape), train=True)["batch_stats"]
tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training loop
num_epochs = 1
batch_size = 128
num_batches = len(train_images) // batch_size

for epoch in range(num_epochs):
    for i in range(num_batches):
        batch_images = train_images[i * batch_size : (i + 1) * batch_size]
        batch_labels = train_labels[i * batch_size : (i + 1) * batch_size]
        batch_images = batch_images.reshape((batch_size, 28, 28, 1))
        state, batch_stats, loss = train_step(state, batch_stats, batch_images, batch_labels)
        print(i)
        print(num_batches)

    # Validation
    test_images = test_images.reshape((len(test_images), 28, 28, 1))
    val_loss, val_accuracy = eval_step(state.params, batch_stats, test_images, test_labels)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test
test_loss, test_accuracy = eval_step(state.params, batch_stats, test_images, test_labels)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Export the forward and backward passes to StableHLO
exported_forward_tr: export.Exported = export.export(jax.jit(forward_pass, static_argnums=(3,)))(
    state.params, batch_stats, jnp.ones(input_shape), True
)
print("Forward Pass StableHLO (Training):")
print(exported_forward_tr.mlir_module())

exported_forward_tst: export.Exported = export.export(jax.jit(forward_pass, static_argnums=(3,)))(
    state.params, batch_stats, jnp.ones(input_shape), False
)
print("Forward Pass StableHLO (Inference):")
print(exported_forward_tst.mlir_module())

exported_backward: export.Exported = export.export(backward_pass)(
    state.params, batch_stats, jnp.ones(input_shape), jnp.ones((1,), dtype=jnp.int32)
)
print("Backward Pass StableHLO:")
print(exported_backward.mlir_module())

# Extract and print unique operations for forward pass (training)
import re

pattern = r"stablehlo\.(\w+)"
operations_forward_tr = re.findall(pattern, exported_forward_tr.mlir_module())
unique_operations_forward_tr = sorted(set(operations_forward_tr))
print("Unique Operations in Forward Pass (Training):", ", ".join(unique_operations_forward_tr))

# Extract and print unique operations for forward pass (inference)
operations_forward_tst = re.findall(pattern, exported_forward_tst.mlir_module())
unique_operations_forward_tst = sorted(set(operations_forward_tst))
print("Unique Operations in Forward Pass (Inference):", ", ".join(unique_operations_forward_tst))

# Extract and print unique operations for backward pass
operations_backward = re.findall(pattern, exported_backward.mlir_module())
unique_operations_backward = sorted(set(operations_backward))
print("Unique Operations in Backward Pass:", ", ".join(unique_operations_backward))
