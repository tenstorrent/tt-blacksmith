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


def load_cifar10():
    ds_builder = tfds.builder("cifar10")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_images, train_labels = train_ds["image"], train_ds["label"]
    test_images, test_labels = test_ds["image"], test_ds["label"]
    train_images = train_images[..., None] / 255.0
    test_images = test_images[..., None] / 255.0
    return train_images, train_labels, test_images, test_labels


train_images, train_labels, test_images, test_labels = load_cifar10()

train_images = train_images[:1000]
train_labels = train_labels[:1000]

test_images = test_images[:1000]
test_labels = test_labels[:1000]


# Define the CNN model with batch normalization and fully connected layers
class ResNet18(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        momentum = 0.8999999761581421
        epsilon = 0.000009999999747378752

        # why use_bias=false? See: [5]
        use_bias = False

        x = nn.Conv(
            features=64, kernel_size=(7, 7), padding=(3, 3), strides=2, use_bias=use_bias, name="resnetv15_conv0"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_batchnorm0",
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Stage 1
        res = x
        x = nn.Conv(
            features=64, kernel_size=(3, 3), padding=(1, 1), strides=1, use_bias=use_bias, name="resnetv15_stage1_conv0"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage1_batchnorm0",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64, kernel_size=(3, 3), padding=(1, 1), strides=1, use_bias=use_bias, name="resnetv15_stage1_conv1"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage1_batchnorm1",
        )(x)
        x = nn.relu(x + res)

        res = x
        x = nn.Conv(
            features=64, kernel_size=(3, 3), padding=(1, 1), strides=1, use_bias=use_bias, name="resnetv15_stage1_conv2"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage1_batchnorm2",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64, kernel_size=(3, 3), padding=(1, 1), strides=1, use_bias=use_bias, name="resnetv15_stage1_conv3"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage1_batchnorm3",
        )(x)
        x = nn.relu(x + res)

        # Stage 2
        res = x
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=2,
            use_bias=use_bias,
            name="resnetv15_stage2_conv0",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage2_batchnorm0",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage2_conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage2_batchnorm1",
        )(x)
        res = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            padding=(0, 0),
            strides=2,
            use_bias=use_bias,
            name="resnetv15_stage2_conv2",
        )(res)
        res = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage2_batchnorm2",
        )(res)
        x = nn.relu(x + res)

        res = x
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage2_conv3",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage2_batchnorm3",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage2_conv4",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage2_batchnorm4",
        )(x)
        x = nn.relu(x + res)

        # Stage 3
        res = x
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=2,
            use_bias=use_bias,
            name="resnetv15_stage3_conv0",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage3_batchnorm0",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage3_conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage3_batchnorm1",
        )(x)
        res = nn.Conv(
            features=256,
            kernel_size=(1, 1),
            padding=(0, 0),
            strides=2,
            use_bias=use_bias,
            name="resnetv15_stage3_conv2",
        )(res)
        res = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage3_batchnorm2",
        )(res)
        x = nn.relu(x + res)

        res = x
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage3_conv3",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage3_batchnorm3",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage3_conv4",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage3_batchnorm4",
        )(x)
        x = nn.relu(x + res)

        # Stage 4
        res = x
        x = nn.Conv(
            features=512,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=2,
            use_bias=use_bias,
            name="resnetv15_stage4_conv0",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage4_batchnorm0",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=512,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage4_conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage4_batchnorm1",
        )(x)
        res = nn.Conv(
            features=512,
            kernel_size=(1, 1),
            padding=(0, 0),
            strides=2,
            use_bias=use_bias,
            name="resnetv15_stage4_conv2",
        )(res)
        res = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage4_batchnorm2",
        )(res)
        x = nn.relu(x + res)

        res = x
        x = nn.Conv(
            features=512,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage4_conv3",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage4_batchnorm3",
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=512,
            kernel_size=(3, 3),
            padding=(1, 1),
            strides=1,
            use_bias=use_bias,
            name="resnetv15_stage4_conv4",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=momentum,
            epsilon=epsilon,
            dtype=jnp.float32,
            name="resnetv15_stage4_batchnorm4",
        )(x)
        x = nn.relu(x + res)

        # Global AVG pool
        x = jnp.mean(x, axis=(1, 2))

        # Flatten
        num_of_classes = 10  # TODO: get this value from constructor or args
        x = nn.Dense(num_of_classes, dtype=jnp.float32, name="resnetv15_dense0")(x)

        x = jnp.asarray(x, jnp.float32)

        return x


# Define the forward pass
@partial(jax.jit, static_argnums=(3,))
def forward_pass(params, batch_stats, x, train: bool):
    def apply_fn(x):
        return ResNet18().apply({"params": params, "batch_stats": batch_stats}, x, train=train, mutable=["batch_stats"])

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
input_shape = (1, 32, 32, 3)
model = ResNet18()
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
        batch_images = batch_images.reshape((batch_size, 32, 32, 3))
        state, batch_stats, loss = train_step(state, batch_stats, batch_images, batch_labels)
        print(i)
        print(num_batches)

    # Validation
    test_images = test_images.reshape((len(test_images), 32, 32, 3))
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
