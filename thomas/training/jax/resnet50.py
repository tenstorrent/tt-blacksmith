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

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import frozen_dict
import optax


class BottleneckBlock(nn.Module):
    in_channels: int
    out_channels: int
    stride: int = 1
    downsample: bool = False
    train: bool = True

    def setup(self):
        # 1x1 convolution to reduce dimensions
        self.conv1 = nn.Conv(
            features=self.out_channels // 4, kernel_size=(1, 1), strides=(self.stride, self.stride), use_bias=False
        )
        self.bn1 = nn.BatchNorm(use_running_average=not self.train)

        # 3x3 convolution for feature extraction
        self.conv2 = nn.Conv(
            features=self.out_channels // 4, kernel_size=(3, 3), padding=(1, 1), strides=1, use_bias=False
        )
        self.bn2 = nn.BatchNorm(use_running_average=not self.train)

        # 1x1 convolution to increase the dimensions
        self.conv3 = nn.Conv(features=self.out_channels, kernel_size=(1, 1), use_bias=False)
        self.bn3 = nn.BatchNorm(use_running_average=not self.train)

        # Downsample if needed (when in_channels != out_channels or stride != 1)
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                [
                    nn.Conv(
                        features=self.out_channels,
                        kernel_size=(1, 1),
                        strides=(self.stride, self.stride),
                        use_bias=False,
                    ),
                    nn.BatchNorm(use_running_average=not self.train),
                ]
            )
        else:
            self.downsample_layer = None

    def __call__(self, x):
        identity = x

        # First convolution (1x1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)

        # Second convolution (3x3)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.relu(x)

        # Third convolution (1x1)
        x = self.conv3(x)
        x = self.bn3(x)

        # If downsampling is required, adjust the identity
        if self.downsample:
            identity = self.downsample_layer(identity)

        # Add the residual connection
        x += identity
        x = nn.relu(x)

        return x


class ResNet50(nn.Module):
    num_classes: int = 10
    train: bool = True

    def setup(self):
        momentum = 0.9
        epsilon = 1e-5
        use_bias = False

        # Initial convolution
        self.conv1 = nn.Conv(
            features=64, kernel_size=(7, 7), strides=2, padding="SAME", use_bias=use_bias, name="conv1"
        )
        self.bn1 = nn.BatchNorm(momentum=momentum, epsilon=epsilon, use_running_average=not self.train, name="bn1")
        self.relu = nn.relu
        self.maxpool = nn.max_pool

        # Stage 1
        self.layer1 = self._make_layer(in_channels=64, out_channels=256, num_blocks=3, stride=1)

        # Stage 2
        self.layer2 = self._make_layer(in_channels=256, out_channels=512, num_blocks=4, stride=2)

        # Stage 3
        self.layer3 = self._make_layer(in_channels=512, out_channels=1024, num_blocks=6, stride=2)

        # Stage 4
        self.layer4 = self._make_layer(in_channels=1024, out_channels=2048, num_blocks=3, stride=2)

        # Global Average Pooling
        self.global_avg_pool = lambda x: jnp.mean(x, axis=(1, 2))

        # Fully connected layer
        self.fc = nn.Dense(self.num_classes, dtype=jnp.float32, name="fc")

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block requires downsampling if stride is not 1 or channels differ
        layers.append(
            BottleneckBlock(
                in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=True, train=self.train
            )
        )

        # Remaining blocks are regular residual blocks
        for _ in range(1, num_blocks):
            layers.append(
                BottleneckBlock(
                    in_channels=out_channels, out_channels=out_channels, stride=1, downsample=False, train=self.train
                )
            )

        return layers

    def __call__(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Stage 1
        for block in self.layer1:
            x = block(x)

        # Stage 2
        for block in self.layer2:
            x = block(x)

        # Stage 3
        for block in self.layer3:
            x = block(x)

        # Stage 4
        for block in self.layer4:
            x = block(x)

        # Global average pooling
        x = self.global_avg_pool(x)

        # Fully connected layer
        x = self.fc(x)

        return x


# Example usage
def create_resnet50_model():
    model = ResNet50(num_classes=10)
    return model


# Define the forward pass
@partial(jax.jit, static_argnums=(3,))
def forward_pass(params, batch_stats, x, train: bool):
    def apply_fn(x):
        return ResNet50().apply({"params": params, "batch_stats": batch_stats}, x, mutable=["batch_stats"])

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
model = create_resnet50_model()
params = model.init(rng, jnp.ones(input_shape))["params"]
batch_stats = model.init(rng, jnp.ones(input_shape))["batch_stats"]
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
    # test_images = test_images.reshape((len(test_images), 32, 32, 3))
    # val_loss, val_accuracy = eval_step(state.params, batch_stats, test_images, test_labels)
    # print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test
# test_loss, test_accuracy = eval_step(state.params, batch_stats, test_images, test_labels)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Export the forward and backward passes to StableHLO
exported_forward_tr: export.Exported = export.export(jax.jit(forward_pass, static_argnums=(3,)))(
    state.params, batch_stats, jnp.ones(input_shape), True
)
print("Forward Pass StableHLO (Training):")
print(exported_forward_tr.mlir_module())

# exported_forward_tst: export.Exported = export.export(jax.jit(forward_pass, static_argnums=(3,)))(state.params, batch_stats, jnp.ones(input_shape), False)
# print("Forward Pass StableHLO (Inference):")
# print(exported_forward_tst.mlir_module())

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
# operations_forward_tst = re.findall(pattern, exported_forward_tst.mlir_module())
# unique_operations_forward_tst = sorted(set(operations_forward_tst))
# print("Unique Operations in Forward Pass (Inference):", ", ".join(unique_operations_forward_tst))

# Extract and print unique operations for backward pass
operations_backward = re.findall(pattern, exported_backward.mlir_module())
unique_operations_backward = sorted(set(operations_backward))
print("Unique Operations in Backward Pass:", ", ".join(unique_operations_backward))
