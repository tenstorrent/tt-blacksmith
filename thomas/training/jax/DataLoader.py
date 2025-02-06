# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from tensorflow import keras


def load_mnist():
    # Load data from MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    train_images = train_images
    train_labels = train_labels

    test_images = test_images
    test_labels = test_labels

    # Normalize and add channel dimension
    train_images = (train_images[..., None] / 255.0).astype(jnp.float32)
    test_images = (test_images[..., None] / 255.0).astype(jnp.float32)

    # Reshape images to (batch_size, 28 * 28)
    train_images = train_images.reshape(-1, 28 * 28).astype(jnp.float32)
    test_images = test_images.reshape(-1, 28 * 28).astype(jnp.float32)

    # One-hot encode the labels
    train_labels = jax.nn.one_hot(train_labels, 10).astype(jnp.float32)
    test_labels = jax.nn.one_hot(test_labels, 10).astype(jnp.float32)

    # Shuffle the training data
    perm = jax.random.permutation(jax.random.PRNGKey(0), len(train_images))
    train_images, train_labels = train_images[perm], train_labels[perm]

    # Split the training data into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(train_images))
    val_size = len(train_images) - train_size

    train_images, val_images = train_images[:train_size], train_images[train_size : train_size + val_size]
    train_labels, val_labels = train_labels[:train_size], train_labels[train_size : train_size + val_size]

    # Return all datasets
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
