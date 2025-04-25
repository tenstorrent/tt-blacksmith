# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST as mnist_dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch


def load_mnist_torch(dtype, batch_size):
    dtype = dtype
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten image
            transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
        ],
    )
    target_transform = transforms.Compose(
        [
            transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long)),
            transforms.Lambda(lambda y: F.one_hot(y, num_classes=10).to(dtype)),
        ]
    )

    train_dataset = mnist_dataset(
        root="data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    test_dataset = mnist_dataset(
        root="data",
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    # Drop last to ensure all batches are the same size as compiled model expects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


def load_mnist_jax():
    import jax
    import jax.numpy as jnp
    # Load MNIST data using torchvision
    mnist = {
        "train": torchvision.datasets.MNIST("./data", train=True, download=True),
        "test": torchvision.datasets.MNIST("./data", train=False, download=True),
    }

    ds = {}

    for split in ["train", "test"]:
        # Get the images and labels
        ds[split] = {"image": mnist[split].data.numpy(), "label": mnist[split].targets.numpy()}

        # Normalize the images by scaling them to [0, 1]
        ds[split]["image"] = jnp.float32(ds[split]["image"]) / 255

        # Cast labels to the appropriate type
        ds[split]["label"] = jnp.int16(ds[split]["label"])

        # Reshape images to (batch_size, 28 * 28)
        ds[split]["image"] = ds[split]["image"].reshape(-1, 28 * 28)

    # One-hot encode the labels
    train_images, train_labels = ds["train"]["image"], ds["train"]["label"]
    test_images, test_labels = ds["test"]["image"], ds["test"]["label"]

    train_labels = jax.nn.one_hot(train_labels, 10).astype(jnp.float32)
    test_labels = jax.nn.one_hot(test_labels, 10).astype(jnp.float32)

    # Shuffle the training data
    perm = jax.random.permutation(jax.random.PRNGKey(42), len(train_images))
    train_images, train_labels = train_images[perm], train_labels[perm]

    # Split the training data into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(train_images))
    val_size = len(train_images) - train_size

    train_images, val_images = train_images[:train_size], train_images[train_size : train_size + val_size]
    train_labels, val_labels = train_labels[:train_size], train_labels[train_size : train_size + val_size]

    # Return all datasets
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
