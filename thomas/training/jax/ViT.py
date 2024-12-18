# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
import math

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random

# Seeding for random operations
main_rng = random.PRNGKey(42)

import flax
from flax import linen as nn
from flax.training import train_state

import optax

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATASET_PATH = "/Users/umales/cifar10"

# Transformations applied on each image => bring them into a numpy array
DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - DATA_MEANS) / DATA_STD
    return img


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


test_transform = image_to_numpy
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        image_to_numpy,
    ]
)
# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

# Loading the test set
test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

# We define a set of data loaders that we can use for training and validation
train_loader = data.DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    collate_fn=numpy_collate,
    num_workers=8,
    persistent_workers=True,
)
val_loader = data.DataLoader(
    val_set,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    collate_fn=numpy_collate,
    num_workers=4,
    persistent_workers=True,
)
test_loader = data.DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    collate_fn=numpy_collate,
    num_workers=4,
    persistent_workers=True,
)


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, H', W', p_H, p_W, C]
    x = x.reshape(B, -1, *x.shape[3:])  # [B, H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.reshape(B, x.shape[1], -1)  # [B, H'*W', p_H*p_W*C]
    return x


class AttentionBlock(nn.Module):
    embed_dim: int = 256  # Dimensionality of input and attention feature vectors
    hidden_dim: int = 512  # Dimensionality of hidden layer in feed-forward network
    num_heads: int = 8  # Number of heads to use in the Multi-Head Attention block
    # dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            # nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim),
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        # self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, train=True):
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + attn_out

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + linear_out
        return x


class VisionTransformer(nn.Module):
    embed_dim: int = 256  # Dimensionality of input and attention feature vectors
    hidden_dim: int = 512  # Dimensionality of hidden layer in feed-forward network
    num_heads: int = 8  # Number of heads to use in the Multi-Head Attention block
    num_channels: int = 3  # Number of channels of the input (3 for RGB)
    num_layers: int = 6  # Number of layers to use in the Transformer
    num_classes: int = 10  # Number of classes to predict
    patch_size: int = 4  # Number of pixels that the patches have per dimension
    num_patches: int = 64  # Maximum number of patches an image can have
    # dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        # Layers/Networks
        self.input_layer = nn.Dense(256)  # HARD CODED 256
        self.transformer = [AttentionBlock(256, 512, 8)]  # HARD CODED 256, 512 and 8
        # self.dropout_prob) for _ in range(self.num_layers)]
        self.mlp_head = nn.Sequential([nn.LayerNorm(), nn.Dense(10)])  # HARD CODED 10
        # self.dropout = nn.Dropout(self.dropout_prob)

        # Parameters/Embeddings
        self.cls_token = self.param("cls_token", nn.initializers.normal(stddev=1.0), (1, 1, 256))  # HARD CODED 256
        self.pos_embedding = self.param(
            "pos_embedding", nn.initializers.normal(stddev=1.0), (1, 1 + 64, 256)
        )  # HARD CODED 64 and 256

    def __call__(self, x, train=True):
        # Preprocess input
        x = img_to_patch(x, 4)  # HARD CODED 4
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        # x = self.dropout(x, deterministic=not train)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)

        # Perform classification prediction
        cls = x[:, 0]
        out = self.mlp_head(cls)
        return out


# Function to calculate the classification loss and accuracy for a model
def calculate_loss(params, rng, batch, model_hparams, train):
    model_ = VisionTransformer(**model_hparams)
    imgs, labels = batch
    # rng, dropout_apply_rng = jax.random.split(rng)
    logits = model_.apply({"params": params}, imgs, train=train)
    # rngs={'dropout': dropout_apply_rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = (logits.argmax(axis=-1) == labels).mean()
    return loss, (acc, rng)


# Training function
@jax.jit
def train_step(
    state, rng, batch, embed_dim, hidden_dim, num_heads, num_layers, patch_size, num_channels, num_patches, num_classes
):
    model_hparams = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "patch_size": patch_size,
        "num_channels": num_channels,
        "num_patches": num_patches,
        "num_classes": num_classes,
    }
    loss_fn = lambda params: calculate_loss(params, rng, batch, model_hparams, train=True)
    # Get loss, gradients for loss, and other outputs of loss function
    (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Update parameters and batch statistics
    state = state.apply_gradients(grads=grads)
    return state, rng, loss, acc


train_step = jax.jit(
    train_step,
    static_argnames=[
        "embed_dim",
        "hidden_dim",
        "num_heads",
        "num_layers",
        "patch_size",
        "num_channels",
        "num_patches",
        "num_classes",
    ],
)

# Eval function
def eval_step(model, state, rng, batch, model_hparams_tuple):
    model_hparams = dict(model_hparams_tuple)
    # Return the accuracy for a single batch
    _, (acc, rng) = calculate_loss(state.params, rng, batch, model_hparams, train=False)
    return rng, acc


def init_model(model, exmp_imgs, rng, model_hparams, seed=42):
    # Initialize model
    model_ = model(**model_hparams)
    rng, init_rng = jax.random.split(rng, 2)
    init_params = model_.init({"params": init_rng}, exmp_imgs, train=True)["params"]
    return init_params, rng


def init_optimizer2(lr, weight_decay, num_epochs, num_steps_per_epoch):
    # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales={
            int(num_steps_per_epoch * num_epochs * 0.6): 0.1,
            int(num_steps_per_epoch * num_epochs * 0.85): 0.1,
        },
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=weight_decay)  # Clip gradients at norm 1
    )
    return optimizer


def init_optimizer(lr, weight_decay, num_epochs, num_steps_per_epoch):
    # Cosine decay schedule
    total_steps = num_epochs * num_steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, weight_decay=weight_decay)  # Clip gradients at norm 1
    )
    return optimizer


def train_model(
    exmp_imgs,
    model,
    train_loader,
    val_loader,
    test_loader,
    embed_dim,
    hidden_dim,
    num_heads,
    num_layers,
    patch_size,
    num_channels,
    num_patches,
    num_classes,
    lr,
):
    rng = jax.random.PRNGKey(0)

    model_hparams = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "patch_size": patch_size,
        "num_channels": num_channels,
        "num_patches": num_patches,
        "num_classes": num_classes,
    }

    init_params, rng = init_model(model, exmp_imgs, rng, model_hparams)
    optimizer = init_optimizer(lr, weight_decay=1e-4, num_epochs=200, num_steps_per_epoch=len(train_loader))
    state = train_state.TrainState.create(apply_fn=model.apply, params=init_params, tx=optimizer)

    best_eval = 0.0
    for epoch_idx in tqdm(range(1, 21)):
        metrics = {"loss": [], "acc": []}
        for batch in tqdm(train_loader, desc="Training", leave=False):
            state, rng, loss, acc = train_step(
                state,
                rng,
                batch,
                embed_dim,
                hidden_dim,
                num_heads,
                num_layers,
                patch_size,
                num_channels,
                num_patches,
                num_classes,
            )
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)

        if epoch_idx % 2 == 0:
            eval_acc = eval_model(model, state, rng, val_loader, tuple(model_hparams.items()))
            print(f"Epoch {epoch_idx}, Eval Acc: {eval_acc:.4f}")

    val_acc = eval_model(model, state, rng, val_loader, tuple(model_hparams.items()))
    test_acc = eval_model(model, state, rng, test_loader, tuple(model_hparams.items()))

    from jax import export
    import re

    exported_all: export.Exported = export.export(train_step)(
        state,
        rng,
        next(iter(train_loader)),
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        patch_size,
        num_channels,
        num_patches,
        num_classes,
    )
    print(exported_all.mlir_module())

    pattern = r"stablehlo\.(\w+)"
    ops = re.findall(pattern, exported_all.mlir_module())
    unique_ops = sorted(set(ops))
    print(unique_ops)
    print(len(unique_ops))

    return {"val": val_acc, "test": test_acc}


def eval_model(model, state, rng, data_loader, model_hparams):
    correct_class, count = 0, 0
    for batch in data_loader:
        rng, acc = eval_step(model, state, rng, batch, model_hparams)
        correct_class += acc * batch[0].shape[0]
        count += batch[0].shape[0]
    eval_acc = (correct_class / count).item()
    return eval_acc


# Example usage of the functions:

if __name__ == "__main__":
    # Load data, model parameters, etc.
    model = VisionTransformer  # Replace with your actual VisionTransformer model
    exmp_imgs = next(iter(train_loader))[0]

    embed_dim = 256
    hidden_dim = 512
    num_heads = 8
    num_layers = 6
    patch_size = 4
    num_channels = 3
    num_patches = 64
    num_classes = 10
    lr = 3e-4

    # Train and evaluate the model
    results = train_model(
        exmp_imgs=exmp_imgs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        patch_size=patch_size,
        num_channels=num_channels,
        num_patches=num_patches,
        num_classes=num_classes,
        lr=lr,
    )
    print("ViT results", results)
