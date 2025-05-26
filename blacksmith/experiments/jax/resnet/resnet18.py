import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import optax
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import flax
import flax.linen as nn
from flax.training import train_state
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10 as cifar10_dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

from blacksmith.tools.jax_utils import init_device

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 0.1
NUM_EPOCHS = 30
NUM_CLASSES = 10  # For CIFAR-10

# Define ResNet-18 model using Flax for CIFAR-10
class ResidualBlock(nn.Module):
    """A ResNet residual block."""
    filters: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        
        # First conv + BN + ReLU
        y = nn.Conv(self.filters, kernel_size=(3, 3), strides=(self.stride, self.stride), 
                   padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        
        # Second conv + BN
        y = nn.Conv(self.filters, kernel_size=(3, 3), strides=(1, 1), 
                   padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        
        # Shortcut connection
        if self.stride > 1 or x.shape[-1] != self.filters:
            residual = nn.Conv(self.filters, kernel_size=(1, 1), 
                              strides=(self.stride, self.stride), padding='SAME', 
                              use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)
            
        # Add residual connection and ReLU
        y = y + residual
        return nn.relu(y)

class ResNet18ForCIFAR10(nn.Module):
    """ResNet-18 implementation in Flax for CIFAR-10."""
    num_classes: int = NUM_CLASSES
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Reshape flat CIFAR10 images to 2D (32x32x3)
        x = x.reshape(-1, 32, 32, 3)
        
        # Create mutable batch_stats collection for batch normalization
        batch_stats_collection = 'batch_stats' if train else None
        mutable = ['batch_stats'] if train else False
        
        # Initial convolution - adapted for CIFAR-10
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # No max pooling to preserve spatial dimensions for small CIFAR-10 images
        
        # Residual blocks
        # Layer 1
        x = ResidualBlock(64)(x, train)
        x = ResidualBlock(64)(x, train)
        
        # Layer 2
        x = ResidualBlock(128, stride=2)(x, train)
        x = ResidualBlock(128)(x, train)
        
        # Layer 3
        x = ResidualBlock(256, stride=2)(x, train)
        x = ResidualBlock(256)(x, train)
        
        # Layer 4
        x = ResidualBlock(512, stride=2)(x, train)
        x = ResidualBlock(512)(x, train)
        
        # Global average pooling and FC layer
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        
        return x

# Load CIFAR-10 data function
def load_cifar10_jax():
    # Load CIFAR-10 data using torchvision
    cifar10 = {
        "train": torchvision.datasets.CIFAR10("./data", train=True, download=True),
        "test": torchvision.datasets.CIFAR10("./data", train=False, download=True),
    }
    ds = {}
    for split in ["train", "test"]:
        # Get the images and labels
        images = cifar10[split].data
        labels = np.array(cifar10[split].targets)
        
        # Convert to JAX arrays and normalize images to [0, 1]
        images = jnp.float32(images) / 255.0
        labels = jnp.int16(labels)
        
        # Transpose images from (N, H, W, C) to (N, H, W, C) (no change needed for CIFAR-10)
        # For CIFAR-10, images are already in the correct format: (N, 32, 32, 3)
        
        # Flatten images for input to the model
        flat_images = images.reshape(-1, 32 * 32 * 3)
        
        ds[split] = {"image": flat_images, "label": labels}
    
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
    val_images = train_images[train_size:train_size + val_size]
    val_labels = train_labels[train_size:train_size + val_size]
    train_images = train_images[:train_size]
    train_labels = train_labels[:train_size]
    
    # Return all datasets
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# Define cross entropy loss (instead of MSE)
def cross_entropy_loss(logits, labels):
    """Cross-entropy loss function."""
    # Get the log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Compute the cross-entropy loss
    return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))

def mse_loss(logits, labels):
    """Mean squared error loss function."""
    return jnp.mean((logits - labels) ** 2)

def accuracy(logits, labels):
    """Calculate accuracy."""
    predicted_class = jnp.argmax(logits, axis=-1)
    true_class = jnp.argmax(labels, axis=-1)
    return jnp.mean(predicted_class == true_class)

# Create and initialize the model
def create_model(rng):
    """Create and initialize the ResNet-18 model."""
    model = ResNet18ForCIFAR10(num_classes=NUM_CLASSES)
    
    # Initialize the model with random parameters
    dummy_input = jnp.ones((1, 32 * 32 * 3))  # CIFAR-10 image shape (flattened)
    variables = model.init(rng, dummy_input, train=False)
    
    # Extract parameters and batch_stats
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    return model, params, batch_stats


# Training functions
@jax.jit
def train_step(state, batch_stats, batch):
    """Single training step."""
    images, labels = batch
    
    def loss_fn(params):
        logits, updated_batch_stats = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            images, 
            train=True,
            mutable=['batch_stats']
        )
        #loss = cross_entropy_loss(logits, labels)  # Using cross-entropy loss
        loss = mse_loss(logits, labels)
        return loss, (logits, updated_batch_stats)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updated_batch_stats)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy(logits, labels),
    }
    
    return state, updated_batch_stats, metrics

@jax.jit
def eval_step(state, batch_stats, batch):
    """Single evaluation step."""
    images, labels = batch
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': batch_stats}, 
        images, 
        train=False
    )
    
    metrics = {
        #'loss': cross_entropy_loss(logits, labels),  # Using cross-entropy loss
        'loss': mse_loss(logits, labels),
        'accuracy': accuracy(logits, labels),
    }
    
    return metrics

# Training loop
def train_model(train_images, train_labels, valid_images, valid_labels, rng_key):
    """Train the ResNet-18 model."""
    # Initialize model and state
    model = ResNet18ForCIFAR10(num_classes=NUM_CLASSES)
    
    with jax.default_device(jax.devices("cpu")[0]):
        dummy_input = jnp.ones((1, 32 * 32 * 3))  # CIFAR-10 image shape (flattened)
        variables_host = model.init(rng_key, dummy_input, train=False)

        # Extract parameters and batch_stats
        params = variables_host['params']
        batch_stats = variables_host['batch_stats']
        
        # Use SGD with momentum and weight decay for better training
        tx = optax.sgd(
            learning_rate=LEARNING_RATE,
            momentum=0.9,
            nesterov=True
        )
        
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )
    
    state = jax.device_put(state, jax.devices("tt")[0]) #ABC
    batch_stats = jax.device_put(batch_stats, jax.devices("tt")[0]) #ABC

    
    # Training loop
    steps_per_epoch = len(train_images) // BATCH_SIZE
    for epoch in range(NUM_EPOCHS):
        # Shuffle training data
        with jax.default_device(jax.devices("cpu")[0]):
            rng_key, shuffle_key = jax.random.split(rng_key)
            perm = jax.random.permutation(shuffle_key, len(train_images))
            train_images_shuffled = train_images[perm]
            train_labels_shuffled = train_labels[perm]
        
        # Training steps
        train_metrics = []
        for step in range(steps_per_epoch):

            with jax.default_device(jax.devices("cpu")[0]):
                start_idx = step * BATCH_SIZE
                end_idx = min((step + 1) * BATCH_SIZE, len(train_images))
                

                train_images_batch_host = train_images_shuffled[start_idx:end_idx]
                train_labels_batch_host = train_labels_shuffled[start_idx:end_idx]
            
            traing_images_batch = jax.device_put(train_images_batch_host, jax.devices("tt")[0]) #ABC
            train_labels_batch = jax.device_put(train_labels_batch_host, jax.devices("tt")[0]) #ABC

            batch = (
                    traing_images_batch,
                    train_labels_batch,
                )
            
            print(train_step.lower(state, batch_stats, batch).as_text())
            #import time
            #time.sleep(1000)
            
            state, batch_stats, metrics = train_step(state, batch_stats, batch)
            #metrics = eval_step(state, batch_stats, batch)
            for key, value in metrics.items():
                print(f"{key}: {value}")
            train_metrics.append(metrics)
            
        # Average training metrics
        train_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *train_metrics)
        
        # Evaluation steps
        eval_metrics = []
        eval_steps = len(valid_images) // BATCH_SIZE
        for step in range(eval_steps):
            start_idx = step * BATCH_SIZE
            end_idx = min((step + 1) * BATCH_SIZE, len(valid_images))
            
            batch = (
                valid_images[start_idx:end_idx],
                valid_labels[start_idx:end_idx],
            )
            
            metrics = eval_step(state.params, batch_stats, model.apply, batch)
            eval_metrics.append(metrics)
            
        # Average evaluation metrics
        eval_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *eval_metrics)
        
        # Print metrics
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Accuracy: {train_metrics['accuracy']:.4f}, "
              f"Val Loss: {eval_metrics['loss']:.4f}, "
              f"Val Accuracy: {eval_metrics['accuracy']:.4f}")
    
    return state, batch_stats

# Main execution
def main():
    with jax.default_device(jax.devices("cpu")[0]):
        print(f"JAX running on device: {jax.devices()[0]}")
    
        # Initialize random key
        rng_key = jax.random.PRNGKey(42)
    
        # Load CIFAR-10 data
        print("Loading and preparing CIFAR-10 data...")
        train_images_host, train_labels_host, val_images_host, val_labels_host, test_images_host, test_labels_host = load_cifar10_jax()
        
        print(f"Training data shape: {train_images_host.shape}, {train_labels_host.shape}")
        print(f"Validation data shape: {val_images_host.shape}, {val_labels_host.shape}")
        print(f"Test data shape: {test_images_host.shape}, {test_labels_host.shape}")
        
    # Train model
    print("Starting training...")
    final_state, final_batch_stats = train_model(train_images_host, train_labels_host, val_images_host, val_labels_host, rng_key)
        
    return final_state, final_batch_stats

if __name__ == "__main__":
    init_device()
    main()