import jax
from jax import random
from jax import export
import jax.numpy as jnp

import optax

from flax import linen as nn
from flax.training import train_state

from model import Models, MLP

import tensorflow_datasets as tfds

from utils import ExportSHLO

def load_mnist():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_images, train_labels = train_ds['image'], train_ds['label']
    test_images, test_labels = test_ds['image'], test_ds['label']
    train_images = train_images[..., None] / 255.0
    test_images = test_images[..., None] / 255.0

    # Shuffle the training dataset
    key_train = random.PRNGKey(0)
    perm_train = random.permutation(key_train, len(train_images))
    train_images = train_images[perm_train]
    train_labels = train_labels[perm_train]
    
    # Shuffle the test dataset
    key_test = random.PRNGKey(1)
    perm_test = random.permutation(key_test, len(test_images))
    test_images = test_images[perm_test]
    test_labels = test_labels[perm_test]

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_mnist()

class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        self.best_accuracy = 0
        self.counter = 0

    def __call__(self, val_accuracy):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=1)

@jax.jit
def forward_pass(params, x):
    return MLP().apply({'params': params}, x, mutable=['params'])

def compute_loss(params, x, y):
    logits, _ = forward_pass(params, x)
    loss = func_optax_loss(logits, y)
    return loss

@jax.jit
def func_optax_loss(logits, labels):
    #one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1]).astype(jnp.float32)
    return optax.losses.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()

@jax.jit
def backward_pass(params,  x, y):
    grad_fn = jax.value_and_grad(compute_loss, has_aux=False)
    loss, grads = grad_fn(params, x, y)
    return grads, loss

@jax.jit
def update_params(state, grads):
    return state.apply_gradients(grads=grads)

@jax.jit
def train_step(state,  x, y):
    grads, loss = backward_pass(state.params, x, y)
    state = update_params(state, grads)
    return state, loss, grads

@jax.jit
def eval_step(params, x):
    logits, _ = forward_pass(params, x)
    return logits

@jax.jit
def calculate_metrics(logits, y):
    loss = func_optax_loss(logits, y)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
    return loss, accuracy


rng = random.PRNGKey(0)
input_shape = (1, 28, 28, 1)
output_shape = jnp.ones((1, 10))
pred_model = Models(model_type='MLP')
params = pred_model.model.init(rng, jnp.ones(input_shape))['params']
tx = optax.sgd(learning_rate=1e-3)
state = train_state.TrainState.create(apply_fn=pred_model.model.apply, params=params, tx=tx)

num_epochs = 10
batch_size = 64
num_batches = len(train_images) // batch_size
for epoch in range(num_epochs):
    for i in range(num_batches):
        batch_images = train_images[i*batch_size:(i+1)*batch_size]
        batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
        state, loss, grads = train_step(state,  batch_images, batch_labels)
    logits = eval_step(state.params, test_images)
    val_loss, val_accuracy = calculate_metrics(logits, test_labels)
    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    if early_stopping(val_accuracy):
        print("Early stopping triggered")
        break

# Test
logits = eval_step(state.params, test_images)
test_loss, test_accuracy = calculate_metrics(logits, test_labels)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

export_it = ExportSHLO()
export_it.export_fwd_train_to_StableHLO_and_get_ops(forward_pass, state, input_shape, print_stablehlo=False)
export_it.export_fwd_tst_to_StableHLO_and_get_ops(eval_step, state, input_shape, print_stablehlo=False)
export_it.export_bwd_to_StableHLO_and_get_ops(backward_pass, state, input_shape, print_stablehlo=False)
export_it.export_loss_to_StableHLO_and_get_ops(func_optax_loss, output_shape, print_stablehlo=False)
export_it.export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=False)