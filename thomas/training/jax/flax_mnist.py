import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import export

import optax

from flax import linen as nn
from flax.training import train_state
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)

from tensorflow import keras
import wandb
import os

from model import Models, MLP
from utils import ExportSHLO
from logg_it import init_wandb, log_metrics, save_checkpoint, load_checkpoint

from train_functions import forward_pass, forward_and_compute_loss, func_optax_loss, compute_loss_and_backward_pass, update_params, train_step, eval_step, calculate_metrics_train, calculate_metrics_val, accumulate_metrics

def load_mnist():
        
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    train_images = train_images[..., None] / 255.0
    test_images = test_images[..., None] / 255.0

    # Shuffle the training data
    perm = jax.random.permutation(jax.random.PRNGKey(0), len(train_images))
    train_images, train_labels = train_images[perm], train_labels[perm]

    # Split the training data into training and validation sets
    train_size = int(0.8 * len(train_images))
    val_size = int(0.2 * len(train_images))

    train_images, val_images = train_images[:train_size], train_images[train_size:train_size + val_size]
    train_labels, val_labels = train_labels[:train_size], train_labels[train_size:train_size + val_size]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels



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

def train(get_best_checkpoint=False, use_export_shlo = False):

    train_images, train_labels, eval_images, eval_labels, test_images, test_labels = load_mnist()

    config = init_wandb(project_name="Flax mnist mlp training", 
                        job_type="Flax mnist mlp training", 
                        dir_path='/proj_sw/user_dev/umales')

    config.learning_rate = 1e-3
    config.batch_size = 64
    config.num_epochs = 30
    config.seed = 0


    rng = random.PRNGKey(config.seed)
    input_shape = (1, 28, 28, 1)
    output_shape = jnp.ones((1, 10))
    pred_model = Models(model_type='MLP')
    params = pred_model.model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.sgd(learning_rate=config.learning_rate)
    state = train_state.TrainState.create(apply_fn=pred_model.model.apply, params=params, tx=tx)

    num_batches = len(train_images) // config.batch_size
    num_eval_batches = len(eval_images) // config.batch_size

    best_epoch = 0
    #early_stopping = EarlyStopping(patience=1)
    best_val_loss = 1e7
    for epoch in range(config.num_epochs):

        train_batch_metrics = []
        for i in range(num_batches):
            batch_images = train_images[i*config.batch_size:(i+1)*config.batch_size]
            batch_labels = train_labels[i*config.batch_size:(i+1)*config.batch_size]
            state, loss, grads = train_step(state,  batch_images, batch_labels)
            
            logits = eval_step(state.params, batch_images)
            metrics = calculate_metrics_train(logits, batch_labels, loss)
            train_batch_metrics.append(metrics)
        train_batch_metrics_avg = accumulate_metrics(train_batch_metrics)

        eval_batch_metrics = []
        for i in range(num_eval_batches):
            batch_images = eval_images[i*config.batch_size:(i+1)*config.batch_size]
            batch_labels = eval_labels[i*config.batch_size:(i+1)*config.batch_size]
            logits = eval_step(state.params, batch_images)
            metrics = calculate_metrics_val(logits, batch_labels)
            eval_batch_metrics.append(metrics)
        eval_batch_metrics_avg = accumulate_metrics(eval_batch_metrics)

        if(eval_batch_metrics_avg['loss'] < best_val_loss):
            best_val_loss = eval_batch_metrics_avg['loss']
            best_epoch = epoch

        log_metrics(grads, state, train_batch_metrics_avg['loss'], train_batch_metrics_avg['accuracy'], eval_batch_metrics_avg['loss'], eval_batch_metrics_avg['accuracy'], epoch)
        
        base_checkpoint_dir = f"/proj_sw/user_dev/umales/checkpoints/{wandb.run.name}"
        epoch_dir = f"epoch={epoch:02d}"
        checkpoint_dir = os.path.join(base_checkpoint_dir, epoch_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file_name = "checkpoint.msgpack"
        checkpoint_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
        save_checkpoint(checkpoint_file_path, state, epoch)
        
    if get_best_checkpoint:
        # For some reason, wandb is unable to load 2 most recent checkpoints
        # So, we load the best checkpoint and the two checkpoints before it
        # I suppose this is a bug in wandb, some sort of latency between saving and loading
        # becaue this pattern repeats no matter the number of epochs
        epoch = best_epoch - 2
        ckpt_file = "checkpoint.msgpack"
        restored_state = load_checkpoint(ckpt_file, state, epoch)
        logits = eval_step(restored_state.params, test_images)
        metrics = calculate_metrics_val(logits, test_labels)
        test_batch_metrics = []
        test_batch_metrics.append(metrics)
        test_batch_metrics_avg = accumulate_metrics(test_batch_metrics)
        wandb.log({"Test Loss": test_batch_metrics_avg["loss"], "Test Accuracy": test_batch_metrics_avg["accuracy"]})
    
    wandb.finish()


    if use_export_shlo:

        from utils import ExportSHLO

        export_it = ExportSHLO()
        #export_it.export_fwd_train_to_StableHLO_and_get_ops(forward_pass, state, input_shape, print_stablehlo=False)
        #export_it.export_fwd_tst_to_StableHLO_and_get_ops(eval_step, state, input_shape, print_stablehlo=False)
        #export_it.export_bwd_to_StableHLO_and_get_ops(backward_pass, state, input_shape, print_stablehlo=False)
        export_it.export_loss_to_StableHLO_and_get_ops(func_optax_loss, output_shape, print_stablehlo=False)
        #export_it.export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=False)

def main():
    train(get_best_checkpoint=True, use_export_shlo=True)

if __name__ == "__main__":
    main()