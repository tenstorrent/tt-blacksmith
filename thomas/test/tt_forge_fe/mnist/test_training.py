# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import os

import torch
from torch import nn

import forge
from forge.verify.compare import compare_with_golden
from forge.op.loss import CrossEntropyLoss
from forge.tensor import to_forge_tensors

from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.models.config import MNISTLinearConfig
from thomas.models.torch.mnist_linear import MNISTLinear
from thomas.tooling.data import load_dataset, DataLoadingConfig
from thomas.training.torch_utils import copy_params, get_param_grads
from thomas.training.pytorch_train.trainer import PyTorchTrainer
from thomas.tooling.config import DataLoadingConfig
from thomas.training.early_stopping import EarlyStopping
from thomas.models.types import Device

from thomas.test.tt_forge_fe.utils import load_tb_writer, train_loop, validation_loop


@pytest.mark.parametrize("freeze_layer", [None, 0, 2, 4])
def test_forge_vs_torch_gradients(freeze_layer):
    disable_forge_logger()
    torch.manual_seed(0)
    batch_size = 64

    dtype = torch.float32
    torch.set_printoptions(precision=10)

    in_features = 28 * 28
    out_features = 10
    hidden_size = 512

    model_config = MNISTLinearConfig(
        batch_size=batch_size, bias=True, input_size=in_features, output_size=out_features, hidden_size=hidden_size
    )
    torch_model = MNISTLinear(model_config)

    forge_model = MNISTLinear(model_config)

    copy_params(torch_model, forge_model)

    if freeze_layer is not None:
        forge_model.linear_relu_stack[freeze_layer].weight.requires_grad = False
        forge_model.linear_relu_stack[freeze_layer].bias.requires_grad = False
        torch_model.linear_relu_stack[freeze_layer].weight.requires_grad = False
        torch_model.linear_relu_stack[freeze_layer].bias.requires_grad = False

    loss_fn = nn.CrossEntropyLoss()

    sample_inputs = [torch.ones(batch_size, in_features, dtype=dtype)]

    tt_model = forge.compile(forge_model, sample_inputs=sample_inputs, training=True)

    X = torch.ones(batch_size, in_features, dtype=dtype)
    y = torch.zeros(batch_size, out_features, dtype=dtype)

    torch_pred = torch_model(X)
    torch_loss = loss_fn(torch_pred, y)
    torch_loss.backward()
    torch_grads = get_param_grads(torch_model.named_parameters)

    X = torch.ones(batch_size, in_features, dtype=dtype)
    y = torch.zeros(batch_size, out_features, dtype=dtype)

    forge_pred = tt_model(X)[0]
    forge_loss = loss_fn(forge_pred, y)
    forge_loss.backward()
    tt_model.backward()
    forge_grads = get_param_grads(forge_model.named_parameters)

    if freeze_layer is not None:
        assert forge_model.linear_relu_stack[freeze_layer].weight.grad is None
        assert forge_model.linear_relu_stack[freeze_layer].bias.grad is None

    # Compare gradients for each parameter
    for name in reversed(list(torch_grads.keys())):
        assert compare_with_golden(torch_grads[name], forge_grads[name])


# For bfloat16, the following line should be added to the test_forge_vs_torch function:
# In file forge/forge/op/eval/forge/eltwise_unary.py:418 should be replaced with: threshold_tensor = ac.tensor(torch.zeros(shape, dtype=torch.bfloat16) + threshold)
# That sets relu threshold to bfloat16 tensor.
# And in file forge/forge/compile.py::compile_main forced bfloat 16 should be added compiler_cfg.default_df_override = DataFormat.Float16_b
# @pytest.mark.skip(reason="Need to be tested with bfloat16 and takes around 10 minutes to run")
def test_forge_vs_torch():
    torch.manual_seed(0)

    batch_size = 64
    learning_rate = 1e-2
    epochs = 10
    verbose = True
    in_features = 28 * 28
    out_features = 10
    hidden_size = 512

    dtype = "float32"

    data_config = DataLoadingConfig(batch_size=batch_size, dtype=dtype, pre_shuffle=False)
    model_config = MNISTLinearConfig(
        batch_size=batch_size, bias=True, input_size=in_features, output_size=out_features, hidden_size=hidden_size
    )
    torch_model = MNISTLinear(model_config)
    forge_model = MNISTLinear(model_config)

    copy_params(torch_model, forge_model)

    torch_writer = load_tb_writer("torch")
    forge_writer = load_tb_writer("forge")

    loss_fn = nn.CrossEntropyLoss()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    forge_optimizer = torch.optim.SGD(forge_model.parameters(), lr=learning_rate)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    tt_model = forge.compile(forge_model, sample_inputs=[torch.ones(batch_size, 784, dtype=torch_dtype)], training=True)

    test_loader, train_loader = load_dataset(data_config)
    step = 0

    early_stop = EarlyStopping(patience=1, mode="max")

    disable_forge_logger()
    for i in range(epochs):
        start_time = time.time()
        torch_loop = train_loop(
            train_loader,
            torch_model,
            loss_fn,
            torch_optimizer,
            batch_size,
            torch_model.named_parameters,
            is_tt=False,
            verbose=verbose,
        )
        forge_loop = train_loop(
            train_loader,
            tt_model,
            loss_fn,
            forge_optimizer,
            batch_size,
            forge_model.named_parameters,
            is_tt=True,
            verbose=verbose,
        )
        for torch_data, forge_data in zip(torch_loop, forge_loop):
            step += 1

            torch_loss, torch_pred, torch_grads = torch_data
            forge_loss, forge_pred, forge_grads = forge_data

            if step % 100 == 0:
                torch_val_loss, torch_val_acc = validation_loop(
                    test_loader, torch_model, loss_fn, batch_size, is_tt=False
                )
                forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)

                torch_writer.add_scalar("train_loss", torch_loss.float(), step)
                forge_writer.add_scalar("train_loss", forge_loss.float(), step)
                torch_writer.add_scalar("validation_acc", torch_val_acc, step)
                forge_writer.add_scalar("validation_acc", forge_val_acc, step)

                torch_writer.flush()
                forge_writer.flush()

        if verbose:
            print(f"Epoch {i} took {time.time() - start_time} seconds")

        forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)
        early_stop.step(forge_val_acc, i)

        if early_stop.is_best():
            torch.save(torch_model.state_dict(), f"runs/models/torch_model_{i}.pth")
            torch.save(forge_model.state_dict(), f"runs/models/forge_model_{i}.pth")

        if early_stop.is_early_stop():
            break

    # Load best model
    torch_model.load_state_dict(torch.load(f"runs/models/torch_model_{early_stop.get_best_model()}.pth"))
    forge_model.load_state_dict(torch.load(f"runs/models/forge_model_{early_stop.get_best_model()}.pth"))

    torch_val_loss, torch_val_acc = validation_loop(test_loader, torch_model, loss_fn, batch_size, is_tt=False)
    forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)

    print(f"Validation accuracy for Torch: {torch_val_acc} in epoch {early_stop.get_best_model()}")
    print(f"Validation accuracy for Forge: {forge_val_acc} in epoch {early_stop.get_best_model()}")


# Test for experiment reproducibility
# Should maybe add one more test that loads a checkpoint and continues training
def test_mnist_training(device_type=Device.cpu, checkpoint_path=None):
    torch.manual_seed(0)

    batch_size = 64
    learning_rate = 1e-2
    epochs = 10
    verbose = True
    in_features = 28 * 28
    out_features = 10
    hidden_size = 512

    dtype = "float32"

    data_config = DataLoadingConfig(batch_size=batch_size, dtype=dtype, pre_shuffle=False)
    model_config = MNISTLinearConfig(
        batch_size=batch_size, bias=True, input_size=in_features, output_size=out_features, hidden_size=hidden_size
    )
    framework_model = MNISTLinear(model_config)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    train_loader, test_loader = load_dataset(data_config)

    if device_type == Device.tt:
        # Compile model with Forge
        model = forge.compile(
            framework_model, sample_inputs=[torch.ones(batch_size, 784, dtype=torch_dtype)], training=True
        )
        loss_inputs = [
            torch.ones(batch_size, 10, dtype=torch_dtype).requires_grad_(True),
            torch.ones(batch_size, 10, dtype=torch_dtype),
        ]
        loss_inputs = to_forge_tensors(loss_inputs)
        forge_loss_fn = CrossEntropyLoss(name="cross_entropy_loss")
        loss_fn = forge.compile(forge_loss_fn, sample_inputs=loss_inputs, attach_to=model, training=True)
    else:
        model = framework_model

    step = 0
    start_epoch = 0

    # Should be a function
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        framework_model.load_state_dict(checkpoint["model"])
        train_loader.load_state_dict(checkpoint["generator_state"])
        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]

    # Initialize tensorboard writer
    writer = load_tb_writer("forge" if device_type == Device.tt else "torch")

    # Training loop with early stopping
    early_stop = EarlyStopping(patience=2, mode="max")

    # TODO: Maybe should not be hardcoded, also should point to /proj_sw/training_artifacts but not sure how to set unique name for each run
    model_path = f"runs/models/{'forge' if device_type == Device.tt else 'torch'}/model_step={{step}}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        # Training phase
        train = train_loop(
            train_loader,
            model,
            loss_fn,
            optimizer,
            batch_size,
            framework_model.named_parameters,
            is_tt=device_type == Device.tt,
        )
        for data in train:
            step += 1

            loss, pred, grads = data

            if step % 100 == 0:
                torch_val_loss, torch_val_acc = validation_loop(
                    test_loader, model, loss_fn, batch_size, is_tt=device_type == Device.tt
                )

                writer.add_scalar("train_loss", loss.float(), step)
                writer.add_scalar("validation_acc", torch_val_acc, step)

                writer.flush()
                torch.save(
                    {
                        "model": framework_model.state_dict(),
                        "generator_state": train_loader.state_dict(),
                        "step": step,
                        "epoch": epoch,
                    },
                    model_path.format(step=step),
                )

        # Validation phase
        val_loss, val_acc = validation_loop(test_loader, model, loss_fn, batch_size, is_tt=device_type == Device.tt)

        # Log epoch metrics
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_accuracy", val_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {time.time() - start_time:.2f}s")
        print("-" * 60)

        # Early stopping
        early_stop.step(val_acc, epoch)
        if early_stop.is_early_stop():
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    writer.close()
