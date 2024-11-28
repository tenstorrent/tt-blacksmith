# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time

import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden

from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.models.torch.mnist_linear import MNISTLinear, ModelConfig
from thomas.tooling.data import load_dataset, DataLoadingConfig
from thomas.training.utils import EarlyStopping, get_param_grads, copy_params


@pytest.mark.push
def test_mnist_training():
    disable_forge_logger()
    torch.manual_seed(0)

    # Config
    num_epochs = 3
    batch_size = 1
    learning_rate = 0.001
    input_size = 28 * 28

    # Limit number of batches to run - quicker test
    limit_num_batches = 1000

    # Load dataset
    model_config = ModelConfig(batch_size=batch_size, bias=False)
    data_config = DataLoadingConfig(batch_size=batch_size, dtype="float32", pre_shuffle=True)

    test_loader, train_loader = load_dataset(data_config)

    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear(model_config)  # bias=False because batch_size=1 with bias=True is not supported

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_model = forge.compile(
        framework_model, sample_inputs=[torch.rand(batch_size, input_size)], loss=loss_fn, optimizer=framework_optimizer
    )

    disable_forge_logger
    for epoch_idx in range(num_epochs):
        # Reset gradients (every epoch) - since our batch size is currently 1,
        # we accumulate gradients across multiple batches (limit_num_batches),
        # and then run the optimizer.
        framework_optimizer.zero_grad()

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            pred = tt_model(data)[0]
            golden_pred = framework_model(data)
            assert compare_with_golden(golden_pred, pred, pcc=0.95)

            # Compute loss on CPU
            loss = loss_fn(pred, target)
            total_loss += loss.item()

            golden_loss = loss_fn(golden_pred, target)
            assert torch.allclose(loss, golden_loss, rtol=5e-2)  # 5% tolerance

            # Run backward pass on device
            loss.backward()

            tt_model.backward()

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")

        # Adjust weights (on CPU)
        framework_optimizer.step()

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).float()

        test_loss += loss_fn(pred, target)

        if batch_idx == 1000:
            break

    print(f"Test (total) loss: {test_loss}")


@pytest.mark.parametrize("freeze_layer", [None, 0, 2, 4])
@pytest.mark.push
def test_forge_vs_torch_gradients(freeze_layer):
    disable_forge_logger()
    torch.manual_seed(0)
    batch_size = 64

    dtype = torch.float32
    torch.set_printoptions(precision=10)

    in_features = 28 * 28
    out_features = 10

    model_config = ModelConfig(batch_size=batch_size, bias=True)
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

    tt_model = forge.compile(forge_model, sample_inputs=sample_inputs, loss=loss_fn)

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
