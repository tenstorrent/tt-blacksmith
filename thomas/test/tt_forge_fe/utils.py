# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# Model definition
class MNISTLinear(nn.Module):
    def __init__(
        self, input_size=784, output_size=10, hidden_size=512, bias=True, dtype=torch.float32
    ):  # changed hidden_size to 512 because matmul 256 x batch_size is not supported in ttnn
        super(MNISTLinear, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=bias, dtype=dtype),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def load_tb_writer(model):
    """
    Load TensorBoard writer for logging
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/gradient_visualization/{model}/{current_time}/"
    writer = SummaryWriter(log_dir)

    return writer


def get_param_grads(named_params):
    return {name: param.grad.detach().clone() for name, param in named_params() if param.grad is not None}


def copy_params(src, dst):
    state_dict = src.state_dict()
    for name, param in dst.named_parameters():
        param.data = state_dict[name].data.detach().clone()

    dst.load_state_dict(state_dict)


def write_grads(writer, named_params, step):
    for name in named_params:
        writer.add_histogram(name, named_params[name].flatten().float(), step)


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, named_params, is_tt=False, verbose=False):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        pred = pred[0] if is_tt else pred

        y = nn.functional.one_hot(y, num_classes=10).to(pred.dtype)
        loss = loss_fn(pred, y)
        loss = loss[0] if is_tt else loss

        loss.backward()
        if is_tt:
            model.backward()

        yield loss, pred, get_param_grads(named_params)

        optimizer.step()
        optimizer.zero_grad()

        if verbose and batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"{'Forge' if is_tt else 'Torch'} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_loop(dataloader, model, loss_fn, batch_size, is_tt=False, verbose=False):
    size = len(dataloader.dataset)
    loss, accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred = pred[0] if is_tt else pred
            y = nn.functional.one_hot(y, num_classes=10).to(pred.dtype)
            loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    loss /= size
    accuracy /= size
    if verbose:
        print(
            f"{'Forge' if is_tt else 'Torch'} Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n"
        )
    return loss, accuracy
