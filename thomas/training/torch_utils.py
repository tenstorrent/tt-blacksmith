# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
def get_param_grads(named_params):
    return {name: param.grad.detach().clone() for name, param in named_params() if param.grad is not None}


def copy_params(src, dst):
    state_dict = src.state_dict()
    for name, param in dst.named_parameters():
        param.data = state_dict[name].data.detach().clone()

    dst.load_state_dict(state_dict)
