# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from forge._C import DataFormat
from forge.module import ForgeModule
from forge.op.constant import Constant
from forge.op.eltwise_binary import Multiply
from forge.op.eltwise_unary import Cast, Log
from forge.op.nn import Softmax
from forge.op.reduce import ReduceAvg, ReduceSum


# Due to softmax issues, we are using a custom loss function
# This softmax uses dim=0 instead of dim=-1
class CrossEntropyLoss(ForgeModule):
    """
    Cross-Entropy Loss (with mean reduction)

    loss = reduce_avg(-1 * sum(labels * log(softmax(predictions)), dim=-1), dim=0)
    """

    def __init__(self, name: str, dtype=DataFormat.Float32):
        super().__init__(name)
        self.is_loss = True
        self.dtype = dtype

    def forward(self, predictions, labels):
        softmax = Softmax("softmax", predictions, dim=0)
        log_softmax = Log("log", softmax)

        product = Multiply("products", labels, log_softmax)
        log_loss = ReduceSum("log_loss", product, dim=0)

        negative_one_constant = Constant("negative_one_const", constant=-1.0, dtype=self.dtype)
        negative_log_loss = Multiply(
            "negative_log_loss",
            log_loss,
            negative_one_constant,
        )

        reduction_mean = ReduceAvg("reduction_mean", negative_log_loss, dim=-1)
        return reduction_mean
