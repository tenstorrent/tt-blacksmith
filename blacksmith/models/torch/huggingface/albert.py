# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from transformers import AlbertModel

from blacksmith.tools.templates.configs import TrainingConfig


class AlbertWithMLP(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained(config.model_name)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Simple MLP - NO dropout
        self.classifier = nn.Sequential(
            nn.Linear(768, config.mlp_hidden_dim), nn.GELU(), nn.Linear(config.mlp_hidden_dim, config.num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        return logits
