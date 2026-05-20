# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

from blacksmith.datasets.torch.alpaca.alpaca_dataset import AlpacaDataset
from blacksmith.datasets.torch.banking77.banking77_dataset import Banking77Dataset
from blacksmith.datasets.torch.BOUNTIES.wikitext.wikitext_dataset import WikitextDataset
from blacksmith.datasets.torch.fusechat.fusechat_dataset import FuseChatDataset
from blacksmith.datasets.torch.mathpreference.math_preference_dataset import (
    MathDPODataset,
    MathSFTDataset,
)
from blacksmith.datasets.torch.metamathqa.metamathqa_dataset import MetaMathQADataset
from blacksmith.datasets.torch.mnist.mnist_dataset import MNISTDataset
from blacksmith.datasets.torch.squadV2.squadV2_dataset import SquadV2Dataset
from blacksmith.datasets.torch.sst2.sst2_dataset import SSTDataset
from blacksmith.datasets.torch.stanfordcars.stanfordcars_dataset import (
    StanfordCarsDataset,
)
from blacksmith.datasets.torch.text2sql.text2sql_dataset import TextToSQLDataset
from blacksmith.datasets.torch.wizardlm_evol.wizardlm_evol_dataset import WizardLMEvolDataset
from blacksmith.tools.templates.configs import TrainingConfig


class AvailableDataset(Enum):
    MNIST = "mnist"
    NERF = "nerf"
    SST2 = "sst2"
    TEXT2SQL = "text2sql"
    BANKING77 = "banking77"
    SQUADV2 = "squadv2"
    MATH_PREFERENCE_DPO = "math_preference_dpo"
    MATH_PREFERENCE_SFT = "math_preference_sft"  # Supervised fine-tuning on chosen responses (stage 1 of DPO pipeline)
    WIKITEXT = "wikitext"
    STANFORDCARS = "stanfordcars"
    FUSECHAT = "fusechat"
    ALPACA = "alpaca"
    METAMATHQA = "metamathqa"
    WIZARDLM_EVOL = "wizardlm_evol"


def get_dataset(config: TrainingConfig, split: str = "train", collate_fn=None):
    """Factory function to get the appropriate dataset based on the config"""
    dataset_id = config.dataset_id.lower()

    if dataset_id == AvailableDataset.MNIST.value:
        return MNISTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.NERF.value:
        # BlenderDataset requires kornia, which has problems with torch 2.7.0 version.
        from blacksmith.datasets.torch.nerf.blender import BlenderDataset

        return BlenderDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.SST2.value:
        return SSTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.TEXT2SQL.value:
        return TextToSQLDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.BANKING77.value:
        return Banking77Dataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.SQUADV2.value:
        return SquadV2Dataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.WIKITEXT.value:
        return WikitextDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.STANFORDCARS.value:
        return StanfordCarsDataset(config, split)
    elif dataset_id == AvailableDataset.FUSECHAT.value:
        return FuseChatDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.ALPACA.value:
        return AlpacaDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.METAMATHQA.value:
        return MetaMathQADataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.MATH_PREFERENCE_DPO.value:
        return MathDPODataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.MATH_PREFERENCE_SFT.value:
        return MathSFTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.WIZARDLM_EVOL.value:
        return WizardLMEvolDataset(config, split, collate_fn=collate_fn)
    else:
        available_datasets = [ds.value for ds in AvailableDataset]
        raise ValueError(f"Unsupported dataset: {dataset_id}. Available options are: {available_datasets}")
