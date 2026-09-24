# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

from blacksmith.configs import TrainingConfig
from blacksmith.datasets.alpaca import AlpacaDataset
from blacksmith.datasets.banking77 import Banking77Dataset
from blacksmith.datasets.BOUNTIES.wikitext import WikitextDataset
from blacksmith.datasets.custom import CustomLLMDataset
from blacksmith.datasets.fusechat import FuseChatDataset
from blacksmith.datasets.gsm8k import GSM8KDataset
from blacksmith.datasets.math_preference import MathDPODataset, MathSFTDataset
from blacksmith.datasets.metamathqa import MetaMathQADataset
from blacksmith.datasets.squadV2 import SquadV2Dataset
from blacksmith.datasets.sst2 import SSTDataset
from blacksmith.datasets.text2sql import TextToSQLDataset
from blacksmith.datasets.wizardlm_evol import WizardLMEvolDataset


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
    GSM8K = "gsm8k"
    WIZARDLM_EVOL = "wizardlm_evol"
    CUSTOM = "custom"


def get_dataset(config: TrainingConfig, split: str = "train", collate_fn=None):
    """Factory function to get the appropriate dataset based on the config"""
    dataset_id = config.dataset_id.lower()

    if dataset_id == AvailableDataset.MNIST.value:
        # torchvision is not part of the crank env; import only when this dataset is used.
        from blacksmith.datasets.mnist import MNISTDataset

        return MNISTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.NERF.value:
        # BlenderDataset requires kornia, which has problems with torch 2.7.0 version.
        from blacksmith.datasets.nerf import BlenderDataset

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
        # torchvision is not part of the crank env; import only when this dataset is used.
        from blacksmith.datasets.stanfordcars import StanfordCarsDataset

        return StanfordCarsDataset(config, split)
    elif dataset_id == AvailableDataset.FUSECHAT.value:
        return FuseChatDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.ALPACA.value:
        return AlpacaDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.METAMATHQA.value:
        return MetaMathQADataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.GSM8K.value:
        return GSM8KDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.MATH_PREFERENCE_DPO.value:
        return MathDPODataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.MATH_PREFERENCE_SFT.value:
        return MathSFTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.WIZARDLM_EVOL.value:
        return WizardLMEvolDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.CUSTOM.value:
        return CustomLLMDataset(config, split, collate_fn=collate_fn)
    else:
        available_datasets = [ds.value for ds in AvailableDataset]
        raise ValueError(f"Unsupported dataset: {dataset_id}. Available options are: {available_datasets}")
