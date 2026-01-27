# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import load_dataset

DATASET_PATH = "tanganke/stanford_cars"


class StanfordCarsDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train"):
        """
        Args:
            config: TrainingConfig (ensure config.dataset_id is set to "stanfordcars")
            split: Dataset split to use
        """
        self.dtype = eval(config.dtype)
        self.num_classes = 196

        super().__init__(config, split)

    def _get_transform_function(self):
        img_transform = transforms.Compose(
            [
                # Dataset contains grayscale images.
                transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                transforms.RandomResizedCrop(self.config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.image_mean,
                    std=self.config.image_std,
                ),
                transforms.Lambda(lambda x: x.to(self.dtype)),
            ]
        )
        label_transform = transforms.Compose(
            [
                transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long)),
            ]
        )

        return lambda batch: {
            "image": [img_transform(img) for img in batch["image"]],
            "label": [label_transform(label) for label in batch["label"]],
        }

    def _prepare_dataset(self):
        transform_function = self._get_transform_function()
        raw_dataset = load_dataset(DATASET_PATH, split=self.split)

        self.dataset = raw_dataset.with_transform(transform_function).shuffle(seed=self.config.seed)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def _get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.split == "train",
            drop_last=True,
        )
