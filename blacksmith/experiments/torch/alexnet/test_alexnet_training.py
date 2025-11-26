# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import traceback
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from blacksmith.tools.cli import generate_config
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_xla_utils import setup_tt_environment
from third_party.tt_forge_models.alexnet.pytorch.loader import ModelLoader, ModelVariant
from blacksmith.experiments.torch.alexnet.configs import TrainingConfig


class TinyImageNetValDataset(Dataset):
    """Dataset for Tiny ImageNet validation set"""
    def __init__(self, val_dir, annotations_file, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        
        # Read annotations
        self.annotations = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.annotations.append((parts[0], parts[1]))
        
        unique_classes = sorted(set(cls for _, cls in self.annotations))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        filename, class_name = self.annotations[idx]
        img_path = os.path.join(self.val_dir, "images", filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label


def create_dataloaders(config: TrainingConfig):
    transform_train = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(config.data_path, "train"), transform=transform_train)
    val_dataset = TinyImageNetValDataset(
        val_dir=os.path.join(config.data_path, "val"),
        annotations_file=os.path.join(config.data_path, "val/val_annotations.txt"),
        transform=transform_val
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader


def setup_model_and_optimizer(config: TrainingConfig, device: torch.device):
    model_loader = ModelLoader(variant=ModelVariant(config.model_name))
    model = model_loader.load_model()
    if hasattr(model, 'classifier') and hasattr(model.classifier, '6'):
        if model.classifier[6].out_features != config.output_size:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, config.output_size)
    model = model.to(torch.float32).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)

    return model, loss_fn, optimizer


def validate(model: nn.Module, val_loader: DataLoader, device: torch.device,
             logger: TrainingLogger, config: TrainingConfig, loss_fn: nn.Module) -> Tuple[float, float]:
    logger.info("Starting validation...")
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    logger.info(f"Validation finished. Avg loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def train(config: TrainingConfig, device: torch.device, logger: TrainingLogger,
          checkpoint_manager: CheckpointManager):
    logger.info("Starting AlexNet training...")
    torch.manual_seed(config.seed)

    train_loader, val_loader = create_dataloaders(config)
    model, loss_fn, optimizer = setup_model_and_optimizer(config, device)

    best_val_acc = 0.0
    global_step = 0
    running_loss = 0.0

    try:
        model.train()
        for epoch in range(config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                torch_xla.sync(wait=True)

                running_loss += loss.item()
                global_step += 1

                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    running_loss = 0.0
                    val_loss, val_acc = validate(model, val_loader, device, logger, config, loss_fn)
                    logger.log_metrics({
                        "train/loss": avg_loss,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc
                    }, step=global_step)
                    model.train()
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Training failed: {e}", tb)
        raise
    finally:
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            model, global_step, config.num_epochs - 1, optimizer, checkpoint_name="final_model.pth"
        )
        logger.log_artifact(final_checkpoint_path, artifact_type="model", name="final_model.pth")
        logger.info("Training finished successfully.")
        logger.finish()


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_alexnet_training.yaml")
    config: TrainingConfig = generate_config(TrainingConfig, config_file_path)

    if config.use_tt:
        setup_tt_environment(config)
        device = torch_xla.device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    logger = TrainingLogger(config)
    checkpoint_manager = CheckpointManager(config, logger)

    train(config, device, logger, checkpoint_manager)
