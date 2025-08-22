# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import wandb
from tqdm import tqdm
from third_party.tt_forge_models.alexnet.pytorch.loader import ModelLoader

from blacksmith.experiments.torch.alexnet.configs import TrainingConfig
from blacksmith.tools.cli import generate_config


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
                    filename = parts[0]
                    class_name = parts[1]
                    self.annotations.append((filename, class_name))
        
        unique_classes = sorted(set(class_name for _, class_name in self.annotations))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        filename, class_name = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.val_dir, "images", filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        
        return image, label


def create_dataloaders(config: TrainingConfig):
    print("Setting up dataloaders...")
    
    transform_train = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(f"{config.data_path}/train", transform=transform_train)
    val_annotations_file = f"{config.data_path}/val/val_annotations.txt"
    val_dataset = TinyImageNetValDataset(
        val_dir=f"{config.data_path}/val",
        annotations_file=val_annotations_file,
        transform=transform_val
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def setup_model_and_optimizer(config: TrainingConfig):
    print("Setting up model, loss function and optimizer...")
    
    model_loader = ModelLoader(variant=config.model_variant)
    model = model_loader.load_model()
    if hasattr(model, 'classifier') and hasattr(model.classifier, '6'):
        if model.classifier[6].out_features != config.num_classes:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, config.num_classes)
    
    if config.dtype == "torch.float16":
        model = model.half()
    elif config.dtype == "torch.bfloat16":
        model = model.to(torch.bfloat16)
    
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    if config.optim.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optim.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optim}")

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, loss_fn, optimizer


def train_epoch(model, train_loader, loss_fn, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
        })
        
        if batch_idx % config.logging_steps == 0:
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": 100 * correct_predictions / total_samples,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, loss_fn, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct_predictions / total_samples
    
    return val_loss, val_acc


def train(config: TrainingConfig):
    print("Starting AlexNet training...")
    
    torch.manual_seed(config.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    train_loader, val_loader = create_dataloaders(config)
    model, loss_fn, optimizer = setup_model_and_optimizer(config)
    model = model.to(device)
    
    if config.report_to == "wandb":
        wandb.init(project=config.wandb_project, config=vars(config))
        wandb.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)
    
    best_val_acc = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 50)
        
        # Train
        if config.do_train:
            train_loss, train_acc = train_epoch(
                model, train_loader, loss_fn, optimizer, device, config
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        if config.do_eval:
            val_loss, val_acc = validate(model, val_loader, loss_fn, device)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Log to wandb
            if config.report_to == "wandb":
                wandb.log({
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': vars(config)
                }, best_model_path)
                print(f"New best model saved with val_acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if config.save_strategy == "epoch":
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss if config.do_train else None,
                'val_acc': val_acc if config.do_eval else None,
                'config': vars(config)
            }, checkpoint_path)
            
            # Keep only the last N checkpoints
            checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.startswith("checkpoint_epoch_")])
            if len(checkpoints) > config.save_total_limit:
                for old_checkpoint in checkpoints[:-config.save_total_limit]:
                    os.remove(os.path.join(checkpoints_dir, old_checkpoint))
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'best_val_acc': best_val_acc
    }, final_model_path)
    
    if config.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_alexnet_training.yaml")
    config = generate_config(TrainingConfig, config_file_path)
    
    train(config) 