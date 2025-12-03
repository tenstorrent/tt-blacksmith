# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Diffusers & Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# PEFT Imports
from peft import LoraConfig, get_peft_model, PeftModel

# --- Configuration ---
class TrainingConfig:
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    RESOLUTION = 1024
    TRAIN_BATCH_SIZE = 1 # Lowered batch size as SDXL requires significant VRAM
    LEARNING_RATE = 1e-4 # LoRA usually needs a higher LR than full finetuning (1e-5 -> 1e-4)
    NUM_EPOCHS = 5
    
    # LoRA Specifics
    LORA_RANK = 8
    LORA_ALPHA = 8 # Usually set equal to rank or rank/2
    
    # We define the global data type here
    DTYPE = torch.bfloat16 
    DATA_DIR = "data/sdxl-chalkboarddrawing-lora"
    OUTPUT_DIR = "output_lora"


class LocalDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_dir = config.DATA_DIR 
        
        # SDXL Transforms
        self.transform = transforms.Compose([
            transforms.Resize(
                config.RESOLUTION, 
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(config.RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])
        
        self.data_pairs = []
        self._prepare_dataset()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, prompt_path = self.data_pairs[idx]

        try:
            # 1. Load and process image
            img = Image.open(image_path).convert("RGB")
            pixel_values = self.transform(img)

            # 2. Load prompt
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                
            return {"pixel_values": pixel_values, "prompt": prompt}
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a fallback or handle appropriately in real training
            return self.__getitem__((idx + 1) % len(self))
    
    def _prepare_dataset(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Created {self.data_dir}. Please put images there.")
            return

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.data_dir, ext)))

        print(f"Found {len(image_files)} images in {self.data_dir}")

        for img_path in image_files:
            path_obj = Path(img_path)
            txt_path = path_obj.with_suffix('.txt')

            if txt_path.exists():
                self.data_pairs.append((str(img_path), str(txt_path)))
            else:
                print(f"Warning: No prompt found for {img_path}, skipping.")

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True, 
            num_workers=2, 
            drop_last=True,
        )


class SDXLTrainer:
    def __init__(self, device='cuda'):
        self.device = device
        self.config = TrainingConfig()
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
    def setup(self):
        self.load_models()
        self.load_tokenizers()
        self.setup_scheduler()
        self.setup_optimizer()

    def load_models(self):
        # 1. Load VAE (Frozen)
        self.vae = AutoencoderKL.from_pretrained(
            self.config.MODEL_ID, 
            subfolder="vae", 
            torch_dtype=self.config.DTYPE
        ).to(self.device)
        self.vae.requires_grad_(False)

        # 2. Load Text Encoders (Frozen)
        self.text_encoder_1 = CLIPTextModel.from_pretrained(
            self.config.MODEL_ID, 
            subfolder="text_encoder", 
            torch_dtype=self.config.DTYPE
        ).to(self.device)
        
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.MODEL_ID, 
            subfolder="text_encoder_2", 
            torch_dtype=self.config.DTYPE
        ).to(self.device)
        
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # 3. Load UNet and Config LoRA
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.MODEL_ID, 
            subfolder="unet", 
            torch_dtype=self.config.DTYPE
        ).to(self.device)
        
        # FREEZE base UNet parameters
        self.unet.requires_grad_(False)
        
        # Enable gradient checkpointing to save memory
        self.unet.enable_gradient_checkpointing()

        # Define LoRA Config
        lora_config = LoraConfig(
            r=self.config.LORA_RANK,
            lora_alpha=self.config.LORA_ALPHA,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Standard attention targets for SDXL
            bias="none"
        )

        # Wrap UNet with PEFT
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Ensure model is in training mode
        self.unet.train()

    def load_tokenizers(self):
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(self.config.MODEL_ID, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.config.MODEL_ID, subfolder="tokenizer_2")

    def setup_scheduler(self):
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.MODEL_ID, subfolder="scheduler")

    def setup_optimizer(self):
        # PEFT model automatically handles `requires_grad`. 
        # Only LoRA layers have requires_grad=True.
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )

    def compute_text_embeddings(self, prompts):
        with torch.no_grad():
            tokenizers = [self.tokenizer_1, self.tokenizer_2]
            text_encoders = [self.text_encoder_1, self.text_encoder_2]
            prompt_embeds_list = []
            
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
                )
                input_ids = text_inputs.input_ids.to(self.device)
                
                output = text_encoder(input_ids, output_hidden_states=True)
                hidden_state = output.hidden_states[-2]
                prompt_embeds_list.append(hidden_state)
                
                if text_encoder == self.text_encoder_2:
                    pooled_prompt_embeds = output.text_embeds

            prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
            return prompt_embeds, pooled_prompt_embeds

    def compute_time_ids(self, batch_size):
        original_size = (self.config.RESOLUTION, self.config.RESOLUTION)
        target_size = (self.config.RESOLUTION, self.config.RESOLUTION)
        crop_coords = (0, 0)
        
        add_time_ids = list(original_size + crop_coords + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.config.DTYPE).to(self.device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def save_lora(self, step):
        # Save only the adapters
        save_path = os.path.join(self.config.OUTPUT_DIR, f"checkpoint-{step}")
        self.unet.save_pretrained(save_path)
        print(f"Saved LoRA weights to {save_path}")

    def train_one_epoch(self, dataloader, epoch_index):
        for step, batch in enumerate(dataloader):
            # 1. Load Data
            pixel_values = batch["pixel_values"].to(device=self.device, dtype=self.config.DTYPE)
            prompts = batch["prompt"]
            bsz = pixel_values.shape[0]

            # 2. Encode to Latents (VAE)
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
            # 3. Sample Noise
            noise = torch.randn_like(latents, dtype=self.config.DTYPE)
            
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device
            ).long()

            # 4. Add Noise
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # 5. Get Conditions
            prompt_embeds, pooled_prompt_embeds = self.compute_text_embeddings(prompts)
            add_time_ids = self.compute_time_ids(bsz)

            # 6. Forward Pass
            # LoRA layers are applied automatically inside self.unet
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                },
            ).sample

            # 7. Loss Calculation
            loss = F.mse_loss(model_pred, noise, reduction="mean")

            # 8. Backward Pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if step % 5 == 0:
                print(f"Epoch {epoch_index} | Step {step} | Loss: {loss.item():.4f}")

    def run(self):
        self.setup()
        
        # Fixed: Initialize Dataset with Config, not num_samples
        dataset = LocalDataset(self.config)
        
        if len(dataset) == 0:
            print("No data found. Exiting.")
            return

        dataloader = dataset.get_dataloader()
        
        total_steps = 0
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"--- Starting Epoch {epoch} ---")
            self.train_one_epoch(dataloader, epoch)
            # Save at the end of epoch
            self.save_lora(f"epoch_{epoch}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        trainer = SDXLTrainer(device='cuda')
        trainer.run()
    else:
        print("CUDA not available. This script requires a GPU.")