import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from PIL import Image
from blacksmith.datasets.torch.torch_dataset import BaseDataset
from datasets import load_dataset
import numpy as np

# --- Configuration ---
class TrainingConfig:
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    RESOLUTION = 1024
    TRAIN_BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 1
    # We define the global data type here
    DTYPE = torch.bfloat16 

class DummyDataset(BaseDataset):
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.split="train"
        self.transform = transforms.Compose([
            transforms.Resize(TrainingConfig.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(TrainingConfig.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self._prepare_dataset()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image
        sample = self.dataset[idx]
        return {"pixel_values": self.transform(sample["images"]), "prompt": sample["prompt"]}
    
    def _prepare_dataset(self):
        raw_dataset = load_dataset(self.config.dataset_id, split=self.split)

        self.dataset = raw_dataset

    def get_dataloader(self) -> DataLoader:

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle = False,
            drop_last=True,
        )

class SDXLTrainer:
    def __init__(self, device='cuda'):
        self.device = device
        self.config = TrainingConfig()
        
    def setup(self):
        print("Loading models in Pure BFloat16...")
        self.load_models()
        self.load_tokenizers()
        self.setup_scheduler()
        self.setup_optimizer()
        # Note: No GradScaler needed for bfloat16

    def load_models(self):
        # 1. Load VAE (Frozen) - Directly in bfloat16
        self.vae = AutoencoderKL.from_pretrained(
            self.config.MODEL_ID, 
            subfolder="vae", 
            torch_dtype=self.config.DTYPE
        ).to(self.device)
        self.vae.requires_grad_(False)

        # 2. Load Text Encoders (Frozen) - Directly in bfloat16
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

        # 3. Load UNet (Trainable) - Directly in bfloat16
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.MODEL_ID, 
            subfolder="unet", 
            torch_dtype=self.config.DTYPE
        ).to(self.device)
        
        self.unet.enable_gradient_checkpointing()
        self.unet.train()

    def load_tokenizers(self):
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(self.config.MODEL_ID, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.config.MODEL_ID, subfolder="tokenizer_2")

    def setup_scheduler(self):
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.MODEL_ID, subfolder="scheduler")

    def setup_optimizer(self):
        # AdamW will handle BF16 parameters natively
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
                
                # The output here is already bfloat16 because the model is bfloat16
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
        # IMPORTANT: Ensure these control parameters are bfloat16
        add_time_ids = torch.tensor([add_time_ids], dtype=self.config.DTYPE).to(self.device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def train_one_epoch(self, dataloader, epoch_index):
        for step, batch in enumerate(dataloader):
            # 1. Load Data and CAST to bfloat16 immediately
            pixel_values = batch["pixel_values"].to(device=self.device, dtype=self.config.DTYPE)
            prompts = batch["prompt"]
            bsz = pixel_values.shape[0]

            # 2. Encode to Latents (VAE)
            with torch.no_grad():
                # VAE is in bf16, input is bf16 -> output is bf16
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
            # 3. Sample Noise (in bf16)
            noise = torch.randn_like(latents, dtype=self.config.DTYPE)
            
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device
            ).long()

            # 4. Add Noise (bf16 math)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # 5. Get Conditions (bf16)
            prompt_embeds, pooled_prompt_embeds = self.compute_text_embeddings(prompts)
            add_time_ids = self.compute_time_ids(bsz)

            # 6. Forward Pass (Pure BF16)
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
            # Both model_pred and noise are bfloat16.
            # MSE Loss works fine in bf16, though sometimes people cast to float32 just for the loss calculation
            # to be ultra-precise, but for SDXL training, pure bf16 loss is usually fine.
            loss = F.mse_loss(model_pred, noise, reduction="mean")

            # 8. Backward Pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if step % 1 == 0:
                print(f"Epoch {epoch_index} | Step {step} | Loss: {loss.item():.4f}")

    def run(self):
        self.setup()
        dataset = DummyDataset(num_samples=4)
        dataloader = DataLoader(dataset, batch_size=self.config.TRAIN_BATCH_SIZE, shuffle=True)
        print("Starting Pure BFloat16 training...")
        for epoch in range(self.config.NUM_EPOCHS):
            self.train_one_epoch(dataloader, epoch)

if __name__ == "__main__":
    trainer = SDXLTrainer(device='cuda')
    trainer.run()