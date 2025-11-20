# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_xla
import torch_xla.runtime as xr
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# Diffusers imports for SDXL
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler

# Maintaining your custom structure
from blacksmith.experiments.torch.phi.configs import TrainingConfig
from blacksmith.datasets.torch.sdxl.image_caption_dataset import ImageCaptionDataset # Renamed from SSTDataset
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.torch_helpers import show_examples, collect_examples # You might need to adapt these for images
# Assuming a custom collate for resizing images and tokenizing prompts
from blacksmith.tools.torch_helpers import collate_fn_for_sdxl 


def validate(model, vae, text_encoders, val_data_loader, loss_fn, device, config, logger, noise_scheduler):
    """
    Validates the UNet by calculating MSE loss on the validation set.
    """
    logger.info(f"\n=== Starting Validation ===")
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    # Ensure text encoders and VAE are in eval mode
    for te in text_encoders:
        te.eval()
    vae.eval()

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            # 1. Prepare inputs
            pixel_values = batch["pixel_values"].to(device)
            prompt_ids_one = batch["input_ids_one"].to(device)
            prompt_ids_two = batch["input_ids_two"].to(device)
            # SDXL specific conditioning (original size, crops, etc)
            add_time_ids = batch["unet_added_conditions"]["time_ids"].to(device)
            
            # 2. Encode Images to Latents (VAE)
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # 3. Encode Text (Text Encoders)
            # Simplified for brevity: assumes utility to get pooled embeds and hidden states
            # In a real script, you run both text_encoders here to get prompt_embeds and pooled_prompt_embeds
            prompt_embeds, pooled_prompt_embeds = encode_prompts(text_encoders, prompt_ids_one, prompt_ids_two)
            
            # 4. Sample Noise
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            # 5. Add Noise (Forward Diffusion)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # 6. Predict Noise (UNet)
            # SDXL requires added_cond_kwargs for aspect ratio bucketing
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            
            model_pred = model(
                noisy_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # 7. Calculate Loss
            # target is usually 'noise' (epsilon-prediction) or 'v_prediction' depending on config
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            total_val_loss += loss.item()

            if config.use_tt:
                torch_xla.sync(wait=True)

            num_val_batches += 1

    # Note: Visualizing generated images during validation is expensive. 
    # Usually simpler to just track MSE loss here.
    
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


def train(
    config: TrainingConfig,
    device: torch.device,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training for SDXL Base 1.0...")

    # 1. Load Helper Models (VAE, Tokenizers, Scheduler)
    # Usually we freeze these and only train the UNet
    # For strict structure, we assume they are loaded here or via get_model
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae").to(device)
    
    # SDXL has two text encoders
    text_encoder_1 = get_model(config, device, subfolder="text_encoder")
    text_encoder_2 = get_model(config, device, subfolder="text_encoder_2")
    
    # Freeze VAE and Text Encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # 2. Load Main Model (UNet)
    # This is the model we actually run optimizer on
    model = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet").to(device)
    
    logger.info(f"Loaded {config.model_name} UNet.")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint()

    # 3. Load Dataset
    # Assuming ImageCaptionDataset returns pixel_values and tokenized input_ids
    train_dataset = ImageCaptionDataset(config=config, collate_fn=collate_fn_for_sdxl)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = ImageCaptionDataset(config=config, split="validation", collate_fn=collate_fn_for_sdxl)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    # Init training components
    # SDXL usually uses a lower LR (e.g. 1e-5 or 1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Loss function is MSE, defined implicitly in loop or via F.mse_loss

    global_step = 0
    running_loss = 0.0
    model.train()
    
    try:
        for epoch in range(config.num_epochs):
            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()

                # --- Data Prep ---
                pixel_values = batch["pixel_values"].to(device)
                prompt_ids_one = batch["input_ids_one"].to(device)
                prompt_ids_two = batch["input_ids_two"].to(device)
                add_time_ids = batch["unet_added_conditions"]["time_ids"].to(device)

                # --- Forward Pass Components ---
                
                # 1. Convert images to latents
                with torch.no_grad():
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor

                    # 2. Encode text prompts
                    # Helper function to run both encoders and concat results
                    prompt_embeds, pooled_prompt_embeds = encode_prompts([text_encoder_1, text_encoder_2], prompt_ids_one, prompt_ids_two)

                # 3. Sample Noise
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()

                # 4. Add noise to the latents (Forward Diffusion Process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # 5. Predict the noise residual (UNet Forward)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                
                model_pred = model(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # 6. Calculate Loss (MSE between actual noise and predicted noise)
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                running_loss += loss.item()

                # --- Backward Pass ---
                loss.backward()
                
                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Update parameters
                optimizer.step()
                
                if config.use_tt:
                    torch_xla.sync(wait=True)

                # --- Logging & Validation ---
                do_validation = global_step % config.val_steps_freq == 0

                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq if global_step > 0 else running_loss
                    logger.log_metrics({"train/loss": avg_loss}, commit=not do_validation, step=global_step)
                    running_loss = 0.0

                # Validation phase
                if do_validation:
                    avg_val_loss = validate(
                        model, vae, [text_encoder_1, text_encoder_2], eval_dataloader, None, device, config, logger, noise_scheduler
                    )
                    model.train()

                    logger.log_metrics(
                        {"epoch": epoch + 1, "val/loss": avg_val_loss},
                        step=global_step,
                    )

                if checkpoint_manager.should_save_checkpoint(global_step):
                    # We only save the UNet
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                global_step += 1

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Save final model
        final_model_path = checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model_unet.pth")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()

# Helper to handle SDXL's dual text encoders
def encode_prompts(text_encoders, token_ids_1, token_ids_2):
    # This is a simplified representation. 
    # SDXL requires encoding with both encoders and concatenating the results.
    # Real implementation would iterate over encoders and concat last_hidden_states.
    with torch.no_grad():
        # Logic to run text_encoders[0] on token_ids_1
        # Logic to run text_encoders[1] on token_ids_2
        # Return concatenated prompt_embeds and pooled_embeds
        # For syntax correctness in this snippet, returning placeholders:
        return torch.randn(token_ids_1.shape[0], 77, 2048).to(token_ids_1.device), torch.randn(token_ids_1.shape[0], 1280).to(token_ids_1.device)


if __name__ == "__main__":
    # Config setup
    config_file_path = os.path.join(os.path.dirname(__file__), "sdxl_finetuning_config.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger)

    # Device setup
    if config.use_tt:
        xr.runtime.set_device_type("TT")
        device = torch_xla.device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Start training
    train(config, device, logger, checkpoint_manager)