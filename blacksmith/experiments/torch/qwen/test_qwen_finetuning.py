# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
from tqdm import tqdm

from blacksmith.experiments.torch.qwen.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.torch.text2sql.text2sql_dataset import TextToSQLDataset
from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset2
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager


def inspect_data_batches(tokenizer, batch):
    for ind in range(len(batch["input_ids"])):
        full_text = tokenizer.decode(batch["input_ids"][ind], skip_special_tokens=False)

        labels_for_decoding = batch["labels"][ind].clone()
        labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
        target_text = tokenizer.decode(labels_for_decoding, skip_special_tokens=False)

        print(f"Full text: {full_text}")
        print(f"Target text: {target_text}")
        print(f"Labels: {batch['labels'][ind]}")
        print(f"Num of tokens: {len(batch['input_ids'][ind])}")
        print("-" * 80)

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Config setup
    config_file_path = os.path.join(os.path.dirname(__file__), "test_qwen_finetuning.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config)

    # Device setup
    if config.use_tt:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        xr.runtime.set_device_type("TT")
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = get_model(config)

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint()

    # Load dataset
    dataset = TextToSQLDataset(config=config)
    train_dataloader = dataset.get_dataloader()
    logger.info(
        f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)}"
    )  # Add eval dataset size

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    global_step = 0
    running_loss = 0.0
    try:
        for epoch in range(config.num_epochs):
            model.train()

            for ind, batch in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()

                # move tensors to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                running_loss += loss.item()
                loss.backward()

                global_step += 1
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    logger.log_metrics({"train/loss": avg_loss}, step=global_step)
                    running_loss = 0.0

                    # Do validation
                    # Log validation loss
                    # Log sample generations

                    # Save checkpoint
                    if checkpoint_manager.should_save(global_step):
                        checkpoint_manager.save_checkpoint(model, optimizer, global_step, epoch)

                if config.use_tt:
                    xm.optimizer_step(optimizer)
                    torch_xla.sync(wait=True)
                else:
                    optimizer.step()

                if ind == 200:
                    break

            if checkpoint_manager.should_save(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, optimizer, global_step, epoch)

        # Save final model
        final_model_path = checkpoint_manager.save_checkpoint(model, optimizer, global_step, epoch)
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        error_msg = f"Training failed with error: {str(e)}"
        traceback_str = traceback.format_exc()

        logger.error(error_msg, traceback_str)
        raise
    finally:
        logger.finish()
