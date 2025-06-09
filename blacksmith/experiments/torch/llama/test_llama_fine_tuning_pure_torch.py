# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback
import re
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.torch.llama.sst_utils import VALUE2LBL


def train(config, model, train_data_loader):
    run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config), save_code=True)
    run.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)

    if config.use_tt:
        import forge

        tt_optimizer = forge.optimizers.AdamW()
        sample_inputs = [torch.randint(0, model.config.vocab_size, (config.batch_size, config.max_length))]
        compiled_model = forge.compile(model, sample_inputs, optimizer=tt_optimizer, training=True)
    else:
        torch_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    # Create a torch loss and leave on CPU
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/18997 resolved
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    try:
        global_step = 0
        running_loss = 0.0
        log_every_n_steps = config.logging_steps

        for epoch in range(config.num_epochs):
            model.train()

            for batch in tqdm(train_data_loader):
                input_ids = batch["input_ids"]
                expected_output = batch["labels"]

                if config.use_tt:
                    logits = compiled_model(input_ids)[0]
                else:
                    input_ids = input_ids.to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    expected_output = expected_output.to(device)
                    logits = model(input_ids, attention_mask=attention_mask).logits

                loss = loss_fn(logits.view(-1, model.config.vocab_size), expected_output.view(-1))
                running_loss += loss.item()

                loss.backward()
                if config.use_tt:
                    compiled_model.backward()
                    tt_optimizer.step()
                else:
                    torch_optimizer.step()
                    torch_optimizer.zero_grad()

                global_step += 1
                if global_step % log_every_n_steps == 0:
                    avg_loss = running_loss / log_every_n_steps
                    run.log({"train/loss": avg_loss, "step": global_step})
                    running_loss = 0.0

            if config.save_strategy == "epoch":
                checkpoint_path = os.path.join(config.output_dir, "checkpoints", f"checkpoint-{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)

                artifact = wandb.Artifact(f"checkpoint-{epoch+1}", type="model")
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact)

        final_model_path = os.path.join(config.output_dir, "checkpoints", "final_model.pth")
        torch.save(model.state_dict(), final_model_path)

        artifact = wandb.Artifact("final_model", type="model")
        artifact.add_file(final_model_path)
        run.log_artifact(artifact)

    except Exception as e:
        error_msg = f"Training failed with error: {str(e)}"
        traceback_str = traceback.format_exc()
        print(error_msg)
        print(traceback_str)
        run.alert(title="Training Failed", text=error_msg, level=wandb.AlertLevel.ERROR)
        run.log({"error": error_msg, "traceback": traceback_str})
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_llama_fine_tuning_pure_torch.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)

    model = get_model(config)

    dataset = SSTDataset(config)
    train_set, eval_set = dataset.load_tokenized_data()
    train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, drop_last=True)
    eval_data_loader = DataLoader(eval_set, batch_size=config.batch_size, shuffle=False, drop_last=True)

    if config.do_train:
        train(config, model, train_data_loader)
