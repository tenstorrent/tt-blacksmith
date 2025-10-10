import os

from tqdm import tqdm

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from blacksmith.experiments.torch.qwen.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.torch.text2sql.text2sql_dataset import TextToSQLDataset


if __name__ == "__main__":
    # Config setup
    config_file_path = os.path.join(os.path.dirname(__file__), "test_qwen_finetuning.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    # Device setup
    xr.runtime.set_device_type("TT")
    device = xm.xla_device()

    # Load model
    model = get_model(config)
    model = model.to(device)

    # Load dataset
    dataset = TextToSQLDataset(config=config)
    train_dataloader = dataset.get_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
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
            loss.backward()
            optimizer.step()

            # xm.optimizer_step(optimizer)
            torch_xla.sync(wait=True)

            if ind == 10:
                break
