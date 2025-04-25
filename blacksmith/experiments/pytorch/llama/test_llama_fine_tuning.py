# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import re
import json
import torch
import wandb
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, set_seed

from blacksmith.experiments.pytorch.llama.configs import TrainingConfig
from blacksmith.models.pytorch.hf_models import get_model
from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.datasets.torch.llama.sst_utils import VALUE2LBL
from blacksmith.tools.cli import generate_config
from blacksmith.tools.hf_callbacks import GradientSavingCallback, ProfilerCallback, WandbMemoryCallback


os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def setup_training(config, model, tokenizer, train_set, eval_set):
    print("Setting up training...")

    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    gradients_dir = os.path.join(config.output_dir, "gradients")
    profiler_dir = os.path.join(config.output_dir, "profiler_logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(gradients_dir, exist_ok=True)
    os.makedirs(profiler_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        optim=config.optim,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        callbacks=[WandbMemoryCallback(), GradientSavingCallback(gradients_dir), ProfilerCallback(profiler_dir)],
    )

    return trainer


def train(config: TrainingConfig, model, tokenizer, train_set, eval_set):
    print("Starting training...")

    try:
        set_seed(config.seed)

        trainer = setup_training(config, model, tokenizer, train_set, eval_set)
        trainer.train()

        final_model_path = os.path.join(config.output_dir, "final_model")
        trainer.save_model(final_model_path)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        wandb.alert(title="Training Failed", text=str(e))
        raise

    print("Training complete!")


def evaluate(model, tokenizer, eval_set):
    print("Evaluating model...")

    model.to("cuda")
    model.eval()
    results = []

    def extract_first_valid_json(text):
        """Extract first valid JSON from generated text."""
        json_pattern = r"\{[^{}]*\}"
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed_json = json.loads(match.group())
                return parsed_json
            except json.JSONDecodeError:
                continue

        return None

    for item in tqdm(eval_set):
        input_ids = tokenizer(item["text"], return_tensors="pt")["input_ids"]
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids.cuda(), max_new_tokens=20)

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = extract_first_valid_json(response_text)
        response = -1 if response is None else VALUE2LBL[response["label"].lower()]
        results.append(
            {
                "prompt": item["text"],
                "generated_text": response_text,
                "response": response,
                "label": item["label"].item(),
            }
        )

    accuracy = sum([r["response"] == r["label"] for r in results]) / len(results)
    wandb.log({"eval_accuracy": accuracy})
    print(f"Eval accuracy: {accuracy}")

    results_path = os.path.join(config.output_dir, "eval_results.json")
    with open(results_path, "w+") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_llama_fine_tuning.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    model = get_model(config)

    dataset = SSTDataset(config)
    train_set, eval_set = dataset.load_tokenized_data()

    wandb.init(project=config.wandb_project, config=vars(config))
    wandb.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)

    if config.do_train:
        train(config, model, dataset.tokenizer, train_set, eval_set)

    if config.do_eval:
        evaluate(model, dataset.tokenizer, eval_set)
