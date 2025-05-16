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


def train(config, model, tokenizer, train_data_loader, eval_data_loader, run):
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

        breakpoint()
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

                    # Decode and print logits, expected_output, and input_ids
                    # decoded_logits = torch.argmax(logits, dim=-1)
                    # decoded_logits_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in decoded_logits]
                    # input_ids_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

                    # print("Decoded logits (text):", decoded_logits_text[0])
                    # print("Input IDs (text):", input_ids_text[0])

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

                    eval_loss = evaluate(model, eval_data_loader, loss_fn, device)
                    run.log({"eval/loss": eval_loss, "epoch": epoch + 1})

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
        run.alert(
            title="Training Failed",
            text=error_msg,
            level=wandb.AlertLevel.ERROR
        )
        run.log({"error": error_msg, "traceback": traceback_str})
        raise
    finally:
        wandb.finish()


@torch.no_grad()
def evaluate_old(model, tokenizer, eval_set, device):
    print("Evaluating model...")
    model.to(device)
    model.eval()
    results = []

    def extract_response(text):
        """Extract first valid JSON from generated text."""
        json_pattern = r"\{[^{}]*\}"
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed_json = json.loads(match.group())
                return parsed_json
                if "label" not in parsed_json:
                    return -1
                return VALUE2LBL.get(parsed_json["label"].lower(), -1)
            except json.JSONDecodeError:
                continue

        return -1

    breakpoint()
    for batch in tqdm(eval_set):
        input_ids = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt")["input_ids"]

        outputs = model.generate(input_ids=input_ids.cuda(), max_new_tokens=20)

        response_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = list(map(extract_response, response_texts))

        for text, response_text, response, label in zip(batch["text"], response_texts, responses, batch["label"]):
            results.append(
                {
                    "prompt": text,
                    "generated_text": response_text,
                    "response": response,
                    "label": label.item(),
                }
            )

    accuracy = sum([r["response"] == r["label"] for r in results]) / len(results)
    print(f"Eval accuracy: {accuracy:.2f}")
    breakpoint()

    # Save results to a file
    with open("blacksmith/experiments/torch/llama/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_llama_fine_tuning_pure_torch.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)

    # run = wandb.init(
    #     project=config.wandb_project,
    #     name=config.wandb_run_name,
    #     config=vars(config),
    #     save_code=True
    # )

    model = get_model(config)
    # run.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)

    dataset = SSTDataset(config)
    train_set, eval_set = dataset.load_tokenized_data()
    train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, drop_last=True)
    eval_data_loader = DataLoader(eval_set, batch_size=config.batch_size, shuffle=False, drop_last=True)

    # train(config, model, dataset.tokenizer, train_data_loader, eval_data_loader, run)
    evaluate_old(model, dataset.tokenizer, eval_data_loader, "cuda")
