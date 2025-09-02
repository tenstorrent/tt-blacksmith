import sys
import os
import shutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback

# Add custom wandb installation path to sys.path
sys.path.insert(0, '/opt/dlami/nvme/python_packages')
import wandb

# Import your custom dataset and config classes
from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig

# Wandb configuration
DEFAULT_EXPERIMENT_NAME = "Llama-FineTuning"
DEFAULT_RUN_NAME = "llama-3.2-1b-sst2"
# Use wandb's default directory (~/.local/share/wandb or WANDB_DIR env var)
DEFAULT_WANDB_DIR = None

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def setup_wandb(config: TrainingConfig):
    """Setup wandb with error handling - only log to wandb, no local files."""
    wandb_run = wandb.init(
        project=DEFAULT_EXPERIMENT_NAME,
        name=DEFAULT_RUN_NAME,
        config={
            "model_name": config.model_name,
            "dataset_id": config.dataset_id,
            "max_length": config.max_length,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lora_r": 4,  # Updated to match your spec
            "lora_alpha": 8,  # Typically 2x the rank
            "target_modules": ["mlp.gate_proj.kernel", "mlp.up_proj.kernel", "mlp.down_proj.kernel"],
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    )
    print(f"Started wandb run: {wandb_run.name}")
    return wandb_run

def log_to_wandb(data_dict, step=None):
    """Helper function to log data to wandb."""
    wandb.log(data_dict, step=step)

def get_lora_target_modules(model):
    """Get LoRA target modules - ONLY MLP layers as specified."""
    target_modules = []
    
    # LoRA spec: ONLY MLP layers (as originally requested!)
    for name, module in model.named_modules():
        # Llama MLP params use gate/up/down proj, no bias
        if ".mlp." in name and (
            ".gate_proj" in name or ".up_proj" in name or ".down_proj" in name
        ):
            # Extract the base module name (remove any submodule parts)
            module_name = name.split('.')[-1]  # e.g., "gate_proj", "up_proj", "down_proj"
            if module_name not in target_modules:
                target_modules.append(module_name)
                print(f"✅ Adding LoRA target module: {module_name} (from {name})")
    
    return target_modules

class WandbCallback(TrainerCallback):
    """Custom callback to log training metrics to wandb."""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log training metrics to wandb."""
        if logs is not None:
            wandb.log(logs, step=state.global_step)
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Log evaluation metrics to wandb."""
        if logs is not None:
            eval_logs = {f"eval/{k}": v for k, v in logs.items()}
            wandb.log(eval_logs, step=state.global_step)
    
    def on_train_end(self, args, state, control, model=None, logs=None, **kwargs):
        """Log final training summary."""
        if logs is not None:
            final_logs = {f"final/{k}": v for k, v in logs.items()}
            wandb.log(final_logs, step=state.global_step)

def main():
    """Main training function with wandb integration."""
    # Create training configuration
    config = TrainingConfig(
        model_name="meta-llama/Llama-3.2-1B",
        dataset_id="stanfordnlp/sst2",
        max_length=512,
        learning_rate=1e-4,  # Much lower learning rate for LoRA
        batch_size=8,  # Reduce to fit in GPU memory
        num_epochs=1  # Just 1 epoch for testing
    )
    
    # Setup wandb
    wandb_run = setup_wandb(config)
    
    try:
        # Load LLaMA 3.2-1B model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            device_map="auto",  # Automatically distribute model across available GPUs
            torch_dtype=torch.float16,  # Use full precision
        )

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load SST dataset using your custom dataset class
        print("Loading dataset...")
        dataset_loader = SSTDataset(config)
        train_dataset, validation_dataset = dataset_loader.load_tokenized_data()

        # Get LoRA target modules dynamically
        target_modules = get_lora_target_modules(model)
        
        # Training arguments - optimized for GPU with wandb integration (no local logging)
        training_args = TrainingArguments(
            output_dir="/opt/dlami/nvme/temp_trainer",  # Required by trainer class but won't be used
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            logging_steps=config.logging_steps,
            learning_rate=config.learning_rate,
            gradient_accumulation_steps=8,  # Effective batch size = 8 × 8 = 64
            eval_strategy="epoch", 
            save_strategy="no",  # Explicitly no saving - this prevents all saves
            save_only_model=False,  # Don't save model
            # Wandb integration - ONLY log to wandb
            report_to="wandb",
            run_name=DEFAULT_RUN_NAME,
            logging_dir=None,  # No local logging
            dataloader_pin_memory=False,  # Reduce memory usage
        )

        # LoRA configuration - targeting ONLY MLP layers with rank 16
        lora_config = LoraConfig(
            r=8,  # Rank 16 as specified
            lora_alpha=16,  # Typically 2x the rank
            target_modules=target_modules,  # Dynamically determined MLP modules
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to the model
        print("Applying LoRA to model...")
        model = get_peft_model(model, lora_config)

        # Print trainable parameters info
        model.print_trainable_parameters()

        # Create trainer with wandb callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            callbacks=[WandbCallback()],
        )

        # Start training
        print("Starting training...")
        trainer.train()

        # Log final model info
        log_to_wandb({
            "training_completed": True,
            "total_steps": trainer.state.global_step,
            "final_loss": trainer.state.log_history[-1].get("train_loss", 0) if trainer.state.log_history else 0,
        })

        print("Training completed successfully! All logs are in wandb.")

    except Exception as e:
        print(f"Error during training: {e}")
        log_to_wandb({"error": str(e), "training_failed": True})
        raise
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared.")
        
        # Clean up temporary trainer directory
        if os.path.exists("/opt/dlami/nvme/temp_trainer"):
            shutil.rmtree("/opt/dlami/nvme/temp_trainer")
            print("Temporary trainer directory cleaned up.")
        
        # Finish wandb run
        wandb.finish()
        print("Finished wandb run - all logs are in wandb only.")

if __name__ == "__main__":
    main()
