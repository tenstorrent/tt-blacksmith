#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    print("Testing all imports...")
    
    try:
        from configs import ExperimentConfig, get_cpu_config, get_tt_config
        print("✅ configs imports work")
    except Exception as e:
        print(f"❌ configs import failed: {e}")
        return False
    
    try:
        from models.gpt_model import create_model
        print("✅ models.gpt_model import works")
    except Exception as e:
        print(f"❌ models.gpt_model import failed: {e}")
        return False
    
    try:
        from datasets.text_dataset import load_text_dataset, create_dataloader
        print("✅ datasets.text_dataset import works")
    except Exception as e:
        print(f"❌ datasets.text_dataset import failed: {e}")
        return False
    
    try:
        from utils.device_utils import create_device_manager, log_device_info
        print("✅ utils.device_utils import works")
    except Exception as e:
        print(f"❌ utils.device_utils import failed: {e}")
        return False
    
    try:
        from utils.training_utils import (
            create_optimizer, create_train_state, training_step, 
            estimate_loss, get_lr, save_checkpoint, load_checkpoint
        )
        print("✅ utils.training_utils import works")
    except Exception as e:
        print(f"❌ utils.training_utils import failed: {e}")
        return False
    
    try:
        from wandb_logging.wandb_utils import init_wandb, log_metrics, finish_wandb
        print("✅ wandb_logging.wandb_utils import works")
    except Exception as e:
        print(f"❌ wandb_logging.wandb_utils import failed: {e}")
        return False
    
    print("🎉 All imports successful!")
    return True

if __name__ == "__main__":
    test_imports()
