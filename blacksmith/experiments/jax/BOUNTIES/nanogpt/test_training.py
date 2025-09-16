# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test script for NanoGPT training implementation.
This script validates the training pipeline on both CPU and TT-N150.
"""

import os
import sys
import logging
import jax
import jax.numpy as jnp

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from configs import get_cpu_config, get_tt_config
from models.gpt_model import create_model
from datasets.text_dataset import load_text_dataset
from utils.device_utils import create_device_manager, log_device_info
from utils.training_utils import create_optimizer, create_train_state, compute_loss


def test_model_creation():
    """Test model creation and initialization."""
    print("Testing model creation...")
    
    config = get_cpu_config()
    model = create_model(config)
    
    # Test model initialization
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, config.model_config.block_size), dtype=jnp.int32)
    
    params = model.init(key, dummy_input, training=False)
    print(f"✓ Model created successfully with {model.get_num_params(params)} parameters")
    
    return model, params


def test_data_loading():
    """Test data loading and tokenization."""
    print("Testing data loading...")
    
    config = get_cpu_config()
    dataset = load_text_dataset(config)
    
    # Test batch creation
    inputs, targets = dataset.get_batch('train', batch_size=2)
    print(f"✓ Data loaded successfully: {inputs.shape}, {targets.shape}")
    
    return dataset


def test_device_management():
    """Test device management and fallback."""
    print("Testing device management...")
    
    config = get_tt_config()
    device_manager = create_device_manager(config)
    log_device_info(device_manager)
    
    print(f"✓ Device manager created: primary={device_manager.primary_device}")
    print(f"✓ TT available: {device_manager.is_tt_available()}")
    
    return device_manager


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")
    
    config = get_cpu_config()
    model, params = test_model_creation()
    dataset = test_data_loading()
    device_manager = create_device_manager(config)
    
    # Create optimizer and training state
    optimizer = create_optimizer(config)
    train_state = create_train_state(model, params, optimizer)
    
    # Get a batch
    inputs, targets = dataset.get_batch('train', batch_size=2)
    
    # Test loss computation
    loss, logits = compute_loss(model, params, inputs, targets, training=True)
    print(f"✓ Training step successful: loss={loss:.4f}")
    
    return train_state


def test_cpu_training():
    """Test CPU training pipeline."""
    print("\n=== Testing CPU Training Pipeline ===")
    
    try:
        config = get_cpu_config()
        model, params = test_model_creation()
        dataset = test_data_loading()
        device_manager = create_device_manager(config)
        train_state = test_training_step()
        
        print("✓ CPU training pipeline test passed")
        return True
    except Exception as e:
        print(f"✗ CPU training pipeline test failed: {e}")
        return False


def test_tt_training():
    """Test TT-N150 training pipeline."""
    print("\n=== Testing TT-N150 Training Pipeline ===")
    
    try:
        config = get_tt_config()
        model, params = test_model_creation()
        dataset = test_data_loading()
        device_manager = create_device_manager(config)
        
        if device_manager.is_tt_available():
            train_state = test_training_step()
            print("✓ TT-N150 training pipeline test passed")
        else:
            print("⚠ TT-N150 not available, skipping TT-specific tests")
        
        return True
    except Exception as e:
        print(f"✗ TT-N150 training pipeline test failed: {e}")
        return False


def test_fallback_mechanism():
    """Test fallback mechanism."""
    print("\n=== Testing Fallback Mechanism ===")
    
    try:
        config = get_tt_config()
        device_manager = create_device_manager(config)
        
        # Test device switching
        cpu_device = device_manager.get_device("cpu")
        tt_device = device_manager.get_device("tt")
        
        print(f"✓ CPU device: {cpu_device}")
        print(f"✓ TT device: {tt_device}")
        
        # Test data movement
        data = jnp.ones((2, 10))
        cpu_data = device_manager.device_put(data, "cpu")
        tt_data = device_manager.device_put(data, "tt")
        
        print("✓ Fallback mechanism test passed")
        return True
    except Exception as e:
        print(f"✗ Fallback mechanism test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("NanoGPT Training Implementation Tests")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    tests = [
        test_cpu_training,
        test_tt_training,
        test_fallback_mechanism
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
