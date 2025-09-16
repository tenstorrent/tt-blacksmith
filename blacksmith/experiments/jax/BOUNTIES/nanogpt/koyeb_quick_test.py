#!/usr/bin/env python3
"""
Quick test script for Koyeb deployment
Tests basic functionality without full training
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import jax
        print(f"✅ JAX version: {jax.__version__}")
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    
    try:
        import flax
        print(f"✅ Flax version: {flax.__version__}")
    except ImportError as e:
        print(f"❌ Flax import failed: {e}")
        return False
    
    try:
        import optax
        print(f"✅ Optax version: {optax.__version__}")
    except ImportError as e:
        print(f"❌ Optax import failed: {e}")
        return False
    
    return True

def test_devices():
    """Test device availability"""
    print("\nTesting devices...")
    
    try:
        import jax
        
        # Get available devices
        devices = jax.devices()
        print(f"Available devices: {devices}")
        
        # Test basic computation
        x = jax.numpy.array([1, 2, 3, 4])
        y = jax.numpy.sum(x)
        print(f"✅ Basic computation test: {y}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False

def test_model_creation():
    """Test that we can create the GPT model"""
    print("\nTesting model creation...")
    
    try:
        from models.gpt_model import create_model
        from configs import NetConfig
        
        # Create a small model for testing
        config = NetConfig(
            vocab_size=1000,
            n_embd=64,
            n_head=2,
            n_layer=2,
            block_size=128
        )
        
        model = create_model(config)
        print(f"✅ Model created successfully")
        print(f"Model parameters: {model.count_params()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        from datasets.text_dataset import TextDataset
        
        # Test with a small dataset
        dataset = TextDataset(
            dataset_name="shakespeare",
            block_size=64,
            batch_size=2
        )
        
        # Get a sample batch
        batch = next(iter(dataset.get_batches()))
        print(f"✅ Data loading successful")
        print(f"Batch shape: {batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Koyeb Quick Test for NanoGPT Implementation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_devices,
        test_model_creation,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
