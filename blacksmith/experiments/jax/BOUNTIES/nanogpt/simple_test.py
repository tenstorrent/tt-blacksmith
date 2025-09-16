#!/usr/bin/env python3
"""
Simple test script to verify the environment without complex imports
"""

import os
import sys

def main():
    print("🚀 Simple Environment Test")
    print("=" * 50)
    
    # Test basic Python
    print(f"✅ Python version: {sys.version}")
    print(f"✅ Current directory: {os.getcwd()}")
    
    # Test JAX
    try:
        import jax
        print(f"✅ JAX version: {jax.__version__}")
        print(f"✅ JAX devices: {jax.devices()}")
        
        # Test basic computation
        import jax.numpy as jnp
        x = jnp.array([1, 2, 3, 4, 5])
        result = jnp.sum(x)
        print(f"✅ Basic computation: {result}")
        
    except Exception as e:
        print(f"❌ JAX error: {e}")
    
    # Test other dependencies
    try:
        import flax
        print(f"✅ Flax version: {flax.__version__}")
    except Exception as e:
        print(f"❌ Flax error: {e}")
    
    try:
        import optax
        print(f"✅ Optax available")
    except Exception as e:
        print(f"❌ Optax error: {e}")
    
    # Test file access
    print("\n📁 Testing file access:")
    files_to_check = [
        'configs.py',
        'models/gpt_model.py',
        'datasets/text_dataset.py',
        'utils/training_utils.py',
        'logging/wandb_utils.py'
    ]
    
    for file in files_to_check:
        exists = os.path.exists(file)
        print(f"  {file}: {'✅' if exists else '❌'}")
    
    print("\n🎯 Simple test complete!")

if __name__ == "__main__":
    main()
