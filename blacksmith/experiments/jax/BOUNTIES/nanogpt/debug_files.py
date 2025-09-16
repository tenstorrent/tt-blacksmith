#!/usr/bin/env python3
"""
Debug script to see what files are available in the workspace
"""

import os
import sys

def main():
    print("🔍 Debugging Koyeb Workspace")
    print("=" * 50)
    
    # Show current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # List all files in current directory
    print("\n📁 Files in current directory:")
    try:
        files = os.listdir('.')
        for file in sorted(files):
            print(f"  - {file}")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Check if specific files exist
    test_files = [
        'simple_koyeb_test.py',
        'koyeb_quick_test.py', 
        'train_nanogpt.py',
        'requirements.txt',
        'config_tt.yaml'
    ]
    
    print("\n🔍 Checking for specific files:")
    for file in test_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✅' if exists else '❌'}")
    
    # Try to import JAX
    print("\n🧪 Testing JAX import:")
    try:
        import jax
        print(f"✅ JAX version: {jax.__version__}")
        print(f"✅ Available devices: {jax.devices()}")
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
    except Exception as e:
        print(f"❌ JAX error: {e}")
    
    print("\n🎯 Debug complete!")

if __name__ == "__main__":
    main()
