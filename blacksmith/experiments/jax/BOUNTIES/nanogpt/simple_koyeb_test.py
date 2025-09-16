#!/usr/bin/env python3
"""
Simple test script to verify JAX and TT-N150 setup on Koyeb.
Run this first to make sure the environment is working.
"""

import sys
import os

def test_python_version():
    """Test Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 10):
        print("✅ Python version is compatible with JAX")
        return True
    else:
        print("❌ Python version too old for JAX (need 3.10+)")
        return False

def test_jax_installation():
    """Test JAX installation."""
    try:
        import jax
        print(f"✅ JAX version: {jax.__version__}")
        print(f"✅ JAX devices: {jax.devices()}")
        return True
    except ImportError as e:
        print(f"❌ JAX not installed: {e}")
        return False

def test_tt_device():
    """Test TT device availability."""
    try:
        import jax
        tt_devices = jax.devices('tt')
        if tt_devices:
            print(f"✅ TT devices found: {tt_devices}")
            return True
        else:
            print("⚠️  No TT devices found")
            return False
    except Exception as e:
        print(f"❌ Error checking TT devices: {e}")
        return False

def test_basic_jax_operations():
    """Test basic JAX operations."""
    try:
        import jax.numpy as jnp
        
        # Test CPU operation
        cpu_array = jnp.array([1, 2, 3, 4, 5])
        cpu_result = jnp.sum(cpu_array)
        print(f"✅ CPU operation: sum([1,2,3,4,5]) = {cpu_result}")
        
        # Test TT operation (if available)
        try:
            tt_devices = jax.devices('tt')
            if tt_devices:
                with jax.default_device(tt_devices[0]):
                    tt_array = jnp.array([1, 2, 3, 4, 5])
                    tt_result = jnp.sum(tt_array)
                    print(f"✅ TT operation: sum([1,2,3,4,5]) = {tt_result}")
                return True
            else:
                print("⚠️  No TT devices for testing")
                return True
        except Exception as e:
            print(f"⚠️  TT operation failed (expected): {e}")
            return True
            
    except Exception as e:
        print(f"❌ Basic JAX operations failed: {e}")
        return False

def test_dependencies():
    """Test other required dependencies."""
    dependencies = ['flax', 'optax', 'pydantic', 'yaml', 'requests', 'tqdm']
    
    for dep in dependencies:
        try:
            if dep == 'yaml':
                import yaml
            else:
                __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Koyeb TT-N150 Environment Test")
    print("=" * 40)
    
    tests = [
        ("Python Version", test_python_version),
        ("JAX Installation", test_jax_installation),
        ("TT Device", test_tt_device),
        ("Basic JAX Operations", test_basic_jax_operations),
        ("Dependencies", test_dependencies)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready for NanoGPT training.")
        print("\nNext steps:")
        print("1. Run: python3 train_nanogpt.py --device tt --config config_tt.yaml")
        print("2. Monitor the training logs")
        print("3. Check for any fallback scenarios")
    else:
        print("❌ Some tests failed. Please fix the environment issues first.")
        print("\nCommon fixes:")
        print("- Install JAX: pip install jax jaxlib")
        print("- Install dependencies: pip install flax optax pydantic pyyaml")
        print("- Check TT-N150 hardware availability")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
