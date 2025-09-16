#!/usr/bin/env python3
"""
Basic test script to validate the implementation structure and imports.
This test runs without requiring JAX to be installed.
"""

import os
import sys
import importlib.util

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "configs.py",
        "config_cpu.yaml", 
        "config_tt.yaml",
        "train_nanogpt.py",
        "test_training.py",
        "compare_cpu_tt.py",
        "README.md",
        "requirements.txt",
        "models/gpt_model.py",
        "datasets/text_dataset.py",
        "utils/device_utils.py",
        "utils/training_utils.py",
        "logging/logger_config.py",
        "logging/wandb_utils.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    python_files = [
        "configs.py",
        "train_nanogpt.py", 
        "test_training.py",
        "compare_cpu_tt.py",
        "models/gpt_model.py",
        "datasets/text_dataset.py",
        "utils/device_utils.py",
        "utils/training_utils.py",
        "logging/logger_config.py",
        "logging/wandb_utils.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, file_path, 'exec')
            print(f"✓ {file_path}")
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"✗ {file_path}: {e}")
        except Exception as e:
            print(f"⚠ {file_path}: {e}")
    
    if syntax_errors:
        print(f"✗ Syntax errors found: {syntax_errors}")
        return False
    else:
        print("✓ All Python files have valid syntax")
        return True

def test_yaml_syntax():
    """Test that YAML files have valid syntax."""
    print("\nTesting YAML syntax...")
    
    yaml_files = ["config_cpu.yaml", "config_tt.yaml"]
    
    try:
        import yaml
    except ImportError:
        print("⚠ PyYAML not available, skipping YAML syntax test")
        return True
    
    yaml_errors = []
    for file_path in yaml_files:
        try:
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            print(f"✓ {file_path}")
        except yaml.YAMLError as e:
            yaml_errors.append(f"{file_path}: {e}")
            print(f"✗ {file_path}: {e}")
        except Exception as e:
            print(f"⚠ {file_path}: {e}")
    
    if yaml_errors:
        print(f"✗ YAML errors found: {yaml_errors}")
        return False
    else:
        print("✓ All YAML files have valid syntax")
        return True

def test_imports():
    """Test that modules can be imported (without JAX dependencies)."""
    print("\nTesting module imports...")
    
    # Test configs.py (should work without JAX)
    try:
        sys.path.insert(0, '.')
        import configs
        print("✓ configs.py imports successfully")
        
        # Test that we can create configs
        cpu_config = configs.get_cpu_config()
        tt_config = configs.get_tt_config()
        print("✓ Config creation works")
        
    except Exception as e:
        print(f"✗ configs.py import failed: {e}")
        return False
    
    return True

def test_documentation():
    """Test that documentation files exist and have content."""
    print("\nTesting documentation...")
    
    doc_files = ["README.md", "IMPLEMENTATION_SUMMARY.md"]
    
    for file_path in doc_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Reasonable minimum length
                    print(f"✓ {file_path} has substantial content ({len(content)} chars)")
                else:
                    print(f"⚠ {file_path} seems short ({len(content)} chars)")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True

def test_requirements():
    """Test that requirements.txt exists and has reasonable content."""
    print("\nTesting requirements...")
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            content = f.read()
            required_packages = ["jax", "flax", "optax", "wandb", "pydantic"]
            
            missing_packages = []
            for package in required_packages:
                if package not in content:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"✗ Missing packages in requirements.txt: {missing_packages}")
                return False
            else:
                print("✓ requirements.txt contains required packages")
                return True
    else:
        print("✗ requirements.txt missing")
        return False

def main():
    """Run all basic tests."""
    print("NanoGPT Implementation - Basic Validation Tests")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_yaml_syntax,
        test_imports,
        test_documentation,
        test_requirements
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
        print("✓ All basic tests passed!")
        print("\nNext steps:")
        print("1. Set up TT-Forge environment: ./scripts/build_frontends.sh --xla")
        print("2. Activate environment: source ./scripts/activate_frontend.sh --xla")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Run full tests: python3 test_training.py")
        return 0
    else:
        print("✗ Some basic tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
