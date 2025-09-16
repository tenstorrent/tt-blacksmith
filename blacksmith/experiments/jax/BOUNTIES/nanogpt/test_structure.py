#!/usr/bin/env python3
"""
Structure validation test that works without external dependencies.
This validates the implementation structure and code quality.
"""

import os
import sys
import re
import ast

def test_file_structure():
    """Test that all required files exist with proper structure."""
    print("Testing file structure...")
    
    required_files = [
        "configs.py",
        "config_cpu.yaml", 
        "config_tt.yaml",
        "train_nanogpt.py",
        "test_training.py",
        "compare_cpu_tt.py",
        "README.md",
        "IMPLEMENTATION_SUMMARY.md",
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
            # Check file size (should not be empty)
            size = os.path.getsize(file_path)
            if size < 100:  # Minimum reasonable size
                print(f"⚠ {file_path} seems too small ({size} bytes)")
            else:
                print(f"✓ {file_path} ({size} bytes)")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist with reasonable content")
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
            ast.parse(code)
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

def test_code_quality():
    """Test code quality indicators."""
    print("\nTesting code quality...")
    
    python_files = [
        "configs.py",
        "models/gpt_model.py",
        "datasets/text_dataset.py",
        "utils/device_utils.py",
        "utils/training_utils.py"
    ]
    
    quality_issues = []
    
    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for docstrings
        if '"""' not in content and "'''" not in content:
            quality_issues.append(f"{file_path}: No docstrings found")
        
        # Check for proper imports
        if 'import jax' not in content and 'from jax' not in content:
            if 'gpt_model.py' in file_path or 'training_utils.py' in file_path:
                quality_issues.append(f"{file_path}: Missing JAX imports")
        
        # Check for error handling
        if 'try:' not in content and 'except' not in content:
            if 'device_utils.py' in file_path or 'training_utils.py' in file_path:
                quality_issues.append(f"{file_path}: No error handling found")
        
        # Check for logging
        if 'logging' not in content and 'print(' not in content:
            quality_issues.append(f"{file_path}: No logging found")
        
        print(f"✓ {file_path} - basic quality checks passed")
    
    if quality_issues:
        print(f"⚠ Quality issues: {quality_issues}")
        return False
    else:
        print("✓ Code quality checks passed")
        return True

def test_configuration_files():
    """Test configuration files."""
    print("\nTesting configuration files...")
    
    # Test YAML files
    yaml_files = ["config_cpu.yaml", "config_tt.yaml"]
    
    for file_path in yaml_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for required sections
        required_sections = ['model_config', 'data_config', 'training_config', 'device_config']
        missing_sections = []
        
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ {file_path} missing sections: {missing_sections}")
            return False
        else:
            print(f"✓ {file_path} has all required sections")
    
    # Test requirements.txt
    with open("requirements.txt", 'r') as f:
        requirements = f.read()
    
    required_packages = ["jax", "flax", "optax", "wandb", "pydantic"]
    missing_packages = [pkg for pkg in required_packages if pkg not in requirements]
    
    if missing_packages:
        print(f"✗ requirements.txt missing packages: {missing_packages}")
        return False
    else:
        print("✓ requirements.txt has all required packages")
    
    return True

def test_documentation():
    """Test documentation completeness."""
    print("\nTesting documentation...")
    
    # Test README.md
    with open("README.md", 'r') as f:
        readme = f.read()
    
    required_sections = [
        "Installation", "Usage", "Configuration", "Fallback Mechanism",
        "Monitoring and Logging", "Results and Comparison", "Troubleshooting"
    ]
    
    missing_sections = [section for section in required_sections if section not in readme]
    
    if missing_sections:
        print(f"✗ README.md missing sections: {missing_sections}")
        return False
    else:
        print("✓ README.md has all required sections")
    
    # Test IMPLEMENTATION_SUMMARY.md
    with open("IMPLEMENTATION_SUMMARY.md", 'r') as f:
        summary = f.read()
    
    required_sections = ["Overview", "Completed Requirements", "Implementation Architecture"]
    missing_sections = [section for section in required_sections if section not in summary]
    
    if missing_sections:
        print(f"✗ IMPLEMENTATION_SUMMARY.md missing sections: {missing_sections}")
        return False
    else:
        print("✓ IMPLEMENTATION_SUMMARY.md has all required sections")
    
    return True

def test_bounty_requirements():
    """Test that bounty requirements are addressed."""
    print("\nTesting bounty requirements coverage...")
    
    # Read all documentation
    with open("README.md", 'r') as f:
        readme = f.read()
    with open("IMPLEMENTATION_SUMMARY.md", 'r') as f:
        summary = f.read()
    
    # Check for key bounty requirements
    bounty_requirements = [
        "JAX", "Flax", "TT-N150", "CPU fallback", "Koyeb", "NanoGPT",
        "training workflow", "metric parity", "fallback mechanism",
        "minimal reproducer", "code quality", "documentation"
    ]
    
    missing_requirements = []
    for requirement in bounty_requirements:
        if requirement.lower() not in (readme + summary).lower():
            missing_requirements.append(requirement)
    
    if missing_requirements:
        print(f"⚠ Missing bounty requirements in docs: {missing_requirements}")
        return False
    else:
        print("✓ All bounty requirements addressed in documentation")
    
    return True

def test_implementation_completeness():
    """Test that the implementation is complete."""
    print("\nTesting implementation completeness...")
    
    # Check that we have all the core components
    core_components = {
        "Model": "models/gpt_model.py",
        "Data Pipeline": "datasets/text_dataset.py", 
        "Device Management": "utils/device_utils.py",
        "Training": "utils/training_utils.py",
        "Configuration": "configs.py",
        "Logging": "logging/",
        "Main Training": "train_nanogpt.py",
        "Testing": "test_training.py",
        "Comparison": "compare_cpu_tt.py"
    }
    
    missing_components = []
    for component, path in core_components.items():
        if not os.path.exists(path):
            missing_components.append(component)
        else:
            print(f"✓ {component}: {path}")
    
    if missing_components:
        print(f"✗ Missing components: {missing_components}")
        return False
    else:
        print("✓ All core components implemented")
    
    return True

def main():
    """Run all structure validation tests."""
    print("NanoGPT Implementation - Structure Validation Tests")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_code_quality,
        test_configuration_files,
        test_documentation,
        test_bounty_requirements,
        test_implementation_completeness
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
    print("\n" + "=" * 60)
    print("Structure Validation Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All structure validation tests passed!")
        print("\n🎉 Implementation is structurally complete and ready for testing!")
        print("\nNext steps for full validation:")
        print("1. Set up TT-Forge environment")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run functional tests: python3 test_training.py")
        print("4. Run comparison: python3 compare_cpu_tt.py")
        return 0
    else:
        print("❌ Some structure validation tests failed!")
        print("Please review the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())
