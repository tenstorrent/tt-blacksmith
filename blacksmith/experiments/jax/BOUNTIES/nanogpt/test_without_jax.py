#!/usr/bin/env python3
"""
Test script that validates the implementation without requiring JAX to be installed.
This is useful for testing on systems where JAX installation is problematic.
"""

import os
import sys
import ast
import yaml
from typing import Dict, Any, List

def test_config_loading():
    """Test that configuration files can be loaded."""
    print("Testing configuration loading...")
    
    try:
        # Test YAML configs
        with open("config_cpu.yaml", 'r') as f:
            cpu_config = yaml.safe_load(f)
        
        with open("config_tt.yaml", 'r') as f:
            tt_config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['model_config', 'data_config', 'training_config', 'device_config']
        
        for config_name, config in [("CPU", cpu_config), ("TT", tt_config)]:
            for section in required_sections:
                if section not in config:
                    print(f"❌ {config_name} config missing section: {section}")
                    return False
            print(f"✅ {config_name} config loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_model_structure():
    """Test that the model structure is correct."""
    print("\nTesting model structure...")
    
    try:
        with open("models/gpt_model.py", 'r') as f:
            model_code = f.read()
        
        # Parse the AST to check for required classes
        tree = ast.parse(model_code)
        
        class_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
        
        required_classes = ['CausalSelfAttention', 'MLP', 'Block', 'GPT']
        missing_classes = [cls for cls in required_classes if cls not in class_names]
        
        if missing_classes:
            print(f"❌ Missing model classes: {missing_classes}")
            return False
        
        print("✅ All required model classes found")
        return True
    except Exception as e:
        print(f"❌ Model structure test failed: {e}")
        return False

def test_training_workflow():
    """Test that the training workflow components exist."""
    print("\nTesting training workflow...")
    
    required_files = [
        "train_nanogpt.py",
        "utils/training_utils.py", 
        "utils/device_utils.py",
        "datasets/text_dataset.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing training files: {missing_files}")
        return False
    
    # Check for key functions in training_utils.py
    try:
        with open("utils/training_utils.py", 'r') as f:
            training_code = f.read()
        
        required_functions = [
            'create_optimizer', 'compute_loss', 'training_step', 
            'create_train_state', 'estimate_loss'
        ]
        
        missing_functions = []
        for func in required_functions:
            if f"def {func}" not in training_code:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing training functions: {missing_functions}")
            return False
        
        print("✅ All training workflow components found")
        return True
    except Exception as e:
        print(f"❌ Training workflow test failed: {e}")
        return False

def test_fallback_mechanism():
    """Test that fallback mechanisms are implemented."""
    print("\nTesting fallback mechanism...")
    
    try:
        with open("utils/device_utils.py", 'r') as f:
            device_code = f.read()
        
        # Check for fallback-related code
        fallback_indicators = [
            'fallback', 'cpu', 'tt', 'try:', 'except', 'DeviceManager'
        ]
        
        found_indicators = [indicator for indicator in fallback_indicators if indicator in device_code.lower()]
        
        if len(found_indicators) < 4:  # At least 4 indicators should be present
            print(f"❌ Insufficient fallback indicators found: {found_indicators}")
            return False
        
        print("✅ Fallback mechanism implementation found")
        return True
    except Exception as e:
        print(f"❌ Fallback mechanism test failed: {e}")
        return False

def test_data_pipeline():
    """Test that the data pipeline is implemented."""
    print("\nTesting data pipeline...")
    
    try:
        with open("datasets/text_dataset.py", 'r') as f:
            data_code = f.read()
        
        # Check for key data pipeline components
        required_components = [
            'TextDataset', 'SimpleTokenizer', 'get_batch', 'prepare_data'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in data_code:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing data pipeline components: {missing_components}")
            return False
        
        print("✅ Data pipeline implementation found")
        return True
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False

def test_comparison_framework():
    """Test that the comparison framework exists."""
    print("\nTesting comparison framework...")
    
    try:
        with open("compare_cpu_tt.py", 'r') as f:
            comparison_code = f.read()
        
        # Check for comparison-related functions
        required_functions = [
            'run_training', 'plot_comparison', 'print_comparison_summary'
        ]
        
        missing_functions = []
        for func in required_functions:
            if f"def {func}" not in comparison_code:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing comparison functions: {missing_functions}")
            return False
        
        print("✅ Comparison framework found")
        return True
    except Exception as e:
        print(f"❌ Comparison framework test failed: {e}")
        return False

def test_documentation():
    """Test that documentation is comprehensive."""
    print("\nTesting documentation...")
    
    doc_files = ["README.md", "IMPLEMENTATION_SUMMARY.md"]
    total_doc_size = 0
    
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            size = os.path.getsize(doc_file)
            total_doc_size += size
            print(f"✅ {doc_file}: {size} bytes")
        else:
            print(f"❌ Missing documentation: {doc_file}")
            return False
    
    if total_doc_size < 15000:  # At least 15KB of documentation
        print(f"❌ Insufficient documentation: {total_doc_size} bytes")
        return False
    
    print("✅ Comprehensive documentation found")
    return True

def main():
    """Run all tests without requiring JAX."""
    print("NanoGPT Implementation - No-JAX Validation Tests")
    print("=" * 60)
    print("Testing implementation structure and completeness...")
    
    tests = [
        test_config_loading,
        test_model_structure,
        test_training_workflow,
        test_fallback_mechanism,
        test_data_pipeline,
        test_comparison_framework,
        test_documentation
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
    print("No-JAX Validation Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All structure tests passed!")
        print("\n🎯 Implementation is ready for JAX environment testing!")
        print("\nNext steps:")
        print("1. Set up proper Python 3.10+ environment (or use Koyeb)")
        print("2. Install JAX and dependencies")
        print("3. Run functional tests: python3 test_training.py")
        print("4. Run comparison: python3 compare_cpu_tt.py")
        print("5. Deploy to Koyeb TT-N150 instances for final testing")
        return 0
    else:
        print("❌ Some structure tests failed!")
        print("Please review the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())
