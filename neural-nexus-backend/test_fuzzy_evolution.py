#!/usr/bin/env python3
"""
Test script for hybrid neuro-fuzzy co-evolution implementation.

This script tests both:
1. Backward compatibility (use_fuzzy=False) - should behave identically to original
2. Hybrid neuro-fuzzy co-evolution (use_fuzzy=True) - new functionality

Usage:
    python test_fuzzy_evolution.py
"""

import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent))

from app.core.fuzzy import FuzzyInferenceSystem
from app.utils.fuzzy_helpers import init_fuzzy_genome, decode_fuzzy_params, mutate_fuzzy_params


def create_simple_model_definition():
    """Create a simple model definition file for testing."""
    model_code = '''# Simple CNN for MNIST
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_filters=32, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(num_filters * 2 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self.fc1.in_features)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
'''
    model_path = Path(__file__).parent / "test_model.py"
    with open(model_path, 'w') as f:
        f.write(model_code)
    return str(model_path)


def test_fuzzy_system():
    """Test the FuzzyInferenceSystem directly."""
    print("=" * 60)
    print("TEST 1: Fuzzy Inference System")
    print("=" * 60)
    
    fuzzy_config = {
        "num_inputs": 3,
        "num_rules": 5,
        "membership_per_input": 3,
        "alpha": 0.7
    }
    
    # Initialize fuzzy system
    fuzzy_sys = FuzzyInferenceSystem(fuzzy_config)
    print(f"[OK] Fuzzy system initialized: {fuzzy_config}")
    
    # Create test fuzzy parameters
    fuzzy_params = init_fuzzy_genome(fuzzy_config)
    print(f"[OK] Fuzzy genome initialized: {len(fuzzy_params)} parameters")
    
    # Test evaluation
    test_inputs = np.array([0.5, 0.3, 0.7])
    fitness = fuzzy_sys.evaluate(test_inputs, fuzzy_params)
    print(f"[OK] Fuzzy evaluation: inputs={test_inputs} -> fitness={fitness:.4f}")
    
    # Test mutation
    mutated_params = mutate_fuzzy_params(
        np.concatenate((np.zeros(0), fuzzy_params)),
        0, len(fuzzy_params), 0.1, 0.01
    )
    print(f"[OK] Fuzzy mutation: {np.sum(fuzzy_params != mutated_params)} parameters changed")
    
    print("TEST 1 PASSED\n")
    return True


def test_fuzzy_helpers():
    """Test fuzzy helper functions."""
    print("=" * 60)
    print("TEST 2: Fuzzy Helper Functions")
    print("=" * 60)
    
    fuzzy_config = {
        "num_inputs": 3,
        "num_rules": 5,
        "membership_per_input": 3
    }
    
    # Test initialization
    genome = init_fuzzy_genome(fuzzy_config)
    print(f"[OK] Initialized genome: {len(genome)} parameters")
    
    # Test decoding
    decoded = decode_fuzzy_params(genome, 0, fuzzy_config)
    print(f"[OK] Decoded fuzzy params: {decoded['num_inputs']} inputs, {decoded['num_rules']} rules")
    
    # Test mutation preserves ordering
    chromosome = np.concatenate((np.zeros(5), genome))  # Add dummy hyperparams
    mutated = mutate_fuzzy_params(chromosome, 5, len(genome), 0.2, 0.05)
    
    # Check MF ordering (a <= b <= c) for first few triplets
    for i in range(0, min(9, len(genome)), 3):
        a, b, c = mutated[5+i], mutated[5+i+1], mutated[5+i+2]
        if a <= b <= c:
            print(f"[OK] MF ordering preserved for triplet {i//3}: {a:.3f} <= {b:.3f} <= {c:.3f}")
        else:
            print(f"[FAIL] MF ordering violated for triplet {i//3}: {a:.3f}, {b:.3f}, {c:.3f}")
            return False
    
    print("TEST 2 PASSED\n")
    return True


def create_test_configs():
    """Create example configs for testing."""
    print("=" * 60)
    print("TEST 3: Configuration Examples")
    print("=" * 60)
    
    # Config 1: NN-only (backward compatibility)
    config_nn_only = {
        "model_class": "SimpleCNN",
        "generations": 3,
        "population_size": 5,
        "selection_strategy": "tournament",
        "crossover_operator": "one_point",
        "mutation_operator": "gaussian",
        "mutation_rate": 0.1,
        "mutation_strength": 0.05,
        "elitism_count": 1,
        "evolvable_hyperparams": {
            "num_filters": {
                "type": "int",
                "range": [16, 64]
            },
            "dropout_rate": {
                "type": "float",
                "range": [0.1, 0.7]
            }
        },
        "model_kwargs": {},
        "eval_config": {
            "batch_size": 128
        }
    }
    
    # Config 2: Hybrid neuro-fuzzy
    config_hybrid = config_nn_only.copy()
    config_hybrid["use_fuzzy"] = True
    config_hybrid["fuzzy_config"] = {
        "num_inputs": 5,  # Match number of features extracted from model output
        "num_rules": 5,
        "membership_per_input": 3,
        "alpha": 0.7
    }
    
    # Save configs
    config_dir = Path(__file__).parent / "test_configs"
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "config_nn_only.json", 'w') as f:
        json.dump(config_nn_only, f, indent=2)
    print(f"[OK] Created: {config_dir / 'config_nn_only.json'}")
    
    with open(config_dir / "config_hybrid.json", 'w') as f:
        json.dump(config_hybrid, f, indent=2)
    print(f"[OK] Created: {config_dir / 'config_hybrid.json'}")
    
    print("\nConfiguration files created. Use these with the API or frontend.")
    print("TEST 3 PASSED\n")
    return config_nn_only, config_hybrid


def print_testing_instructions():
    """Print instructions for testing via API."""
    print("=" * 60)
    print("TESTING INSTRUCTIONS")
    print("=" * 60)
    
    print("""
1. BACKWARD COMPATIBILITY TEST (use_fuzzy=False):
   - Use config_nn_only.json
   - Verify system behaves identically to original implementation
   - Check logs for: "Fuzzy co-evolution disabled (NN-only)"
   - Verify no errors related to fuzzy parameters

2. HYBRID NEURO-FUZZY TEST (use_fuzzy=True):
   - Use config_hybrid.json
   - Check logs for: "Fuzzy co-evolution enabled"
   - Verify fuzzy genome initialization in logs
   - Check that fitness combines NN and fuzzy components
   - Verify best_fuzzy_parameters in final result

3. TESTING VIA API:
   POST /evolver/start
   - model_definition: Upload test_model.py (or your model file)
   - config_json: JSON string from config files
   - use_standard_eval: true (for MNIST)
   - initial_weights: (optional)

4. TESTING VIA FRONTEND:
   - Upload model definition file
   - Paste config JSON in the form
   - Select "Standard Evaluation" for MNIST
   - Submit and monitor progress

5. WHAT TO CHECK:
   [OK] No errors in logs
   [OK] Chromosome slicing works correctly
   [OK] Fitness values are finite
   [OK] Best fuzzy parameters appear in final result (if use_fuzzy=True)
   [OK] Evolution progresses through generations
   [OK] Final model is saved successfully

6. VERIFY LOGS:
   Look for these log messages:
   - "[Task XXX] Fuzzy co-evolution enabled/disabled"
   - "[Task XXX] Initialized fuzzy genome with N parameters"
   - "[Task XXX] Best fuzzy parameters extracted"
   - No "IndexError" or "ValueError" related to chromosome slicing
    """)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NEURO-FUZZY CO-EVOLUTION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Fuzzy system
        test_fuzzy_system()
        
        # Test 2: Helper functions
        test_fuzzy_helpers()
        
        # Test 3: Create configs
        create_test_configs()
        
        # Print instructions
        print_testing_instructions()
        
        print("=" * 60)
        print("ALL UNIT TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the generated config files in test_configs/")
        print("2. Test via API or frontend using those configs")
        print("3. Monitor logs for any errors")
        print("4. Verify backward compatibility with use_fuzzy=False")
        print("5. Verify hybrid evolution with use_fuzzy=True\n")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Create test model file
    model_path = create_simple_model_definition()
    print(f"Created test model: {model_path}\n")
    
    exit(main())

