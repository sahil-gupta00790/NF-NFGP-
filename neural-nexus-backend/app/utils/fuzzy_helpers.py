# app/utils/fuzzy_helpers.py
"""
Fuzzy genome utilities for initialization, decoding, and mutation.
"""

import numpy as np
import random
from typing import Dict, Any


def init_fuzzy_genome(fuzzy_config: dict) -> np.ndarray:
    """
    Initialize fuzzy genome with valid parameters.
    
    Args:
        fuzzy_config: Dictionary containing:
            - num_inputs: Number of input variables
            - num_rules: Number of fuzzy rules
            - membership_per_input: Number of membership functions per input
            - num_output_mfs: Number of output membership functions (default: membership_per_input)
    
    Returns:
        Initialized fuzzy parameter array with valid MF ordering (a <= b <= c)
    """
    num_inputs = fuzzy_config.get("num_inputs", 3)
    num_rules = fuzzy_config.get("num_rules", 5)
    membership_per_input = fuzzy_config.get("membership_per_input", 3)
    num_output_mfs = fuzzy_config.get("num_output_mfs", membership_per_input)
    
    # Calculate total parameters
    # 1. Input MFs: num_inputs * membership_per_input * 3 (a, b, c for each MF)
    input_mfs_params = num_inputs * membership_per_input * 3
    
    # 2. Rules: num_rules * (num_inputs + 1) (antecedents + consequent)
    rules_params = num_rules * (num_inputs + 1)
    
    # 3. Output MFs: num_output_mfs * 3 (a, b, c for each MF)
    output_mfs_params = num_output_mfs * 3
    
    total_params = input_mfs_params + rules_params + output_mfs_params
    fuzzy_genome = np.zeros(total_params, dtype=np.float64)
    
    idx = 0
    
    # Initialize input MFs with valid ordering (a <= b <= c)
    # Distribute MFs across [0, 1] range for each input
    for inp in range(num_inputs):
        for mf in range(membership_per_input):
            # Create overlapping triangular MFs
            center = mf / max(1, membership_per_input - 1) if membership_per_input > 1 else 0.5
            spread = 0.3 / membership_per_input
            
            a = max(0.0, center - spread)
            b = center
            c = min(1.0, center + spread)
            
            # Ensure ordering
            a, b, c = min(a, b, c), sorted([a, b, c])[1], max(a, b, c)
            
            fuzzy_genome[idx] = a
            fuzzy_genome[idx + 1] = b
            fuzzy_genome[idx + 2] = c
            idx += 3
    
    # Initialize rules (antecedents: MF indices, consequent: output MF index)
    for r in range(num_rules):
        # Random antecedents
        for inp in range(num_inputs):
            fuzzy_genome[idx] = random.uniform(0, membership_per_input - 1)
            idx += 1
        # Random consequent
        fuzzy_genome[idx] = random.uniform(0, num_output_mfs - 1)
        idx += 1
    
    # Initialize output MFs with valid ordering
    for mf in range(num_output_mfs):
        center = mf / max(1, num_output_mfs - 1) if num_output_mfs > 1 else 0.5
        spread = 0.3 / num_output_mfs
        
        a = max(0.0, center - spread)
        b = center
        c = min(1.0, center + spread)
        
        # Ensure ordering
        a, b, c = min(a, b, c), sorted([a, b, c])[1], max(a, b, c)
        
        fuzzy_genome[idx] = a
        fuzzy_genome[idx + 1] = b
        fuzzy_genome[idx + 2] = c
        idx += 3
    
    return fuzzy_genome


def decode_fuzzy_params(chromosome: np.ndarray, start_idx: int, fuzzy_config: dict) -> dict:
    """
    Decode fuzzy parameters from chromosome.
    
    Args:
        chromosome: Full chromosome array
        start_idx: Starting index of fuzzy parameters
        fuzzy_config: Fuzzy configuration dictionary
    
    Returns:
        Dictionary with decoded fuzzy parameters structure
    """
    num_inputs = fuzzy_config.get("num_inputs", 3)
    num_rules = fuzzy_config.get("num_rules", 5)
    membership_per_input = fuzzy_config.get("membership_per_input", 3)
    num_output_mfs = fuzzy_config.get("num_output_mfs", membership_per_input)
    
    # Calculate parameter counts
    input_mfs_params = num_inputs * membership_per_input * 3
    rules_params = num_rules * (num_inputs + 1)
    output_mfs_params = num_output_mfs * 3
    
    end_idx = start_idx + input_mfs_params + rules_params + output_mfs_params
    
    if len(chromosome) < end_idx:
        raise ValueError(f"Chromosome too short: {len(chromosome)} < {end_idx}")
    
    fuzzy_params = chromosome[start_idx:end_idx].copy()
    
    return {
        'params': fuzzy_params,
        'num_inputs': num_inputs,
        'num_rules': num_rules,
        'membership_per_input': membership_per_input,
        'num_output_mfs': num_output_mfs,
        'input_mfs_start': 0,
        'rules_start': input_mfs_params,
        'output_mfs_start': input_mfs_params + rules_params
    }


def mutate_fuzzy_params(
    chromosome: np.ndarray,
    start_idx: int,
    num_fuzzy_params: int,
    mutation_rate: float,
    mutation_strength: float
) -> np.ndarray:
    """
    Mutate fuzzy parameters while preserving MF ordering (a <= b <= c).
    
    Args:
        chromosome: Full chromosome array
        start_idx: Starting index of fuzzy parameters
        num_fuzzy_params: Number of fuzzy parameters
        mutation_rate: Probability of mutating each parameter
        mutation_strength: Standard deviation for Gaussian mutation
    
    Returns:
        Mutated chromosome
    """
    if num_fuzzy_params == 0:
        return chromosome
    
    if mutation_rate <= 0 or mutation_strength <= 0:
        return chromosome
    
    mutated_chromosome = chromosome.copy()
    fuzzy_section = mutated_chromosome[start_idx:start_idx + num_fuzzy_params]
    
    # Apply mutation
    mutation_mask = np.random.rand(num_fuzzy_params) < mutation_rate
    num_mutations = np.sum(mutation_mask)
    
    if num_mutations > 0:
        noise = np.random.normal(0, mutation_strength, size=num_mutations)
        fuzzy_section[mutation_mask] += noise
    
    # Preserve MF ordering for triangular MFs (a <= b <= c)
    # This is a simplified approach - assumes MFs are stored as triplets
    # For a more robust solution, we'd need fuzzy_config to know exact structure
    # For now, we'll enforce ordering on all triplets in the fuzzy section
    
    # Group parameters into triplets and enforce ordering
    num_triplets = num_fuzzy_params // 3
    for i in range(num_triplets):
        triplet_start = start_idx + i * 3
        if triplet_start + 2 < start_idx + num_fuzzy_params:
            a, b, c = (
                mutated_chromosome[triplet_start],
                mutated_chromosome[triplet_start + 1],
                mutated_chromosome[triplet_start + 2]
            )
            # Enforce ordering: a <= b <= c
            sorted_vals = sorted([a, b, c])
            mutated_chromosome[triplet_start] = sorted_vals[0]
            mutated_chromosome[triplet_start + 1] = sorted_vals[1]
            mutated_chromosome[triplet_start + 2] = sorted_vals[2]
    
    # Clamp all fuzzy parameters to reasonable range [0, 1] for MFs
    # Rule parameters can go slightly outside for flexibility
    mutated_chromosome[start_idx:start_idx + num_fuzzy_params] = np.clip(
        mutated_chromosome[start_idx:start_idx + num_fuzzy_params],
        -1.0, 2.0  # Allow some flexibility for rule indices
    )
    
    return mutated_chromosome

