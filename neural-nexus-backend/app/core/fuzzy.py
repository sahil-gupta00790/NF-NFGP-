# app/core/fuzzy.py
"""
Pure-Python Mamdani Fuzzy Inference System for hybrid neuro-fuzzy co-evolution.
Uses triangular membership functions and deterministic inference.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class FuzzyInferenceSystem:
    """
    Mamdani Fuzzy Inference System with triangular membership functions.
    Deterministic and compatible with evolutionary optimization.
    """
    
    def __init__(self, fuzzy_config: dict):
        """
        Initialize the fuzzy inference system.
        
        Args:
            fuzzy_config: Dictionary containing:
                - num_inputs: Number of input variables
                - num_rules: Number of fuzzy rules
                - membership_per_input: Number of membership functions per input
                - alpha: Weight for hybrid fitness (0.7 default)
                - num_output_mfs: Number of output membership functions (default: membership_per_input)
        """
        self.num_inputs = fuzzy_config.get("num_inputs", 3)
        self.num_rules = fuzzy_config.get("num_rules", 5)
        self.membership_per_input = fuzzy_config.get("membership_per_input", 3)
        self.num_output_mfs = fuzzy_config.get("num_output_mfs", self.membership_per_input)
        self.alpha = fuzzy_config.get("alpha", 0.7)
        
        # Validate config
        if self.num_inputs <= 0 or self.num_rules <= 0 or self.membership_per_input <= 0:
            raise ValueError("Fuzzy config must have positive num_inputs, num_rules, and membership_per_input")
    
    def _triangular_mf(self, x: float, a: float, b: float, c: float) -> float:
        """
        Triangular membership function.
        
        Args:
            x: Input value
            a: Left vertex (where MF = 0)
            b: Peak vertex (where MF = 1)
            c: Right vertex (where MF = 0)
        
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:  # b < x < c
            return (c - x) / (c - b) if c != b else 1.0
    
    def _decode_membership_functions(self, params: np.ndarray, num_mfs: int, start_idx: int) -> List[Tuple[float, float, float]]:
        """
        Decode triangular membership function parameters.
        
        Args:
            params: Parameter array
            num_mfs: Number of membership functions
            start_idx: Starting index in params array
        
        Returns:
            List of (a, b, c) tuples for each MF
        """
        mfs = []
        for i in range(num_mfs):
            idx = start_idx + i * 3
            a = params[idx]
            b = params[idx + 1]
            c = params[idx + 2]
            # Ensure ordering: a <= b <= c
            a, b, c = min(a, b, c), sorted([a, b, c])[1], max(a, b, c)
            mfs.append((a, b, c))
        return mfs
    
    def _decode_rules(self, params: np.ndarray, rules_start_idx: int) -> List[Dict[str, Any]]:
        """
        Decode fuzzy rules from parameters.
        
        Args:
            params: Parameter array
            rules_start_idx: Starting index for rule parameters
        
        Returns:
            List of rule dictionaries with antecedents and consequent
        """
        rules = []
        for r in range(self.num_rules):
            rule = {
                'antecedents': [],  # Which MF for each input
                'consequent': 0     # Which output MF
            }
            # Decode antecedents (one MF index per input)
            for inp in range(self.num_inputs):
                idx = rules_start_idx + r * (self.num_inputs + 1) + inp
                mf_idx = int(np.clip(params[idx], 0, self.membership_per_input - 1))
                rule['antecedents'].append(mf_idx)
            # Decode consequent (output MF index)
            idx = rules_start_idx + r * (self.num_inputs + 1) + self.num_inputs
            rule['consequent'] = int(np.clip(params[idx], 0, self.num_output_mfs - 1))
            rules.append(rule)
        return rules
    
    def evaluate(self, inputs: np.ndarray, fuzzy_params: np.ndarray) -> float:
        """
        Evaluate fuzzy system and return fitness score.
        
        Args:
            inputs: Input features (model-output-derived behavioral features)
            fuzzy_params: Encoded fuzzy parameters from chromosome
        
        Returns:
            Fuzzy fitness score [0, 1] (normalized)
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Input size {len(inputs)} != num_inputs {self.num_inputs}")
        
        # Decode parameters
        # 1. Input MFs: num_inputs * membership_per_input * 3 parameters
        input_mfs_start = 0
        input_mfs_params = self.num_inputs * self.membership_per_input * 3
        
        # 2. Rules: num_rules * (num_inputs + 1) parameters
        rules_start = input_mfs_params
        rules_params = self.num_rules * (self.num_inputs + 1)
        
        # 3. Output MFs: num_output_mfs * 3 parameters
        output_mfs_start = rules_start + rules_params
        output_mfs_params = self.num_output_mfs * 3
        
        # Decode input MFs
        input_mfs = []
        for inp in range(self.num_inputs):
            mf_start = input_mfs_start + inp * self.membership_per_input * 3
            mfs = self._decode_membership_functions(fuzzy_params, self.membership_per_input, mf_start)
            input_mfs.append(mfs)
        
        # Decode output MFs
        output_mfs = self._decode_membership_functions(fuzzy_params, self.num_output_mfs, output_mfs_start)
        
        # Decode rules
        rules = self._decode_rules(fuzzy_params, rules_start)
        
        # Fuzzification: Compute membership degrees for each input
        input_memberships = []
        for inp_idx, input_value in enumerate(inputs):
            memberships = []
            for mf_idx, (a, b, c) in enumerate(input_mfs[inp_idx]):
                mu = self._triangular_mf(input_value, a, b, c)
                memberships.append(mu)
            input_memberships.append(memberships)
        
        # Rule evaluation: Compute rule firing strengths
        rule_strengths = []
        for rule in rules:
            # Min (AND) operation for antecedents
            strength = 1.0
            for inp_idx, mf_idx in enumerate(rule['antecedents']):
                mu = input_memberships[inp_idx][mf_idx]
                strength = min(strength, mu)
            rule_strengths.append((strength, rule['consequent']))
        
        # Aggregation: Combine output MFs using max (OR) operation
        # For each output MF, take max of all rule strengths that fire it
        aggregated_output = np.zeros(100)  # Discretized output universe [0, 1]
        output_universe = np.linspace(0, 1, 100)
        
        for strength, consequent_mf_idx in rule_strengths:
            if strength > 0:
                a, b, c = output_mfs[consequent_mf_idx]
                # Clip MFs to [0, 1] range
                a, b, c = max(0, min(1, a)), max(0, min(1, b)), max(0, min(1, c))
                
                # Compute clipped MF for each point in output universe
                for i, x in enumerate(output_universe):
                    mf_value = self._triangular_mf(x, a, b, c)
                    aggregated_output[i] = max(aggregated_output[i], min(strength, mf_value))
        
        # Defuzzification: Centroid method
        numerator = np.sum(output_universe * aggregated_output)
        denominator = np.sum(aggregated_output)
        
        if denominator == 0:
            return 0.5  # Default if no rules fire
        
        fuzzy_output = numerator / denominator
        # Normalize to [0, 1] and return as fitness
        return float(np.clip(fuzzy_output, 0.0, 1.0))

