# app/utils/evolution_helpers.py
# Contains helper functions for the Celery evolution task.

import torch
import torch.nn as nn
import numpy as np
import os
import importlib.util
import time
import random
import logging
from typing import List, Dict, Any, Tuple # Added typing

logger = logging.getLogger(__name__)

# --- Utility Functions (Model Loading, Weight Handling, Evaluation) ---

def flatten_weights(model: nn.Module) -> np.ndarray:
    """ Flattens all model parameters into a single numpy vector. """
    try:
        weights = []
        for param in model.parameters():
            if param.requires_grad:
                weights.append(param.data.cpu().numpy().flatten())
        if not weights:
            logger.warning("No trainable parameters found in the model to flatten.")
            return np.array([], dtype=np.float32) # Return typed empty array
        return np.concatenate(weights).astype(np.float32) # Ensure float32
    except Exception as e:
        logger.error(f"Error during weight flattening: {e}", exc_info=True)
        raise

def load_weights_from_flat(model: nn.Module, flat_weights: np.ndarray):
    """ Loads flattened weights back into a model instance. Assumes flat_weights contains ONLY weights. """
    try:
        offset = 0
        # Ensure input is numpy float32
        if not isinstance(flat_weights, np.ndarray): flat_weights = np.array(flat_weights)
        flat_weights = flat_weights.astype(np.float32)

        flat_weights_tensor = torch.from_numpy(flat_weights)
        model_device = next(model.parameters()).device

        # Compare against model parameter count
        total_elements_in_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_elements_in_model != len(flat_weights_tensor):
             # This is now more likely if num_hyperparams is wrong, make it an error
             error_msg = f"Weight Size mismatch: Model requires {total_elements_in_model} elements, but flat_weights has {len(flat_weights_tensor)}. Check hyperparam count or model architecture."
             logger.error(error_msg)
             raise ValueError(error_msg)

        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param_shape = param.size()
                if offset + numel > len(flat_weights_tensor):
                    error_msg = f"Shape mismatch: Not enough data in flat_weights (len {len(flat_weights_tensor)}) to fill parameter {param_shape} (needs {numel} at offset {offset})"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Slice, reshape, move to device, and copy
                param_slice = flat_weights_tensor[offset:offset + numel].view(param_shape).to(model_device)
                with torch.no_grad():
                    param.data.copy_(param_slice)
                offset += numel

        # Final check (should always match if initial check passed)
        if offset != len(flat_weights_tensor):
            logger.error(f"Critical Size mismatch after loading weights. Offset {offset} != weights length {len(flat_weights_tensor)}.")
            # This indicates a logic error, raise it
            raise RuntimeError("Internal error during weight loading: size mismatch detected.")

    except Exception as e:
        logger.error(f"Error loading weights from flat vector: {e}", exc_info=True)
        raise

# NEW Helper to decode hyperparameters
def decode_hyperparameters(
    hyperparam_vector: np.ndarray,
    hyperparam_keys: List[str],
    evolvable_hyperparams_config: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Decodes the hyperparameter vector based on config (including type conversion)."""
    decoded = {}
    if len(hyperparam_vector) != len(hyperparam_keys):
        raise ValueError(f"Mismatch between hyperparam vector size ({len(hyperparam_vector)}) and keys ({len(hyperparam_keys)})")

    for i, key in enumerate(hyperparam_keys):
        value = hyperparam_vector[i]
        h_config = evolvable_hyperparams_config.get(key, {})
        h_type = h_config.get('type', 'float') # Default to float if not specified

        try:
            if h_type == 'int':
                # Round and convert to int, potentially clamp if range specified
                min_val, max_val = h_config.get('range', [None, None])
                decoded_val = int(round(value))
                if min_val is not None: decoded_val = max(int(min_val), decoded_val)
                if max_val is not None: decoded_val = min(int(max_val), decoded_val)
                decoded[key] = decoded_val
            elif h_type == 'float':
                 # Potentially clamp if range specified
                min_val, max_val = h_config.get('range', [None, None])
                decoded_val = float(value)
                if min_val is not None: decoded_val = max(float(min_val), decoded_val)
                if max_val is not None: decoded_val = min(float(max_val), decoded_val)
                decoded[key] = decoded_val
            # Add cases for other types like 'categorical' if needed
            else:
                logger.warning(f"Unsupported hyperparameter type '{h_type}' for key '{key}'. Treating as float.")
                decoded[key] = float(value)
        except Exception as e:
            logger.error(f"Error decoding hyperparameter '{key}' with value {value}: {e}", exc_info=True)
            # Fallback: Use midpoint of range or default? For now, error out.
            raise ValueError(f"Failed to decode hyperparameter '{key}'") from e
    return decoded


def load_pytorch_model(
    model_definition_path: str,
    class_name: str,
    state_dict_path: str | None,
    device: torch.device,
    *model_args_static: Any, # Renamed for clarity
    **model_kwargs_combined: Any # Receives static + dynamic kwargs merged
) -> nn.Module:
    """ Loads the model class, instantiates it with combined kwargs, and loads state_dict if provided. """
    try:
        norm_model_path = os.path.normpath(model_definition_path)
        if not os.path.exists(norm_model_path):
             raise FileNotFoundError(f"Model definition file not found at {norm_model_path}")

        module_name = f"model_module_{random.randint(1000, 9999)}_{os.path.basename(norm_model_path).split('.')[0]}" # Unique name
        spec = importlib.util.spec_from_file_location(module_name, norm_model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {norm_model_path}")

        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        if not hasattr(model_module, class_name):
            available_classes = [name for name, obj in model_module.__dict__.items() if isinstance(obj, type)]
            raise AttributeError(f"Class '{class_name}' not found in {norm_model_path}. Available: {available_classes}")

        ModelClass = getattr(model_module, class_name)

        # Instantiate using static args and COMBINED kwargs (static + dynamic/evolved)
        logger.info(f"Instantiating '{class_name}' | Args: {model_args_static} | Kwargs: {model_kwargs_combined}")
        model = ModelClass(*model_args_static, **model_kwargs_combined)
        model.to(device)

        # Load state_dict if provided (typically only for the very first model/generation)
        if state_dict_path:
            norm_weights_path = os.path.normpath(state_dict_path)
            if os.path.exists(norm_weights_path):
                logger.info(f"Loading state_dict from: {norm_weights_path}")
                try:
                    state_dict = torch.load(norm_weights_path, map_location=device)
                    model.load_state_dict(state_dict, strict=False) # Use strict=False initially? Be careful.
                    logger.info("State_dict loaded.")
                except Exception as load_err:
                    logger.error(f"Error loading state_dict: {load_err}", exc_info=True)
                    # Decide whether to raise or just warn
            else:
                logger.warning(f"state_dict path '{norm_weights_path}' provided but not found.")
        # else: logger.info("No state_dict path provided, using model init weights.") # Less verbose

        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error in load_pytorch_model: {e}", exc_info=True)
        raise

def load_task_eval_function(task_module_path: str) -> callable:
    """ Loads the fitness evaluation function (expecting name 'evaluate_model'). """
    try:
        norm_eval_path = os.path.normpath(task_module_path)
        if not os.path.exists(norm_eval_path):
            raise FileNotFoundError(f"Evaluation script not found at {norm_eval_path}")

        module_name = f"eval_module_{random.randint(1000, 9999)}_{os.path.basename(norm_eval_path).split('.')[0]}" # Unique name
        spec = importlib.util.spec_from_file_location(module_name, norm_eval_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {norm_eval_path}")

        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)

        # Standard expected function name
        eval_func_name = 'evaluate_model'
        if not hasattr(task_module, eval_func_name):
            # Backward compatibility check (optional)
            alt_func_name = 'evaluate_network_on_task'
            if hasattr(task_module, alt_func_name):
                 logger.warning(f"Found legacy eval func name '{alt_func_name}'. Recommend renaming to '{eval_func_name}'.")
                 eval_func_name = alt_func_name
            else:
                 raise AttributeError(f"Function '{eval_func_name}' (or legacy '{alt_func_name}') not found in {norm_eval_path}")

        logger.info(f"Loaded evaluation function '{eval_func_name}' from {norm_eval_path}")
        eval_func = getattr(task_module, eval_func_name)

        # Check signature (optional but recommended)
        import inspect
        sig = inspect.signature(eval_func)
        if len(sig.parameters) < 3:
             logger.warning(f"Evaluation function '{eval_func_name}' has < 3 parameters ({len(sig.parameters)}). Expected signature like `(model, device, config)`.")
             # Decide whether to raise or just warn based on strictness needed

        return eval_func
    except Exception as e:
        logger.error(f"Error loading task evaluation function: {e}", exc_info=True)
        raise

# MODIFIED: Handles hyperparameter evolution and fuzzy co-evolution
def evaluate_population_step(
    population: List[np.ndarray], # List of full chromosomes (hyperparams + fuzzy_params + weights)
    model_definition_path: str,
    class_name: str,
    task_eval_func: callable,
    device: torch.device,
    model_args_static: List[Any],
    model_kwargs_static: Dict[str, Any],
    eval_config: Dict[str, Any], # Passed from task
    num_hyperparams: int, # Number of hyperparameter genes
    evolvable_hyperparams_config: Dict[str, Dict[str, Any]], # Config for decoding
    use_fuzzy: bool = False, # Whether to use fuzzy co-evolution
    num_fuzzy_params: int = 0, # Number of fuzzy parameters
    fuzzy_config: Dict[str, Any] = {} # Fuzzy configuration
) -> List[float]:
    """ Evaluates fitness, handling hyperparameter decoding and dynamic model instantiation. """
    fitness_scores = [-float('inf')] * len(population) # Initialize with failure sentinel
    num_individuals = len(population)
    evaluation_times = []
    hyperparam_keys = list(evolvable_hyperparams_config.keys()) # Get keys for decoding

    logger.info(f"Evaluating {num_individuals} individuals with {num_hyperparams} hyperparameters...")
    if use_fuzzy:
        logger.info(f"Fuzzy co-evolution enabled with {num_fuzzy_params} fuzzy parameters")
    
    # Import fuzzy components if needed
    if use_fuzzy:
        from app.core.fuzzy import FuzzyInferenceSystem
        from app.utils.fuzzy_helpers import decode_fuzzy_params

    # Pre-load the model CLASS definition once (reduces overhead in loop)
    # This part needs careful error handling as failure here affects all individuals
    try:
        module_name = f"model_module_eval_{random.randint(1000, 9999)}"
        spec = importlib.util.spec_from_file_location(module_name, model_definition_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        ModelClass = getattr(model_module, class_name)
        logger.debug(f"Successfully pre-loaded ModelClass '{class_name}' for evaluation.")
    except Exception as e:
         logger.error(f"Failed to preload ModelClass '{class_name}' for evaluation: {e}", exc_info=True)
         # Cannot proceed if model class cannot be loaded
         raise RuntimeError(f"Failed to load ModelClass '{class_name}' for evaluation.") from e


    for i, chromosome in enumerate(population):
        individual_start_time = time.time()
        current_model = None
        decoded_hparams = {}
        try:
            # 1. Separate and Decode Hyperparameters
            if num_hyperparams > 0:
                 if len(chromosome) < num_hyperparams:
                      raise ValueError(f"Chromosome length ({len(chromosome)}) is less than num_hyperparams ({num_hyperparams})")
                 hyperparam_vector = chromosome[:num_hyperparams]
                 decoded_hparams = decode_hyperparameters(hyperparam_vector, hyperparam_keys, evolvable_hyperparams_config)
            else:
                 # If no hyperparams evolved, decoded_hparams remains empty
                 pass

            # 2. Instantiate Model with Evolved Hyperparameters + Static Config
            combined_kwargs = {**model_kwargs_static, **decoded_hparams}
            # Note: load_pytorch_model now handles instantiation with combined kwargs
            current_model = ModelClass(*model_args_static, **combined_kwargs)
            current_model.to(device)

            # 3. Extract fuzzy parameters (if enabled)
            fuzzy_params = None
            if use_fuzzy and num_fuzzy_params > 0:
                fuzzy_start = num_hyperparams
                fuzzy_end = fuzzy_start + num_fuzzy_params
                if len(chromosome) >= fuzzy_end:
                    fuzzy_params = chromosome[fuzzy_start:fuzzy_end]
                else:
                    logger.warning(f"Ind {i+1}: Chromosome too short for fuzzy params")
            
            # 4. Load Weights (accounting for fuzzy params)
            weights_start = num_hyperparams + num_fuzzy_params
            if len(chromosome) > weights_start:
                weights_vector = chromosome[weights_start:]
                load_weights_from_flat(current_model, weights_vector)
            # else: only hyperparams/fuzzy evolved, use initial model weights (already set by ModelClass init)

            # 5. Evaluate Neural Network
            current_model.eval()
            with torch.no_grad():
                # Call user's function: (model, device, config_dict)
                # Ensure eval_config contains the device before calling
                config_for_eval = {**eval_config, 'device': device} # Merge device into the config copy
                nn_fitness = task_eval_func(current_model, config_for_eval) # Pass model and combined config

            # Validate NN fitness value
            if not isinstance(nn_fitness, (float, int)):
                logger.warning(f"Ind {i+1}: Fitness non-numeric ({type(nn_fitness)}). Setting -inf.")
                nn_fitness = -float('inf')
            elif not np.isfinite(nn_fitness):
                logger.warning(f"Ind {i+1}: Fitness non-finite ({nn_fitness}). Setting -inf.")
                nn_fitness = -float('inf')
            
            # 6. Compute Fuzzy Fitness (if enabled)
            if use_fuzzy and num_fuzzy_params > 0 and fuzzy_params is not None:
                try:
                    # Extract model-output-derived behavioral features
                    # Use a sample forward pass to get behavioral features
                    current_model.eval()
                    with torch.no_grad():
                        # Get sample input from eval_config if available
                        sample_input = eval_config.get('sample_input', None)
                        if sample_input is None:
                            # Create dummy input based on model's expected input shape
                            # This is a fallback - ideally eval_config should provide sample_input
                            input_shape = eval_config.get('input_shape', (1, 28, 28))  # Default MNIST-like
                            sample_input = torch.randn(input_shape).to(device)
                        
                        # Forward pass to get model output
                        model_output = current_model(sample_input)
                        
                        # Extract behavioral features from model output
                        # Use statistical features: mean, std, min, max, median
                        output_np = model_output.cpu().numpy().flatten()
                        features = [
                            float(np.mean(output_np)),
                            float(np.std(output_np)),
                            float(np.min(output_np)),
                            float(np.max(output_np)),
                            float(np.median(output_np))
                        ]
                        
                        # Pad or truncate to match num_inputs
                        num_inputs = fuzzy_config.get("num_inputs", 3)
                        if len(features) < num_inputs:
                            # Pad with zeros
                            features.extend([0.0] * (num_inputs - len(features)))
                        else:
                            # Truncate to num_inputs
                            features = features[:num_inputs]
                        
                        # Normalize features to [0, 1] range
                        features = np.array(features)
                        if np.max(features) != np.min(features):
                            features = (features - np.min(features)) / (np.max(features) - np.min(features))
                        else:
                            features = np.ones_like(features) * 0.5
                        
                        # Evaluate fuzzy system
                        fuzzy_system = FuzzyInferenceSystem(fuzzy_config)
                        fuzzy_fitness = fuzzy_system.evaluate(features, fuzzy_params)
                        
                        # Combine NN and fuzzy fitness
                        alpha = fuzzy_config.get("alpha", 0.7)
                        fitness = alpha * nn_fitness + (1 - alpha) * fuzzy_fitness
                        
                except Exception as fuzzy_err:
                    logger.warning(f"Ind {i+1}: Fuzzy evaluation failed: {fuzzy_err}. Using NN fitness only.")
                    fitness = nn_fitness
            else:
                # No fuzzy evaluation, use NN fitness only
                fitness = nn_fitness

            fitness_scores[i] = float(fitness) # Store valid float or -inf

        except Exception as e:
            logger.error(f"Error evaluating individual {i+1} (HParams: {decoded_hparams}): {e}", exc_info=True)
            # fitness_scores[i] remains -inf (already initialized)
        finally:
            # Cleanup
            del current_model # Ensure model object is released
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Clear CUDA cache
            eval_time = time.time() - individual_start_time
            evaluation_times.append(eval_time)

    avg_eval_time = np.mean(evaluation_times) if evaluation_times else 0
    valid_count = sum(1 for f in fitness_scores if f > -float('inf'))
    logger.info(f"Finished evaluation ({valid_count}/{num_individuals} valid). Avg time/ind: {avg_eval_time:.3f}s")
    return fitness_scores

# --- Genetic Algorithm Operators (Modified for Hyperparams + Weights) ---

# --- Selection Operators ---
# These operate on fitness scores and return selected *chromosomes* (which contain hyperparams+weights)
# No fundamental change needed to tournament/roulette logic itself, just ensure they receive/return full chromosomes

def select_parents_tournament(
    population_chromosomes: List[np.ndarray], # Changed name for clarity
    fitness_scores: List[float],
    num_parents: int,
    tournament_size: int = 3
) -> List[np.ndarray]:
    """ Selects parent chromosomes using tournament selection. """
    parents = []
    population_size = len(population_chromosomes)
    if population_size == 0 or num_parents <= 0: return []
    # ... (Rest of tournament logic remains the same, operating on indices) ...
    actual_tournament_size = max(2, min(population_size, tournament_size))
    valid_indices = [i for i, f in enumerate(fitness_scores) if f > -float('inf')]
    if not valid_indices:
        logger.warning("No valid individuals for tournament selection.")
        return []
    num_valid = len(valid_indices)
    for _ in range(num_parents):
        participants_count = min(num_valid, actual_tournament_size)
        if participants_count < 1: break
        replace = num_valid < actual_tournament_size
        tournament_indices = np.random.choice(valid_indices, size=participants_count, replace=replace).tolist()
        winner_idx_in_pop = -1
        best_fit_tourn = -float('inf')
        for idx in tournament_indices:
            if fitness_scores[idx] > best_fit_tourn:
                best_fit_tourn = fitness_scores[idx]
                winner_idx_in_pop = idx
        if winner_idx_in_pop != -1:
            parents.append(population_chromosomes[winner_idx_in_pop]) # Return selected chromosome
        elif valid_indices:
            logger.warning("Tournament failed to select winner, picking random valid.")
            parents.append(population_chromosomes[random.choice(valid_indices)])
    return parents

def select_parents_roulette(
    population_chromosomes: List[np.ndarray], # Changed name for clarity
    fitness_scores: List[float],
    num_parents: int
) -> List[np.ndarray]:
    """ Selects parent chromosomes using Roulette Wheel selection. """
    # ... (Input validation remains the same) ...
    if not population_chromosomes or not fitness_scores or len(population_chromosomes) != len(fitness_scores): raise ValueError("Population/fitness mismatch.")
    if num_parents <= 0: return []

    fitness_np = np.array(fitness_scores, dtype=np.float64)
    valid_indices = np.where(fitness_np > -np.inf)[0]
    # ... (Handling invalid/non-positive fitness remains the same) ...
    if len(valid_indices) == 0:
         logger.warning("Roulette: All invalid fitness. Selecting uniformly.")
         indices = np.random.choice(len(population_chromosomes), size=num_parents, replace=True)
         return [population_chromosomes[i] for i in indices]
    valid_fitness = fitness_np[valid_indices]
    if np.all(valid_fitness <= 0):
        min_valid = np.min(valid_fitness)
        if min_valid > -np.inf: # Check if there's at least one non -inf score
            shifted_fitness = valid_fitness - min_valid + 1e-9 # Shift to positive
        else: # All valid scores are -inf, select uniformly
             logger.warning("Roulette: All valid fitness scores are -inf. Selecting uniformly.")
             selected_valid_indices = np.random.choice(valid_indices, size=num_parents, replace=True)
             return [population_chromosomes[i] for i in selected_valid_indices]
    else: # At least one positive score exists
        shifted_fitness = valid_fitness.copy()
        shifted_fitness[shifted_fitness < 0] = 0 # Zero out any remaining negatives

    total_fitness = np.sum(shifted_fitness)
    if total_fitness <= 0:
         logger.warning("Roulette: Total fitness zero after shift. Selecting uniformly among valid.")
         selected_valid_indices = np.random.choice(valid_indices, size=num_parents, replace=True)
         return [population_chromosomes[i] for i in selected_valid_indices]

    probabilities = shifted_fitness / total_fitness
    probabilities /= np.sum(probabilities) # Re-normalize

    # Select indices from the *valid* subset based on probability
    selected_indices_in_valid = np.random.choice(len(valid_indices), size=num_parents, replace=True, p=probabilities)
    # Map back to original population indices
    selected_original_indices = valid_indices[selected_indices_in_valid]
    selected_parents = [population_chromosomes[i] for i in selected_original_indices]
    return selected_parents


# --- Crossover Operators ---
# MODIFIED: Accept num_hyperparams and operate separately

def crossover_average(parent1: np.ndarray, parent2: np.ndarray, num_hyperparams: int, num_fuzzy_params: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """ Average crossover for hyperparams, fuzzy params, and weights separately. """
    p1, p2 = np.asarray(parent1), np.asarray(parent2)
    weights_start = num_hyperparams + num_fuzzy_params
    if p1.shape != p2.shape or len(p1) <= weights_start:
        logger.warning("Shape mismatch or insufficient length for average crossover. Returning copies.")
        return p1.copy(), p2.copy()

    # Average hyperparams
    h_child = (p1[:num_hyperparams] + p2[:num_hyperparams]) / 2.0 if num_hyperparams > 0 else np.array([])
    # Average fuzzy params (as contiguous block)
    f_child = (p1[num_hyperparams:weights_start] + p2[num_hyperparams:weights_start]) / 2.0 if num_fuzzy_params > 0 else np.array([])
    # Average weights
    w_child = (p1[weights_start:] + p2[weights_start:]) / 2.0

    child = np.concatenate((h_child, f_child, w_child))
    return child.copy(), child.copy() # Returns two identical children

def crossover_one_point(parent1: np.ndarray, parent2: np.ndarray, num_hyperparams: int, num_fuzzy_params: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """ One-point crossover for hyperparams, fuzzy params (as block), and weights separately. """
    p1, p2 = np.asarray(parent1), np.asarray(parent2)
    weights_start = num_hyperparams + num_fuzzy_params
    if p1.shape != p2.shape or len(p1) <= weights_start:
        logger.warning("Shape mismatch/insufficient length for one-point crossover. Returning copies.")
        return p1.copy(), p2.copy()

    # Crossover Hyperparameters
    if num_hyperparams > 1:
        hp_cross_pt = random.randint(1, num_hyperparams - 1)
        h1_new = np.concatenate((p1[:hp_cross_pt], p2[hp_cross_pt:num_hyperparams]))
        h2_new = np.concatenate((p2[:hp_cross_pt], p1[hp_cross_pt:num_hyperparams]))
    elif num_hyperparams == 1: # Swap single hyperparam
        h1_new, h2_new = p2[:1], p1[:1]
    else: # No hyperparams
        h1_new, h2_new = np.array([]), np.array([])

    # Crossover Fuzzy Params (as contiguous block - swap entire block)
    if num_fuzzy_params > 0:
        # Randomly decide whether to swap fuzzy block
        if random.random() < 0.5:
            f1_new = p1[num_hyperparams:weights_start]
            f2_new = p2[num_hyperparams:weights_start]
        else:
            f1_new = p2[num_hyperparams:weights_start]
            f2_new = p1[num_hyperparams:weights_start]
    else:
        f1_new, f2_new = np.array([]), np.array([])

    # Crossover Weights
    weight_size = len(p1) - weights_start
    if weight_size > 1:
        wt_cross_pt = random.randint(1, weight_size - 1)
        w1_new = np.concatenate((p1[weights_start : weights_start + wt_cross_pt], p2[weights_start + wt_cross_pt :]))
        w2_new = np.concatenate((p2[weights_start : weights_start + wt_cross_pt], p1[weights_start + wt_cross_pt :]))
    elif weight_size == 1: # Swap single weight
        w1_new, w2_new = p2[weights_start:], p1[weights_start:]
    else: # No weights (unlikely)
        w1_new, w2_new = np.array([]), np.array([])

    child1 = np.concatenate((h1_new, f1_new, w1_new))
    child2 = np.concatenate((h2_new, f2_new, w2_new))
    return child1, child2

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray, num_hyperparams: int, crossover_prob: float = 0.5, num_fuzzy_params: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """ Uniform Crossover for hyperparams, fuzzy params (as block), and weights separately. """
    p1, p2 = np.asarray(parent1), np.asarray(parent2)
    weights_start = num_hyperparams + num_fuzzy_params
    if p1.shape != p2.shape or len(p1) <= weights_start:
        logger.warning("Shape mismatch/insufficient length for uniform crossover. Returning copies.")
        return p1.copy(), p2.copy()

    # Crossover Hyperparameters
    h1_new, h2_new = p1[:num_hyperparams], p2[:num_hyperparams]
    if num_hyperparams > 0:
        hp_swap_mask = np.random.rand(num_hyperparams) < crossover_prob
        h1_new = np.where(hp_swap_mask, p2[:num_hyperparams], p1[:num_hyperparams])
        h2_new = np.where(hp_swap_mask, p1[:num_hyperparams], p2[:num_hyperparams])

    # Crossover Fuzzy Params (as contiguous block - swap entire block based on probability)
    if num_fuzzy_params > 0:
        if random.random() < crossover_prob:
            f1_new = p2[num_hyperparams:weights_start]
            f2_new = p1[num_hyperparams:weights_start]
        else:
            f1_new = p1[num_hyperparams:weights_start]
            f2_new = p2[num_hyperparams:weights_start]
    else:
        f1_new, f2_new = np.array([]), np.array([])

    # Crossover Weights
    w1_new, w2_new = p1[weights_start:], p2[weights_start:]
    weight_size = len(w1_new)
    if weight_size > 0:
        wt_swap_mask = np.random.rand(weight_size) < crossover_prob
        w1_new = np.where(wt_swap_mask, p2[weights_start:], p1[weights_start:])
        w2_new = np.where(wt_swap_mask, p1[weights_start:], p2[weights_start:])

    child1 = np.concatenate((h1_new, f1_new, w1_new))
    child2 = np.concatenate((h2_new, f2_new, w2_new))
    return child1, child2


# --- Mutation Operators ---
# MODIFIED: Accept num_hyperparams, operate on weights only, accept current_mutation_rate
# NEW: Function for hyperparameter mutation

def mutate_hyperparams_gaussian(chromosome: np.ndarray, mutation_strength: float, num_hyperparams: int) -> np.ndarray:
    """ Adds Gaussian noise to hyperparameter genes. Simple version, no rate param. """
    if num_hyperparams <= 0 or mutation_strength <= 0: return chromosome # Return unchanged if no hyperparams or no strength

    mutated_chromosome = chromosome.copy()
    hyperparams = mutated_chromosome[:num_hyperparams]

    # Apply Gaussian noise scaled by strength to ALL hyperparams for simplicity
    noise = np.random.normal(0, mutation_strength, size=num_hyperparams).astype(hyperparams.dtype)
    hyperparams += noise

    # Clamp based on ranges? Decoding function handles clamping, maybe not needed here.
    # Or, potentially use relative mutation: hyperparams *= (1 + noise)

    mutated_chromosome[:num_hyperparams] = hyperparams
    return mutated_chromosome

def mutate_weights_gaussian(
    chromosome: np.ndarray,
    current_mutation_rate: float, # Use rate passed from task
    mutation_strength: float,
    num_hyperparams: int,
    num_fuzzy_params: int = 0
) -> np.ndarray:
    """ Adds Gaussian noise to a fraction of weights based on current mutation rate. """
    weights_start = num_hyperparams + num_fuzzy_params
    if current_mutation_rate <= 0 or mutation_strength <= 0 or len(chromosome) <= weights_start:
        return chromosome # Return unchanged if no mutation needed or no weights

    mutated_chromosome = chromosome.copy()
    weights = mutated_chromosome[weights_start:] # Get view/copy of weights
    num_weights = weights.size
    if num_weights == 0: return chromosome

    num_weights_to_mutate = int(num_weights * current_mutation_rate)
    if num_weights_to_mutate == 0 and current_mutation_rate > 0: num_weights_to_mutate = 1 # Ensure at least one mutation
    num_weights_to_mutate = min(num_weights_to_mutate, num_weights) # Cap at total

    if num_weights_to_mutate > 0:
        indices_to_mutate = np.random.choice(num_weights, num_weights_to_mutate, replace=False)
        noise = np.random.normal(0, mutation_strength, size=num_weights_to_mutate).astype(weights.dtype)
        weights[indices_to_mutate] += noise # Mutate weights in place

    # No need to re-concatenate if 'weights' was a view, but copy ensures safety
    mutated_chromosome[weights_start:] = weights
    return mutated_chromosome

def mutate_weights_uniform_random(
    chromosome: np.ndarray,
    current_mutation_rate: float, # Use rate passed from task
    value_range: Tuple[float, float],
    num_hyperparams: int,
    num_fuzzy_params: int = 0
) -> np.ndarray:
    """ Mutates weights by replacing selected genes with a new random value. """
    weights_start = num_hyperparams + num_fuzzy_params
    if current_mutation_rate <= 0 or len(chromosome) <= weights_start:
        return chromosome

    mutated_chromosome = chromosome.copy()
    weights = mutated_chromosome[weights_start:]
    num_weights = weights.size
    if num_weights == 0: return chromosome

    min_val, max_val = value_range
    mutation_mask = np.random.rand(num_weights) < current_mutation_rate
    num_mutations = np.sum(mutation_mask)

    if num_mutations > 0:
        new_values = np.random.uniform(min_val, max_val, size=num_mutations).astype(weights.dtype)
        weights[mutation_mask] = new_values # Replace weights in place

    mutated_chromosome[weights_start:] = weights
    return mutated_chromosome

def fast_non_dominated_sort(obj_scores: np.ndarray) -> List[List[int]]:
    """
    Computes Pareto Fronts for Multi-Objective Optimization.
    obj_scores: Array of shape (pop_size, num_objectives)
    Returns: List of lists, where each inner list contains indices of a front.
    """
    pop_size = obj_scores.shape[0]
    domination_count = np.zeros(pop_size)
    dominated_solutions = [[] for _ in range(pop_size)]
    fronts = [[]]

    for p in range(pop_size):
        for q in range(pop_size):
            # p dominates q if it is better or equal in all objectives 
            # and strictly better in at least one.
            if all(obj_scores[p] >= obj_scores[q]) and any(obj_scores[p] > obj_scores[q]):
                dominated_solutions[p].append(q)
            elif all(obj_scores[q] >= obj_scores[p]) and any(obj_scores[q] > obj_scores[p]):
                domination_count[p] += 1
        
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

def calculate_crowding_distance(obj_scores: np.ndarray, front: List[int]) -> np.ndarray:
    """
    Measures how close an individual is to its neighbors in objective space.
    Higher distance = More unique/diverse solution.
    """
    distances = np.zeros(len(front))
    num_objs = obj_scores.shape[1]
    
    for m in range(num_objs):
        # Sort front by objective m
        sorted_indices = sorted(range(len(front)), key=lambda k: obj_scores[front[k], m])
        
        # Boundary points always get infinite distance (highly prioritized)
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        obj_range = obj_scores[front[sorted_indices[-1]], m] - obj_scores[front[sorted_indices[0]], m]
        if obj_range == 0: continue
            
        for i in range(1, len(front) - 1):
            distances[sorted_indices[i]] += (obj_scores[front[sorted_indices[i+1]], m] - 
                                           obj_scores[front[sorted_indices[i-1]], m]) / obj_range
    return distances

def evaluate_population_multi_objective(
    population: List[np.ndarray],
    model_definition_path: str,
    class_name: str,
    task_eval_func: callable,
    device: torch.device,
    model_args_static: List[Any],
    model_kwargs_static: Dict[str, Any],
    eval_config: Dict[str, Any],
    num_hyperparams: int,
    evolvable_hyperparams_config: Dict[str, Dict[str, Any]],
    use_fuzzy: bool = False,
    num_fuzzy_params: int = 0
) -> np.ndarray:
    """
    Modified evaluation that expects a LIST of objectives from the task_eval_func.
    Preserves Fuzzy Rule passing and Hyperparameter decoding.
    """
    all_obj_scores = []
    hyperparam_keys = list(evolvable_hyperparams_config.keys()) if evolvable_hyperparams_config else []

    for i, chromosome in enumerate(population):
        try:
            # 1. Decode Hyperparameters
            current_hparams = {}
            if num_hyperparams > 0:
                hparam_genes = chromosome[:num_hyperparams]
                current_hparams = decode_hyperparameters(hparam_genes, hyperparam_keys, evolvable_hyperparams_config)

            # 2. Slice Fuzzy & Weights
            fuzzy_start = num_hyperparams
            weights_start = fuzzy_start + num_fuzzy_params
            
            fuzzy_genes = chromosome[fuzzy_start:weights_start] if use_fuzzy else None
            weight_genes = chromosome[weights_start:]

            # 3. Instantiate Model
            merged_kwargs = {**model_kwargs_static, **current_hparams}
            model = load_pytorch_model(model_definition_path, class_name, None, device, *model_args_static, **merged_kwargs)
            
            load_weights_from_flat(model, weight_genes)
            model.to(device)

            # 4. Multi-Objective Evaluation
            # Your evaluation script must return a list/tuple: [Acc, Conf, -Latency]
            scores = task_eval_func(model, {
                **eval_config, 
                'device': device, 
                'fuzzy_rules': fuzzy_genes
            })
            
            # Ensure scores is a list of floats
            all_obj_scores.append([float(s) for s in scores])

        except Exception as e:
            logger.error(f"Error evaluating individual {i}: {e}")
            # Dynamically match the number of objectives if possible, 
            # or use a safe fallback based on the first successful individual
            if all_obj_scores:
                num_objs = len(all_obj_scores[0])
                all_obj_scores.append([0.0] * num_objs)
            else:
                all_obj_scores.append([0.0, 0.0, 0.0]) # Default fallback

    return np.array(all_obj_scores)

def select_parents_nsga2(population: List[np.ndarray], ranks: np.ndarray, distances: np.ndarray, num_parents: int) -> List[np.ndarray]:
    """
    Binary Tournament Selection based on Rank and Crowding Distance.
    """
    parents = []
    pop_size = len(population)
    for _ in range(num_parents):
        idx1, idx2 = random.sample(range(pop_size), 2)
        # Winner has lower rank (better front) or higher distance (more diverse)
        if ranks[idx1] < ranks[idx2]:
            winner_idx = idx1
        elif ranks[idx2] < ranks[idx1]:
            winner_idx = idx2
        else:
            winner_idx = idx1 if distances[idx1] > distances[idx2] else idx2
        parents.append(population[winner_idx].copy())
    return parents
# --- End of Helper Functions ---
