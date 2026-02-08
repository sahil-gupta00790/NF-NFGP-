# tasks/evolution_tasks.py

from app.core.celery_app import celery_app
from celery import current_task, Task
import time
import random
import os
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Tuple
import redis
from celery.exceptions import SoftTimeLimitExceeded

# Use config settings imported via Celery context or directly
from app.core.config import settings
# --- Import ALL helper functions ---
from app.utils.evolution_helpers import (
    load_pytorch_model, flatten_weights, load_task_eval_function,
    evaluate_population_step, load_weights_from_flat,
    decode_hyperparameters,
    # Selection
    select_parents_tournament,
    select_parents_roulette,
    # Crossover
    crossover_one_point,
    crossover_uniform,
    crossover_average,
    # Mutation
    mutate_hyperparams_gaussian,
    mutate_weights_gaussian,
    mutate_weights_uniform_random,
    #NSGA things
    evaluate_population_multi_objective,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    select_parents_nsga2,
)
# --- Import fuzzy components ---
from app.core.fuzzy import FuzzyInferenceSystem
from app.utils.fuzzy_helpers import init_fuzzy_genome, decode_fuzzy_params, mutate_fuzzy_params

logger = logging.getLogger(__name__)

STANDARD_EVAL_SCRIPT_PATH = settings.STANDARD_EVAL_SCRIPT_PATH
RESULT_DIR = settings.RESULT_DIR
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Helper Function for Diversity (Modified) ---
def calculate_population_diversity(population: list[np.ndarray], num_hyperparams: int, num_fuzzy_params: int = 0) -> float:
    """Calculates average pairwise Euclidean distance between the weight vectors."""
    weights_start = num_hyperparams + num_fuzzy_params
    if not population or len(population) < 2 or weights_start < 0: return 0.0
    num_individuals = len(population)
    weight_vectors = [ind[weights_start:] for ind in population if isinstance(ind, np.ndarray) and len(ind) > weights_start]
    if len(weight_vectors) < 2: return 0.0
    flat_weights = [w.flatten() for w in weight_vectors]
    if len(flat_weights) < 2: return 0.0
    first_len = len(flat_weights[0])
    consistent_weights = [w for w in flat_weights if len(w) == first_len]
    num_consistent = len(consistent_weights)
    if num_consistent < 2:
        logger.debug("Less than 2 consistent weight vectors for diversity calc.") # Less alarming
        return 0.0
    if num_consistent < num_individuals:
         logger.debug(f"Using {num_consistent}/{num_individuals} individuals for diversity.") # Less alarming
    total_distance = 0.0
    num_pairs = 0
    for i in range(num_consistent):
        for j in range(i + 1, num_consistent):
            try:
                # Using float32 for potentially better performance on norm
                distance = np.linalg.norm(consistent_weights[i].astype(np.float32) - consistent_weights[j].astype(np.float32))
                total_distance += distance
                num_pairs += 1
            except (ValueError, FloatingPointError) as e: # Catch potential FP errors
                 logger.warning(f"Error calculating distance between individuals {i} and {j}: {e}")
    return float(total_distance / num_pairs) if num_pairs > 0 else 0.0

# --- Task Base Class with Retry ---
class EvolutionTaskWithRetry(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 2}
    retry_backoff = True
    retry_backoff_max = 7000
    retry_jitter = False
    acks_late = True
    reject_on_worker_lost = True

@celery_app.task(bind=True, base=EvolutionTaskWithRetry)
def run_evolution_task(self: Task, model_definition_path: str, task_evaluation_path: str | None, use_standard_eval: bool, initial_weights_path: str | None, config: Dict[str, Any]):
    """ Celery task for evolution with dynamic rates, hyperparameter optimization, and cooperative halt check. """
    task_id = None
    redis_client = None # Initialize redis client variable
    halt_key = None

    try:
         task_id = self.request.id
         if not task_id: logger.error("CRITICAL: Failed to get task_id"); raise ValueError("Task ID missing")
         # Construct key early, now that halt_key exists in this scope
         halt_key = f"task:halt:{task_id}"
         logger.info(f"[Task {task_id}] Starting evolution...")
         config_preview = {k: v for i, (k, v) in enumerate(config.items()) if i < 15}
         logger.info(f"[Task {task_id}] Received config preview: {config_preview}")

         # --- NEW: Initialize Redis Client ---
         try:
             redis_url = str(settings.REDIS_URL)
             # Use decode_responses=True for easier handling of keys/values as strings
             redis_client = redis.from_url(redis_url, decode_responses=True)
             # Test connection (optional but good practice)
             redis_client.ping()
             logger.info(f"[Task {task_id}] Successfully connected to Redis at {redis_url}")
         except Exception as redis_err:
             # Log warning but allow task to continue if Redis isn't essential for core logic (only for halting)
             logger.warning(f"[Task {task_id}] Failed to connect to Redis for halt check: {redis_err}. Halt feature disabled for this run.", exc_info=False)
             redis_client = None # Ensure client is None if connection failed
         # --- End Redis Init ---

    except Exception as entry_err:
         logger.critical(f"CRITICAL ERROR AT TASK ENTRY: {entry_err}", exc_info=True)
         raise

    # --- Determine Evaluation Script ---
    if use_standard_eval:
        actual_eval_script_path = STANDARD_EVAL_SCRIPT_PATH
        logger.info(f"[Task {task_id}] Using standard evaluation script: {actual_eval_script_path}")
        if not os.path.exists(actual_eval_script_path):
            raise FileNotFoundError(f"Standard eval script not found: {actual_eval_script_path}")
    elif task_evaluation_path and os.path.exists(task_evaluation_path):
        actual_eval_script_path = task_evaluation_path
        logger.info(f"[Task {task_id}] Using custom evaluation script: {actual_eval_script_path}")
    else:
        raise FileNotFoundError(f"Custom eval script path invalid or not found: {task_evaluation_path}")

    # --- Setup & Config Extraction ---
    try:
        # Helper function (remains mostly the same)
        def safe_convert(key, target_type, default_value, is_tuple=False, expected_len=None):
            value = config.get(key, default_value)
            if value is None:
                if default_value is None and not is_tuple: return None
                logger.debug(f"Config '{key}' is None, using default: {default_value}") # Less alarming
                value = default_value
            try:
                if is_tuple:
                    if not isinstance(value, (list, tuple)): raise TypeError(f"'{key}' must be list/tuple")
                    if expected_len is not None and len(value) != expected_len: raise ValueError(f"'{key}' needs length {expected_len}")
                    return tuple(float(v) for v in value)
                if target_type is float: return float(value)
                if target_type is int: return int(value)
                if target_type is str: return str(value).lower() if key in ["selection_strategy", "crossover_operator", "mutation_operator", "dynamic_mutation_heuristic"] else str(value)
                if target_type is bool: return str(value).lower() in ['true', '1', 'yes'] if isinstance(value, str) else bool(value)
                return value # list, dict assumed okay
            except (TypeError, ValueError) as conv_err:
                logger.error(f"Invalid type for config '{key}'. Expected {target_type.__name__}, got {type(value)} ({value}). Error: {conv_err}. Using default: {default_value}", exc_info=False)
                try:
                    if is_tuple: return tuple(float(v) for v in default_value)
                    else: return target_type(default_value)
                except Exception as default_err:
                    logger.critical(f"Default value '{default_value}' for '{key}' is invalid! Error: {default_err}", exc_info=True)
                    raise ValueError(f"Invalid config for '{key}' and invalid default.")

        # --- General Config ---
        generations = safe_convert('generations', int, 10)
        population_size = safe_convert('population_size', int, 20)
        model_class = safe_convert('model_class', str, None)
        if not model_class: raise ValueError("'model_class' is required.")
        model_args = safe_convert('model_args', list, [])
        model_kwargs_static = safe_convert('model_kwargs', dict, {})
        eval_config = safe_convert('eval_config', dict, {})

        # --- Hyperparameter Evolution Config ---
        evolvable_hyperparams_config: Dict[str, Dict[str, Any]] = config.get('evolvable_hyperparams', {})
        hyperparam_keys: List[str] = list(evolvable_hyperparams_config.keys())
        num_hyperparams: int = len(hyperparam_keys)
        logger.info(f"[Task {task_id}] Evolving {num_hyperparams} hyperparameters: {hyperparam_keys}")
        hyperparam_mutation_strength: float = safe_convert('hyperparam_mutation_strength', float, 0.02)

        # --- GA Operator Config ---
        selection_strategy = safe_convert("selection_strategy", str, "tournament")
        crossover_operator = safe_convert("crossover_operator", str, "one_point")
        mutation_operator = safe_convert("mutation_operator", str, "gaussian")
        elitism_count = safe_convert("elitism_count", int, 1)
        elitism_count = max(0, min(elitism_count, population_size - 1))

        # --- Operator-Specific Params (Weights) ---
        mutation_rate = safe_convert('mutation_rate', float, 0.1) # Base fixed rate
        mutation_strength = safe_convert('mutation_strength', float, 0.05)
        tournament_size = safe_convert('tournament_size', int, 3)
        uniform_crossover_prob = safe_convert('uniform_crossover_prob', float, 0.5)
        uniform_mutation_range = safe_convert('uniform_mutation_range', tuple, (-1.0, 1.0), is_tuple=True, expected_len=2)

        # --- Dynamic Weight Mutation Rate Params ---
        use_dynamic_mutation_rate = safe_convert('use_dynamic_mutation_rate', bool, False)
        dynamic_mutation_heuristic = safe_convert('dynamic_mutation_heuristic', str, 'time_decay')
        initial_mutation_rate = safe_convert('initial_mutation_rate', float, mutation_rate) # Default based on fixed rate
        final_mutation_rate = safe_convert('final_mutation_rate', float, 0.01)
        # Renamed for clarity based on implementation: high_fitness_rate is the 'normal' rate
        normal_fitness_mutation_rate = safe_convert('high_fitness_mutation_rate', float, 0.05)
        stagnation_mutation_rate = safe_convert('low_fitness_mutation_rate', float, 0.25)
        stagnation_threshold: float = safe_convert('stagnation_threshold', float, 0.001) # Threshold for fitness-based heuristic

        base_mutation_rate = safe_convert('base_mutation_rate', float, mutation_rate) # Default based on fixed rate
        diversity_threshold_low = safe_convert('diversity_threshold_low', float, 0.1)
        mutation_rate_increase_factor = safe_convert('mutation_rate_increase_factor', float, 1.5)
        if mutation_rate_increase_factor < 1.0:
             logger.warning(f"mutation_rate_increase_factor ({mutation_rate_increase_factor}) < 1.0. Setting to 1.0")
             mutation_rate_increase_factor = 1.0

        # --- Other Config ---
        final_model_filename = f"evolved_{task_id}.pth"
        final_model_path = os.path.join(RESULT_DIR, final_model_filename)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Task {task_id}] Using device: {device}")
        if device.type == 'cuda': logger.info(f"[Task {task_id}] CUDA Device Name: {torch.cuda.get_device_name(0)}")
        # --- Fuzzy Config Extraction ---
        use_fuzzy = config.get("use_fuzzy", False)
        fuzzy_config = config.get("fuzzy_config", {})
        if use_fuzzy:
            fuzzy_config.setdefault("alpha", 0.7)
            logger.info(f"[Task {task_id}] Fuzzy co-evolution enabled: {fuzzy_config}")
        else:
            logger.info(f"[Task {task_id}] Fuzzy co-evolution disabled (NN-only)")
        
        logger.info(f"[Task {task_id}] Config Summary: Gens={generations}, Pop={population_size}, Elitism={elitism_count}")
        logger.info(f"[Task {task_id}] Operators: Select='{selection_strategy}', Cross='{crossover_operator}', Mut(W)='{mutation_operator}'")
        if use_dynamic_mutation_rate: logger.info(f"[Task {task_id}] Dynamic Rate(W) Enabled: H='{dynamic_mutation_heuristic}'")
        else: logger.info(f"[Task {task_id}] Fixed Rate(W): {mutation_rate:.3f}")
        logger.info(f"[Task {task_id}] Strengths: Weights={mutation_strength:.4f}, Hyperparams={hyperparam_mutation_strength:.4f}")
        logger.info(f"[Task {task_id}] Eval Config: {eval_config}")

        # --- Load Task Evaluation Function ---
        task_eval_func = load_task_eval_function(actual_eval_script_path)

        # --- Initialize Population (Hyperparams + Weights) ---
        self.update_state(state='PROGRESS', meta={'progress': 0.0, 'message': 'Initializing population...', 'fitness_history': [], 'avg_fitness_history': [], 'diversity_history': []})
        logger.info(f"[Task {task_id}] Initializing population ({population_size})...")

        # 1. Get initial/reference weights
        initial_weights = None
        try:
            # Instantiate model once to get weight structure/size
            # Use static kwargs + mid-range placeholder dynamic kwargs for structure definition
            placeholder_hparams = {k: (evolvable_hyperparams_config[k]['range'][0] + evolvable_hyperparams_config[k]['range'][1]) / 2 for k in hyperparam_keys if 'range' in evolvable_hyperparams_config[k]}
            ref_model = load_pytorch_model(model_definition_path, model_class, initial_weights_path, device, *model_args, **model_kwargs_static, **placeholder_hparams)
            initial_weights = flatten_weights(ref_model)
            if initial_weights_path and os.path.exists(initial_weights_path):
                logger.info(f"[Task {task_id}] Loaded initial weights. Weight vector size: {initial_weights.shape[0]}")
            else:
                logger.info(f"[Task {task_id}] Using default model weights. Weight vector size: {initial_weights.shape[0]}")
            del ref_model
        except Exception as e:
             error_msg = f"Failed to instantiate model '{model_class}' to get weight structure: {e}"
             logger.error(f"[Task {task_id}] {error_msg}", exc_info=True)
             raise ValueError(error_msg) from e

        if initial_weights.size == 0:
            raise ValueError("Initial weight vector size is zero. Cannot proceed.")

        # 3. Initialize fuzzy genome (if enabled)
        fuzzy_genome_template = None
        num_fuzzy_params = 0
        if use_fuzzy:
            fuzzy_genome_template = init_fuzzy_genome(fuzzy_config)
            num_fuzzy_params = len(fuzzy_genome_template)
            logger.info(f"[Task {task_id}] Initialized fuzzy genome with {num_fuzzy_params} parameters")
        else:
            fuzzy_genome_template = np.array([], dtype=np.float64)
        
        # 4. Generate initial population chromosomes
        population: List[np.ndarray] = []
        init_rate_for_spread = initial_mutation_rate if use_dynamic_mutation_rate else mutation_rate # Use initial rate if dynamic for spread
        for i in range(population_size):
            # Generate random initial hyperparams
            hyperparam_values = np.zeros(num_hyperparams, dtype=np.float64)
            for idx, key in enumerate(hyperparam_keys):
                 h_config = evolvable_hyperparams_config[key]
                 h_min, h_max = h_config.get('range', [0.0, 1.0])
                 hyperparam_values[idx] = random.uniform(h_min, h_max)

            # Generate fuzzy genome (copy template, optionally mutate for diversity)
            if use_fuzzy and num_fuzzy_params > 0:
                if i == 0:
                    fuzzy_genome = fuzzy_genome_template.copy()
                else:
                    # Slight mutation for diversity
                    fuzzy_genome = mutate_fuzzy_params(
                        np.concatenate((np.zeros(num_hyperparams), fuzzy_genome_template)),
                        num_hyperparams,
                        num_fuzzy_params,
                        mutation_rate=0.1,
                        mutation_strength=0.01
                    )[num_hyperparams:]
            else:
                fuzzy_genome = np.array([], dtype=np.float64)

            # Generate initial weights (copy or mutated copy)
            if i == 0: # First individual keeps initial weights
                 individual_weights = initial_weights.copy()
            else: # Mutate initial weights for diversity
                 if mutation_operator == "gaussian":
                     individual_weights = mutate_weights_gaussian(initial_weights, init_rate_for_spread, mutation_strength, num_hyperparams=0, num_fuzzy_params=0)
                 elif mutation_operator == "uniform_random":
                     individual_weights = mutate_weights_uniform_random(initial_weights, init_rate_for_spread, uniform_mutation_range, num_hyperparams=0, num_fuzzy_params=0)
                 else:
                     individual_weights = mutate_weights_gaussian(initial_weights, init_rate_for_spread, mutation_strength, num_hyperparams=0, num_fuzzy_params=0)

            chromosome = np.concatenate((hyperparam_values, fuzzy_genome, individual_weights))
            population.append(chromosome)

        if device.type == 'cuda': torch.cuda.empty_cache()
        self.update_state(state='PROGRESS', meta={'progress': 0.01, 'message': 'Initialization complete. Starting generations...', 'fitness_history': [], 'avg_fitness_history': [], 'diversity_history': []})

    except Exception as init_err:
        error_msg = f"Initialization failed: {init_err}"
        logger.error(f"[Task {task_id}] {error_msg}", exc_info=True)
        self.update_state(state='FAILURE', meta={'message': error_msg, 'error': str(init_err)})
        raise

    # --- Evolution Loop ---
    best_fitness_overall = -float('inf')
    best_chromosome_overall = population[0].copy()
    fitness_history_overall = []
    avg_fitness_history_overall = []
    diversity_history_overall = []
    # No need for last_fitness_scores, calculate current avg directly

    try:
        # Store final front size before loop ends
        final_front_0_size = 0
        for gen in range(generations):
            gen_num = gen + 1
            logger.info(f"[Task {task_id}] --- Generation {gen_num}/{generations} ---")
            gen_start_time = time.time()

            # --- Check for Halt Request ---
            if redis_client and halt_key: # Check if client is valid and key was constructed
                try:
                    # --- ADDED Debug Logs ---
                    logger.debug(f"[Task {task_id}] Checking Redis for halt key: {halt_key}")
                    exists = redis_client.exists(halt_key)
                    logger.debug(f"[Task {task_id}] Redis exists result for {halt_key}: {exists}")
                    # --- End Debug Logs ---
                    if exists: # exists returns number of keys found (1 if found)
                        logger.warning(f"[Task {task_id}] Halt flag detected. Stopping evolution.")
                        try:
                            deleted_count = redis_client.delete(halt_key)
                            logger.info(f"[Task {task_id}] Deleted halt key {halt_key} (Count: {deleted_count})")
                        except Exception as del_err:
                            logger.error(f"[Task {task_id}] Failed to delete halt flag {halt_key}: {del_err}")

                        halt_message = "Task halted by user request."
                        current_progress = (gen / generations) if generations > 0 else 0
                        best_hparams_halt = {}
                        if best_chromosome_overall is not None and num_hyperparams > 0:
                             try: best_hparams_halt = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)
                             except Exception as decode_err: logger.error(f"Error decoding best hparams during halt: {decode_err}")
                        halt_meta = {
                            'message': halt_message, 'progress': current_progress,
                            'fitness_history': fitness_history_overall,
                            'avg_fitness_history': avg_fitness_history_overall,
                            'diversity_history': diversity_history_overall,
                            'best_hyperparameters': best_hparams_halt
                        }
                        self.update_state(state='HALTED', meta=halt_meta)
                        logger.info(f"[Task {task_id}] Task state set to HALTED.")
                        return {'message': halt_message, 'status': 'HALTED_BY_USER'}
                except redis.exceptions.ConnectionError as redis_conn_err:
                     logger.error(f"[Task {task_id}] Redis connection error during halt check: {redis_conn_err}. Halt check disabled.", exc_info=False)
                     redis_client = None # Disable further checks
                except Exception as check_err:
                     logger.error(f"[Task {task_id}] Unexpected error checking halt flag: {check_err}. Continuing run.", exc_info=True)
            # --- End Halt Check ---

            # 1. Evaluate Population
            # --- 1. MULTI-OBJECTIVE EVALUATION (NSGA-II Upgrade) ---
            try:
                # This returns a matrix: [[Acc1, Conf1, -Lat1], [Acc2, Conf2, -Lat2], ...]
                obj_scores = evaluate_population_multi_objective(
                    population, model_definition_path, model_class, task_eval_func, device,
                    model_args, model_kwargs_static, eval_config,
                    num_hyperparams=num_hyperparams,
                    evolvable_hyperparams_config=evolvable_hyperparams_config,
                    use_fuzzy=use_fuzzy,
                    num_fuzzy_params=num_fuzzy_params
                )
            except Exception as eval_err:
                logger.error(f"[Task {task_id}] Critical Multi-Objective evaluation error in Gen {gen_num}: {eval_err}", exc_info=True)
                raise RuntimeError(f"NSGA-II Evaluation failed critically in Gen {gen_num}") from eval_err

            # --- 2. PARETO FRONT CALCULATIONS ---
            # Group individuals into layers of dominance
            fronts = fast_non_dominated_sort(obj_scores)
            
            # Calculate Ranks and Crowding Distances for selection diversity
            ranks = np.zeros(population_size)
            distances = np.zeros(population_size)
            for rank_idx, front in enumerate(fronts):
                ranks[front] = rank_idx
                if len(front) > 0:
                    distances[front] = calculate_crowding_distance(obj_scores, front)

            # --- 3. METRICS & DYNAMIC SETTINGS ---
            # We track 'Accuracy' (Objective 0) as the primary fitness for legacy logs
            valid_primary_scores = obj_scores[:, 0]
            max_fitness = np.max(valid_primary_scores)
            avg_obj_scores = np.mean(obj_scores, axis=0) # Average of all objectives
            
            diversity = calculate_population_diversity(population, num_hyperparams, num_fuzzy_params)
            
            fitness_history_overall.append(float(max_fitness))
            avg_fitness_history_overall.append(avg_obj_scores.tolist())
            diversity_history_overall.append(float(diversity))

            # Determine Mutation Rate (Preserving your existing Dynamic Mutation Logic)
            current_mutation_rate = mutation_rate 
            if use_dynamic_mutation_rate:
                try:
                    if dynamic_mutation_heuristic == 'time_decay':
                        progress = gen / max(1, generations - 1)
                        current_mutation_rate = max(final_mutation_rate, initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * progress)
                    elif dynamic_mutation_heuristic == 'diversity_based' and diversity < diversity_threshold_low:
                        current_mutation_rate = base_mutation_rate * mutation_rate_increase_factor
                    # (Other heuristics like 'fitness_based' can be added here)
                except Exception as dyn_err:
                    logger.error(f"Error calculating dynamic mutation rate: {dyn_err}. Using fallback.")

            # Log Generation Stats
            logger.info(f"[Task {task_id}] Gen {gen_num} | Front0 Size: {len(fronts[0])} | Max Acc: {max_fitness:.4f} | Avg Conf: {avg_obj_scores[1] if len(avg_obj_scores) > 1 else 0.0:.4f} | Mutation: {current_mutation_rate:.3f}")
            
            # Store final front size
            final_front_0_size = len(fronts[0])

            # --- 4. UPDATE BEST INDIVIDUAL (From Front 0) ---
            # We pick the most accurate individual from the elite Pareto Front
            best_idx_in_front_0 = fronts[0][np.argmax(obj_scores[fronts[0], 0])]
            current_best_fitness = obj_scores[best_idx_in_front_0, 0]
            
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                best_chromosome_overall = population[best_idx_in_front_0].copy()
                logger.info(f"[Task {task_id}] *** New Pareto Leader Found (Acc: {best_fitness_overall:.4f}) ***")
                try:
                    best_hparams_decoded = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)
                    logger.info(f"[Task {task_id}] Leader Hyperparams: {best_hparams_decoded}")
                except Exception: pass

            # Update Celery Progress
            current_progress = gen_num / generations
            progress_message = f"Gen {gen_num}/{generations} | Front0: {len(fronts[0])} | Best Acc: {best_fitness_overall:.4f}"
            self.update_state(state='PROGRESS', meta={
                'generation': gen_num,           # Added
                'total_generations': generations, # Added
                'progress': current_progress, 
                'message': progress_message,
                'front_0_count': len(fronts[0]),
                'fitness_history': fitness_history_overall,
                'avg_objectives': avg_obj_scores.tolist()
            })

            # --- 5. NSGA-II SELECTION ---
            # Replaces Tournament/Roulette with Pareto-based Crowding Tournament
            logger.debug(f"[Task {task_id}] Selecting parents via Crowding Distance Tournament...")
            parents = select_parents_nsga2(population, ranks, distances, population_size)

            # --- 6. REPRODUCTION (Elitism + Crossover + Mutation) ---
            next_population = []
            
            # Elitism: Carry over the top models from Front 0
            if elitism_count > 0:
                elite_indices = fronts[0][:elitism_count]
                for e_idx in elite_indices:
                    if len(next_population) < population_size:
                        next_population.append(population[e_idx].copy())

            # Offspring Loop
            while len(next_population) < population_size:
                p1_chrom, p2_chrom = random.sample(parents, 2)

                # Crossover (Preserves your existing operator logic)
                if crossover_operator == "uniform":
                    child1, child2 = crossover_uniform(p1_chrom, p2_chrom, num_hyperparams, crossover_prob=uniform_crossover_prob, num_fuzzy_params=num_fuzzy_params)
                else: # Default to one_point
                    child1, child2 = crossover_one_point(p1_chrom, p2_chrom, num_hyperparams, num_fuzzy_params)

                # --- HYBRID MUTATION (Bible 1.2 & 2.2) ---
                for child in [child1, child2]:
                    if len(next_population) >= population_size: break
                    
                    # 1. Mutate Hyperparams
                    mutated = mutate_hyperparams_gaussian(child, hyperparam_mutation_strength, num_hyperparams)
                    
                    # 2. Mutate Fuzzy Rules (Co-Evolution)
                    if use_fuzzy and num_fuzzy_params > 0:
                        mutated = mutate_fuzzy_params(mutated, num_hyperparams, num_fuzzy_params, current_mutation_rate, mutation_strength)
                    
                    # 3. Mutate Model Weights
                    if mutation_operator == "uniform_random":
                        mutated = mutate_weights_uniform_random(mutated, current_mutation_rate, uniform_mutation_range, num_hyperparams, num_fuzzy_params)
                    else: # Default Gaussian
                        mutated = mutate_weights_gaussian(mutated, current_mutation_rate, mutation_strength, num_hyperparams, num_fuzzy_params)
                    
                    next_population.append(mutated)

            population = next_population[:population_size]

        # --- FINALIZATION (After All Generations) ---
        logger.info(f"[Task {task_id}] Evolution Complete. Saving optimal Pareto individual...")
        
        final_save_path = os.path.join(RESULT_DIR, f"best_model_{task_id}.pth")
        final_model = load_pytorch_model(model_definition_path, model_class, None, device, *model_args, **model_kwargs_static)
        
        # Load the best weights (skipping hparams and fuzzy segments)
        weights_start = num_hyperparams + num_fuzzy_params
        load_weights_from_flat(final_model, best_chromosome_overall[weights_start:])
        
        torch.save(final_model.state_dict(), final_save_path)
        
        # Decode final best hparams for the result
        final_hparams = {}
        if num_hyperparams > 0:
            final_hparams = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)

        return {
            'status': 'SUCCESS',
            'model_path': final_save_path,
            'best_accuracy': float(best_fitness_overall),
            'best_hyperparameters': final_hparams,
            'fitness_history': fitness_history_overall,
            'front_0_final_size': final_front_0_size
        }

    # --- Exception Handling (Added SoftTimeLimitExceeded) ---
    except SoftTimeLimitExceeded: # Catch Celery's exception if SIGUSR1 was used (alternative)
        logger.warning(f"[Task {task_id}] Soft time limit exceeded. Task halting.")
        # Update state to HALTED or a custom state
        halt_message = "Task halted due to time limit or signal."
        halt_meta = { 'message': halt_message, 'fitness_history': fitness_history_overall, 'avg_fitness_history': avg_fitness_history_overall, 'diversity_history': diversity_history_overall, 'best_hyperparameters': decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config) if num_hyperparams > 0 else {} }
        self.update_state(state='HALTED', meta=halt_meta)
        # Return a result
        return {'message': halt_message, 'status': 'HALTED_BY_SIGNAL'}
    except Exception as e:
        error_message = f'Evolution task failed: {str(e)}'
        logger.error(f"[Task {task_id}] {error_message}", exc_info=True)
        # ... (existing failure handling remains the same) ...
        best_hyperparams_fail = {};
        if num_hyperparams > 0 and len(best_chromosome_overall) >= num_hyperparams:
            try: best_hyperparams_fail = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)
            except Exception as decode_err: logger.error(f"Error decoding best hparams on failure: {decode_err}")
        meta_fail = { 'message': error_message, 'error': str(e), 'best_hyperparameters': best_hyperparams_fail, 'fitness_history': fitness_history_overall, 'avg_fitness_history': avg_fitness_history_overall, 'diversity_history': diversity_history_overall }
        self.update_state(state='FAILURE', meta=meta_fail)
        raise

    finally:
        # --- NEW: Close Redis connection ---
        if redis_client:
            try:
                redis_client.close()
                logger.info(f"[Task {task_id}] Closed Redis connection.")
            except Exception as close_err:
                logger.error(f"[Task {task_id}] Error closing Redis connection: {close_err}")
        # --- End Redis Close ---