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
import json

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
    calculate_crowding_distance,
    fast_non_dominated_sort
)

logger = logging.getLogger(__name__)

STANDARD_EVAL_SCRIPT_PATH = settings.STANDARD_EVAL_SCRIPT_PATH
RESULT_DIR = settings.RESULT_DIR
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Helper Function for Diversity (Modified) ---
def calculate_population_diversity(population: list[np.ndarray], num_hyperparams: int) -> float:
    """Calculates average pairwise Euclidean distance between the weight vectors."""
    if not population or len(population) < 2 or num_hyperparams < 0: return 0.0
    num_individuals = len(population)
    weight_vectors = [ind[num_hyperparams:] for ind in population if isinstance(ind, np.ndarray) and len(ind) > num_hyperparams]
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

def select_parents_nsga2(population: List[np.ndarray], ranks: np.ndarray, distances: np.ndarray, num_parents: int) -> List[np.ndarray]:
    """
    Binary Tournament Selection for NSGA-II.
    Criteria: 1. Lower Rank (Better Front) | 2. Higher Crowding Distance (Diversity)
    """
    parents = []
    pop_size = len(population)
    for _ in range(num_parents):
        idx1, idx2 = random.sample(range(pop_size), 2)
        # Check Rank first
        if ranks[idx1] < ranks[idx2]:
            winner_idx = idx1
        elif ranks[idx2] < ranks[idx1]:
            winner_idx = idx2
        else:
            # If ranks are tied, use Crowding Distance
            winner_idx = idx1 if distances[idx1] > distances[idx2] else idx2
        parents.append(population[winner_idx].copy())
    return parents
# --- Task Base Class with Retry ---
class EvolutionTaskWithRetry(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 2}
    retry_backoff = True
    retry_backoff_max = 7000
    retry_jitter = False
    acks_late = True
    reject_on_worker_lost = True

# --- Redis Progress Helper (Preserved from original) ---
def update_redis_status(redis_client, task_id, data):
    if redis_client:
        try:
            redis_client.set(f"evolution_status:{task_id}", json.dumps(data), ex=3600)
        except Exception as e:
            logger.error(f"Redis Update Error: {e}")

@celery_app.task(bind=True, base=EvolutionTaskWithRetry)
def run_evolution_task(self, model_definition_path, task_evaluation_path, config):
    task_id = self.request.id
    logger.info(f"[Task {task_id}] Initializing NeuroForge Multi-Objective Engine...")

    # 1. SETUP & CONFIGURATION
    redis_client = None
    try:
        redis_client = redis.from_url(settings.CELERY_BROKER_URL)
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")

    # Parameters from Bible & Config
    pop_size = config.get('population_size', 20)
    generations = config.get('generations', 10)
    elitism_count = config.get('elitism_count', 2)
    
    # Neuro-Fuzzy Config (Bible 2.2)
    num_hparams = len(config.get('evolvable_hyperparams_config', {}))
    num_fuzzy_params = config.get('num_fuzzy_params', 10) # Number of membership/rule genes
    
    # Load Core Assets
    model_class = load_pytorch_model(model_definition_path, config.get('model_class'))
    task_eval_func = load_task_eval_function(task_evaluation_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine total chromosome length
    sample_model = model_class()
    num_weights = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
    total_gene_count = num_hparams + num_fuzzy_params + num_weights
    
    # Initialization
    population = [np.random.randn(total_gene_count).astype(np.float32) for _ in range(pop_size)]
    
    history = {
        'avg_obj_scores': [], # [Acc, Confidence, -Latency]
        'best_fitness_history': [], 
        'front_0_size_history': []
    }

    try:
        for gen in range(generations):
            obj_scores = []
            
            # --- EVALUATION STEP (Section 1.1 & 2.1) ---
            for i, chromosome in enumerate(population):
                # Slice Chromosome: [HParams | Fuzzy | Weights]
                # Fuzzy parameters are passed to the eval script for behavior scoring
                fuzzy_genes = chromosome[num_hparams : num_hparams + num_fuzzy_params]
                weight_genes = chromosome[num_hparams + num_fuzzy_params :]
                
                model = model_class()
                load_weights_from_flat(model, weight_genes)
                
                # Evaluation returns: [Accuracy, Confidence, -Latency]
                scores = task_eval_func(model, {
                    'device': device, 
                    'fuzzy_rules': fuzzy_genes, # Section 2.2: Pass rules to evaluation
                    'task_id': task_id
                })
                obj_scores.append(scores)

                # Celery State Update
                self.update_state(state='PROGRESS', meta={
                    'generation': gen, 'current': i + 1, 'total': pop_size
                })

            obj_scores = np.array(obj_scores)

            # --- NSGA-II LOGIC (Section 1.1) ---
            fronts = fast_non_dominated_sort(obj_scores)
            
            # Rank and Distance calculation
            ranks = np.zeros(pop_size)
            distances = np.zeros(pop_size)
            for rank_idx, front in enumerate(fronts):
                ranks[front] = rank_idx
                if len(front) > 0:
                    distances[front] = calculate_crowding_distance(obj_scores, front)

            # --- LOGGING & ANALYTICS (Preserved Redis Logic) ---
            avg_gen_scores = np.mean(obj_scores, axis=0).tolist()
            history['avg_obj_scores'].append(avg_gen_scores)
            history['best_fitness_history'].append(float(np.max(obj_scores[:, 0])))
            history['front_0_size_history'].append(len(fronts[0]))

            update_redis_status(redis_client, task_id, {
                'generation': gen,
                'avg_accuracy': avg_gen_scores[0],
                'front_0_count': len(fronts[0]),
                'status': 'EVOLVING'
            })

            logger.info(f"Gen {gen} | Front0: {len(fronts[0])} | Avg Acc: {avg_gen_scores[0]:.4f}")

            # --- REPRODUCTION ---
            next_population = []

            # 1. Elitism (Keep top of Front 0)
            elites = fronts[0][:elitism_count]
            for idx in elites:
                next_population.append(population[idx].copy())

            # 2. Selection & Crossover (Section 1.1)
            num_to_breed = pop_size - len(next_population)
            parents = select_parents_nsga2(population, ranks, distances, num_to_breed)

            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = crossover_uniform(parents[i], parents[i+1])
                    
                    # 3. Mutation (Section 1.2 & 2.2)
                    # Gaussian mutation for both model weights and fuzzy rules
                    child1 = mutate_weights_gaussian(child1, config.get('mutation_rate', 0.1), config.get('mutation_strength', 0.05), 0)
                    child2 = mutate_weights_gaussian(child2, config.get('mutation_rate', 0.1), config.get('mutation_strength', 0.05), 0)
                    
                    next_population.extend([child1, child2])
                else:
                    next_population.append(parents[i])

            population = next_population[:pop_size]

        # --- FINALIZATION ---
        # Select best individual from Front 0 (highest accuracy)
        best_idx = fronts[0][np.argmax(obj_scores[fronts[0], 0])]
        best_chromosome = population[best_idx]
        
        save_path = os.path.join(settings.RESULT_DIR, f"final_model_{task_id}.pth")
        final_model = model_class()
        load_weights_from_flat(final_model, best_chromosome[num_hparams + num_fuzzy_params:])
        torch.save(final_model.state_dict(), save_path)

        update_redis_status(redis_client, task_id, {'status': 'COMPLETED', 'result_path': save_path})

        return {
            'status': 'SUCCESS',
            'accuracy': float(obj_scores[best_idx, 0]),
            'confidence': float(obj_scores[best_idx, 1]),
            'history': history,
            'model_path': save_path
        }

    except Exception as e:
        logger.error(f"Evolution Critical Failure: {str(e)}", exc_info=True)
        update_redis_status(redis_client, task_id, {'status': 'FAILED', 'error': str(e)})
        raise e
    finally:
        if redis_client:
            redis_client.close()