# octmnist_evaluation.py

import torch
import torch.nn.functional as F # Needed if model uses it, good practice to have
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast # Import AMP for potential GPU speedup
import torchvision.transforms as transforms

# Import specific MedMNIST dataset
import medmnist
from medmnist import OCTMNIST # Use OCTMNIST explicitly

import numpy as np
import os
import time
import logging # Use logging

logger = logging.getLogger(__name__)

# --- Configuration for OCTMNIST Evaluation ---

# Directory to download/load MedMNIST dataset
DATA_ROOT = '/code/medmnist_data'

# --- OPTIMIZATION: Batch Size ---
# Adjust based on GPU VRAM.
BATCH_SIZE = 256 # Keeping similar to MNIST template, adjust as needed

# --- OPTIMIZATION: Subset Evaluation ---
# Set to an integer for faster, noisier evaluation on a subset.
# Set to None to evaluate on the full test set.
SAMPLES_TO_EVALUATE = None # Default: Evaluate on the full dataset

# --- Global variable to cache dataset/loader ---
_test_loader = None

def get_octmnist_test_loader(device):
    """ Loads or retrieves the OCTMNIST test dataloader. """
    global _test_loader

    if _test_loader is not None:
        # Check if device properties changed (edge case)
        if _test_loader.pin_memory != (device.type == 'cuda'):
            logger.info(" (Device type changed, reloading DataLoader for OCTMNIST)")
            _test_loader = None
        else:
            return _test_loader

    logger.info(f" (Loading OCTMNIST test dataset [Root: {DATA_ROOT}, Batch Size: {BATCH_SIZE}]...)")

    try:
        os.makedirs(DATA_ROOT, exist_ok=True)
    except OSError as e:
        logger.error(f"!!! ERROR: Could not create DATA_ROOT directory '{DATA_ROOT}': {e}", exc_info=True)
        raise # Re-raise critical error

    # Define transformations:
    # 1. ToTensor(): Converts PIL Image (0-255) to FloatTensor (0.0-1.0) and moves channel dim.
    # 2. Normalize(): Uses mean/std calculated from notebook (scaled to 0-1 range).
    #    Mean = 47.9222 / 255.0 = 0.1879
    #    Std = 49.7935 / 255.0 = 0.1953
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1879,), std=(0.1953,))
    ])

    # Download and load the test dataset using OCTMNIST
    try:
        # Use as_rgb=False if the model expects 1 channel (likely)
        test_dataset = OCTMNIST(
            split='test',
            transform=transform,
            download=True,
            root=DATA_ROOT,
            as_rgb=False # Ensure grayscale images (1 channel)
        )
        logger.info(f" OCTMNIST download/load successful from {DATA_ROOT}")

    except Exception as e:
        logger.error(f"\n!!! ERROR: Failed to download or load OCTMNIST dataset from {DATA_ROOT}.")
        logger.error("!!! Please check internet connection, directory permissions, and medmnist installation.")
        logger.error(f"!!! Error details: {e}\n", exc_info=True)
        raise

    # Create the DataLoader
    pin_memory_setting = (device.type == 'cuda')
    num_workers_setting = min(4, os.cpu_count() // 2) if os.cpu_count() else 2

    _test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No shuffling needed for evaluation
        num_workers=num_workers_setting,
        pin_memory=pin_memory_setting
    )

    dataset_size = len(test_dataset)
    subset_info = f"(evaluating subset: {SAMPLES_TO_EVALUATE})" if SAMPLES_TO_EVALUATE is not None else "(evaluating full dataset)"
    logger.info(f" (OCTMNIST test dataset loaded: {dataset_size} samples {subset_info})")

    return _test_loader

# --- The Fitness Function ---

# Keep the function signature the same as the template
def evaluate_network_on_task(model_instance: torch.nn.Module, config: dict) -> float:
    """
    Evaluates the performance (accuracy) of the given model instance
    on the OCTMNIST test dataset (or a subset). Uses AMP if on CUDA.

    Args:
        model_instance (torch.nn.Module): An instance of your network with weights loaded.
        device (torch.device): The device ('cuda' or 'cpu') to run evaluation on.
        config (dict): Configuration dictionary (currently unused here but required by caller).

    Returns:
        float: The test accuracy (between 0.0 and 1.0). Higher is better.
               Returns -float('inf') if evaluation fails.
    """
    eval_start_time = time.time()
    fitness = 0.0 # Default fitness

    try:
        device = config.get('device')
        if not isinstance(device, torch.device): # Basic check
             logger.error(f"Invalid or missing 'device' in config: {device}. Falling back to CPU.")
             # Decide: Fallback or raise? Fallback might hide issues. Raising is safer.
             # device = torch.device('cpu')
             raise ValueError(f"Invalid or missing 'device' in config dict: {config}")

        # 1. Get the test data loader for OCTMNIST
        test_loader = get_octmnist_test_loader(device)

        # 2. Set up model for evaluation
        model_instance.to(device)
        model_instance.eval()

        correct_predictions = 0
        total_samples = 0
        samples_processed_count = 0 # Counter for subset evaluation

        # 3. Iterate through the test dataset without calculating gradients
        with torch.no_grad():
            for inputs, labels in test_loader:
                # --- Subset Evaluation Check ---
                if SAMPLES_TO_EVALUATE is not None and samples_processed_count >= SAMPLES_TO_EVALUATE:
                    break # Stop iterating early

                # Move data to the specified device
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze() # Squeeze label tensor if it's [N, 1] -> [N]

                # --- Automatic Mixed Precision (AMP) ---
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        outputs = model_instance(inputs)
                else:
                    outputs = model_instance(inputs) # Run without autocast on CPU

                # Get predictions (class with the highest logit)
                _, predicted_classes = torch.max(outputs.data, 1)

                # Update counts
                batch_size_actual = labels.size(0)
                total_samples += batch_size_actual
                correct_predictions += (predicted_classes == labels).sum().item()
                samples_processed_count += batch_size_actual # Update subset counter

        # 4. Calculate accuracy based on samples evaluated
        if total_samples > 0:
            accuracy = correct_predictions / total_samples
            fitness = float(accuracy)
        else:
            logger.warning(" Warning: Zero samples evaluated. Check SAMPLES_TO_EVALUATE and dataset.")
            fitness = 0.0

        eval_time = time.time() - eval_start_time
        amp_info = "(AMP enabled)" if device.type == 'cuda' else "(AMP disabled/CPU)"
        subset_info = f"on {total_samples} samples" if SAMPLES_TO_EVALUATE is not None else "on full dataset"
        logger.debug(f" (Eval {task_id if 'task_id' in locals() else ''} took {eval_time:.3f}s {amp_info}, Accuracy: {fitness:.4f} {subset_info})")

    except Exception as e:
        logger.error(f" ERROR during fitness evaluation: {e}", exc_info=True)
        return -float('inf') # Return very low fitness on error

    # Ensure the return value is a float
    return float(fitness)

# --- Optional: Add a main block for testing this file directly ---
if __name__ == '__main__':
    # This allows you to test the evaluation function independently
    logging.basicConfig(level=logging.INFO) # Configure logging for testing
    logger.info("--- Testing octmnist_evaluation.py directly ---")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using test device: {test_device}")

    # You need a model definition available to test
    try:
        # Assuming your adapted model file is named 'octmnist_cnn_hp_ready.py'
        # and located where Python can find it (e.g., same dir or in PYTHONPATH)
        from octmnist_cnn_hp_ready import OCTMNIST_CNN # Import your adapted model class

        logger.info("Successfully imported model definition (OCTMNIST_CNN).")

        # Instantiate a dummy model (with default or example evolved params)
        # Ensure num_classes matches OCTMNIST (which is 4)
        test_model = OCTMNIST_CNN(input_channels=1, num_classes=4)
        logger.info("Model instantiated.")

        # Evaluate the randomly initialized model
        logger.info("Running evaluation function...")
        test_config = {} # Empty config for testing
        test_fitness = evaluate_network_on_task(test_model, test_device, test_config)
        logger.info(f"\nDirect test result: Fitness (Accuracy) = {test_fitness:.4f}")

        # Test again to check caching
        logger.info("\nRunning evaluation function again (testing loader cache)...")
        test_fitness_2 = evaluate_network_on_task(test_model, test_device, test_config)
        logger.info(f"Second run result: Fitness (Accuracy) = {test_fitness_2:.4f}")

    except ImportError as imp_err:
        logger.error(f"\nError: Could not import model definition for testing.")
        logger.error(f"Ensure 'octmnist_cnn_hp_ready.py' exists and is runnable.")
        logger.error(f"Import Error: {imp_err}")
    except Exception as e:
        logger.error(f"\nAn error occurred during direct testing: {e}", exc_info=True)

