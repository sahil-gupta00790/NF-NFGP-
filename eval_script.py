
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
def evaluate_network_on_task(model_instance: torch.nn.Module, config: dict) -> list[float]:
    """
    Evaluates the network across multiple objectives for NSGA-II.
    
    Objectives returned: [Accuracy, Mean_Confidence, Latency_ms]
    Aligns with Feature Bible Sections 1.1 (Multi-Objective) and 2.1 (Fuzzy Behavioral).
    """
    eval_start_time = time.time()
    
    # Initialize trackers
    total_samples = 0
    correct_predictions = 0
    confidences = []
    inference_times = []

    try:
        device = config.get('device', torch.device('cpu'))
        # Using the loader defined in your previous file
        test_loader = get_octmnist_test_loader(device)

        model_instance.to(device)
        model_instance.eval()

        with torch.no_grad():
            for inputs, labels in test_loader:
                # --- Subset Evaluation Check (from your config) ---
                if SAMPLES_TO_EVALUATE is not None and total_samples >= SAMPLES_TO_EVALUATE:
                    break

                inputs = inputs.to(device)
                labels = labels.to(device).squeeze()

                # 1. LATENCY TRACKING (Objective 3)
                # We time only the forward pass to measure pure model efficiency
                start_batch = time.perf_counter()
                
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        outputs = model_instance(inputs)
                else:
                    outputs = model_instance(inputs)
                
                end_batch = time.perf_counter()
                inference_times.append((end_batch - start_batch) / labels.size(0))

                # 2. ACCURACY TRACKING (Objective 1)
                _, predicted_classes = torch.max(outputs.data, 1)
                correct_predictions += (predicted_classes == labels).sum().item()
                total_samples += labels.size(0)

                # 3. CONFIDENCE TRACKING (Objective 2) [Bible 2.1 Behavioral Signal]
                # High confidence indicates a 'stable' and 'certain' model
                probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                confidences.extend(max_probs.cpu().numpy().tolist())

        # Aggregate Results
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_latency_ms = (np.mean(inference_times) * 1000) if inference_times else 999.0

        logger.info(f"Eval Complete - Acc: {accuracy:.4f}, Conf: {avg_confidence:.4f}, Latency: {avg_latency_ms:.4f}ms")

        # RETURN LIST FOR NSGA-II [Bible 1.1]
        # Important: NSGA-II expects to MAXIMIZE values. 
        # Since we want to MINIMIZE latency, we return it as a negative value or 1/latency.
        # We will use negative latency so "higher" is better.
        return [float(accuracy), float(avg_confidence), float(-avg_latency_ms)]

    except Exception as e:
        logger.error(f"Critical error in evaluate_network_on_task: {e}", exc_info=True)
        # Return worst-case scores on failure so individual is penalized but doesn't crash evolution
        return [0.0, 0.0, -999.0]

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