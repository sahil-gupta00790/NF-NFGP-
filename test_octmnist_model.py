# test_octmnist_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast

# Import MedMNIST dataset tools and specific dataset
import medmnist
from medmnist import OCTMNIST # Use OCTMNIST explicitly

import numpy as np
import os
import argparse
import time
import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Import your model definition ---
# Ensure 'octmnist_cnn_hp_ready.py' is accessible
try:
    from octmnist_cnn_hp_ready import MyCNN
except ImportError:
    print("ERROR: Could not import 'MyCNN' from 'octmnist_cnn_hp_ready.py'.")
    print("Please ensure the file exists and is in the same directory or Python path.")
    exit(1)

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Loading (Adapted from your evaluation script) ---
# Global variable to cache dataset/loader
_test_loader = None
_test_dataset_info = None # Cache dataset info too

def get_octmnist_test_loader(data_root: str, batch_size: int, device: torch.device):
    """ Loads or retrieves the OCTMNIST test dataloader. """
    global _test_loader, _test_dataset_info

    if _test_loader is not None:
        # Basic check if loader settings are compatible
        if _test_loader.batch_size == batch_size and \
           _test_loader.pin_memory == (device.type == 'cuda'):
            logger.info(" (Reusing cached OCTMNIST test loader)")
            return _test_loader, _test_dataset_info
        else:
            logger.info(" (Loader parameters changed, reloading OCTMNIST DataLoader)")
            _test_loader = None # Force reload

    logger.info(f" (Loading OCTMNIST test dataset [Root: {data_root}, Batch Size: {batch_size}]...)")
    try:
        os.makedirs(data_root, exist_ok=True)
    except OSError as e:
        logger.error(f"!!! ERROR: Could not create data directory '{data_root}': {e}", exc_info=True)
        raise

    # Define transformations (Matches your evaluation script)
    # Mean = 47.9222 / 255.0 = 0.1879
    # Std = 49.7935 / 255.0 = 0.1953
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1879,), std=(0.1953,))
    ])

    # Download and load the test dataset using OCTMNIST
    try:
        # Ensure grayscale images (1 channel)
        test_dataset = OCTMNIST(
            split='test',
            transform=transform,
            download=True,
            root=data_root,
            as_rgb=False
        )
        _test_dataset_info = medmnist.INFO['octmnist'] # Store dataset info
        logger.info(f" OCTMNIST download/load successful from {data_root}")
    except Exception as e:
        logger.error(f"\n!!! ERROR: Failed to download or load OCTMNIST dataset from {data_root}.", exc_info=True)
        raise

    # Create the DataLoader
    pin_memory_setting = (device.type == 'cuda')
    # Simple heuristic for num_workers
    num_workers_setting = min(4, os.cpu_count() // 2 if os.cpu_count() else 2, batch_size // 32 if batch_size > 32 else 1)
    num_workers_setting = max(1, num_workers_setting) # Ensure at least 1 worker


    _test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # No shuffling needed for evaluation
        num_workers=num_workers_setting,
        pin_memory=pin_memory_setting
    )

    dataset_size = len(test_dataset)
    logger.info(f" (OCTMNIST test dataset loaded: {dataset_size} samples)")
    logger.info(f" (Using {num_workers_setting} workers for DataLoader)")
    return _test_loader, _test_dataset_info


# --- Evaluation Function ---
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """ Evaluates the model and returns predictions and true labels. """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_labels = []

    logger.info("Starting model evaluation on the test set...")
    eval_start_time = time.time()

    with torch.no_grad(): # Disable gradient calculations for efficiency
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze() # Ensure labels are 1D

            # --- MODIFICATION HERE ---
            # Use torch.amp.autocast (or torch.autocast) and pass device.type as the first argument
            # The 'enabled' flag should only be True if the device is actually 'cuda' or potentially 'cpu'
            # if using bfloat16, but for simplicity, let's enable only for CUDA.
            # Use device.type which will be 'cuda' or 'cpu'
            use_autocast = (device.type == 'cuda') # Only use autocast for CUDA in this common case

            # Use the modern torch.amp.autocast (or torch.autocast)
            # Pass device.type ('cuda' or 'cpu') as the FIRST positional argument
            with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
                 outputs = model(inputs)
            # --- END MODIFICATION ---


            # Get predicted class indices (highest logit)
            _, predicted_classes = torch.max(outputs.data, 1)

            all_preds.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0: # Log progress every 10 batches
                 logger.debug(f"  Processed batch {i+1}/{len(test_loader)}")


    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation finished in {eval_time:.2f} seconds.")

    return np.array(all_labels), np.array(all_preds)


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OCTMNIST model (.pt file).")
    parser.add_argument("model_path", type=str, help="Path to the saved model state_dict (.pt file).")
    parser.add_argument("--data_root", type=str, default="./medmnist_data", help="Directory for MedMNIST data.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation.")
    # Add arguments for hyperparameters IF your saved model uses non-defaults
    # parser.add_argument("--out_channels_conv1", type=int, default=32, help="...")
    # ... add others as needed ...

    args = parser.parse_args()

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    # --- Data Loader ---
    try:
        test_loader, dataset_info = get_octmnist_test_loader(args.data_root, args.batch_size, device)
        num_classes = len(dataset_info['label'])
        class_names = [dataset_info['label'][str(i)] for i in range(num_classes)] # Get class names
        logger.info(f"Dataset has {num_classes} classes: {class_names}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        return # Exit if data loading fails

    # --- Model Instantiation ---
    # !! IMPORTANT !!
    # If your saved model used non-default hyperparameters, you MUST provide them here.
    # Example:
    # model = MyCNN(num_classes=num_classes,
    #               out_channels_conv1=args.out_channels_conv1, # Assuming you added argparse arguments
    #               out_channels_conv2=args.out_channels_conv2,
    #               # ... other evolved params ...
    #               )
    logger.info("Instantiating model architecture (MyCNN)...")
    try:
        # Using default hyperparameters from the class definition for now.
        # MODIFY THIS if your .pt file was saved with non-default parameters.
        model = MyCNN(num_classes=num_classes)
        logger.info("Model instantiated with default hyperparameters.")
        # logger.info(f"Model structure:\n{model}") # Optional: print model structure
    except Exception as e:
        logger.error(f"Failed to instantiate model: {e}", exc_info=True)
        return

    # --- Load Model Weights ---
    logger.info(f"Loading model weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return

    try:
        # Load state dict, mapping to the correct device
        state_dict = torch.load(args.model_path, map_location=device)

        # Handle potential keys mismatch (e.g., if saved with DataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
            logger.info("Detected 'module.' prefix in state_dict keys, removing it.")
            state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}", exc_info=True)
        logger.error("Ensure the .pt file contains a valid state_dict for the MyCNN architecture with matching hyperparameters.")
        return

    # --- Run Evaluation ---
    try:
        true_labels, predicted_labels = evaluate_model(model, test_loader, device)
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
        return

    # --- Calculate and Display Metrics ---
    logger.info("\n--- Performance Metrics ---")

    # 1. Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)") # [3][4][5][6]

    # 2. Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Labels:", class_names)
    print(cm)

    # 3. Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    # Use target_names for better readability if available
    report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
    print(report)

    logger.info("--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
