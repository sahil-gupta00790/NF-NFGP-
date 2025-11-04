# octmnist_cnn_hp_ready.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Optional: Use logging instead of print for better practice

logger = logging.getLogger(__name__)

class MyCNN(nn.Module):
    """
    OCTMNIST CNN adapted for hyperparameter evolution.
    Assumes input 1x28x28 images.
    Kernel sizes, padding, strides, and pooling are fixed.
    """
    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 4, # Your original model had 4 output classes
                 # --- Evolvable Hyperparameters (Defaults from your original code) ---
                 out_channels_conv1: int = 32,
                 out_channels_conv2: int = 64,
                 out_channels_conv3: int = 128,
                 neurons_fc1: int = 64,
                 dropout_rate: float = 0.5
                 ):
        super().__init__()

        # --- Validate and Process Hyperparameters ---
        # Round potential floats from GA and ensure minimum values
        out_channels_conv1 = max(1, int(round(out_channels_conv1)))
        out_channels_conv2 = max(1, int(round(out_channels_conv2)))
        out_channels_conv3 = max(1, int(round(out_channels_conv3)))
        neurons_fc1 = max(1, int(round(neurons_fc1)))
        # Clamp dropout rate between 0.0 and 1.0
        dropout_rate = max(0.0, min(1.0, float(dropout_rate)))

        logger.info(f"--- Initializing OCTMNIST_CNN (Evolvable) ---")
        logger.info(f" Input Channels: {input_channels}")
        logger.info(f" Num Classes: {num_classes}")
        logger.info(f" Conv1 Out Channels: {out_channels_conv1}")
        logger.info(f" Conv2 Out Channels: {out_channels_conv2}")
        logger.info(f" Conv3 Out Channels: {out_channels_conv3}")
        logger.info(f" FC1 Neurons: {neurons_fc1}")
        logger.info(f" Dropout Rate: {dropout_rate:.2f}")
        # --------------------------------------------------

        # --- Define Layers using Hyperparameters ---

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=out_channels_conv1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels_conv1) # BN matches conv output

        # Shared MaxPooling Layer (as in your original code)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Input 28x28 -> Conv1 (no change) -> Pool -> 14x14

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=out_channels_conv1, out_channels=out_channels_conv2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels_conv2) # BN matches conv output
        # Pool -> 7x7

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=out_channels_conv2, out_channels=out_channels_conv3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels_conv3) # BN matches conv output
        # Pool -> 3x3 (assuming input starts near 28x28)

        # Calculate the flattened size dynamically based on conv3 output channels and final spatial size (3x3)
        # This calculation depends on input size ~28x28 and the fixed pooling layers.
        # If input size or pooling changes, this calculation MUST be updated.
        flattened_size = out_channels_conv3 * 3 * 3
        logger.debug(f"Calculated flattened size for FC1 input: {flattened_size}")
        if flattened_size == 0: # Add check for safety
            raise ValueError(f"Calculated flattened_size is zero. Check conv3 output channels ({out_channels_conv3})")

        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, neurons_fc1)
        self.dropout = nn.Dropout(dropout_rate) # Use the hyperparameter
        self.fc2 = nn.Linear(neurons_fc1, num_classes) # Output layer size fixed by num_classes

        logger.info("--- OCTMNIST_CNN (Evolvable) Initialization Complete ---")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass."""
        try:
            # Apply layers with activations and pooling
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))

            # Flatten the output for the fully connected layers
            # x = x.view(x.size(0), -1) # Original flatten method
            x = torch.flatten(x, 1) # More explicit flatten (start_dim=1)

            # Apply fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x) # Final layer output (logits)

            return x
        except Exception as e:
            input_shape = x.shape if isinstance(x, torch.Tensor) else 'Unknown'
            logger.error(f"Error during forward pass: {e}. Input shape: {input_shape}", exc_info=True)
            # Optionally, log intermediate shapes if debugging size mismatches
            # Example: logger.error(f"Shape after conv3 pool: {x.shape}") before flatten
            raise # Re-raise after logging


# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)

    # Example: Instantiate with default hyperparameters
    print("Instantiating with default hyperparameters:")
    model_default = MyCNN()
    print(model_default)

    # Example: Instantiate with evolved hyperparameters
    print("\nInstantiating with example evolved hyperparameters:")
    evolved_params = {
        'out_channels_conv1': 40,
        'out_channels_conv2': 70,
        'out_channels_conv3': 110,
        'neurons_fc1': 55,
        'dropout_rate': 0.3
    }
    model_evolved = MyCNN(input_channels=1, num_classes=4, **evolved_params)
    print(model_evolved)

    # Test forward pass with dummy data (assuming 1x28x28 input)
    print("\nTesting forward pass with dummy data (batch size 2):")
    dummy_input = torch.randn(2, 1, 28, 28) # Batch size 2
    try:
        output = model_evolved(dummy_input)
        print(f"Output shape: {output.shape}") # Expected: [2, 4]
    except Exception as e:
        print(f"Forward pass failed: {e}")

