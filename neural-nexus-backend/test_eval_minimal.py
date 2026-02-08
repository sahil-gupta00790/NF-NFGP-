import torch
import numpy as np

def evaluate_model(model, config):
    """Minimal evaluation - returns fitness based on model output."""
    device = config.get('device', torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        output = model(dummy_input)
        # Return fitness based on output statistics
        fitness = float(torch.mean(torch.abs(output)).item()) * 10.0
        return max(0.0, min(100.0, fitness))
