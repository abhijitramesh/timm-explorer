from taipy.gui import Gui
import pandas as pd
import timm
import torch
from typing import Dict, List
import numpy as np

def count_parameters(model: torch.nn.Module) -> float:
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_model_info() -> Dict[str, List]:
    """Fetch model information from TIMM"""
    models_data = {
        'name': [],
        'parameters': [],
    }
    
    # Get all pretrained model names
    model_names = timm.list_models(pretrained=True)[:30]
    
    # Sample a subset of models for faster loading (remove this limit for production)
    for model_name in model_names:  # Limiting to 30 models for demonstration
        try:
            # Create model instance
            model = timm.create_model(model_name, pretrained=False)
            
            # Get model details
            models_data['name'].append(model_name)
            models_data['parameters'].append(count_parameters(model))
                
            
        except Exception as e:
            print(f"Skipping {model_name} due to: {str(e)}")
            continue
    
    return models_data

# Fetch real model data
print("Fetching model information from TIMM...")
models_data = get_model_info()
df = pd.DataFrame(models_data)

# Calculate max parameters for slider
max_params = np.ceil(df['parameters'].max())

# Initial state variables
param_range = [0, max_params]  # Initial range for parameters (in millions)
filtered_df = df  # Initially show all models

def filter_models(state):
    """Filter models based on parameter range"""
    min_param, max_param = state.param_range
    state.filtered_df = df[
        (df['parameters'] >= min_param) & 
        (df['parameters'] <= max_param)].copy()

# Create the page layout
page = """
# TIMM Model Explorer

## Parameter Range Selection
Select model size range (in millions of parameters):
<|{param_range}|slider|min=0|max={max_params}|value_by_id=False|on_change=filter_models|>


## Models in Selected Range
<|{filtered_df}|table|width=100%|>
"""

if __name__ == "__main__":
    print("Starting Taipy GUI...")
    Gui(page).run(host="127.0.0.1", port=5050) 