from taipy.gui import Gui, State
import pandas as pd
import numpy as np
from pathlib import Path

# Load the pre-generated model stats
MODEL_STATS_PATH = "model_stats.parquet"

def load_model_stats():
    """Load model statistics from parquet file"""
    if not Path(MODEL_STATS_PATH).exists():
        raise FileNotFoundError(
            f"Model stats file not found at {MODEL_STATS_PATH}. "
            "Please run model_stats_generator.py first!"
        )
    return pd.read_parquet(MODEL_STATS_PATH)

# Initialize data
print("Loading model information...")
df = load_model_stats()
filtered_df = df.copy()

# Initial state variables
max_params = np.ceil(df['parameters'].max())
param_range = [0, max_params]
search_text = ""
min_param_input = 0
max_param_input = max_params

def on_init(state: State):
    """Initialize the state when the app starts"""
    state.df = df
    state.filtered_df = df.copy()
    state.param_range = param_range
    state.search_text = search_text
    state.max_params = max_params
    state.min_param_input = min_param_input
    state.max_param_input = max_param_input

def filter_models(state: State):
    """Filter models based on parameter range and categories"""
    min_param, max_param = state.param_range
    
    # Apply filters
    mask = (
        (state.df['parameters'] >= min_param) & 
        (state.df['parameters'] <= max_param)
    )
    
    # Apply search filter if there's search text
    if state.search_text:
        mask &= state.df['name'].str.contains(state.search_text, case=False)
    
    state.filtered_df = state.df[mask].copy()

def on_param_input_change(state: State):
    """Handle changes to the parameter input boxes"""
    try:
        # Convert input values to float
        min_val = float(state.min_param_input)
        max_val = float(state.max_param_input)
        
        # Ensure values are within valid range
        state.min_param_input = max(0, min(min_val, state.max_params))
        state.max_param_input = max(0, min(max_val, state.max_params))
        
        # Update slider
        state.param_range = [state.min_param_input, state.max_param_input]
        filter_models(state)
    except (ValueError, TypeError):
        # Reset to previous values if conversion fails
        state.min_param_input = state.param_range[0]
        state.max_param_input = state.param_range[1]

def on_slider_change(state: State):
    """Handle changes to the slider"""
    # Update input boxes
    state.min_param_input = state.param_range[0]
    state.max_param_input = state.param_range[1]
    filter_models(state)


page = """
<|container|
<|part|class_name=model-explorer-container|
# TIMM Model Explorer

<|layout|columns=1 gap=30px|
<|part|class_name=stats-card|
## Parameter Range
Select model size range (in millions of parameters):
<|{param_range}|slider|min=0|max={max_params}|value_by_id=False|on_change=on_slider_change|>

<|layout|columns=2 gap=10px|class_name=parameter-inputs|
<|{min_param_input}|input|label=Min Parameters (M)|type=number|on_change=on_param_input_change|>
<|{max_param_input}|input|label=Max Parameters (M)|type=number|on_change=on_param_input_change|>
|>
|>

<|part|class_name=stats-card|
## Search & Results
<|{search_text}|input|label=Search models by name|on_change=filter_models|>

Found <|{len(filtered_df)}|> models matching your criteria

<|{filtered_df}|table|width=100%|page_size=10|>
|>
|>
|>
|>
"""

if __name__ == "__main__":
    print("Starting Taipy GUI...")
    gui = Gui(page)
    gui.run(
        host="127.0.0.1", 
        port=5050, 
        on_init=on_init,
        title="TIMM Model Explorer"
    ) 