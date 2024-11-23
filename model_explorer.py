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

def on_init(state: State):
    """Initialize the state when the app starts"""
    state.df = df
    state.filtered_df = df.copy()
    state.param_range = param_range
    state.search_text = search_text
    state.max_params = max_params

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

page = """
<|container|
# TIMM Model Explorer

<|layout|columns=1|gap=30px|
<|
## Parameter Range
Select model size range (in millions of parameters):
<|{param_range}|slider|min=0|max={max_params}|value_by_id=False|on_change=filter_models|>
|>
|>

## Search & Results
<|{search_text}|input|label=Search models by name|on_change=filter_models|>

### Found <|{len(filtered_df)}|> models matching your criteria

<|{filtered_df}|table|width=100%|page_size=10|>
|>
"""

if __name__ == "__main__":
    print("Starting Taipy GUI...")
    gui = Gui(page)
    gui.run(host="127.0.0.1", port=5050, on_init=on_init) 