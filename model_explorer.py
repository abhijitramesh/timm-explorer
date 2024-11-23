from taipy.gui import Gui, State, notify
import pandas as pd
import timm
import torch
import numpy as np

# Initialize data
print("Fetching model information from TIMM...")
model_names = timm.list_models(pretrained=True)[:30]  # Limit to 30 models

data = []
for name in model_names:
    try:
        model = timm.create_model(name, pretrained=False)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        category = name.split('_')[0]
        data.append({'name': name, 'parameters': params, 'category': category})
    except Exception as e:
        print(f"Error loading {name}: {e}")

df = pd.DataFrame(data)
filtered_df = df.copy()

# Initial state variables
max_params = np.ceil(df['parameters'].max())
param_range = [0, max_params]
selected_categories = list(df['category'].unique())
search_text = ""

def on_init(state: State):
    """Initialize the state when the app starts"""
    state.df = df
    state.filtered_df = df.copy()
    state.param_range = param_range
    state.selected_categories = selected_categories
    state.search_text = search_text
    state.max_params = max_params

def filter_models(state: State):
    """Filter models based on parameter range and categories"""
    min_param, max_param = state.param_range
    
    # Apply filters
    mask = (
        (state.df['parameters'] >= min_param) & 
        (state.df['parameters'] <= max_param) &
        (state.df['category'].isin(state.selected_categories))
    )
    
    # Apply search filter if there's search text
    if state.search_text:
        mask &= state.df['name'].str.contains(state.search_text, case=False)
    
    state.filtered_df = state.df[mask].copy()

page = """
<|container|
# TIMM Model Explorer

<|layout|columns=1 1|gap=30px|
<|
## Parameter Range
Select model size range (in millions of parameters):
<|{param_range}|slider|min=0|max={max_params}|value_by_id=False|on_change=filter_models|>
|>

<|
## Model Categories
<|{selected_categories}|selector|lov={list(df['category'].unique())}|on_change=filter_models|label=Select model categories|multiple|>
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