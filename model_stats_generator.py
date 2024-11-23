import pandas as pd
import timm
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp

def count_parameters(model: torch.nn.Module) -> float:
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def process_model(name: str) -> dict:
    """Process a single model - used by the process pool"""
    try:
        model = timm.create_model(name, pretrained=False)
        params = count_parameters(model)
        category = name.split('_')[0]
        return {
            'name': name,
            'parameters': params,
            'category': category,
            'error': None
        }
    except Exception as e:
        return {
            'name': name,
            'parameters': None,
            'category': name.split('_')[0],
            'error': str(e)
        }

def generate_model_stats(output_path: str = "model_stats.parquet", num_workers: int = None):
    """Generate statistics for all TIMM models and save to parquet"""
    print("Fetching model information from TIMM...")
    model_names = timm.list_models(pretrained=True)
    total = len(model_names)
    
    # If num_workers not specified, use number of CPU cores minus 1 (leave one for system)
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Processing {total} models using {num_workers} workers...")
    
    data = []
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=total, desc="Processing models") as pbar:
            futures = [executor.submit(process_model, name) for name in model_names]
            for future in futures:
                result = future.result()
                if result['error'] is None:
                    data.append({
                        'name': result['name'],
                        'parameters': result['parameters'],
                        'category': result['category']
                    })
                else:
                    errors.append({
                        'name': result['name'],
                        'error': result['error']
                    })
                pbar.update(1)
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"- {error['name']}: {error['error']}")
    
    print("\nCreating DataFrame...")
    df = pd.DataFrame(data)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(df)} models to {output_path}")
    df.to_parquet(output_path)
    
    print("\nSummary:")
    print(f"- Total models processed: {total}")
    print(f"- Successful: {len(data)}")
    print(f"- Failed: {len(errors)}")
    print("Done!")

if __name__ == "__main__":
    generate_model_stats(num_workers=4)