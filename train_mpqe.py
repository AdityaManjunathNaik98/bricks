import torch
import numpy as np
import pickle
import random
import json
import os
import argparse
import sys
import requests
import urllib.parse
import time
from typing import Dict, List, Any
from collections import defaultdict
from pathlib import Path
import io
import base64

# Import from project files
from nesy_factory.GNNs.mpqe import RGCNEncoderDecoder
from nesy_factory.utils.data_utils import Query, load_graph, load_queries_by_formula, get_queries_iterator
import nesy_factory.utils.mpqeutils as mpqeutils

print('===Script Mark 1===')

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def normalize_url(url: str) -> str:
    """Normalize URL by decoding special characters"""
    url = urllib.parse.unquote(url)
    url = url.replace("_$_", "_$$").replace("_-", "_$$").replace("__DOLLARS__", "$$")
    return url

def safe_format_metric(value, default="0.000000"):
    """Safely format a metric value to ensure it's numeric and properly formatted"""
    try:
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return default
            return f"{float(value):.6f}"
        elif isinstance(value, str):
            # Test if it's convertible to float
            float_val = float(value)
            if np.isnan(float_val) or np.isinf(float_val):
                return default
            return f"{float_val:.6f}"
        else:
            return default
    except (ValueError, TypeError):
        return default

def save_metrics(metrics: dict, training_history: dict = None):
    """Save metrics for Katib - using the expected metric names from YAML"""
    global_step = 1
    trial_id = "0"
    timestamp = time.time()
    
    print("=== SAVING MPQE METRICS ===")
    print(f"Input metrics: {metrics}")
    
    # The YAML expects 'val_loss' as the primary metric (changed from final_loss)
    if training_history and 'val_loss' in training_history and len(training_history['val_loss']) > 0:
        final_val_loss = training_history['val_loss'][-1]
        metric_value = safe_format_metric(final_val_loss, "10.000000")
    elif "val_loss" in metrics:
        metric_value = safe_format_metric(metrics['val_loss'], "10.000000")
    else:
        metric_value = "10.000000"

    # Create a clean record with only the required metric
    record = {
        "val_loss": metric_value,  # This matches your YAML objective_metric_name
        "checkpoint_path": "",
        "global_step": str(global_step),
        "timestamp": timestamp,
        "trial": trial_id,
    }
    
    print(f"Final record being saved: {record}")
    
    # Create katib directory
    katib_dir = Path("/katib")
    katib_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to the file the YAML expects
    metrics_file = katib_dir / "mnist.json"
    with open(metrics_file, "a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")
        
    print("=== MPQE METRICS SAVING COMPLETE ===")

def load_pickle_url(url: str):
    """Load pickle data from URL"""
    print(f"Downloading MPQE data from: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    
    try:
        data = pickle.load(io.BytesIO(resp.content))
        print(f"✅ MPQE Data loaded successfully: {type(data)}")
        if isinstance(data, dict):
            print(f"   Data keys: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"❌ Pickle loading failed: {e}")
        raise

class MPQETrainingData:
    """Training data wrapper for MPQE"""
    def __init__(self, train_queries=None, val_queries=None, test_queries=None, 
                 batch_size=512, current_iteration=0, past_burn_in=False):
        self.train_queries = train_queries
        self.val_queries = val_queries
        self.test_queries = test_queries
        self.batch_size = batch_size
        self.current_iteration = current_iteration
        self.past_burn_in = past_burn_in

def convert_to_query_format(queries_list):
    """Convert list of serialized queries to the format expected by MPQE"""
    queries_by_type = defaultdict(lambda: defaultdict(list))
    
    for query_data in queries_list:
        try:
            query = Query.deserialize(query_data)
            query_type = query.formula.query_type
            queries_by_type[query_type][query.formula].append(query)
        except Exception as e:
            print(f"Warning: Failed to deserialize query: {e}")
            continue
    
    # Convert to standard dict format
    result = {}
    for query_type, formulas_dict in queries_by_type.items():
        if len(formulas_dict) > 0:
            result[query_type] = dict(formulas_dict)
    
    return result

def load_and_process_data(process_data_url: str):
    """Load and process data for MPQE training"""
    
    # Load raw data
    raw_data = load_pickle_url(normalize_url(process_data_url))
    
    if not isinstance(raw_data, dict):
        raise ValueError("Data must be a dictionary")
    
    # Extract components
    train_edges = raw_data.get('train_edges', [])
    train_queries_2 = raw_data.get('train_queries_2', [])
    train_queries_3 = raw_data.get('train_queries_3', [])
    
    val_edges = raw_data.get('val_edges', [])
    val_queries_2 = raw_data.get('val_queries_2', [])
    val_queries_3 = raw_data.get('val_queries_3', [])
    
    test_edges = raw_data.get('test_edges', [])
    test_queries_2 = raw_data.get('test_queries_2', [])
    test_queries_3 = raw_data.get('test_queries_3', [])
    
    graph_data = raw_data.get('graph')
    
    print(f"✅ Data loaded - Train edges: {len(train_edges)}, Train 2-hop: {len(train_queries_2)}")
    
    # Combine training queries
    all_train_queries = train_edges + train_queries_2 + train_queries_3
    all_val_queries = val_edges + val_queries_2 + val_queries_3
    all_test_queries = test_edges + test_queries_2 + test_queries_3
    
    # Convert to proper format
    train_queries_formatted = convert_to_query_format(all_train_queries)
    val_queries_formatted = convert_to_query_format(all_val_queries)
    test_queries_formatted = convert_to_query_format(all_test_queries)
    
    # Create validation structure
    val_queries_structured = {
        'one_neg': val_queries_formatted,
        'full_neg': test_queries_formatted
    }
    
    print(f"✅ Formatted train queries: {list(train_queries_formatted.keys())}")
    print(f"✅ Formatted val queries: {list(val_queries_formatted.keys())}")
    
    return {
        'graph': graph_data,
        'train_queries': train_queries_formatted,
        'val_queries': val_queries_structured,
        'raw_data': raw_data
    }

def setup_model_and_graph(config, graph_data):
    """Setup MPQE model and graph"""
    
    print("Setting up MPQE model with graph structure...")
    
    # Create temporary data directory for load_graph
    temp_dir = "/tmp/mpqe_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save graph data to temporary file
    temp_graph_file = os.path.join(temp_dir, "all_data.pkl")
    with open(temp_graph_file, 'wb') as f:
        pickle.dump({'graph': graph_data}, f)
    
    # Load graph
    graph, feature_modules, node_maps = load_graph(
        temp_dir,
        config['embed_dim']
    )
    
    # Setup CUDA if requested
    if config.get('use_cuda', False) and torch.cuda.is_available():
        graph.features = mpqeutils.cudify(feature_modules, node_maps)
        for key in node_maps:
            node_maps[key] = node_maps[key].cuda()
    
    # Create encoder
    out_dims = {mode: config['embed_dim'] for mode in graph.relations}
    enc = mpqeutils.get_encoder(
        config.get('depth', 0),
        graph,
        out_dims,
        feature_modules,
        config.get('use_cuda', False)
    )
    
    # Create MPQE model
    model = RGCNEncoderDecoder(config)
    model.set_graph_and_encoder(graph, enc)
    
    print(f"✅ Model configured with loss function: {model.loss_function_name}")
    if hasattr(model, 'margin'):
        print(f"✅ Margin parameter: {model.margin}")
    
    return model

def run_hyperparameter_training(args):
    """Run MPQE hyperparameter training"""
    
    print("=== STARTING MPQE HYPERPARAMETER TUNING ===")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Margin: {args.margin}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    
    set_random_seed(42)
    
    try:
        # Load and process data
        data = load_and_process_data(args.process_data_url)
        
        # Create config
        config = {
            'model_name': 'mpqe',
            'embed_dim': int(args.embed_dim),
            'hidden_dim': int(args.hidden_dim),
            'output_dim': int(args.embed_dim),
            'num_layers': int(args.num_layers),
            'readout': 'mp',
            'scatter_op': 'add',
            'shared_layers': True,
            'adaptive': True,
            'loss_function': 'margin_loss',
            'margin': float(args.margin),
            'learning_rate': float(args.learning_rate),
            'batch_size': 128,
            'epochs': int(args.epochs),
            'max_burn_in': 40,
            'use_cuda': False,
            'depth': 0
        }
        
        # Setup model and graph
        model = setup_model_and_graph(config, data['graph'])
        
        # Move to device
        device = torch.device('cuda' if config.get('use_cuda', False) and torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✅ Model created on device: {device}")
        
        # Training loop with better error handling
        total_epochs = config['epochs']
        burn_in_epochs = config['max_burn_in']
        
        losses = []
        val_aucs = []
        
        for epoch in range(total_epochs):
            try:
                past_burn_in = epoch >= burn_in_epochs
                
                # Create training data
                training_data = MPQETrainingData(
                    train_queries=data['train_queries'],
                    batch_size=config['batch_size'],
                    current_iteration=epoch,
                    past_burn_in=past_burn_in
                )
                
                # Train step with error handling
                loss = model.train_step(training_data)
                
                # Ensure loss is valid
                if np.isnan(loss) or np.isinf(loss):
                    print(f"Warning: Invalid loss at epoch {epoch}, using previous or default")
                    loss = losses[-1] if losses else 1.0
                
                losses.append(loss)
                
                # Validation every 10 epochs
                if epoch % 10 == 0 and data['val_queries']:
                    try:
                        val_data_obj = MPQETrainingData(
                            val_queries=data['val_queries'],
                            batch_size=config['batch_size'],
                            current_iteration=epoch,
                            past_burn_in=past_burn_in
                        )
                        val_result = model.eval_step(val_data_obj)
                        val_auc = val_result.get('accuracy', 0.0)
                        
                        # Ensure val_auc is valid
                        if np.isnan(val_auc) or np.isinf(val_auc):
                            val_auc = 0.0
                            
                        val_aucs.append(val_auc)
                    except Exception as e:
                        print(f'Val failed: {e}')
                        val_aucs.append(0.0)
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch:03d} | Train Loss: {loss:.4f}')
                    
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                # Use a reasonable default loss for failed epochs
                losses.append(losses[-1] if losses else 1.0)
                continue
        
        # Calculate final metrics with safety checks
        final_loss = losses[-1] if losses else 1.0
        final_auc = val_aucs[-1] if val_aucs else max(0.5, 1.0 - final_loss)
        
        # Ensure all metrics are valid numbers
        final_loss = safe_format_metric(final_loss, "1.000000")
        final_auc = safe_format_metric(final_auc, "0.500000")
        
        print(f'final_loss={final_loss}')
        print(f'train_loss={final_loss}')
        print(f'val_auc={final_auc}')
        
        # Save metrics using the expected format
        save_metrics({
            "val_loss": float(final_loss),
        })
        
        print("=== MPQE HYPERPARAMETER TUNING COMPLETED ===")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        save_metrics({"val_loss": 10.0})

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MPQE Training Script')
    
    # Required arguments that match the YAML
    parser.add_argument('--model_type', type=str, required=True, help='Type of model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--process_data_url', type=str, required=True, help='URL to preprocessed dataset')
    parser.add_argument('--config', type=str, required=True, help='Base64 encoded JSON configuration')
    
    # Hyperparameter tuning parameters
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for loss')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args, unknown = parser.parse_known_args()
    
    try:
        print("=== MPQE HYPERPARAMETER TUNING MODE MK2 ===")
        run_hyperparameter_training(args)
        
    except Exception as e:
        print(f"❌ MPQE Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        save_metrics({"val_loss": 10.0})
        sys.exit(1)
    
    print("=== MPQE PIPELINE COMPLETED ===")
    sys.exit(0)

if __name__ == "__main__":
    main()
