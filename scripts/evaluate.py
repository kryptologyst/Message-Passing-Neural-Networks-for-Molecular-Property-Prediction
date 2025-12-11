"""Evaluation script for comparing multiple models."""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import yaml
import torch
import numpy as np

from src.data.molecular_dataset import MolecularDataset, create_data_loaders
from src.models.mpnn import MPNN
from src.models.baselines import GCN, GraphSAGE, GAT
from src.eval.evaluator import MolecularEvaluator
from src.utils.device import get_device, set_seed


def create_all_models(node_dim: int, edge_dim: int, hidden_dim: int = 64) -> Dict[str, torch.nn.Module]:
    """Create all models for comparison.
    
    Args:
        node_dim: Number of node features.
        edge_dim: Number of edge features.
        hidden_dim: Hidden dimension for models.
        
    Returns:
        Dictionary of model names to models.
    """
    models = {}
    
    # MPNN
    models['MPNN'] = MPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1,
        aggregation='mean',
        activation='relu',
        use_gru=True,
        pooling='set2set',
    )
    
    # GCN
    models['GCN'] = GCN(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1,
        pooling='mean',
    )
    
    # GraphSAGE
    models['GraphSAGE'] = GraphSAGE(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1,
        aggregation='mean',
        pooling='mean',
    )
    
    # GAT
    models['GAT'] = GAT(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        pooling='mean',
    )
    
    return models


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate multiple models on molecular property prediction')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, mps, cpu, auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = MolecularDataset(
        root=args.data_dir,
        dataset_name="qm9",
        target_property=0,
        max_samples=args.max_samples,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Node features: {dataset.num_node_features}")
    print(f"Edge features: {dataset.num_edge_features}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle_train=False,
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Create models
    print("Creating models...")
    models = create_all_models(
        node_dim=dataset.num_node_features,
        edge_dim=dataset.num_edge_features,
        hidden_dim=64,
    )
    
    # Move models to device
    for name, model in models.items():
        model.to(device)
        print(f"{name}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Create evaluator
    evaluator = MolecularEvaluator(device=device)
    
    # Evaluate on test set
    print("\nEvaluating models on test set...")
    test_results = evaluator.compare_models(models, test_loader)
    
    # Create leaderboard
    evaluator.create_leaderboard(test_results)
    
    # Plot comparison
    evaluator.plot_model_comparison(
        test_results, 
        metric='mae',
        save_path=os.path.join(args.output_dir, 'model_comparison_mae.png')
    )
    
    evaluator.plot_model_comparison(
        test_results, 
        metric='r2',
        save_path=os.path.join(args.output_dir, 'model_comparison_r2.png')
    )
    
    # Save results
    results_dir = os.path.join(args.output_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'test_results.yaml'), 'w') as f:
        yaml.dump(test_results, f, default_flow_style=False)
    
    print(f"\nResults saved to: {results_dir}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
