"""Main training script for MPNN molecular property prediction."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from omegaconf import OmegaConf

from src.data.molecular_dataset import MolecularDataset, create_data_loaders
from src.models.mpnn import MPNN
from src.models.baselines import GCN, GraphSAGE, GAT
from src.train.trainer import Trainer
from src.eval.evaluator import MolecularEvaluator
from src.utils.device import get_device, set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], node_dim: int, edge_dim: int) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Configuration dictionary.
        node_dim: Number of node features.
        edge_dim: Number of edge features.
        
    Returns:
        PyTorch model.
    """
    model_config = config['model']
    model_name = model_config['name'].lower()
    
    if model_name == 'mpnn':
        model = MPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            aggregation=model_config['aggregation'],
            activation=model_config['activation'],
            use_gru=model_config['use_gru'],
            pooling=model_config['pooling'],
            processing_steps=model_config['processing_steps'],
        )
    elif model_name == 'gcn':
        model = GCN(
            node_dim=node_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            pooling=model_config['pooling'],
        )
    elif model_name == 'graphsage':
        model = GraphSAGE(
            node_dim=node_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            aggregation=model_config['aggregation'],
            pooling=model_config['pooling'],
        )
    elif model_name == 'gat':
        model = GAT(
            node_dim=node_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config['dropout'],
            pooling=model_config['pooling'],
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MPNN for molecular property prediction')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to store data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, mps, cpu, auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
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
    print(f"Configuration: {config}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = MolecularDataset(
        root=args.data_dir,
        dataset_name=config['data']['dataset_name'],
        target_property=config['data']['target_property'],
        max_samples=config['data']['max_samples'],
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Node features: {dataset.num_node_features}")
    print(f"Edge features: {dataset.num_edge_features}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle_train=config['data']['shuffle_train'],
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Create model
    print("Creating model...")
    model = create_model(config, dataset.num_node_features, dataset.num_edge_features)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler=config['training']['scheduler'],
        loss_fn=config['training']['loss_fn'],
        device=device,
        use_wandb=config['logging']['use_wandb'],
        project_name=config['logging']['project_name'],
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        save_dir=os.path.join(args.output_dir, 'checkpoints'),
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = MolecularEvaluator(device=device)
    
    # Test evaluation
    test_metrics = evaluator.evaluate_model(model, test_loader, return_predictions=True)
    metrics, predictions, targets = test_metrics
    
    print(f"Test Results:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Save results
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    if config['evaluation']['save_predictions']:
        import numpy as np
        np.savez(
            os.path.join(results_dir, 'predictions.npz'),
            predictions=predictions,
            targets=targets,
        )
    
    # Plot results
    if config['evaluation']['plot_results']:
        evaluator.plot_predictions(
            predictions, targets, 
            model_name=model.__class__.__name__,
            save_path=os.path.join(results_dir, 'predictions_plot.png')
        )
        evaluator.plot_residuals(
            predictions, targets,
            model_name=model.__class__.__name__,
            save_path=os.path.join(results_dir, 'residuals_plot.png')
        )
    
    # Save configuration and results
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    with open(os.path.join(results_dir, 'results.yaml'), 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    print(f"Results saved to: {results_dir}")
    print("Training completed!")


if __name__ == "__main__":
    main()
