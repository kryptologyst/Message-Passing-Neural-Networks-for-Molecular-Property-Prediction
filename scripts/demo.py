#!/usr/bin/env python3
"""Demonstration script for the MPNN molecular property prediction project.

This script showcases the key features of the modernized project:
- Model creation and training
- Evaluation and comparison
- Visualization and results
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.data.molecular_dataset import generate_synthetic_molecular_data
from src.models.mpnn import MPNN
from src.models.baselines import GCN, GraphSAGE, GAT
from src.train.trainer import Trainer
from src.eval.evaluator import MolecularEvaluator
from src.utils.device import get_device, set_seed, count_parameters


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def demonstrate_data_generation():
    """Demonstrate synthetic data generation."""
    print_section("Data Generation")
    
    print("Generating synthetic molecular data...")
    data = generate_synthetic_molecular_data(
        num_samples=50,
        num_nodes_range=(5, 15),
        num_features=9,
        num_edge_features=3,
        seed=42
    )
    
    print(f"Generated {len(data)} molecular graphs")
    print(f"Sample graph: {data[0].num_nodes} nodes, {data[0].num_edges} edges")
    print(f"Node features shape: {data[0].x.shape}")
    print(f"Edge features shape: {data[0].edge_attr.shape}")
    print(f"Target value: {data[0].y.item():.4f}")


def demonstrate_models():
    """Demonstrate model creation and comparison."""
    print_section("Model Creation")
    
    # Create test data
    data = generate_synthetic_molecular_data(num_samples=1, seed=42)[0]
    node_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]
    hidden_dim = 64
    
    # Create models
    models = {
        'MPNN': MPNN(node_dim, edge_dim, hidden_dim),
        'GCN': GCN(node_dim, hidden_dim),
        'GraphSAGE': GraphSAGE(node_dim, hidden_dim),
        'GAT': GAT(node_dim, hidden_dim),
    }
    
    print("Model Information:")
    for name, model in models.items():
        num_params = count_parameters(model)
        print(f"  {name:12}: {num_params:,} parameters")
    
    # Test forward pass
    print_section("Forward Pass Test")
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(data)
            print(f"  {name:12}: Output shape {output.shape}, Value: {output.item():.4f}")


def demonstrate_training():
    """Demonstrate model training."""
    print_section("Model Training")
    
    # Generate training data
    train_data = generate_synthetic_molecular_data(
        num_samples=100,
        num_nodes_range=(5, 15),
        seed=42
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data[:80], batch_size=8, shuffle=True)
    val_loader = DataLoader(train_data[80:90], batch_size=8, shuffle=False)
    test_loader = DataLoader(train_data[90:], batch_size=8, shuffle=False)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create and train MPNN model
    model = MPNN(
        node_dim=train_data[0].x.shape[1],
        edge_dim=train_data[0].edge_attr.shape[1],
        hidden_dim=64,
        num_layers=2,  # Smaller for demo
        dropout=0.1
    )
    
    device = get_device()
    model.to(device)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=0.001,
        device=device,
        use_wandb=False
    )
    
    print("Training MPNN model...")
    start_time = time.time()
    
    # Quick training for demo
    history = trainer.train(num_epochs=10, early_stopping_patience=5)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Show training history
    print("Training History:")
    for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
        print(f"  Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def demonstrate_evaluation():
    """Demonstrate model evaluation."""
    print_section("Model Evaluation")
    
    # Generate test data
    test_data = generate_synthetic_molecular_data(
        num_samples=50,
        num_nodes_range=(5, 15),
        seed=123
    )
    
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    
    # Create models for comparison
    node_dim = test_data[0].x.shape[1]
    edge_dim = test_data[0].edge_attr.shape[1]
    
    models = {
        'MPNN': MPNN(node_dim, edge_dim, 64),
        'GCN': GCN(node_dim, 64),
        'GraphSAGE': GraphSAGE(node_dim, 64),
        'GAT': GAT(node_dim, 64),
    }
    
    # Quick training for each model
    device = get_device()
    evaluator = MolecularEvaluator(device=device)
    
    print("Training and evaluating models...")
    results = {}
    
    for name, model in models.items():
        model.to(device)
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.L1Loss()
        
        model.train()
        for _ in range(5):  # Quick training
            for batch in DataLoader(test_data[:20], batch_size=4, shuffle=True):
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        metrics = evaluator.evaluate_model(model, test_loader)
        results[name] = metrics
        
        print(f"  {name:12}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Create leaderboard
    evaluator.create_leaderboard(results)
    
    return results


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print_section("Visualization")
    
    # Generate data for visualization
    data = generate_synthetic_molecular_data(
        num_samples=20,
        num_nodes_range=(8, 12),
        seed=42
    )
    
    # Create and train a model
    model = MPNN(
        node_dim=data[0].x.shape[1],
        edge_dim=data[0].edge_attr.shape[1],
        hidden_dim=64
    )
    
    device = get_device()
    model.to(device)
    
    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()
    
    model.train()
    for _ in range(3):
        for batch in DataLoader(data[:10], batch_size=4, shuffle=True):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
    
    # Get predictions
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for sample in data[:15]:
            sample = sample.to(device)
            pred = model(sample)
            predictions.append(pred.cpu().item())
            targets.append(sample.y.cpu().item())
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Predictions vs targets
    plt.subplot(1, 2, 1)
    plt.scatter(targets, predictions, alpha=0.7)
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Targets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(1, 2, 2)
    residuals = np.array(predictions) - np.array(targets)
    plt.scatter(predictions, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/demo_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to assets/demo_visualization.png")
    
    # Calculate and display metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((np.array(targets) - np.mean(targets))**2)
    
    print(f"Visualization Metrics:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")


def main():
    """Main demonstration function."""
    print_header("MPNN Molecular Property Prediction - Demonstration")
    
    print("This demonstration showcases the modernized MPNN project with:")
    print("- Synthetic molecular data generation")
    print("- Multiple GNN model architectures")
    print("- Training and evaluation pipelines")
    print("- Comprehensive visualization")
    print("- Model comparison and leaderboards")
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        # Run demonstrations
        demonstrate_data_generation()
        demonstrate_models()
        demonstrate_training()
        results = demonstrate_evaluation()
        demonstrate_visualization()
        
        print_header("Demonstration Complete")
        print("All features have been successfully demonstrated!")
        print("\nKey Features Showcased:")
        print("✓ Modern project structure with proper organization")
        print("✓ Type hints and comprehensive documentation")
        print("✓ Multiple GNN architectures (MPNN, GCN, GraphSAGE, GAT)")
        print("✓ Robust data pipeline with synthetic data generation")
        print("✓ Comprehensive evaluation with multiple metrics")
        print("✓ Interactive visualization and plotting")
        print("✓ Model comparison and leaderboards")
        print("✓ Production-ready code with testing and CI/CD")
        
        print("\nNext Steps:")
        print("1. Run the interactive demo: streamlit run demo/app.py")
        print("2. Train on real data: python scripts/train.py")
        print("3. Compare models: python scripts/evaluate.py")
        print("4. Explore the codebase in src/ directory")
        print("5. Check out the comprehensive README.md")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check your installation and dependencies.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
