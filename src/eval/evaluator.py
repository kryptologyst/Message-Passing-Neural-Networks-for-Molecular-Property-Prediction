"""Evaluation utilities for molecular property prediction models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class MolecularEvaluator:
    """Evaluator class for molecular property prediction models."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize the evaluator.
        
        Args:
            device: Device to use for evaluation.
        """
        self.device = device or torch.device('cpu')
        
        # Initialize metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
    
    def evaluate_model(
        self,
        model: nn.Module,
        data_loader,
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate.
            data_loader: Data loader for evaluation.
            return_predictions: Whether to return predictions and targets.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                pred = model(batch)
                
                all_predictions.append(pred.cpu())
                all_targets.append(batch.y.cpu())
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        mae = self.mae(predictions, targets).item()
        mse = self.mse(predictions, targets).item()
        rmse = np.sqrt(mse)
        r2 = self.r2(predictions, targets).item()
        
        # Additional metrics
        mape = self._calculate_mape(predictions, targets)
        
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
        }
        
        if return_predictions:
            return metrics, predictions.numpy(), targets.numpy()
        else:
            return metrics
    
    def _calculate_mape(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Mean Absolute Percentage Error.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            MAPE value.
        """
        # Avoid division by zero
        mask = targets != 0
        if mask.sum() == 0:
            return float('inf')
        
        mape = torch.mean(torch.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        return mape.item()
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model names to models.
            data_loader: Data loader for evaluation.
            model_names: Optional list of model names to use.
            
        Returns:
            Dictionary containing metrics for each model.
        """
        if model_names is None:
            model_names = list(models.keys())
        
        results = {}
        
        for name in model_names:
            if name in models:
                metrics = self.evaluate_model(models[name], data_loader)
                results[name] = metrics
                print(f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        return results
    
    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot predictions vs targets.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            model_name: Name of the model for the plot title.
            save_path: Optional path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(targets, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predictions vs Targets\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot residuals (predictions - targets).
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            model_name: Name of the model for the plot title.
            save_path: Optional path to save the plot.
        """
        residuals = predictions - targets
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs predictions
        ax1.scatter(predictions, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'{model_name} - Residuals vs Predictions')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{model_name} - Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "mae",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot comparison of multiple models.
        
        Args:
            results: Results from compare_models.
            metric: Metric to plot.
            save_path: Optional path to save the plot.
        """
        model_names = list(results.keys())
        metric_values = [results[name][metric] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, metric_values, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
    ) -> None:
        """Create a leaderboard table.
        
        Args:
            results: Results from compare_models.
            metrics: List of metrics to include in the leaderboard.
        """
        if metrics is None:
            metrics = ["mae", "rmse", "r2", "mape"]
        
        print("\n" + "="*80)
        print("MODEL LEADERBOARD")
        print("="*80)
        
        # Header
        header = f"{'Model':<20}"
        for metric in metrics:
            header += f"{metric.upper():<12}"
        print(header)
        print("-" * 80)
        
        # Sort models by MAE (lower is better)
        sorted_models = sorted(results.items(), key=lambda x: x[1].get('mae', float('inf')))
        
        # Data rows
        for model_name, model_results in sorted_models:
            row = f"{model_name:<20}"
            for metric in metrics:
                value = model_results.get(metric, 0.0)
                if metric == 'r2':
                    row += f"{value:<12.4f}"
                else:
                    row += f"{value:<12.4f}"
            print(row)
        
        print("="*80)
        print("Note: Lower values are better for MAE, RMSE, MAPE. Higher values are better for R².")
        print("="*80)
