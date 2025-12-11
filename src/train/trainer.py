"""Training utilities for molecular property prediction models."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import wandb

from src.utils.device import get_device, count_parameters, get_model_size_mb


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially restore weights.
            
        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class Trainer:
    """Trainer class for molecular property prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        scheduler: str = "plateau",
        loss_fn: str = "mae",
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
        project_name: str = "mpnn-molecular-prediction",
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader.
            optimizer: Optimizer type ('adam', 'adamw').
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            scheduler: Learning rate scheduler ('plateau', 'cosine').
            loss_fn: Loss function ('mae', 'mse', 'huber').
            device: Device to use for training.
            use_wandb: Whether to use Weights & Biases logging.
            project_name: Project name for wandb.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or get_device()
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer.lower() == "adam":
            self.optimizer = Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer.lower() == "adamw":
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Setup scheduler
        if scheduler.lower() == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
            )
        elif scheduler.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6,
            )
        else:
            self.scheduler = None
        
        # Setup loss function
        if loss_fn.lower() == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_fn.lower() == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn.lower() == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                config={
                    "model": model.__class__.__name__,
                    "optimizer": optimizer,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "scheduler": scheduler,
                    "loss_fn": loss_fn,
                    "num_parameters": count_parameters(model),
                    "model_size_mb": get_model_size_mb(model),
                }
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}
    
    def test(self) -> Dict[str, float]:
        """Test the model.
        
        Returns:
            Dictionary containing test metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"test_loss": avg_loss}
    
    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            early_stopping_patience: Patience for early stopping.
            save_dir: Directory to save checkpoints.
            
        Returns:
            Dictionary containing training history.
        """
        # Setup early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "learning_rate": [],
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Test (for monitoring)
            test_metrics = self.test()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()
            
            # Record metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["test_loss"].append(test_metrics["test_loss"])
            history["learning_rate"].append(current_lr)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "test_loss": test_metrics["test_loss"],
                    "learning_rate": current_lr,
                })
            
            # Print progress
            print(
                f"Epoch {epoch+1:3d}/{num_epochs}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Test Loss: {test_metrics['test_loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics["val_loss"],
                        'test_loss': test_metrics["test_loss"],
                    }, os.path.join(save_dir, 'best_model.pt'))
            
            # Early stopping
            if early_stopping(val_metrics["val_loss"], self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final test
        final_test_metrics = self.test()
        print(f"Final Test Loss: {final_test_metrics['test_loss']:.4f}")
        
        return history
    
    def save_model(self, path: str) -> None:
        """Save the model.
        
        Args:
            path: Path to save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load the model.
        
        Args:
            path: Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
