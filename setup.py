#!/usr/bin/env python3
"""Setup script for the MPNN molecular property prediction project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up MPNN Molecular Property Prediction Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        return 1
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
            break
    
    if not success:
        print("\nSetup failed. Please check the error messages above.")
        return 1
    
    # Create necessary directories
    directories = ["data", "outputs", "assets", "checkpoints"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Test installation
    print("\nTesting installation...")
    test_commands = [
        ("python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"", "Testing PyTorch"),
        ("python -c \"import torch_geometric; print(f'PyG: {torch_geometric.__version__}')\"", "Testing PyTorch Geometric"),
        ("python -c \"from src.models.mpnn import MPNN; print('MPNN import successful')\"", "Testing MPNN import"),
        ("python -c \"from src.models.baselines import GCN; print('Baseline models import successful')\"", "Testing baseline models"),
    ]
    
    for command, description in test_commands:
        if not run_command(command, description):
            print(f"Warning: {description} failed")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demonstration: python scripts/demo.py")
    print("2. Launch the interactive demo: streamlit run demo/app.py")
    print("3. Train a model: python scripts/train.py")
    print("4. Compare models: python scripts/evaluate.py")
    print("\nFor more information, see README.md")
    
    return 0


if __name__ == "__main__":
    exit(main())
