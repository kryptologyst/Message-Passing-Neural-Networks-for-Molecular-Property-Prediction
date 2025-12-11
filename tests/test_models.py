"""Unit tests for the MPNN molecular property prediction project."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from src.models.mpnn import MPNN, MPNNLayer
from src.models.baselines import GCN, GraphSAGE, GAT
from src.data.molecular_dataset import generate_synthetic_molecular_data
from src.utils.device import get_device, set_seed, count_parameters, get_model_size_mb


class TestMPNN:
    """Test cases for MPNN model."""
    
    def test_mpnn_layer_forward(self):
        """Test MPNN layer forward pass."""
        node_dim = 9
        edge_dim = 3
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_dim)
        
        # Create layer
        layer = MPNNLayer(node_dim, edge_dim, hidden_dim)
        
        # Forward pass
        output = layer(x, edge_index, edge_attr)
        
        assert output.shape == (num_nodes, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_mpnn_forward(self):
        """Test MPNN model forward pass."""
        node_dim = 9
        edge_dim = 3
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_dim)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=torch.randn(1))
        
        # Create model
        model = MPNN(node_dim, edge_dim, hidden_dim)
        
        # Forward pass
        output = model(data)
        
        assert output.shape == (1,)  # Single graph prediction
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_mpnn_embeddings(self):
        """Test MPNN embeddings extraction."""
        node_dim = 9
        edge_dim = 3
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_dim)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Create model
        model = MPNN(node_dim, edge_dim, hidden_dim)
        
        # Get embeddings
        embeddings = model.get_embeddings(data)
        
        assert embeddings.shape == (num_nodes, hidden_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()


class TestBaselineModels:
    """Test cases for baseline models."""
    
    def test_gcn_forward(self):
        """Test GCN model forward pass."""
        node_dim = 9
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, batch=batch, y=torch.randn(1))
        
        # Create model
        model = GCN(node_dim, hidden_dim)
        
        # Forward pass
        output = model(data)
        
        assert output.shape == (1,)  # Single graph prediction
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_graphsage_forward(self):
        """Test GraphSAGE model forward pass."""
        node_dim = 9
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, batch=batch, y=torch.randn(1))
        
        # Create model
        model = GraphSAGE(node_dim, hidden_dim)
        
        # Forward pass
        output = model(data)
        
        assert output.shape == (1,)  # Single graph prediction
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gat_forward(self):
        """Test GAT model forward pass."""
        node_dim = 9
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, batch=batch, y=torch.randn(1))
        
        # Create model
        model = GAT(node_dim, hidden_dim)
        
        # Forward pass
        output = model(data)
        
        assert output.shape == (1,)  # Single graph prediction
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gat_attention_weights(self):
        """Test GAT attention weights extraction."""
        node_dim = 9
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        data = Data(x=x, edge_index=edge_index)
        
        # Create model
        model = GAT(node_dim, hidden_dim)
        
        # Get attention weights
        attention_weights = model.get_attention_weights(data)
        
        assert attention_weights.shape[0] == num_edges
        assert not torch.isnan(attention_weights).any()
        assert not torch.isinf(attention_weights).any()


class TestDataGeneration:
    """Test cases for data generation."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic molecular data generation."""
        num_samples = 10
        num_nodes_range = (5, 15)
        num_features = 9
        num_edge_features = 3
        
        data_list = generate_synthetic_molecular_data(
            num_samples=num_samples,
            num_nodes_range=num_nodes_range,
            num_features=num_features,
            num_edge_features=num_edge_features,
            seed=42
        )
        
        assert len(data_list) == num_samples
        
        for data in data_list:
            assert data.num_nodes >= num_nodes_range[0]
            assert data.num_nodes <= num_nodes_range[1]
            assert data.x.shape[1] == num_features
            assert data.edge_attr.shape[1] == num_edge_features
            assert data.y.shape == (1,)
            assert not torch.isnan(data.x).any()
            assert not torch.isnan(data.edge_attr).any()
            assert not torch.isnan(data.y).any()
    
    def test_synthetic_data_deterministic(self):
        """Test that synthetic data generation is deterministic with same seed."""
        data1 = generate_synthetic_molecular_data(num_samples=5, seed=42)
        data2 = generate_synthetic_molecular_data(num_samples=5, seed=42)
        
        assert len(data1) == len(data2)
        
        for d1, d2 in zip(data1, data2):
            assert torch.allclose(d1.x, d2.x)
            assert torch.allclose(d1.edge_index, d2.edge_index)
            assert torch.allclose(d1.edge_attr, d2.edge_attr)
            assert torch.allclose(d1.y, d2.y)


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        import random
        assert random.getstate()[1][0] == 42
        
        # Test numpy seed
        np.random.seed(42)
        val1 = np.random.random()
        np.random.seed(42)
        val2 = np.random.random()
        assert val1 == val2
        
        # Test torch seed
        torch.manual_seed(42)
        val1 = torch.randn(1)
        torch.manual_seed(42)
        val2 = torch.randn(1)
        assert torch.allclose(val1, val2)
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        model = MPNN(node_dim=9, edge_dim=3, hidden_dim=64)
        num_params = count_parameters(model)
        
        assert isinstance(num_params, int)
        assert num_params > 0
        
        # Count manually
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == manual_count
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model = MPNN(node_dim=9, edge_dim=3, hidden_dim=64)
        size_mb = get_model_size_mb(model)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
        
        # Should be reasonable size (not too large)
        assert size_mb < 100  # Less than 100MB for this small model


class TestModelComparison:
    """Test cases for model comparison."""
    
    def test_all_models_forward(self):
        """Test that all models can perform forward pass."""
        node_dim = 9
        edge_dim = 3
        hidden_dim = 64
        num_nodes = 10
        num_edges = 20
        
        # Create test data
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_dim)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=torch.randn(1))
        
        # Test all models
        models = {
            'MPNN': MPNN(node_dim, edge_dim, hidden_dim),
            'GCN': GCN(node_dim, hidden_dim),
            'GraphSAGE': GraphSAGE(node_dim, hidden_dim),
            'GAT': GAT(node_dim, hidden_dim),
        }
        
        for name, model in models.items():
            output = model(data)
            assert output.shape == (1,), f"{name} output shape incorrect"
            assert not torch.isnan(output).any(), f"{name} output contains NaN"
            assert not torch.isinf(output).any(), f"{name} output contains Inf"
    
    def test_model_parameter_counts(self):
        """Test that models have reasonable parameter counts."""
        node_dim = 9
        edge_dim = 3
        hidden_dim = 64
        
        models = {
            'MPNN': MPNN(node_dim, edge_dim, hidden_dim),
            'GCN': GCN(node_dim, hidden_dim),
            'GraphSAGE': GraphSAGE(node_dim, hidden_dim),
            'GAT': GAT(node_dim, hidden_dim),
        }
        
        for name, model in models.items():
            num_params = count_parameters(model)
            assert num_params > 1000, f"{name} has too few parameters: {num_params}"
            assert num_params < 100000, f"{name} has too many parameters: {num_params}"


if __name__ == "__main__":
    pytest.main([__file__])
