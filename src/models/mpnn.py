"""Message Passing Neural Network implementation for molecular property prediction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool, global_max_pool


class MPNNLayer(nn.Module):
    """Single Message Passing Neural Network layer.
    
    This layer implements the message passing framework where nodes update
    their states by receiving and aggregating messages from neighbors.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        aggregation: str = "mean",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """Initialize the MPNN layer.
        
        Args:
            node_dim: Dimension of node features.
            edge_dim: Dimension of edge features.
            hidden_dim: Hidden dimension for message passing.
            aggregation: Aggregation method ('mean', 'sum', 'max').
            activation: Activation function ('relu', 'gelu', 'tanh').
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.dropout = dropout
        
        # Edge network for computing messages
        self.edge_network = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim),
        )
        
        # Node update network
        self.node_network = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
        )
        
        # Message aggregation
        self.conv = NNConv(
            node_dim, hidden_dim, self.edge_network, aggr=aggregation
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the MPNN layer.
        
        Args:
            x: Node features [num_nodes, node_dim].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge features [num_edges, edge_dim].
            
        Returns:
            Updated node features [num_nodes, hidden_dim].
        """
        # Message passing
        messages = self.conv(x, edge_index, edge_attr)
        
        # Concatenate original features with messages
        combined = torch.cat([x, messages], dim=-1)
        
        # Update node features
        updated = self.node_network(combined)
        
        return updated


class MPNN(nn.Module):
    """Message Passing Neural Network for molecular property prediction.
    
    This model implements a multi-layer MPNN with GRU-based state updates
    and Set2Set pooling for graph-level predictions.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        aggregation: str = "mean",
        activation: str = "relu",
        use_gru: bool = True,
        pooling: str = "set2set",
        processing_steps: int = 3,
    ):
        """Initialize the MPNN model.
        
        Args:
            node_dim: Dimension of node features.
            edge_dim: Dimension of edge features.
            hidden_dim: Hidden dimension for message passing.
            num_layers: Number of MPNN layers.
            dropout: Dropout rate.
            aggregation: Aggregation method for message passing.
            activation: Activation function.
            use_gru: Whether to use GRU for state updates.
            pooling: Pooling method ('set2set', 'mean', 'max', 'attention').
            processing_steps: Number of processing steps for Set2Set.
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gru = use_gru
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # MPNN layers
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(
                hidden_dim,
                edge_dim,
                hidden_dim,
                aggregation=aggregation,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # GRU for state updates (optional)
        if use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Pooling layer
        if pooling == "set2set":
            self.pool = Set2Set(hidden_dim, processing_steps=processing_steps)
            pool_dim = 2 * hidden_dim
        elif pooling == "mean":
            self.pool = global_mean_pool
            pool_dim = hidden_dim
        elif pooling == "max":
            self.pool = global_max_pool
            pool_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass of the MPNN model.
        
        Args:
            data: PyTorch Geometric data object.
            
        Returns:
            Graph-level predictions [batch_size, 1].
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Message passing layers
        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)
        
        # GRU-based state updates (optional)
        if self.use_gru:
            # Reshape for GRU: [batch_size, seq_len, hidden_dim]
            x_reshaped = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            h = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device)
            
            for _ in range(3):  # Multiple GRU steps
                x_reshaped, h = self.gru(x_reshaped, h)
            
            x = x_reshaped.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Graph-level pooling
        if self.pooling == "set2set":
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)
        
        # Output prediction
        out = self.output_layers(x)
        
        return out.squeeze(-1)
    
    def get_embeddings(self, data) -> torch.Tensor:
        """Get node embeddings from the model.
        
        Args:
            data: PyTorch Geometric data object.
            
        Returns:
            Node embeddings [num_nodes, hidden_dim].
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Input projection
        x = self.input_proj(x)
        
        # Message passing layers
        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)
        
        return x
