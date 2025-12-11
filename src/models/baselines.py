"""Baseline Graph Neural Network models for comparison."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, global_mean_pool, global_max_pool


class GCN(nn.Module):
    """Graph Convolutional Network baseline model.
    
    Implements the standard GCN architecture for molecular property prediction.
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        """Initialize the GCN model.
        
        Args:
            node_dim: Dimension of node features.
            hidden_dim: Hidden dimension for GCN layers.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            pooling: Pooling method ('mean', 'max').
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass of the GCN model.
        
        Args:
            data: PyTorch Geometric data object.
            
        Returns:
            Graph-level predictions [batch_size, 1].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # GCN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Graph-level pooling
        x = self.pool(x, batch)
        
        # Output prediction
        out = self.output_layers(x)
        
        return out.squeeze(-1)


class GraphSAGE(nn.Module):
    """GraphSAGE baseline model.
    
    Implements the GraphSAGE architecture for molecular property prediction.
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        aggregation: str = "mean",
        pooling: str = "mean",
    ):
        """Initialize the GraphSAGE model.
        
        Args:
            node_dim: Dimension of node features.
            hidden_dim: Hidden dimension for GraphSAGE layers.
            num_layers: Number of GraphSAGE layers.
            dropout: Dropout rate.
            aggregation: Aggregation method ('mean', 'max', 'lstm').
            pooling: Pooling method ('mean', 'max').
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # GraphSAGE layers
        self.sage = GraphSAGE(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout,
            aggr=aggregation,
        )
        
        # Pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass of the GraphSAGE model.
        
        Args:
            data: PyTorch Geometric data object.
            
        Returns:
            Graph-level predictions [batch_size, 1].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # GraphSAGE layers
        x = self.sage(x, edge_index)
        
        # Graph-level pooling
        x = self.pool(x, batch)
        
        # Output prediction
        out = self.output_layers(x)
        
        return out.squeeze(-1)


class GAT(nn.Module):
    """Graph Attention Network baseline model.
    
    Implements the GAT architecture for molecular property prediction.
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        """Initialize the GAT model.
        
        Args:
            node_dim: Dimension of node features.
            hidden_dim: Hidden dimension for GAT layers.
            num_layers: Number of GAT layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            pooling: Pooling method ('mean', 'max').
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    GATConv(node_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
            else:
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass of the GAT model.
        
        Args:
            data: PyTorch Geometric data object.
            
        Returns:
            Graph-level predictions [batch_size, 1].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # GAT layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Graph-level pooling
        x = self.pool(x, batch)
        
        # Output prediction
        out = self.output_layers(x)
        
        return out.squeeze(-1)
    
    def get_attention_weights(self, data) -> torch.Tensor:
        """Get attention weights from the GAT model.
        
        Args:
            data: PyTorch Geometric data object.
            
        Returns:
            Attention weights [num_edges, num_heads].
        """
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        
        # Get attention weights from first layer
        _, attention_weights = self.convs[0](x, edge_index, return_attention_weights=True)
        
        return attention_weights[1]  # Return attention weights only
