"""Interactive Streamlit demo for MPNN molecular property prediction."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64

from src.data.molecular_dataset import MolecularDataset, generate_synthetic_molecular_data
from src.models.mpnn import MPNN
from src.models.baselines import GCN, GraphSAGE, GAT
from src.utils.device import get_device, set_seed


def create_model(model_name: str, node_dim: int, edge_dim: int, hidden_dim: int = 64) -> torch.nn.Module:
    """Create a model based on the name.
    
    Args:
        model_name: Name of the model to create.
        node_dim: Number of node features.
        edge_dim: Number of edge features.
        hidden_dim: Hidden dimension.
        
    Returns:
        PyTorch model.
    """
    if model_name == "MPNN":
        return MPNN(
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
    elif model_name == "GCN":
        return GCN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
        )
    elif model_name == "GraphSAGE":
        return GraphSAGE(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=0.1,
            aggregation='mean',
            pooling='mean',
        )
    elif model_name == "GAT":
        return GAT(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            pooling='mean',
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def visualize_molecular_graph(data, title: str = "Molecular Graph"):
    """Visualize a molecular graph using NetworkX and Plotly.
    
    Args:
        data: PyTorch Geometric data object.
        title: Title for the plot.
        
    Returns:
        Plotly figure.
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i, features=data.x[i].numpy())
    
    # Add edges
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract coordinates
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines',
        name='Bonds'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=[f'Node {i}' for i in G.nodes()],
        hoverinfo='text',
        name='Atoms'
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Interactive molecular graph visualization",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="MPNN Molecular Property Prediction",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Message Passing Neural Networks for Molecular Property Prediction")
    st.markdown("Interactive demo for predicting molecular properties using Graph Neural Networks")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["MPNN", "GCN", "GraphSAGE", "GAT"],
        index=0
    )
    
    # Parameters
    hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 128, 64)
    num_samples = st.sidebar.slider("Number of Samples", 10, 100, 50)
    
    # Device selection
    device_name = st.sidebar.selectbox(
        "Device",
        ["cpu", "cuda", "mps"],
        index=0
    )
    
    if device_name == "cuda" and not torch.cuda.is_available():
        st.sidebar.warning("CUDA not available, using CPU")
        device_name = "cpu"
    elif device_name == "mps" and not hasattr(torch.backends, "mps"):
        st.sidebar.warning("MPS not available, using CPU")
        device_name = "cpu"
    
    device = torch.device(device_name)
    
    # Set seed
    set_seed(42)
    
    # Generate synthetic data
    st.header("📊 Dataset Overview")
    
    with st.spinner("Generating synthetic molecular data..."):
        synthetic_data = generate_synthetic_molecular_data(
            num_samples=num_samples,
            num_nodes_range=(5, 15),
            num_features=9,
            num_edge_features=3,
            seed=42
        )
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(synthetic_data))
    
    with col2:
        avg_nodes = np.mean([data.num_nodes for data in synthetic_data])
        st.metric("Avg Nodes per Graph", f"{avg_nodes:.1f}")
    
    with col3:
        avg_edges = np.mean([data.num_edges for data in synthetic_data])
        st.metric("Avg Edges per Graph", f"{avg_edges:.1f}")
    
    with col4:
        st.metric("Node Features", synthetic_data[0].x.shape[1])
    
    # Model training and evaluation
    st.header("🤖 Model Training & Evaluation")
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Create model
            model = create_model(
                model_name=model_name,
                node_dim=synthetic_data[0].x.shape[1],
                edge_dim=synthetic_data[0].edge_attr.shape[1],
                hidden_dim=hidden_dim
            )
            model.to(device)
            
            # Create data loaders
            from torch_geometric.loader import DataLoader
            loader = DataLoader(synthetic_data, batch_size=8, shuffle=True)
            
            # Simple training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = torch.nn.L1Loss()
            
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            st.success(f"Model trained! Average Loss: {avg_loss:.4f}")
            
            # Store model in session state
            st.session_state.model = model
            st.session_state.model_name = model_name
    
    # Model evaluation
    if 'model' in st.session_state:
        st.subheader("📈 Model Performance")
        
        # Evaluate on synthetic data
        model = st.session_state.model
        model.eval()
        
        with torch.no_grad():
            predictions = []
            targets = []
            
            for data in synthetic_data[:20]:  # Evaluate on first 20 samples
                data = data.to(device)
                pred = model(data)
                predictions.append(pred.cpu().item())
                targets.append(data.y.cpu().item())
        
        # Calculate metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))
        r2 = 1 - np.sum((np.array(targets) - np.array(predictions))**2) / np.sum((np.array(targets) - np.mean(targets))**2)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{mae:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("R²", f"{r2:.4f}")
        
        # Plot predictions vs targets
        fig = px.scatter(
            x=targets, y=predictions,
            labels={'x': 'True Values', 'y': 'Predicted Values'},
            title=f'{st.session_state.model_name} - Predictions vs Targets'
        )
        
        # Add perfect prediction line
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive molecular graph visualization
    st.header("🔬 Molecular Graph Visualization")
    
    if synthetic_data:
        # Select a sample to visualize
        sample_idx = st.selectbox(
            "Select Sample to Visualize",
            range(len(synthetic_data)),
            format_func=lambda x: f"Sample {x} (Nodes: {synthetic_data[x].num_nodes}, Edges: {synthetic_data[x].num_edges})"
        )
        
        selected_data = synthetic_data[sample_idx]
        
        # Show graph properties
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Nodes", selected_data.num_nodes)
        with col2:
            st.metric("Number of Edges", selected_data.num_edges)
        with col3:
            st.metric("Target Value", f"{selected_data.y.item():.4f}")
        
        # Visualize the graph
        fig = visualize_molecular_graph(selected_data, f"Sample {sample_idx}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show node features
        st.subheader("Node Features")
        node_features_df = pd.DataFrame(
            selected_data.x.numpy(),
            columns=[f"Feature {i}" for i in range(selected_data.x.shape[1])]
        )
        st.dataframe(node_features_df)
        
        # Show edge features
        st.subheader("Edge Features")
        edge_features_df = pd.DataFrame(
            selected_data.edge_attr.numpy(),
            columns=[f"Edge Feature {i}" for i in range(selected_data.edge_attr.shape[1])]
        )
        st.dataframe(edge_features_df)
    
    # Model comparison
    st.header("⚖️ Model Comparison")
    
    if st.button("Compare All Models"):
        with st.spinner("Training and comparing all models..."):
            models = {}
            results = {}
            
            for model_type in ["MPNN", "GCN", "GraphSAGE", "GAT"]:
                # Create and train model
                model = create_model(
                    model_name=model_type,
                    node_dim=synthetic_data[0].x.shape[1],
                    edge_dim=synthetic_data[0].edge_attr.shape[1],
                    hidden_dim=hidden_dim
                )
                model.to(device)
                
                # Quick training
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss_fn = torch.nn.L1Loss()
                
                model.train()
                for _ in range(5):  # Quick training
                    for batch in DataLoader(synthetic_data[:10], batch_size=4, shuffle=True):
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        pred = model(batch)
                        loss = loss_fn(pred, batch.y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    predictions = []
                    targets = []
                    
                    for data in synthetic_data[:20]:
                        data = data.to(device)
                        pred = model(data)
                        predictions.append(pred.cpu().item())
                        targets.append(data.y.cpu().item())
                
                mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
                rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))
                r2 = 1 - np.sum((np.array(targets) - np.array(predictions))**2) / np.sum((np.array(targets) - np.mean(targets))**2)
                
                results[model_type] = {"MAE": mae, "RMSE": rmse, "R²": r2}
            
            # Display results
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df)
            
            # Plot comparison
            fig = go.Figure()
            
            for metric in ["MAE", "RMSE", "R²"]:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=list(results.keys()),
                    y=[results[model][metric] for model in results.keys()]
                ))
            
            fig.update_layout(
                title="Model Comparison",
                xaxis_title="Models",
                yaxis_title="Metric Value",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Demo
    
    This interactive demo showcases Message Passing Neural Networks (MPNNs) for molecular property prediction:
    
    - **MPNN**: Uses edge-conditioned message passing with GRU updates
    - **GCN**: Standard Graph Convolutional Network
    - **GraphSAGE**: Inductive graph learning with neighbor sampling
    - **GAT**: Graph Attention Network with multi-head attention
    
    The demo uses synthetic molecular data to demonstrate the capabilities of different GNN architectures.
    """)
    
    st.markdown("""
    ### Key Features
    
    - Interactive model training and evaluation
    - Real-time molecular graph visualization
    - Model comparison and performance metrics
    - Synthetic data generation for demonstration
    """)
    
    st.markdown("""
    ### Technical Details
    
    - Built with PyTorch Geometric for graph neural networks
    - Streamlit for interactive web interface
    - Plotly for dynamic visualizations
    - NetworkX for graph analysis
    """)
    
    st.markdown("""
    ### Ethical Considerations
    
    - This demo uses synthetic data for educational purposes
    - Real molecular data should be handled with appropriate privacy and safety considerations
    - Model predictions should be validated with domain experts before use in drug discovery
    """)


if __name__ == "__main__":
    main()
