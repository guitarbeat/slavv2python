"""
Visualization module for SLAVV results

This module provides comprehensive visualization capabilities for vascular networks
including 2D/3D plotting, statistical analysis, and export functionality.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from .utils import calculate_path_length

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkVisualizer:
    """
    Comprehensive visualization class for SLAVV results
    """
    
    def __init__(self):
        self.color_schemes = {
            'energy': 'RdBu_r',
            'depth': 'Viridis',
            'strand_id': 'Set3',
            'radius': 'Plasma',
            'random': 'Set1'
        }
    
    def plot_2d_network(self, vertices: Dict[str, Any], edges: Dict[str, Any], 
                       network: Dict[str, Any], parameters: Dict[str, Any],
                       color_by: str = 'energy', projection_axis: int = 2,
                       show_vertices: bool = True, show_edges: bool = True,
                       show_bifurcations: bool = True) -> go.Figure:
        """
        Create 2D network visualization with projection
        
        Args:
            vertices: Vertex data
            edges: Edge data  
            network: Network data
            parameters: Processing parameters
            color_by: Coloring scheme ('energy', 'depth', 'strand_id', 'radius')
            projection_axis: Axis to project along (0=Y, 1=X, 2=Z)
            show_vertices: Whether to show vertices
            show_edges: Whether to show edges
            show_bifurcations: Whether to highlight bifurcations
        """
        logger.info(f"Creating 2D network plot with {color_by} coloring")
        
        fig = go.Figure()
        
        vertex_positions = vertices['positions']
        vertex_energies = vertices['energies']
        vertex_radii = vertices['radii']
        edge_traces = edges['traces']
        bifurcations = network.get('bifurcations', [])
        
        # Determine projection axes
        axes = [0, 1, 2]
        axes.remove(projection_axis)
        x_axis, y_axis = axes
        
        axis_names = ['Y', 'X', 'Z']
        x_label = f"{axis_names[x_axis]} (μm)"
        y_label = f"{axis_names[y_axis]} (μm)"
        
        # Convert to physical units
        microns_per_voxel = parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])
        
        # Plot edges
        if show_edges and edge_traces:
            for i, trace in enumerate(edge_traces):
                if len(trace) < 2:
                    continue
                    
                trace = np.array(trace)
                x_coords = trace[:, x_axis] * microns_per_voxel[x_axis]
                y_coords = trace[:, y_axis] * microns_per_voxel[y_axis]
                
                # Color by specified scheme
                if color_by == 'energy':
                    # Use average energy along trace (simplified)
                    color = 'blue'
                elif color_by == 'depth':
                    depth = np.mean(trace[:, projection_axis]) * microns_per_voxel[projection_axis]
                    color = f'rgba(0, 0, 255, {0.3 + 0.7 * depth / 100})'
                elif color_by == 'strand_id':
                    color_idx = i % 10
                    colors = px.colors.qualitative.Set3
                    color = colors[color_idx]
                else:
                    color = 'blue'
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'Edge {i}' if i < 10 else '',
                    showlegend=i < 10,
                    hovertemplate=f'Edge {i}<br>Length: {calculate_path_length(trace):.1f} μm<extra></extra>'
                ))
        
        # Plot vertices
        if show_vertices and len(vertex_positions) > 0:
            x_coords = vertex_positions[:, x_axis] * microns_per_voxel[x_axis]
            y_coords = vertex_positions[:, y_axis] * microns_per_voxel[y_axis]
            
            # Color vertices
            if color_by == 'energy':
                colors = vertex_energies
                colorscale = 'RdBu_r'
            elif color_by == 'depth':
                colors = vertex_positions[:, projection_axis] * microns_per_voxel[projection_axis]
                colorscale = 'Viridis'
            elif color_by == 'radius':
                colors = vertex_radii
                colorscale = 'Plasma'
            else:
                colors = 'red'
                colorscale = None
            
            marker_dict = dict(
                size=8,
                color=colors,
                colorscale=colorscale,
                showscale=True if colorscale else False,
                colorbar=dict(title=color_by.title()) if colorscale else None,
                line=dict(width=1, color='black')
            )
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers',
                marker=marker_dict,
                name='Vertices',
                hovertemplate='Vertex<br>Energy: %{customdata[0]:.3f}<br>Radius: %{customdata[1]:.2f} μm<extra></extra>',
                customdata=np.column_stack([vertex_energies, vertex_radii])
            ))
        
        # Highlight bifurcations
        if show_bifurcations and len(bifurcations) > 0:
            bif_positions = vertex_positions[bifurcations]
            x_coords = bif_positions[:, x_axis] * microns_per_voxel[x_axis]
            y_coords = bif_positions[:, y_axis] * microns_per_voxel[y_axis]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers',
                marker=dict(
                    size=12,
                    color='yellow',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name='Bifurcations',
                hovertemplate='Bifurcation<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"2D Vascular Network (Projection along {axis_names[projection_axis]})",
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=True,
            hovermode='closest',
            width=800,
            height=600
        )
        
        return fig
    
    def plot_3d_network(self, vertices: Dict[str, Any], edges: Dict[str, Any], 
                       network: Dict[str, Any], parameters: Dict[str, Any],
                       color_by: str = 'energy', show_vertices: bool = True, 
                       show_edges: bool = True, show_bifurcations: bool = True) -> go.Figure:
        """
        Create 3D network visualization
        """
        logger.info(f"Creating 3D network plot with {color_by} coloring")
        
        fig = go.Figure()
        
        vertex_positions = vertices['positions']
        vertex_energies = vertices['energies']
        vertex_radii = vertices['radii']
        edge_traces = edges['traces']
        bifurcations = network.get('bifurcations', [])
        
        # Convert to physical units
        microns_per_voxel = parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])
        
        # Plot edges as 3D lines
        if show_edges and edge_traces:
            for i, trace in enumerate(edge_traces):
                if len(trace) < 2:
                    continue
                    
                trace = np.array(trace)
                x_coords = trace[:, 1] * microns_per_voxel[1]  # X
                y_coords = trace[:, 0] * microns_per_voxel[0]  # Y
                z_coords = trace[:, 2] * microns_per_voxel[2]  # Z
                
                # Color by specified scheme
                if color_by == 'energy':
                    color = 'blue'
                elif color_by == 'depth':
                    depth = np.mean(z_coords)
                    color = f'rgba(0, 0, 255, {0.3 + 0.7 * depth / 100})'
                elif color_by == 'strand_id':
                    color_idx = i % 10
                    colors = px.colors.qualitative.Set3
                    color = colors[color_idx]
                else:
                    color = 'blue'
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f'Edge {i}' if i < 10 else '',
                    showlegend=i < 10,
                    hovertemplate=f'Edge {i}<br>Length: {calculate_path_length(trace):.1f} μm<extra></extra>'
                ))
        
        # Plot vertices
        if show_vertices and len(vertex_positions) > 0:
            x_coords = vertex_positions[:, 1] * microns_per_voxel[1]  # X
            y_coords = vertex_positions[:, 0] * microns_per_voxel[0]  # Y
            z_coords = vertex_positions[:, 2] * microns_per_voxel[2]  # Z
            
            # Color vertices
            if color_by == 'energy':
                colors = vertex_energies
                colorscale = 'RdBu_r'
            elif color_by == 'depth':
                colors = z_coords
                colorscale = 'Viridis'
            elif color_by == 'radius':
                colors = vertex_radii
                colorscale = 'Plasma'
            else:
                colors = 'red'
                colorscale = None
            
            marker_dict = dict(
                size=6,
                color=colors,
                colorscale=colorscale,
                showscale=True if colorscale else False,
                colorbar=dict(title=color_by.title()) if colorscale else None,
                line=dict(width=1, color='black')
            )
            
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='markers',
                marker=marker_dict,
                name='Vertices',
                hovertemplate='Vertex<br>Energy: %{customdata[0]:.3f}<br>Radius: %{customdata[1]:.2f} μm<extra></extra>',
                customdata=np.column_stack([vertex_energies, vertex_radii])
            ))
        
        # Highlight bifurcations
        if show_bifurcations and len(bifurcations) > 0:
            bif_positions = vertex_positions[bifurcations]
            x_coords = bif_positions[:, 1] * microns_per_voxel[1]
            y_coords = bif_positions[:, 0] * microns_per_voxel[0]
            z_coords = bif_positions[:, 2] * microns_per_voxel[2]
            
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='markers',
                marker=dict(
                    size=10,
                    color='yellow',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name='Bifurcations',
                hovertemplate='Bifurcation<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title="3D Vascular Network",
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)",
                zaxis_title="Z (μm)",
                aspectmode='data'
            ),
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def plot_energy_field(self, energy_data: Dict[str, Any], slice_axis: int = 2, 
                         slice_index: Optional[int] = None) -> go.Figure:
        """
        Visualize energy field as 2D slice
        """
        logger.info("Creating energy field visualization")
        
        energy = energy_data['energy']
        
        if slice_index is None:
            slice_index = energy.shape[slice_axis] // 2
        
        # Extract slice
        if slice_axis == 0:  # Y slice
            energy_slice = energy[slice_index, :, :]
            x_label, y_label = "X", "Z"
        elif slice_axis == 1:  # X slice
            energy_slice = energy[:, slice_index, :]
            x_label, y_label = "Y", "Z"
        else:  # Z slice
            energy_slice = energy[:, :, slice_index]
            x_label, y_label = "Y", "X"
        
        fig = go.Figure(data=go.Heatmap(
            z=energy_slice,
            colorscale='RdBu_r',
            colorbar=dict(title="Energy"),
            hovertemplate=f'{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Energy: %{{z:.3f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Energy Field ({['Y', 'X', 'Z'][slice_axis]} slice at index {slice_index})",
            xaxis_title=f"{x_label} (voxels)",
            yaxis_title=f"{y_label} (voxels)",
            width=600,
            height=500
        )
        
        return fig
    
    def plot_strand_analysis(self, network: Dict[str, Any], vertices: Dict[str, Any],
                           parameters: Dict[str, Any]) -> go.Figure:
        """
        Create strand length and connectivity analysis
        """
        logger.info("Creating strand analysis plot")
        
        strands = network['strands']
        vertex_positions = vertices['positions']
        microns_per_voxel = parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])
        
        # Calculate strand lengths
        strand_lengths = []
        for strand in strands:
            if len(strand) > 1:
                length = 0
                for i in range(len(strand) - 1):
                    pos1 = vertex_positions[strand[i]] * microns_per_voxel
                    pos2 = vertex_positions[strand[i+1]] * microns_per_voxel
                    length += np.linalg.norm(pos2 - pos1)
                strand_lengths.append(length)
        
        if not strand_lengths:
            # Return empty plot if no strands
            fig = go.Figure()
            fig.add_annotation(text="No strands found", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create histogram
        fig = go.Figure(data=go.Histogram(
            x=strand_lengths,
            nbinsx=min(20, len(strand_lengths)),
            name='Strand Lengths',
            hovertemplate='Length: %{x:.1f} μm<br>Count: %{y}<extra></extra>'
        ))
        
        # Add statistics
        mean_length = np.mean(strand_lengths)
        median_length = np.median(strand_lengths)
        
        fig.add_vline(x=mean_length, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_length:.1f} μm")
        fig.add_vline(x=median_length, line_dash="dot", line_color="green",
                     annotation_text=f"Median: {median_length:.1f} μm")
        
        fig.update_layout(
            title="Strand Length Distribution",
            xaxis_title="Length (μm)",
            yaxis_title="Count",
            showlegend=True,
            width=600,
            height=400
        )
        
        return fig
    
    def plot_depth_statistics(self, vertices: Dict[str, Any], edges: Dict[str, Any],
                            parameters: Dict[str, Any], n_bins: int = 10) -> go.Figure:
        """
        Create depth-resolved statistics
        """
        logger.info("Creating depth statistics plot")
        
        vertex_positions = vertices['positions']
        edge_traces = edges['traces']
        microns_per_voxel = parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])
        
        # Get Z coordinates (depth)
        vertex_depths = vertex_positions[:, 2] * microns_per_voxel[2]
        
        # Calculate edge lengths per depth bin
        edge_depths = []
        edge_lengths = []
        
        for trace in edge_traces:
            if len(trace) < 2:
                continue
            trace = np.array(trace)
            depth = np.mean(trace[:, 2]) * microns_per_voxel[2]
            length = calculate_path_length(trace * microns_per_voxel)
            edge_depths.append(depth)
            edge_lengths.append(length)
        
        if not edge_depths:
            fig = go.Figure()
            fig.add_annotation(text="No edges found", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create depth bins
        min_depth = min(min(vertex_depths), min(edge_depths))
        max_depth = max(max(vertex_depths), max(edge_depths))
        depth_bins = np.linspace(min_depth, max_depth, n_bins + 1)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
        
        # Bin vertices
        vertex_counts, _ = np.histogram(vertex_depths, bins=depth_bins)
        
        # Bin edge lengths
        length_sums, _ = np.histogram(edge_depths, bins=depth_bins, weights=edge_lengths)
        
        # Create subplot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add vertex count
        fig.add_trace(
            go.Bar(x=bin_centers, y=vertex_counts, name="Vertex Count", opacity=0.7),
            secondary_y=False,
        )
        
        # Add total length
        fig.add_trace(
            go.Scatter(x=bin_centers, y=length_sums, name="Total Length", 
                      mode='lines+markers', line=dict(color='red')),
            secondary_y=True,
        )
        
        # Update axes
        fig.update_xaxes(title_text="Depth (μm)")
        fig.update_yaxes(title_text="Vertex Count", secondary_y=False)
        fig.update_yaxes(title_text="Total Length (μm)", secondary_y=True)
        
        fig.update_layout(
            title="Depth-Resolved Network Statistics",
            showlegend=True,
            width=700,
            height=400
        )
        
        return fig
    
    def plot_radius_distribution(self, vertices: Dict[str, Any]) -> go.Figure:
        """
        Create vessel radius distribution plot
        """
        logger.info("Creating radius distribution plot")
        
        radii = vertices['radii']
        
        fig = go.Figure(data=go.Histogram(
            x=radii,
            nbinsx=min(25, len(radii)),
            name='Vessel Radii',
            hovertemplate='Radius: %{x:.2f} μm<br>Count: %{y}<extra></extra>'
        ))
        
        # Add statistics
        mean_radius = np.mean(radii)
        median_radius = np.median(radii)
        
        fig.add_vline(x=mean_radius, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_radius:.2f} μm")
        fig.add_vline(x=median_radius, line_dash="dot", line_color="green",
                     annotation_text=f"Median: {median_radius:.2f} μm")
        
        fig.update_layout(
            title="Vessel Radius Distribution",
            xaxis_title="Radius (μm)",
            yaxis_title="Count",
            width=600,
            height=400
        )
        
        return fig
    
    def plot_degree_distribution(self, network: Dict[str, Any]) -> go.Figure:
        """
        Create vertex degree distribution plot
        """
        logger.info("Creating degree distribution plot")
        
        vertex_degrees = network.get('vertex_degrees', [])
        
        if len(vertex_degrees) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No degree data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Count degrees
        unique_degrees, counts = np.unique(vertex_degrees, return_counts=True)
        
        fig = go.Figure(data=go.Bar(
            x=unique_degrees,
            y=counts,
            name='Degree Distribution',
            hovertemplate='Degree: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Vertex Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Count",
            width=500,
            height=400
        )
        
        return fig
    
    def create_summary_dashboard(self, processing_results: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive summary dashboard
        """
        logger.info("Creating summary dashboard")
        
        vertices = processing_results['vertices']
        edges = processing_results['edges']
        network = processing_results['network']
        parameters = processing_results['parameters']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Overview', 'Strand Lengths', 'Radius Distribution', 'Depth Statistics'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"secondary_y": True}]]
        )
        
        # Network overview (2D projection)
        vertex_positions = vertices['positions']
        if len(vertex_positions) > 0:
            microns_per_voxel = parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])
            x_coords = vertex_positions[:, 1] * microns_per_voxel[1]
            y_coords = vertex_positions[:, 0] * microns_per_voxel[0]
            
            fig.add_trace(
                go.Scatter(x=x_coords, y=y_coords, mode='markers',
                          marker=dict(size=4, color='red'), name='Vertices'),
                row=1, col=1
            )
        
        # Strand lengths
        strands = network['strands']
        strand_lengths = []
        for strand in strands:
            if len(strand) > 1:
                length = 0
                for i in range(len(strand) - 1):
                    pos1 = vertex_positions[strand[i]] * parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])
                    pos2 = vertex_positions[strand[i+1]] * parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])
                    length += np.linalg.norm(pos2 - pos1)
                strand_lengths.append(length)
        
        if strand_lengths:
            fig.add_trace(
                go.Histogram(x=strand_lengths, nbinsx=15, name='Strand Lengths'),
                row=1, col=2
            )
        
        # Radius distribution
        radii = vertices['radii']
        if len(radii) > 0:
            fig.add_trace(
                go.Histogram(x=radii, nbinsx=15, name='Radii'),
                row=2, col=1
            )
        
        # Depth statistics (simplified)
        if len(vertex_positions) > 0:
            depths = vertex_positions[:, 2] * parameters.get('microns_per_voxel', [1.0, 1.0, 1.0])[2]
            depth_counts, depth_bins = np.histogram(depths, bins=10)
            bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
            
            fig.add_trace(
                go.Bar(x=bin_centers, y=depth_counts, name='Vertex Count by Depth'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="SLAVV Processing Summary Dashboard",
            showlegend=False,
            height=600,
            width=1000
        )
        
        return fig
    
    def export_network_data(self, processing_results: Dict[str, Any], 
                           output_path: str, format: str = 'csv') -> str:
        """
        Export network data in various formats
        
        Args:
            processing_results: Complete SLAVV processing results
            output_path: Output file path
            format: Export format ('csv', 'json', 'vmv', 'casx')
        
        Returns:
            Path to exported file
        """
        logger.info(f"Exporting network data in {format} format")
        
        vertices = processing_results['vertices']
        edges = processing_results['edges']
        network = processing_results['network']
        parameters = processing_results['parameters']
        
        if format == 'csv':
            return self._export_csv(vertices, edges, network, parameters, output_path)
        elif format == 'json':
            return self._export_json(processing_results, output_path)
        elif format == 'vmv':
            return self._export_vmv(vertices, edges, network, parameters, output_path)
        elif format == 'casx':
            return self._export_casx(vertices, edges, network, parameters, output_path)
        elif format == 'mat':
            return self._export_mat(vertices, edges, network, parameters, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_csv(self, vertices: Dict[str, Any], edges: Dict[str, Any], 
                   network: Dict[str, Any], parameters: Dict[str, Any], 
                   output_path: str) -> str:
        """Export data as CSV files"""
        base_path = Path(output_path).with_suffix('')
        
        # Export vertices
        vertex_df = pd.DataFrame({
            'vertex_id': range(len(vertices['positions'])),
            'y_position': vertices['positions'][:, 0],
            'x_position': vertices['positions'][:, 1],
            'z_position': vertices['positions'][:, 2],
            'energy': vertices['energies'],
            'radius': vertices['radii'],
            'scale': vertices['scales']
        })
        vertex_path = f"{base_path}_vertices.csv"
        vertex_df.to_csv(vertex_path, index=False)
        
        # Export edges
        edge_data = []
        for i, (trace, connection) in enumerate(
            zip(edges['traces'], edges['connections'])
        ):
            start_vertex, end_vertex = connection
            trace = np.array(trace)
            length = calculate_path_length(trace)
            
            edge_data.append({
                'edge_id': i,
                'start_vertex': start_vertex,
                'end_vertex': end_vertex,
                'length': length,
                'n_points': len(trace)
            })
        
        edge_df = pd.DataFrame(edge_data)
        edge_path = f"{base_path}_edges.csv"
        edge_df.to_csv(edge_path, index=False)
        
        logger.info(f"CSV export complete: {vertex_path}, {edge_path}")
        return vertex_path
    
    def _export_json(self, processing_results: Dict[str, Any], output_path: str) -> str:
        """Export complete results as JSON"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_data = convert_numpy(processing_results)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"JSON export complete: {output_path}")
        return output_path
    
    def _export_vmv(self, vertices: Dict[str, Any], edges: Dict[str, Any], 
                   network: Dict[str, Any], parameters: Dict[str, Any], 
                   output_path: str) -> str:
        """Export in VMV (Vascular Modeling Visualization) format"""
        # Simplified VMV export - would need full specification for complete implementation
        with open(output_path, 'w') as f:
            f.write("# VMV Format Export\n")
            f.write(f"# Generated by SLAVV Python Implementation\n")
            f.write(f"# Vertices: {len(vertices['positions'])}\n")
            f.write(f"# Edges: {len(edges['traces'])}\n")
            f.write("\n[VERTICES]\n")
            
            for i, (pos, energy, radius) in enumerate(zip(
                vertices['positions'], vertices['energies'], vertices['radii']
            )):
                f.write(f"{i} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} {radius:.3f} {energy:.6f}\n")
            
            f.write("\n[EDGES]\n")
            for i, (trace, connection) in enumerate(zip(edges['traces'], edges['connections'])):
                start_vertex, end_vertex = connection
                f.write(f"{i} {start_vertex} {end_vertex}\n")
        
        logger.info(f"VMV export complete: {output_path}")
        return output_path
    
    def _export_casx(self, vertices: Dict[str, Any], edges: Dict[str, Any], 
                    network: Dict[str, Any], parameters: Dict[str, Any], 
                    output_path: str) -> str:
        """Export in CASX format"""
        # Simplified CASX export - would need full specification for complete implementation
        with open(output_path, 'w') as f:
            f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write("<CasX>\n")
            f.write("  <Network>\n")
            
            # Write vertices
            f.write("    <Vertices>\n")
            for i, (pos, radius) in enumerate(zip(vertices['positions'], vertices['radii'])):
                f.write(f"      <Vertex id=\"{i}\" x=\"{pos[1]:.3f}\" y=\"{pos[0]:.3f}\" z=\"{pos[2]:.3f}\" radius=\"{radius:.3f}\"/>\n")
            f.write("    </Vertices>\n")
            
            # Write edges
            f.write("    <Edges>\n")
            for i, connection in enumerate(edges['connections']):
                start_vertex, end_vertex = connection
                if start_vertex is not None and end_vertex is not None:
                    f.write(f"      <Edge id=\"{i}\" start=\"{start_vertex}\" end=\"{end_vertex}\"/>\n")
            f.write("    </Edges>\n")
            
            f.write("  </Network>\n")
            f.write("</CasX>\n")
        
        logger.info(f"CASX export complete: {output_path}")
        return output_path

    def _export_mat(self, vertices: Dict[str, Any], edges: Dict[str, Any],
                    network: Dict[str, Any], parameters: Dict[str, Any],
                    output_path: str) -> str:
        """Export data to MATLAB .mat format using scipy.io.savemat"""
        try:
            from scipy.io import savemat
        except ImportError as e:
            raise ImportError("scipy is required for MAT export. Please install scipy.") from e

        data = {
            'vertices': {
                'positions': np.asarray(vertices.get('positions', [])),
                'energies': np.asarray(vertices.get('energies', [])),
                'radii': np.asarray(vertices.get('radii', [])),
                'scales': np.asarray(vertices.get('scales', [])),
            },
            'edges': {
                'connections': np.asarray(edges.get('connections', []), dtype=object),
                'traces': np.array([np.asarray(t) for t in edges.get('traces', [])], dtype=object),
            },
            'network': {
                'strands': np.asarray(network.get('strands', []), dtype=object),
                'bifurcations': np.asarray(network.get('bifurcations', [])),
                'vertex_degrees': np.asarray(network.get('vertex_degrees', [])),
            },
            'parameters': parameters,
        }

        savemat(output_path, data, do_compression=True)
        logger.info(f"MAT export complete: {output_path}")
        return output_path

