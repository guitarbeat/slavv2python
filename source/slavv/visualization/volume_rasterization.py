"""
Volume Rasterization Module

This module provides functionality to rasterize vascular network elements (vertices, edges)
into 3D volumes. This corresponds to the legacy MATLAB functions `visualize_edges_annuli.m`
and `paint_vertex_image.m`.
"""

import numpy as np
from typing import Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def paint_vertices_to_volume(
    vertices: Dict[str, Any],
    volume_shape: Tuple[int, int, int],
    fill_value: Union[float, str] = 'energy',
    background_value: float = 0.0
) -> np.ndarray:
    """
    Paint vertices as spheres into a 3D volume.

    Args:
        vertices: Vertex data dictionary containing 'positions' and 'radii_pixels' (or 'radii').
                  Positions should be in voxel coordinates (0-indexed).
        volume_shape: Shape of the output volume (Z, Y, X) or (Y, X, Z) depending on convention.
                      SLAVV typically uses (Y, X, Z).
        fill_value: Value to fill the spheres with. 
                    - If 'energy', uses vertex['energies'].
                    - If 'radius', uses vertex radii.
                    - If a float, uses that constant value.
        background_value: Value for the background (default 0.0).

    Returns:
        np.ndarray: The rasterized 3D volume.
    """
    # Initialize volume
    volume = np.full(volume_shape, background_value, dtype=np.float32)
    
    positions = vertices['positions']
    radii = vertices.get('radii_pixels', vertices.get('radii', []))
    energies = vertices.get('energies', [])
    
    if len(positions) == 0:
        return volume
        
    # Determine fill values
    if isinstance(fill_value, str):
        if fill_value == 'energy':
            values = energies
        elif fill_value == 'radius':
            values = radii
        else:
            logger.warning(f"Unknown fill_value '{fill_value}', using 1.0")
            values = np.ones(len(positions))
    else:
        values = np.full(len(positions), fill_value)
        
    # Rasterize each vertex
    # Note: Optimizing this with full vectorization is hard due to varying radii.
    # We iterate but use localized slice indexing for speed.
    
    for i, pos in enumerate(positions):
        radius = radii[i] if len(radii) > i else 1.0
        val = values[i] if len(values) > i else 1.0
        
        # Define bounding box for the sphere
        r_ceil = int(np.ceil(radius))
        center = np.round(pos).astype(int)
        
        # Check bounds
        y_min = max(0, center[0] - r_ceil)
        y_max = min(volume_shape[0], center[0] + r_ceil + 1)
        x_min = max(0, center[1] - r_ceil)
        x_max = min(volume_shape[1], center[1] + r_ceil + 1)
        z_min = max(0, center[2] - r_ceil)
        z_max = min(volume_shape[2], center[2] + r_ceil + 1)
        
        # Create grid for sphere check
        # Meshgrid coordinates need to match the sliced volume indices
        y_range = np.arange(y_min, y_max)
        x_range = np.arange(x_min, x_max)
        z_range = np.arange(z_min, z_max)
        
        if len(y_range) == 0 or len(x_range) == 0 or len(z_range) == 0:
            continue
            
        Y, X, Z = np.meshgrid(y_range, x_range, z_range, indexing='ij')
        
        # Calculate distance squared from exact floating point center
        dist_sq = (Y - pos[0])**2 + (X - pos[1])**2 + (Z - pos[2])**2
        
        # Sphere mask
        mask = dist_sq <= radius**2
        
        # Apply value (max logic to handle overlaps, similar to MATLAB implementation often choosing strongest signal)
        # However, simple overriding is faster. We'll stick to overwriting for now or max if requested.
        # MATLAB code often did replacement.
        
        # Extract current slice
        volume_slice = volume[y_min:y_max, x_min:x_max, z_min:z_max]
        
        # Update only uniform mask area
        volume_slice[mask] = val
        
    return volume


def paint_edges_to_volume(
    edges: Dict[str, Any],
    vertices: Dict[str, Any],
    volume_shape: Tuple[int, int, int],
    thickness_padding: float = 0.0,
    fill_value: Union[float, str] = 'energy',
    background_value: float = 0.0
) -> np.ndarray:
    """
    Paint edges as tubes/ellipsoids into a 3D volume.
    
    Corresponds to `visualize_edges_annuli.m`.
    
    Args:
        edges: Edge data dictionary containing 'traces' and 'connections'.
        vertices: Vertex data (needed for radii if variable thickness is desired).
        volume_shape: Shape of the output volume.
        thickness_padding: Additional thickness added to the radius (vessel wall thickness).
        fill_value: 'energy', 'radius', or constant.
        background_value: Default 0.0.
        
    Returns:
        np.ndarray: The rasterized volume.
    """
    volume = np.full(volume_shape, background_value, dtype=np.float32)
    
    traces = edges['traces']
    edge_energies = edges.get('energies', [])
    vertex_radii = vertices.get('radii_pixels', vertices.get('radii', []))
    connections = edges.get('connections', [])
    
    # Pre-calculate fill values if they are per-edge
    if isinstance(fill_value, str) and fill_value == 'energy':
        # Use edge energy
        pass # Handle inside loop
    elif isinstance(fill_value, float):
        constant_fill = fill_value
    else:
        constant_fill = 1.0

    count = 0 
    for i, trace in enumerate(traces):
        if len(trace) == 0:
            continue
            
        points = np.array(trace)
        
        # Determine value to paint
        if isinstance(fill_value, str) and fill_value == 'energy':
            val = edge_energies[i] if len(edge_energies) > i else 1.0
        else:
            val = constant_fill
            
        # Determine radius at each point
        # If we have connections, we can interpolate radius from start/end vertices
        # If not, we might need a stored radius per point (rarely available in this structure)
        # Default fallback: uniform radius or interpolate vertex radii
        
        radii_at_points = np.ones(len(points))
        if len(connections) > i:
            start_idx, end_idx = int(connections[i][0]), int(connections[i][1])
            if start_idx >= 0 and end_idx >= 0 and len(vertex_radii) > max(start_idx, end_idx):
                r0 = vertex_radii[start_idx]
                r1 = vertex_radii[end_idx]
                # Linear interpolation along the trace
                dists = np.linspace(0, 1, len(points))
                radii_at_points = r0 * (1 - dists) + r1 * dists
        
        # Add padding
        radii_at_points += thickness_padding
        
        # Rasterize each point in the trace as a sphere
        # This approximates a tube. For perfect tubes, one would use cylinder segments, 
        # but dense sphere painting is the standard heuristic here matching `visualize_edges_annuli`.
        
        for j, pos in enumerate(points):
            radius = radii_at_points[j]
            r_ceil = int(np.ceil(radius))
            center = np.round(pos).astype(int)
            
            y_min = max(0, center[0] - r_ceil)
            y_max = min(volume_shape[0], center[0] + r_ceil + 1)
            x_min = max(0, center[1] - r_ceil)
            x_max = min(volume_shape[1], center[1] + r_ceil + 1)
            z_min = max(0, center[2] - r_ceil)
            z_max = min(volume_shape[2], center[2] + r_ceil + 1)
            
            y_range = np.arange(y_min, y_max)
            x_range = np.arange(x_min, x_max)
            z_range = np.arange(z_min, z_max)
            
            if len(y_range) == 0 or len(x_range) == 0 or len(z_range) == 0:
                continue
                
            Y, X, Z = np.meshgrid(y_range, x_range, z_range, indexing='ij')
            dist_sq = (Y - pos[0])**2 + (X - pos[1])**2 + (Z - pos[2])**2
            mask = dist_sq <= radius**2
            
            volume_slice = volume[y_min:y_max, x_min:x_max, z_min:z_max]
            # Use max to blend overlaps instead of simple overwrite?
            # MATLAB `visualize_edges_annuli` overwrites based on energy sorting order.
            # Here we simply overwrite or take max if dealing with positive intensities.
            # Let's take max to ensure thickest part wins if painting generic mask.
            # But for plotting specific 'energy', overwriting might be desired.
            # We will default to max magnitude.
            
            np.maximum(volume_slice, val, out=volume_slice, where=mask)
            
    return volume
