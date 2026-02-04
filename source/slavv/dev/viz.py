"""
Visualization tools for SLAVV validation.

This module provides plotting functions for comparing MATLAB and Python results.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def set_plot_style():
    """Set consistent matplotlib style for all plots."""
    available = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'ggplot' in available:
        plt.style.use('ggplot')
    else:
        plt.style.use('default')
        
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_count_comparison(comparison: Dict[str, Any], output_path: Optional[Path] = None):
    """Create bar chart comparing counts (vertices, edges, strands)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = []
    matlab_counts = []
    python_counts = []
    
    if 'vertices' in comparison and comparison['vertices']:
        categories.append('Vertices')
        matlab_counts.append(comparison['vertices'].get('matlab_count', 0))
        python_counts.append(comparison['vertices'].get('python_count', 0))
    
    if 'edges' in comparison and comparison['edges']:
        categories.append('Edges')
        matlab_counts.append(comparison['edges'].get('matlab_count', 0))
        python_counts.append(comparison['edges'].get('python_count', 0))
    
    if 'network' in comparison and comparison['network']:
        categories.append('Strands')
        matlab_counts.append(comparison['network'].get('matlab_strand_count', 0))
        python_counts.append(comparison['network'].get('python_strand_count', 0))
    
    if not categories:
        return fig
    
    x = np.arange(len(categories))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, matlab_counts, width, label='MATLAB', color='#2E86AB', alpha=0.9)
    bars2 = ax.bar(x + width/2, python_counts, width, label='Python', color='#E63946', alpha=0.9)
    
    ax.set_ylabel('Count')
    ax.set_title('MATLAB vs Python: Component Counts Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig

def plot_radius_distributions(comparison: Dict[str, Any], matlab_radii: Optional[np.ndarray], python_radii: Optional[np.ndarray], output_path: Optional[Path] = None):
    """Create overlaid histograms of radius distributions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Flatten arrays if needed
    if matlab_radii is not None and len(matlab_radii) > 0:
        matlab_radii = np.array(matlab_radii).flatten()
    else:
        matlab_radii = np.array([])
        
    if python_radii is not None and len(python_radii) > 0:
        python_radii = np.array(python_radii).flatten()
    else:
        python_radii = np.array([])
    
    all_radii = np.concatenate([matlab_radii, python_radii]) if (len(matlab_radii) > 0 or len(python_radii) > 0) else []
    
    if len(all_radii) > 0:
        bins = np.histogram_bin_edges(all_radii, bins=50)
        
        if len(matlab_radii) > 0:
            ax.hist(matlab_radii, bins=bins, alpha=0.6, label=f'MATLAB (n={len(matlab_radii)})', color='#2E86AB')
        if len(python_radii) > 0:
            ax.hist(python_radii, bins=bins, alpha=0.6, label=f'Python (n={len(python_radii)})', color='#A23B72')
            
        ax.set_xlabel('Radius (microns)')
        ax.set_ylabel('Frequency')
        ax.set_title('Vessel Radius Distributions')
        ax.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig
