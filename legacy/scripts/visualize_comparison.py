#!/usr/bin/env python3
"""
Visualization tools for MATLAB-Python comparison results.

This script generates plots and charts to visualize the comparison between
MATLAB and Python implementations of SLAVV vectorization.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from matlab_output_parser import load_matlab_batch_results


def set_plot_style():
    """Set consistent matplotlib style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot' if 'ggplot' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_count_comparison(comparison: Dict[str, Any], output_path: Path):
    """Create bar chart comparing counts (vertices, edges, strands).
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data from compare_matlab_python.py
    output_path : Path
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract counts
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
        print("No count data available for plotting")
        return
    
    x = np.arange(len(categories))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, matlab_counts, width, label='MATLAB', 
                   color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, python_counts, width, label='Python', 
                   color='#E63946', alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Component', fontweight='bold', fontsize=14)
    ax.set_ylabel('Count', fontweight='bold', fontsize=14)
    ax.set_title('MATLAB vs Python: Component Counts Comparison', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars with better formatting
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add difference annotations between bars
    for i, (m_count, p_count) in enumerate(zip(matlab_counts, python_counts)):
        if m_count > 0 or p_count > 0:
            diff = abs(p_count - m_count)
            avg = (m_count + p_count) / 2.0
            if avg > 0:
                pct_diff = (diff / avg) * 100.0
                y_pos = max(m_count, p_count) * 1.15
                ax.text(i, y_pos, f'Δ {pct_diff:.1f}%',
                       ha='center', va='bottom', fontsize=9, 
                       style='italic', color='#555555')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved count comparison plot: {output_path}")


def plot_timing_breakdown(comparison: Dict[str, Any], output_path: Path):
    """Create stacked bar chart showing timing breakdown.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data from compare_matlab_python.py
    output_path : Path
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract timing data
    matlab_time = comparison.get('matlab', {}).get('elapsed_time', 0)
    python_time = comparison.get('python', {}).get('elapsed_time', 0)
    
    if matlab_time == 0 and python_time == 0:
        print("No timing data available for plotting")
        return
    
    implementations = ['MATLAB', 'Python']
    times = [matlab_time, python_time]
    colors = ['#2E86AB', '#E63946']
    
    bars = ax.bar(implementations, times, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=1.2, width=0.6)
    
    ax.set_ylabel('Time (seconds)', fontweight='bold', fontsize=14)
    ax.set_title('MATLAB vs Python: Execution Time Comparison', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Format time labels
    def format_time_label(seconds):
        if seconds < 60:
            return f'{seconds:.1f}s'
        elif seconds < 3600:
            return f'{int(seconds//60)}m {int(seconds%60)}s'
        else:
            return f'{int(seconds//3600)}h {int((seconds%3600)//60)}m'
    
    # Add value labels with better formatting
    for bar, time in zip(bars, times):
        if time > 0:
            label = format_time_label(time)
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02,
                   label,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
            # Add seconds as secondary label
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                   f'({time:.0f}s)',
                   ha='center', va='center', fontsize=10, color='white', 
                   fontweight='bold')
    
    # Add speedup annotation with better styling
    if matlab_time > 0 and python_time > 0:
        speedup = matlab_time / python_time
        faster = "Python" if python_time < matlab_time else "MATLAB"
        ax.text(0.5, max(times) * 0.85,
               f'{faster} is {abs(speedup):.1f}x faster',
               ha='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFD166', 
                        edgecolor='black', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved timing breakdown plot: {output_path}")


def plot_radius_distributions(comparison: Dict[str, Any], matlab_data: Optional[Dict], python_data: Optional[Dict], output_path: Path):
    """Create overlaid histograms of radius distributions.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data
    matlab_data : Optional[Dict]
        Parsed MATLAB data
    python_data : Optional[Dict]
        Python results data
    output_path : Path
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    matlab_radii = None
    python_radii = None
    
    # Extract MATLAB radii
    if matlab_data and 'vertices' in matlab_data:
        matlab_radii = matlab_data['vertices'].get('radii', np.array([]))
        if isinstance(matlab_radii, np.ndarray) and matlab_radii.size > 0:
            matlab_radii = matlab_radii.flatten()
    
    # Extract Python radii
    if python_data and 'vertices' in python_data:
        python_radii = python_data['vertices'].get('radii', np.array([]))
        if isinstance(python_radii, np.ndarray) and python_radii.size > 0:
            python_radii = python_radii.flatten()
    
    if (matlab_radii is None or matlab_radii.size == 0) and (python_radii is None or python_radii.size == 0):
        print("No radius data available for plotting")
        return
    
    # Determine common bins
    all_radii = []
    if matlab_radii is not None and matlab_radii.size > 0:
        all_radii.extend(matlab_radii)
    if python_radii is not None and python_radii.size > 0:
        all_radii.extend(python_radii)
    
    if all_radii:
        bins = np.histogram_bin_edges(all_radii, bins=50)
        
        # Plot histograms
        if matlab_radii is not None and matlab_radii.size > 0:
            ax.hist(matlab_radii, bins=bins, alpha=0.6, label=f'MATLAB (n={len(matlab_radii)})',
                   color='#2E86AB', edgecolor='black', linewidth=0.5)
        
        if python_radii is not None and python_radii.size > 0:
            ax.hist(python_radii, bins=bins, alpha=0.6, label=f'Python (n={len(python_radii)})',
                   color='#A23B72', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Radius (microns)')
        ax.set_ylabel('Frequency')
        ax.set_title('Vessel Radius Distributions: MATLAB vs Python')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics as text
        stats_text = []
        if matlab_radii is not None and matlab_radii.size > 0:
            stats_text.append(f'MATLAB: μ={np.mean(matlab_radii):.2f}, σ={np.std(matlab_radii):.2f}')
        if python_radii is not None and python_radii.size > 0:
            stats_text.append(f'Python: μ={np.mean(python_radii):.2f}, σ={np.std(python_radii):.2f}')
        
        if stats_text:
            ax.text(0.98, 0.97, '\n'.join(stats_text),
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved radius distributions plot: {output_path}")


def plot_radius_correlation(comparison: Dict[str, Any], output_path: Path):
    """Create scatter plot of matched vertex radii correlation.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data
    output_path : Path
        Path to save the plot
    """
    if 'vertices' not in comparison or not comparison['vertices'].get('radius_correlation'):
        print("No radius correlation data available for plotting")
        return
    
    # This requires access to matched radii pairs, which we don't have in the comparison dict
    # We would need to modify the comparison script to include this data
    # For now, just show the correlation coefficient
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    corr = comparison['vertices']['radius_correlation']
    pearson_r = corr.get('pearson_r', 0)
    
    # Create a text-based visualization
    ax.text(0.5, 0.5, f'Radius Correlation\n\nPearson r = {pearson_r:.3f}',
           ha='center', va='center', fontsize=16,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved radius correlation plot: {output_path}")


def plot_summary_dashboard(comparison: Dict[str, Any], output_path: Path):
    """Create a comprehensive dashboard with multiple subplots.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data
    output_path : Path
        Path to save the plot
    """
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle('MATLAB vs Python: Comprehensive Comparison Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, 
                  top=0.94, bottom=0.06, left=0.08, right=0.96)
    
    # 1. Count comparison (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_counts_subplot(ax1, comparison)
    
    # 2. Timing comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_timing_subplot(ax2, comparison)
    
    # 3. Vertex matching (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_vertex_matching_subplot(ax3, comparison)
    
    # 4. Position errors (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_position_errors_subplot(ax4, comparison)
    
    # 5. Radius stats (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_radius_stats_subplot(ax5, comparison)
    
    # 6. Summary metrics (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    plot_summary_metrics_subplot(ax6, comparison)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary dashboard: {output_path}")


def plot_counts_subplot(ax, comparison):
    """Helper: plot counts in a subplot."""
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
    
    if not categories:
        return
    
    x = np.arange(len(categories))
    width = 0.38
    bars1 = ax.bar(x - width/2, matlab_counts, width, label='MATLAB', 
                   color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, python_counts, width, label='Python', 
                   color='#E63946', alpha=0.85, edgecolor='black', linewidth=1)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Component Counts', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')


def plot_timing_subplot(ax, comparison):
    """Helper: plot timing in a subplot."""
    matlab_time = comparison.get('matlab', {}).get('elapsed_time', 0)
    python_time = comparison.get('python', {}).get('elapsed_time', 0)
    
    if matlab_time > 0 or python_time > 0:
        implementations = ['MATLAB', 'Python']
        times = [matlab_time, python_time]
        colors = ['#2E86AB', '#E63946']
        bars = ax.bar(implementations, times, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1)
        ax.set_ylabel('Time (s)', fontweight='bold')
        ax.set_title('Execution Time', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels
        for bar, time in zip(bars, times):
            if time > 0:
                # Format time nicely
                if time < 60:
                    label = f'{time:.1f}s'
                elif time < 3600:
                    label = f'{int(time//60)}m {int(time%60)}s'
                else:
                    label = f'{int(time//3600)}h {int((time%3600)//60)}m'
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.05,
                       label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add speedup annotation
        if matlab_time > 0 and python_time > 0:
            speedup = matlab_time / python_time
            y_pos = max(times) * 0.7
            ax.text(0.5, y_pos, f'{speedup:.1f}x',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFD166',
                            edgecolor='black', linewidth=1.5, alpha=0.9))


def plot_vertex_matching_subplot(ax, comparison):
    """Helper: plot vertex matching stats in a subplot."""
    if 'vertices' not in comparison:
        return
    
    verts = comparison['vertices']
    matched = verts.get('matched_vertices', 0)
    unmatched_matlab = verts.get('unmatched_matlab', 0)
    unmatched_python = verts.get('unmatched_python', 0)
    
    if matched + unmatched_matlab + unmatched_python == 0:
        return
    
    labels = ['Matched', 'MATLAB only', 'Python only']
    sizes = [matched, unmatched_matlab, unmatched_python]
    colors = ['#4CAF50', '#2E86AB', '#A23B72']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Vertex Matching')


def plot_position_errors_subplot(ax, comparison):
    """Helper: plot position error statistics."""
    if 'vertices' not in comparison or comparison['vertices'].get('position_rmse') is None:
        ax.text(0.5, 0.5, 'No position\nerror data', ha='center', va='center')
        ax.axis('off')
        return
    
    verts = comparison['vertices']
    rmse = verts.get('position_rmse', 0)
    mean_dist = verts.get('position_mean_distance', 0)
    median_dist = verts.get('position_median_distance', 0)
    p95 = verts.get('position_95th_percentile', 0)
    
    metrics = ['RMSE', 'Mean', 'Median', '95th %ile']
    values = [rmse, mean_dist, median_dist, p95]
    
    ax.barh(metrics, values, color='#FF6B6B', alpha=0.7)
    ax.set_xlabel('Distance (voxels)')
    ax.set_title('Position Errors')
    ax.grid(axis='x', alpha=0.3)


def plot_radius_stats_subplot(ax, comparison):
    """Helper: plot radius statistics."""
    if 'vertices' not in comparison or not comparison['vertices'].get('radius_stats'):
        ax.text(0.5, 0.5, 'No radius\nstats data', ha='center', va='center')
        ax.axis('off')
        return
    
    stats = comparison['vertices']['radius_stats']
    
    text = f"Radius Comparison\n\n"
    text += f"MATLAB: {stats.get('matlab_mean', 0):.2f} ± {stats.get('matlab_std', 0):.2f} μm\n"
    text += f"Python: {stats.get('python_mean', 0):.2f} ± {stats.get('python_std', 0):.2f} μm\n"
    text += f"RMSE: {stats.get('rmse', 0):.3f} μm"
    
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Radius Statistics')


def plot_summary_metrics_subplot(ax, comparison):
    """Helper: plot summary metrics table."""
    ax.axis('off')
    
    # Collect key metrics
    metrics = []
    
    if 'performance' in comparison and comparison['performance']:
        perf = comparison['performance']
        if 'speedup' in perf:
            metrics.append(['Performance', f"{perf['faster']} is {abs(perf['speedup']):.2f}x faster"])
    
    if 'vertices' in comparison and comparison['vertices']:
        verts = comparison['vertices']
        if verts.get('count_percent_difference') is not None:
            metrics.append(['Vertex Count Diff', f"{verts['count_percent_difference']:.1f}%"])
        if verts.get('position_rmse') is not None:
            metrics.append(['Position RMSE', f"{verts['position_rmse']:.3f} voxels"])
    
    if 'edges' in comparison and comparison['edges']:
        edges = comparison['edges']
        if edges.get('count_percent_difference') is not None:
            metrics.append(['Edge Count Diff', f"{edges['count_percent_difference']:.1f}%"])
    
    if metrics:
        table_data = metrics
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='left', loc='center',
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Summary Metrics', pad=20)


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization plots for MATLAB-Python comparison'
    )
    parser.add_argument(
        '--comparison-report',
        required=True,
        help='Path to comparison_report.json'
    )
    parser.add_argument(
        '--matlab-batch',
        help='Path to MATLAB batch folder (for loading detailed data)'
    )
    parser.add_argument(
        '--python-results',
        help='Path to Python results directory'
    )
    parser.add_argument(
        '--output-dir',
        default='comparison_output/visualizations',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    # Load comparison report
    report_path = Path(args.comparison_report)
    if not report_path.exists():
        print(f"ERROR: Comparison report not found: {report_path}")
        return 1
    
    with open(report_path, 'r') as f:
        comparison = json.load(f)
    
    # Load detailed MATLAB data if available
    matlab_data = None
    if args.matlab_batch:
        try:
            matlab_data = load_matlab_batch_results(args.matlab_batch)
        except Exception as e:
            print(f"Warning: Could not load MATLAB data: {e}")
    
    # Load Python data if available
    python_data = None
    if args.python_results:
        python_results_dir = Path(args.python_results)
        # Try to load from pickle files
        import pickle
        for pkl_file in python_results_dir.glob('*.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    python_data = pickle.load(f)
                break
            except:
                pass
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Comparison Visualizations")
    print("="*60)
    
    set_plot_style()
    
    # Generate plots
    plot_count_comparison(comparison, output_dir / 'count_comparison.png')
    plot_timing_breakdown(comparison, output_dir / 'timing_breakdown.png')
    plot_radius_distributions(comparison, matlab_data, python_data, output_dir / 'radius_distributions.png')
    plot_radius_correlation(comparison, output_dir / 'radius_correlation.png')
    plot_summary_dashboard(comparison, output_dir / 'summary_dashboard.png')
    
    print("\n" + "="*60)
    print("Visualization Complete")
    print("="*60)
    print(f"\nPlots saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
