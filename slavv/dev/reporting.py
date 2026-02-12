"""
Reporting tools for SLAVV comparison.

This module generates summary files and reports for comparison runs.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def generate_summary(run_dir: Path, output_file: Path):
    """Generate summary.txt for a comparison run."""
    
    # Load comparison report if exists
    report_path = run_dir / 'comparison_report.json'
    report = {}
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
    
    # Extract metadata
    run_name = run_dir.name
    
    # Try to infer date from directory name or file modification times
    date_str = "Unknown"
    if run_name.startswith('2026'):
        # Parse YYYYMMDD from name
        try:
            date_part = run_name.split('_')[0]
            date_obj = datetime.strptime(date_part, '%Y%m%d')
            date_str = date_obj.strftime('%Y-%m-%d')
        except:
            pass
    
    # Build summary
    lines = []
    lines.append("="*70)
    lines.append("SLAVV Comparison Summary")
    lines.append("="*70)
    lines.append(f"Run: {run_name}")
    lines.append(f"Date: {date_str}")
    lines.append("")
    
    # Performance section
    if report and 'performance' in report:
        lines.append("Performance")
        lines.append("-" * 70)
        perf = report['performance']
        
        if report.get('matlab', {}).get('elapsed_time'):
            matlab_time = report['matlab']['elapsed_time']
            lines.append(f"MATLAB:  {format_time(matlab_time):>12}")
        
        if report.get('python', {}).get('elapsed_time'):
            python_time = report['python']['elapsed_time']
            lines.append(f"Python:  {format_time(python_time):>12}")
        
        if 'speedup' in perf:
            speedup = perf['speedup']
            faster = perf.get('faster', 'Unknown')
            lines.append(f"Speedup: {speedup:>12.2f}x ({faster} faster)")
        lines.append("")
    
    # Results section
    lines.append("Results")
    lines.append("-" * 70)
    
    if report:
        # Header
        lines.append(f"{'Component':<15} {'MATLAB':>12} {'Python':>12} {'Difference':>15}")
        lines.append("-" * 70)
        
        # Vertices
        matlab_verts = report.get('matlab', {}).get('vertices_count', 0)
        python_verts = report.get('python', {}).get('vertices_count', 0)
        diff_verts = python_verts - matlab_verts
        lines.append(f"{'Vertices':<15} {matlab_verts:>12,} {python_verts:>12,} {diff_verts:>+15,}")
        
        # Edges
        matlab_edges = report.get('matlab', {}).get('edges_count', 0)
        python_edges = report.get('python', {}).get('edges_count', 0)
        diff_edges = python_edges - matlab_edges
        lines.append(f"{'Edges':<15} {matlab_edges:>12,} {python_edges:>12,} {diff_edges:>+15,}")
        
        # Strands
        matlab_strands = report.get('matlab', {}).get('strand_count', 0)
        python_strands = report.get('python', {}).get('network_strands_count', 0)
        diff_strands = python_strands - matlab_strands
        lines.append(f"{'Strands':<15} {matlab_strands:>12,} {python_strands:>12,} {diff_strands:>+15,}")
    else:
        lines.append("No comparison report found.")
    
    lines.append("")
    
    # Status/notes
    matlab_dir = run_dir / 'matlab_results'
    python_dir = run_dir / 'python_results'
    
    has_matlab = matlab_dir.exists() and any(matlab_dir.iterdir())
    has_python = python_dir.exists() and any(python_dir.iterdir())
    has_plots = (run_dir / 'visualizations').exists()
    
    lines.append("Status")
    lines.append("-" * 70)
    if has_matlab:
        lines.append("- MATLAB results: Present")
    if has_python:
        lines.append("- Python results: Present")
    if has_plots:
        lines.append("- Visualizations: Present")
    
    if report:
        matlab_verts = report.get('matlab', {}).get('vertices_count', 0)
        python_verts = report.get('python', {}).get('vertices_count', 0)
        
        if matlab_verts == 0 and has_matlab:
            lines.append("- WARNING: MATLAB produced 0 vertices (possible config issue)")
        if python_verts == 0 and has_python:
            lines.append("- WARNING: Python produced 0 vertices (possible config issue)")
        if matlab_verts > 0 and python_verts > 0:
            lines.append("- SUCCESS: Both implementations produced vertices")
    
    lines.append("")
    lines.append("="*70)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated summary: {output_file}")
