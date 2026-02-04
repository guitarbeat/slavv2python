#!/usr/bin/env python3
"""
List and inspect comparison runs.

This script provides an easy way to view all past comparison runs,
their results, and disk usage.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


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
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except PermissionError:
        pass
    return total


def load_run_info(run_dir: Path) -> Dict[str, Any]:
    """Load information about a comparison run."""
    info = {
        'name': run_dir.name,
        'path': run_dir,
        'size': get_directory_size(run_dir),
        'has_matlab': (run_dir / 'matlab_results').exists(),
        'has_python': (run_dir / 'python_results').exists(),
        'has_report': (run_dir / 'comparison_report.json').exists(),
        'has_plots': (run_dir / 'visualizations').exists(),
        'has_summary': (run_dir / 'summary.txt').exists(),
    }
    
    # Load comparison report if exists
    if info['has_report']:
        try:
            with open(run_dir / 'comparison_report.json', 'r') as f:
                report = json.load(f)
                info['matlab_time'] = report.get('matlab', {}).get('elapsed_time', 0)
                info['python_time'] = report.get('python', {}).get('elapsed_time', 0)
                info['matlab_vertices'] = report.get('matlab', {}).get('vertices_count', 0)
                info['python_vertices'] = report.get('python', {}).get('vertices_count', 0)
                info['matlab_edges'] = report.get('matlab', {}).get('edges_count', 0)
                info['python_edges'] = report.get('python', {}).get('edges_count', 0)
                info['speedup'] = report.get('performance', {}).get('speedup', 0)
        except:
            pass
    
    return info


def list_runs(comparisons_dir: Path = Path('comparisons')):
    """List all comparison runs."""
    if not comparisons_dir.exists():
        print(f"No {comparisons_dir} directory found.")
        return []
    
    # Find all run directories (starting with 2026)
    runs = []
    for run_dir in sorted(comparisons_dir.glob('2026*')):
        if run_dir.is_dir():
            runs.append(load_run_info(run_dir))
    
    return runs


def print_run_list(runs: List[Dict[str, Any]]):
    """Print formatted list of comparison runs."""
    if not runs:
        print("No comparison runs found.")
        return
    
    print("\n" + "="*80)
    print(" "*30 + "COMPARISON RUNS")
    print("="*80)
    
    total_size = sum(r['size'] for r in runs)
    
    for run in runs:
        print(f"\n{run['name']}")
        print("-" * 80)
        print(f"  Size: {format_size(run['size'])}")
        
        # Content
        content = []
        if run['has_matlab']:
            content.append("MATLAB")
        if run['has_python']:
            content.append("Python")
        if run['has_report']:
            content.append("Report")
        if run['has_plots']:
            content.append("Plots")
        if run['has_summary']:
            content.append("Summary")
        print(f"  Content: {', '.join(content)}")
        
        # Results if available
        if run.get('matlab_time') or run.get('python_time'):
            print(f"  Performance:")
            if run.get('matlab_time'):
                print(f"    MATLAB: {format_time(run['matlab_time'])}")
            if run.get('python_time'):
                print(f"    Python: {format_time(run['python_time'])}")
            if run.get('speedup') and run['speedup'] > 0:
                faster = "Python" if run['python_time'] < run['matlab_time'] else "MATLAB"
                print(f"    Speedup: {run['speedup']:.1f}x ({faster} faster)")
        
        if run.get('matlab_vertices') is not None or run.get('python_vertices') is not None:
            print(f"  Results:")
            if run.get('matlab_vertices') is not None:
                print(f"    MATLAB: {run['matlab_vertices']:,} vertices, {run.get('matlab_edges', 0):,} edges")
            if run.get('python_vertices') is not None:
                print(f"    Python: {run['python_vertices']:,} vertices, {run.get('python_edges', 0):,} edges")
    
    print("\n" + "-"*80)
    print(f"Total: {len(runs)} runs, {format_size(total_size)}")
    print("="*80 + "\n")


def show_run_details(run_name: str, comparisons_dir: Path = Path('comparisons')):
    """Show detailed information for a specific run."""
    run_dir = comparisons_dir / run_name
    
    if not run_dir.exists():
        print(f"ERROR: Run not found: {run_name}")
        return
    
    # Check for summary.txt
    summary_file = run_dir / 'summary.txt'
    if summary_file.exists():
        print("\n" + "="*70)
        with open(summary_file, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        # Fallback to manual summary
        info = load_run_info(run_dir)
        print(f"\n{info['name']}")
        print(f"Size: {format_size(info['size'])}")
        print(f"Path: {info['path']}")


def main():
    parser = argparse.ArgumentParser(
        description='List and inspect comparison runs'
    )
    parser.add_argument(
        '--show',
        metavar='RUN_NAME',
        help='Show detailed info for specific run'
    )
    parser.add_argument(
        '--dir',
        default='comparisons',
        help='Comparisons directory (default: comparisons)'
    )
    
    args = parser.parse_args()
    
    comparisons_dir = Path(args.dir)
    
    if args.show:
        show_run_details(args.show, comparisons_dir)
    else:
        runs = list_runs(comparisons_dir)
        print_run_list(runs)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
