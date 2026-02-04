"""
Management utilities for SLAVV comparison data.
"""

import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


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
                info['speedup'] = report.get('performance', {}).get('speedup', 0)
        except:
            pass
    
    return info

def list_runs(comparisons_dir: Path) -> List[Dict[str, Any]]:
    """List all comparison runs in the directory."""
    if not comparisons_dir.exists():
        return []
    
    runs = []
    # Look for both dated folders and any folder starting with 'comparison_output'
    candidates = sorted(list(comparisons_dir.glob('2026*')) + list(comparisons_dir.glob('comparison_output*')))
    
    for run_dir in candidates:
        if run_dir.is_dir():
            runs.append(load_run_info(run_dir))
    return runs

def analyze_checkpoints(comparisons_dir: Path) -> List[Dict[str, Any]]:
    """Analyze checkpoint files usage."""
    runs = list_runs(comparisons_dir)
    results = []
    
    for run in runs:
        path = run['path']
        pkl_files = list(path.rglob('*.pkl'))
        pkl_size = sum(f.stat().st_size for f in pkl_files)
        
        results.append({
            'name': run['name'],
            'path': path,
            'total_size': run['size'],
            'pkl_size': pkl_size,
            'pkl_count': len(pkl_files),
            'pkl_files': pkl_files
        })
    return results

def cleanup_checkpoints(run_data: Dict[str, Any]) -> int:
    """Remove checkpoints from a specific run. Returns bytes freed."""
    freed = 0
    for pkl_file in run_data['pkl_files']:
        try:
            size = pkl_file.stat().st_size
            pkl_file.unlink()
            freed += size
        except Exception:
            pass
    return freed
