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

def create_experiment_path(base_dir: Path, label: str = "run") -> Path:
    """Create a hierarchical, timestamped experiment path."""
    now = datetime.now()
    year = now.strftime('%Y')
    month_val = now.strftime('%m')
    month_name = now.strftime('%B')
    month_folder = f"{month_val}-{month_name}"
    day_time = now.strftime('%d_%H%M%S')
    
    # Sanitize label
    safe_label = "".join([c if c.isalnum() or c in ("-", "_") else "-" for c in label])
    
    return base_dir / year / month_folder / f"{day_time}_{safe_label}"

def list_runs(experiment_dir: Path) -> List[Dict[str, Any]]:
    """List all comparison runs in the directory hierarchy."""
    if not experiment_dir.exists():
        return []
    
    runs = []
    # Find all directories that contain a manifestation of a run (e.g., comparison_report.json or results.json)
    # We look for folders that have 'matlab_results' or 'python_results' subdirs
    for report in experiment_dir.rglob('comparison_report.json'):
        runs.append(load_run_info(report.parent))
        
    # Also find standalone python runs if they don't have a report yet
    for results in experiment_dir.rglob('python_results'):
        parent = results.parent
        if not (parent / 'comparison_report.json').exists():
             runs.append(load_run_info(parent))

    return sorted(runs, key=lambda x: x['name'], reverse=True)

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

def get_file_inventory(directory: Path) -> Dict[str, List[Path]]:
    """Get inventory of files organized by type."""
    inventory = {
        'vmv': [],
        'casx': [],
        'csv': [],
        'json': [],
        'mat': [],
        'png': [],
        'txt': [],
        'other': []
    }
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower().lstrip('.')
            if ext in inventory:
                inventory[ext].append(file_path)
            else:
                inventory['other'].append(file_path)
    
    return inventory

def generate_manifest(comparison_dir: Path, output_file: Path = None) -> str:
    """Generate manifest/README for a comparison directory."""
    if output_file is None:
        output_file = comparison_dir / 'MANIFEST.md'
    
    # Get directory name and timestamp
    dir_name = comparison_dir.name
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Load comparison report
    report_file = comparison_dir / 'comparison_report.json'
    report = {}
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
        except:
            pass
    
    # Get file inventory
    inventory = get_file_inventory(comparison_dir)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in comparison_dir.rglob('*') if f.is_file())
    
    # Build manifest content
    lines = []
    lines.append(f"# SLAVV Comparison Run: {dir_name}")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Total Size:** {format_size(total_size)}")
    lines.append("")
    
    # Comparison Summary
    if report:
        lines.append("## Comparison Summary")
        lines.append("")
        
        if 'performance' in report:
            perf = report['performance']
            lines.append("### Performance")
            lines.append(f"- **MATLAB:** {perf.get('matlab_time_seconds', 0):.1f}s")
            lines.append(f"- **Python:** {perf.get('python_time_seconds', 0):.1f}s")
            lines.append(f"- **Speedup:** {perf.get('speedup', 0):.2f}x ({perf.get('faster', 'N/A')} faster)")
            lines.append("")
        
        if 'vertices' in report:
            verts = report['vertices']
            lines.append("### Vertices")
            lines.append(f"- **MATLAB:** {verts.get('matlab_count', 0):,}")
            lines.append(f"- **Python:** {verts.get('python_count', 0):,}")
            lines.append("")
        
        if 'edges' in report:
            edges = report['edges']
            lines.append("### Edges")
            lines.append(f"- **MATLAB:** {edges.get('matlab_count', 0):,}")
            lines.append(f"- **Python:** {edges.get('python_count', 0):,}")
            lines.append("")
    
    # File Inventory
    lines.append("## File Inventory")
    lines.append("")
    
    # 3D Visualization Files
    vmv_files = inventory.get('vmv', [])
    casx_files = inventory.get('casx', [])
    if vmv_files or casx_files:
        lines.append("### 3D Visualization Files")
        lines.append("")
        if vmv_files:
            lines.append("**VMV Files** (VessMorphoVis/Blender):")
            for f in sorted(vmv_files):
                rel_path = f.relative_to(comparison_dir)
                size = format_size(f.stat().st_size)
                lines.append(f"- `{rel_path}` ({size})")
            lines.append("")
        if casx_files:
            lines.append("**CASX Files** (CASX format):")
            for f in sorted(casx_files):
                rel_path = f.relative_to(comparison_dir)
                size = format_size(f.stat().st_size)
                lines.append(f"- `{rel_path}` ({size})")
            lines.append("")
    
    # Data Files
    csv_files = inventory.get('csv', [])
    json_files = inventory.get('json', [])
    mat_files = inventory.get('mat', [])
    if csv_files or json_files or mat_files:
        lines.append("### Data Files")
        lines.append("")
        if csv_files:
            lines.append("**CSV Files:**")
            for f in sorted(csv_files):
                rel_path = f.relative_to(comparison_dir)
                lines.append(f"- `{rel_path}`")
            lines.append("")
        if json_files:
            lines.append("**JSON Files:**")
            for f in sorted(json_files):
                rel_path = f.relative_to(comparison_dir)
                lines.append(f"- `{rel_path}`")
            lines.append("")
        if mat_files:
            lines.append("**MATLAB Files:**")
            for f in sorted(mat_files):
                rel_path = f.relative_to(comparison_dir)
                lines.append(f"- `{rel_path}`")
            lines.append("")
    
    # Visualizations
    png_files = inventory.get('png', [])
    if png_files:
        lines.append("### Visualization Images")
        lines.append("")
        for f in sorted(png_files):
            rel_path = f.relative_to(comparison_dir)
            lines.append(f"- `{rel_path}`")
        lines.append("")
        
    # Write to file
    manifest_content = '\n'.join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(manifest_content)
    
    return manifest_content
