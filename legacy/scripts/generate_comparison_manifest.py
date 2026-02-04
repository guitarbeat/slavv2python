#!/usr/bin/env python3
"""
Generate manifest and README files for comparison output directories.

This script creates human-readable documentation for each comparison run,
listing all generated files and providing instructions for viewing results.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


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


def load_comparison_report(directory: Path) -> Dict[str, Any]:
    """Load comparison report if it exists."""
    report_file = directory / 'comparison_report.json'
    if report_file.exists():
        with open(report_file, 'r') as f:
            return json.load(f)
    return {}


def generate_manifest(comparison_dir: Path, output_file: Path = None) -> str:
    """Generate manifest/README for a comparison directory.
    
    Parameters
    ----------
    comparison_dir : Path
        Path to comparison output directory
    output_file : Path, optional
        Path to write manifest (default: comparison_dir/MANIFEST.md)
        
    Returns
    -------
    str
        Generated manifest content
    """
    if output_file is None:
        output_file = comparison_dir / 'MANIFEST.md'
    
    # Get directory name and timestamp
    dir_name = comparison_dir.name
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Load comparison report
    report = load_comparison_report(comparison_dir)
    
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
    
    # Reports
    txt_files = inventory.get('txt', [])
    if txt_files:
        lines.append("### Reports")
        lines.append("")
        for f in sorted(txt_files):
            rel_path = f.relative_to(comparison_dir)
            lines.append(f"- `{rel_path}`")
        lines.append("")
    
    # How to View
    lines.append("## How to View Results")
    lines.append("")
    
    if vmv_files:
        lines.append("### Viewing in Blender with VessMorphoVis")
        lines.append("")
        lines.append("1. Install [Blender](https://www.blender.org/download/)")
        lines.append("2. Install [VessMorphoVis plugin](https://github.com/BlueBrain/VessMorphoVis)")
        lines.append("3. Open Blender")
        lines.append("4. Enable VessMorphoVis in Edit > Preferences > Add-ons")
        lines.append("5. In VessMorphoVis panel, load the VMV file:")
        for f in sorted(vmv_files)[:3]:  # Show first 3
            rel_path = f.relative_to(comparison_dir)
            lines.append(f"   - `{rel_path}`")
        lines.append("6. Adjust visualization settings and render")
        lines.append("")
    
    if csv_files:
        lines.append("### Analyzing Data")
        lines.append("")
        lines.append("**CSV Files** can be opened in:")
        lines.append("- Microsoft Excel")
        lines.append("- Python pandas: `pd.read_csv('network_vertices.csv')`")
        lines.append("- R: `read.csv('network_vertices.csv')`")
        lines.append("")
    
    if png_files:
        lines.append("### Viewing Comparison Plots")
        lines.append("")
        lines.append("PNG files in `visualizations/` directory can be opened with any image viewer.")
        lines.append("")
    
    # Quick Commands
    lines.append("## Quick Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("# View summary")
    lines.append("cat summary.txt")
    lines.append("")
    lines.append("# List all VMV files")
    lines.append("find . -name '*.vmv'")
    lines.append("")
    lines.append("# List all CASX files")
    lines.append("find . -name '*.casx'")
    lines.append("```")
    lines.append("")
    
    # Directory Structure
    lines.append("## Directory Structure")
    lines.append("")
    lines.append("```")
    
    def print_tree(path: Path, prefix: str = "", is_last: bool = True, max_depth: int = 3, current_depth: int = 0):
        """Recursively print directory tree."""
        if current_depth >= max_depth:
            return []
        
        tree_lines = []
        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{path.name}")
        
        if path.is_dir():
            children = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            children = children[:10]  # Limit to first 10 items
            
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                extension = "    " if is_last else "│   "
                child_lines = print_tree(child, prefix + extension, is_last_child, max_depth, current_depth + 1)
                tree_lines.extend(child_lines)
        
        return tree_lines
    
    tree_lines = [comparison_dir.name]
    children = sorted(comparison_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    for i, child in enumerate(children):
        is_last = (i == len(children) - 1)
        tree_lines.extend(print_tree(child, "", is_last, max_depth=2))
    
    lines.extend(tree_lines)
    lines.append("```")
    lines.append("")
    
    # Write to file
    manifest_content = '\n'.join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(manifest_content)
    
    return manifest_content


def main():
    parser = argparse.ArgumentParser(description='Generate comparison manifest')
    parser.add_argument('comparison_dir', type=str, help='Comparison output directory')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: MANIFEST.md)')
    
    args = parser.parse_args()
    
    comparison_dir = Path(args.comparison_dir)
    output_file = Path(args.output) if args.output else None
    
    if not comparison_dir.exists():
        print(f"Error: Directory not found: {comparison_dir}")
        return 1
    
    print(f"Generating manifest for: {comparison_dir}")
    manifest_content = generate_manifest(comparison_dir, output_file)
    
    output_path = output_file or comparison_dir / 'MANIFEST.md'
    print(f"Manifest written to: {output_path}")
    print(f"Length: {len(manifest_content)} characters")
    
    return 0


if __name__ == '__main__':
    exit(main())
