#!/usr/bin/env python3
"""
Cleanup and manage comparison output directories.

This script helps manage disk space by removing checkpoints,
archiving old runs, and providing disk usage analysis.
"""

import argparse
import shutil
import sys
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


def get_pkl_files(path: Path) -> List[Path]:
    """Find all .pkl files in directory."""
    return list(path.rglob('*.pkl'))


def analyze_comparisons(root: Path = Path('.')) -> List[Dict[str, Any]]:
    """Analyze all comparison_output* directories."""
    comparison_dirs = sorted(root.glob('comparison_output*'))
    
    results = []
    for dir_path in comparison_dirs:
        if not dir_path.is_dir():
            continue
            
        total_size = get_directory_size(dir_path)
        pkl_files = get_pkl_files(dir_path)
        pkl_size = sum(f.stat().st_size for f in pkl_files)
        
        # Try to find comparison report
        report_file = dir_path / 'comparison_report.json'
        has_report = report_file.exists()
        
        # Check for matlab/python results
        has_matlab = (dir_path / 'matlab_results').exists()
        has_python = (dir_path / 'python_results').exists()
        
        results.append({
            'path': dir_path,
            'name': dir_path.name,
            'total_size': total_size,
            'pkl_size': pkl_size,
            'pkl_count': len(pkl_files),
            'pkl_files': pkl_files,
            'has_report': has_report,
            'has_matlab': has_matlab,
            'has_python': has_python,
        })
    
    return results


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def print_analysis(results: List[Dict[str, Any]]):
    """Print disk usage analysis."""
    print("\n" + "="*70)
    print("Comparison Directory Analysis")
    print("="*70)
    
    if not results:
        print("No comparison_output* directories found.")
        return
    
    total_size = sum(r['total_size'] for r in results)
    total_pkl = sum(r['pkl_size'] for r in results)
    
    for result in results:
        print(f"\n{result['name']}")
        print(f"  Total size: {format_size(result['total_size'])}")
        print(f"  Checkpoints: {format_size(result['pkl_size'])} ({result['pkl_count']} .pkl files)")
        print(f"  Content: ", end="")
        content_parts = []
        if result['has_matlab']:
            content_parts.append("MATLAB results")
        if result['has_python']:
            content_parts.append("Python results")
        if result['has_report']:
            content_parts.append("comparison report")
        print(", ".join(content_parts) if content_parts else "empty")
        
        if result['pkl_size'] > 0:
            savings = (result['pkl_size'] / result['total_size']) * 100
            print(f"  Can save: {format_size(result['pkl_size'])} ({savings:.1f}%)")
    
    print(f"\n{'-'*70}")
    print(f"Total across {len(results)} directories: {format_size(total_size)}")
    print(f"Total checkpoint size: {format_size(total_pkl)}")
    if total_pkl > 0:
        savings_pct = (total_pkl / total_size) * 100
        print(f"Potential savings: {format_size(total_pkl)} ({savings_pct:.1f}%)")
    print("="*70 + "\n")


def remove_checkpoints(results: List[Dict[str, Any]], dry_run: bool = True):
    """Remove all checkpoint .pkl files."""
    total_files = sum(r['pkl_count'] for r in results)
    total_size = sum(r['pkl_size'] for r in results)
    
    if total_files == 0:
        print("No checkpoint files found to remove.")
        return
    
    print(f"\n{'DRY RUN: Would remove' if dry_run else 'Removing'} {total_files} checkpoint files ({format_size(total_size)})")
    
    removed_count = 0
    removed_size = 0
    
    for result in results:
        if result['pkl_count'] == 0:
            continue
            
        print(f"\n{result['name']}:")
        for pkl_file in result['pkl_files']:
            file_size = pkl_file.stat().st_size
            rel_path = pkl_file.relative_to(result['path'])
            
            if dry_run:
                print(f"  Would remove: {rel_path} ({format_size(file_size)})")
            else:
                try:
                    pkl_file.unlink()
                    print(f"  Removed: {rel_path} ({format_size(file_size)})")
                    removed_count += 1
                    removed_size += file_size
                except Exception as e:
                    print(f"  Error removing {rel_path}: {e}")
    
    if not dry_run:
        print(f"\nRemoved {removed_count} files, freed {format_size(removed_size)}")
    else:
        print(f"\nRun with --confirm to actually remove files.")


def archive_directories(results: List[Dict[str, Any]], archive_root: Path, dry_run: bool = True):
    """Archive comparison directories to a separate location."""
    if not results:
        print("No directories to archive.")
        return
    
    archive_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'DRY RUN: Would archive' if dry_run else 'Archiving'} {len(results)} directories to {archive_root}")
    
    for result in results:
        dest = archive_root / result['name']
        
        if dry_run:
            print(f"  Would move: {result['name']} -> {dest}")
        else:
            try:
                if dest.exists():
                    print(f"  Skipping {result['name']} (already exists in archive)")
                else:
                    shutil.move(str(result['path']), str(dest))
                    print(f"  Archived: {result['name']} -> {dest}")
            except Exception as e:
                print(f"  Error archiving {result['name']}: {e}")
    
    if dry_run:
        print(f"\nRun with --confirm to actually archive directories.")


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup and manage comparison output directories'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze disk usage (default action)'
    )
    parser.add_argument(
        '--remove-checkpoints',
        action='store_true',
        help='Remove all checkpoint .pkl files'
    )
    parser.add_argument(
        '--archive-old',
        metavar='PATTERN',
        help='Archive directories matching pattern (e.g., "comparison_output*")'
    )
    parser.add_argument(
        '--archive-dir',
        default='comparisons/archive',
        help='Archive destination directory (default: comparisons/archive)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Actually perform actions (without this, only dry-run)'
    )
    
    args = parser.parse_args()
    
    # Analyze is default action
    if not any([args.analyze, args.remove_checkpoints, args.archive_old]):
        args.analyze = True
    
    # Get comparison directories
    root = Path('.')
    results = analyze_comparisons(root)
    
    if args.analyze:
        print_analysis(results)
    
    if args.remove_checkpoints:
        remove_checkpoints(results, dry_run=not args.confirm)
    
    if args.archive_old:
        archive_root = Path(args.archive_dir)
        archive_directories(results, archive_root, dry_run=not args.confirm)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
