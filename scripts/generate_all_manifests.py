#!/usr/bin/env python3
"""
Generate manifests for all existing comparison directories.

This utility script will scan the comparisons/ directory and generate
MANIFEST.md files for any comparison runs that don't have one yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_comparison_manifest import generate_manifest


def main():
    # Find comparisons directory
    project_root = Path(__file__).parent.parent
    comparisons_dir = project_root / 'comparisons'
    
    if not comparisons_dir.exists():
        print(f"Comparisons directory not found: {comparisons_dir}")
        return 1
    
    print(f"Scanning: {comparisons_dir}")
    print("="*60)
    
    # Find all comparison directories
    comparison_dirs = [
        d for d in comparisons_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    if not comparison_dirs:
        print("No comparison directories found.")
        return 0
    
    print(f"Found {len(comparison_dirs)} comparison director{'y' if len(comparison_dirs) == 1 else 'ies'}:")
    for d in sorted(comparison_dirs):
        print(f"  - {d.name}")
    print()
    
    # Generate manifests
    generated = 0
    skipped = 0
    
    for comp_dir in sorted(comparison_dirs):
        manifest_file = comp_dir / 'MANIFEST.md'
        
        if manifest_file.exists():
            print(f"[SKIP] {comp_dir.name} (manifest already exists)")
            skipped += 1
            continue
        
        try:
            print(f"[GEN] Generating manifest for {comp_dir.name}...")
            generate_manifest(comp_dir, manifest_file)
            print(f"  [OK] Created: {manifest_file}")
            generated += 1
        except Exception as e:
            print(f"  [FAIL] Failed: {e}")
    
    print()
    print("="*60)
    print(f"Summary: {generated} generated, {skipped} skipped")
    
    return 0


if __name__ == '__main__':
    exit(main())
