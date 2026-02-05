#!/usr/bin/env python3
"""Fix notebook issues found during review"""
import json
from pathlib import Path

def fix_notebooks():
    scripts_dir = Path(__file__).parent / "scripts"
    
    # Fix 5_Tutorial.ipynb - wrong data path
    tutorial_path = scripts_dir / "5_Tutorial.ipynb"
    if tutorial_path.exists():
        print(f"Fixing {tutorial_path.name}...")
        with open(tutorial_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and 'source' in cell:
                new_source = []
                for line in cell['source']:
                    # Fix the input_path reference
                    if '../data/slavv_test_volume.tif' in line:
                        new_line = line.replace(
                            '../data/slavv_test_volume.tif',
                            '../tests/data/slavv_test_volume.tif'
                        )
                        # Also fix variable name to be consistent
                        new_line = new_line.replace('input_path =', 'input_file =')
                        new_source.append(new_line)
                        print(f"  ✓ Fixed data path")
                    elif 'input_path' in line and 'input_file' not in line:
                        # Replace remaining input_path references with input_file
                        new_source.append(line.replace('input_path', 'input_file'))
                    else:
                        new_source.append(line)
                cell['source'] = new_source
        
        # Write back
        with open(tutorial_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=4, ensure_ascii=False)
        print(f"  ✓ Saved {tutorial_path.name}")
    
    # Clear outputs from 1_Run_Comparison.ipynb (still has old error outputs)
    comparison_path = scripts_dir / "1_Run_Comparison.ipynb"
    if comparison_path.exists():
        print(f"Clearing outputs from {comparison_path.name}...")
        with open(comparison_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=4, ensure_ascii=False)
        print(f"  ✓ Cleared outputs from {comparison_path.name}")

if __name__ == '__main__':
    fix_notebooks()
    print("\n✅ All notebooks fixed!")
