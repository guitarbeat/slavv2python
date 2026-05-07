
import os
import shutil
from pathlib import Path
import re

ROOT = Path(r"D:\2P_Data\Aaron\slavv2python")
PACKAGE = ROOT / "slavv_python"

INTERNAL_DIRS = [
    PACKAGE / "analysis" / "geometry_internal",
    PACKAGE / "core" / "edges_internal",
    PACKAGE / "core" / "edge_candidates_internal",
    PACKAGE / "core" / "energy_internal",
    PACKAGE / "core" / "vertices_internal",
    PACKAGE / "runtime" / "run_tracking",
]

# Mapping of old module paths to new ones for import replacement
# e.g. "slavv_python.core.edge_cleanup" -> "slavv_python.core.edge_cleanup"
import_map = {}

def fix_imports(content, file_path):
    # This is a bit tricky. We need to handle both absolute and relative imports.
    new_content = content
    for old, new in import_map.items():
        new_content = new_content.replace(old, new)
    
    # Also handle relative imports within the moved files
    # If a file was in core/edges_internal and imported from .edge_cleanup
    # and now it's in core, it should import from .edge_cleanup (unchanged)
    # BUT if it imported from ..edge_candidates, it should now import from .edge_candidates
    
    if "_internal" in str(file_path) or "run_tracking" in str(file_path):
        # We are currently in the internal dir (before move)
        # from ..something -> from .something
        new_content = re.sub(r"from \.\.([a-zA-Z_])", r"from .\1", new_content)
        new_content = re.sub(r"import \.\.([a-zA-Z_])", r"import .\1", new_content)
        
    return new_content

# 1. Build the import map and move files
for internal_dir in INTERNAL_DIRS:
    if not internal_dir.exists():
        continue
    
    parent_dir = internal_dir.parent
    internal_name = internal_dir.name
    
    for item in internal_dir.iterdir():
        if item.is_dir() and item.name == "__pycache__":
            shutil.rmtree(item)
            continue
        if item.is_file() and item.suffix == ".py":
            if item.name == "__init__.py":
                # Check if we should merge __init__ logic
                continue
            
            dest_name = item.name
            # Handle collisions
            if (parent_dir / dest_name).exists():
                # If parent is a facade, we will overwrite it later
                # For now, move it to a temp name
                dest_name = f"{internal_name}_{item.name}"
            
            dest_path = parent_dir / dest_name
            
            # Record for import map
            old_mod = str(internal_dir.relative_to(ROOT)).replace(os.sep, ".") + "." + item.stem
            new_mod = str(parent_dir.relative_to(ROOT)).replace(os.sep, ".") + "." + Path(dest_name).stem
            import_map[old_mod] = new_mod
            
            # Also record the parent-folder style access if it was used
            # e.g. from .edges_internal import edge_cleanup
            old_parent_mod = str(internal_dir.relative_to(ROOT)).replace(os.sep, ".")
            new_parent_mod = str(parent_dir.relative_to(ROOT)).replace(os.sep, ".")
            # This one is trickier as it depends on context, but let's try
            
            shutil.move(item, dest_path)

# 2. Update all files in the project
for root, dirs, files in os.walk(ROOT):
    if any(s in root for s in [".git", ".mypy_cache", ".pytest_cache", "ui", "external", "workspace"]):
        continue
    for file in files:
        if file.endswith(".py"):
            file_path = Path(root) / file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = fix_imports(content, file_path)
            
            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

# 3. Clean up empty internal dirs and facades
# We need to manually check if facades can be removed
# Facades to check: 
# slavv_python/analysis/geometry.py
# slavv_python/core/edge_candidates.py
# slavv_python/core/energy.py
# slavv_python/core/vertices.py
# slavv_python/core/edges.py
# slavv_python/core/edge_selection.py

facades = [
    PACKAGE / "analysis" / "geometry.py",
    PACKAGE / "core" / "edge_candidates.py",
    PACKAGE / "core" / "energy.py",
    PACKAGE / "core" / "vertices.py",
    PACKAGE / "core" / "edges.py",
    PACKAGE / "core" / "edge_selection.py",
]

for facade in facades:
    if facade.exists():
        # Check if we moved a file with the same name (e.g. core/edge_selection.py)
        # Our script moved internal/edge_selection.py to parent/edges_internal_edge_selection.py
        internal_name = facade.parent / f"{facade.parent.name}_internal_{facade.name}"
        # wait, the naming logic above was f"{internal_name}_{item.name}"
        # e.g. edges_internal_edge_selection.py
        
        # Let's find if there's a renamed version
        candidates = list(facade.parent.glob(f"*_internal_{facade.name}"))
        if candidates:
            # We have a collision. The new one has the logic.
            # Delete facade and rename the new one to the original name.
            facade.unlink()
            shutil.move(candidates[0], facade)
            print(f"Replaced facade {facade.name} with internal logic.")
        else:
            # No collision, check if it's still just a facade for things that moved
            # Many facades might still be useful as aggregate __init__ style files, 
            # but we want to flatten.
            pass

# Cleanup empty internal dirs
for internal_dir in INTERNAL_DIRS:
    if internal_dir.exists() and not os.listdir(internal_dir):
        internal_dir.rmdir()
    elif internal_dir.exists():
        print(f"Dir not empty: {internal_dir}")

print("Flattening complete.")
