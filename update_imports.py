import os
import re

replacements = [
    ("slavv_python.engine.orchestrator", "slavv_python.engine.orchestrator"),
    ("slavv_python.engine.lifecycle", "slavv_python.engine.lifecycle"),
    ("slavv_python.engine.state.tracker", "slavv_python.engine.state.tracker"),
    ("slavv_python.engine.state.snapshots", "slavv_python.engine.state.snapshots"),
    ("slavv_python.engine.state.resume_policy", "slavv_python.engine.state.resume_policy"),
    ("slavv_python.engine.environment", "slavv_python.engine.environment"),
    ("slavv_python.engine.constants", "slavv_python.engine.constants"),
    ("slavv_python.engine.context", "slavv_python.engine.context"),
    ("slavv_python.engine.state.status", "slavv_python.engine.state.status"),
    ("slavv_python.engine.state.progress", "slavv_python.engine.state.progress"),
    ("slavv_python.processing.image.normalization", "slavv_python.processing.image.normalization"),
    ("slavv_python.processing.image.tiling", "slavv_python.processing.image.tiling"),
    ("slavv_python.processing.stages.energy", "slavv_python.processing.stages.energy"),
    ("slavv_python.processing.stages.vertices", "slavv_python.processing.stages.vertices"),
    ("slavv_python.processing.stages.edges", "slavv_python.processing.stages.edges"),
    ("slavv_python.processing.stages.network", "slavv_python.processing.stages.network"),
    ("slavv_python.engine.graph", "slavv_python.processing.stages.network"),
    ("slavv_python.analytics.curation.automated", "slavv_python.analytics.curation.automated"),
    ("slavv_python.analytics.curation.machine_learning", "slavv_python.analytics.curation.machine_learning"),
    ("slavv_python.analytics.metrics.topology", "slavv_python.analytics.metrics.topology"),
    ("slavv_python.analytics.metrics.intensity", "slavv_python.analytics.metrics.intensity"),
    ("slavv_python.analytics.math", "slavv_python.analytics.math"),
    ("slavv_python.analytics.registration", "slavv_python.analytics.registration"),
    ("slavv_python.storage.loaders.tiff", "slavv_python.storage.loaders.tiff"),
    ("slavv_python.storage.exporters.json_v1", "slavv_python.storage.exporters.json_v1"),
    ("slavv_python.storage.loaders.network", "slavv_python.storage.loaders.network"),
    ("slavv_python.schema.results", "slavv_python.schema.results"),
    ("slavv_python.interface", "slavv_python.interface"),
]

# Sort by length of the old string, descending to replace longer paths first
replacements.sort(key=lambda x: len(x[0]), reverse=True)

def update_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return False
    
    new_content = content
    for old, new in replacements:
        # Match the old path only if it's not followed by an alphanumeric character or underscore
        # This allows it to be followed by a dot (for submodules) or other non-word characters.
        pattern = re.escape(old) + r'(?![a-zA-Z0-9_])'
        new_content = re.sub(pattern, new, new_content)
    
    if new_content != content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Could not write to {file_path}: {e}")
            return False
    return False

dirs = ['slavv_python', 'scripts', 'tests']
count = 0
for d in dirs:
    if not os.path.exists(d):
        continue
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith('.py'):
                if update_file(os.path.join(root, file)):
                    count += 1
print(f"Updated {count} files.")
