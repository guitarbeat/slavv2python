import os
import re

MAPPING = [
    # Most specific first
    (r'from slavv_python\.schema import \(\s*([A-Za-z]+Surface|RunCounts)', r'from .models import (\1'),
    (r'from slavv_python\.schema import ([A-Za-z]+Surface|RunCounts)', r'from .models import \1'),
    (r'from slavv_python\.utils import \(\s*(entity_id_from_path|now_iso|resolve_python_commit|string_or_none|write_json_with_hash|fingerprint_file)', r'from .utils import (\1'),
    (r'from slavv_python\.utils import (entity_id_from_path|now_iso|resolve_python_commit|string_or_none|write_json_with_hash|fingerprint_file)', r'from .utils import \1'),
    (r'slavv_python\.core\.pipeline', 'slavv_python.engine'),
    (r'slavv_python\.core', 'slavv_python.engine'), # Fallback
    (r'slavv_python\.runtime\.run_state', 'slavv_python.engine.state'),
    (r'slavv_python\.runtime\.io', 'slavv_python.engine.state'),
    (r'slavv_python\.runtime', 'slavv_python.engine.state'),
    (r'slavv_python\.io\.tiff', 'slavv_python.storage'),
    (r'slavv_python\.io', 'slavv_python.storage'),
    (r'slavv_python\.analysis', 'slavv_python.analytics'),
]

def update_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False
    
    new_content = content
    for pattern, replacement in MAPPING:
        new_content = re.sub(pattern, replacement, new_content)
    
    if new_content != content:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(new_content)
        return True
    return False

count = 0
for root, dirs, files in os.walk('slavv_python/analytics/parity'):
    for f in files:
        if f.endswith('.py'):
            if update_file(os.path.join(root, f)):
                count += 1
print(f"Fixed {count} files in analytics/parity.")
