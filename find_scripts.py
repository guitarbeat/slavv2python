import os
import glob

def find_scripts():
    matlab_files = glob.glob('external/Vectorization-Public/**/*.m', recursive=True) + glob.glob('workspace/**/*.m', recursive=True)
    scripts = []
    
    for f in matlab_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                
            is_function = False
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                if line.startswith('function ') or line.startswith('function['):
                    is_function = True
                break
                
            if not is_function:
                scripts.append(f)
        except Exception as e:
            try:
                # Fallback to latin-1
                with open(f, 'r', encoding='latin-1') as file:
                    content = file.read()
                is_function = False
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue
                    if line.startswith('function ') or line.startswith('function['):
                        is_function = True
                    break
                if not is_function:
                    scripts.append(f)
            except Exception as e2:
                print(f"Error reading {f}: {e2}")

    print("--- SCRIPTS FOUND ---")
    for s in scripts:
        print(s)

if __name__ == '__main__':
    find_scripts()
