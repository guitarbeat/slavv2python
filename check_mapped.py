import os

mapped_content = open("MATLAB_MAPPING.md", "r", encoding="utf-8").read()
matlab_dir = "external/Vectorization-Public/source/"
files = sorted([f for f in os.listdir(matlab_dir) if f.endswith(".m")])
missing = [f for f in files if f not in mapped_content]

with open("MATLAB_MAPPING.md", "a", encoding="utf-8") as f:
    f.write("\n\n## 12. Unmapped / Obsolete Scripts\n\n")
    f.write("These files were present in the MATLAB source but never mapped. They are considered obsolete or user-specific scripts.\n\n")
    f.write("| MATLAB File | Python File | Status | Notes |\n")
    f.write("|---|---|---|---|\n")
    for m in missing:
        f.write(f"| `{m}` | — | 🚫 | Unmapped/Obsolete |\n")

print(f"Appended {len(missing)} unmapped files to MATLAB_MAPPING.md!")
