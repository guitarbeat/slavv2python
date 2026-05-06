"""CLI tool to audit the SLAVV workspace structure."""

import sys
from pathlib import Path

# Add source to path if running from dev/scripts/maintenance
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from source.runtime.workspace import WorkspaceAuditor
except ImportError:
    print("Error: Could not import source.runtime.workspace. Are you running from the repo root?")
    sys.exit(1)


def main():
    """Run the workspace audit and report results."""
    auditor = WorkspaceAuditor(repo_root)
    results = auditor.run_full_audit()

    any_violations = False

    print(f"--- SLAVV Workspace Audit: {repo_root} ---")

    for section, violations in results.items():
        if violations:
            any_violations = True
            print(f"\n[!] {section.upper()} Violations:")
            for v in violations:
                print(f"  - {v}")
        else:
            print(f"\n[OK] {section.upper()} is clean.")

    if any_violations:
        print("\nAudit failed. Please clean up the reported items.")
        sys.exit(1)
    else:
        print("\nAudit passed! Repository structure is canonical.")
        sys.exit(0)


if __name__ == "__main__":
    main()
