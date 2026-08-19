#!/usr/bin/env python3
"""
Comprehensive Documentation Integrity Verification Suite for SLAVV.

Performs multi-tier automated validation:
  Tier 1: Feature Coverage & Authority Invariants
          - ONE TRUTH exclusivity in EXACT_PROOF_FINDINGS.md (no duplicates)
          - Deprecation/archive warning banner presence in historical files
          - Phase 2 Performance & Translation Paper publication wiring
          - Operator commands & task dashboard scope boundaries
  Tier 2: Boundary & Corner Cases
          - Absolute file:/// URIs & Windows drive path detection
          - Malformed or empty markdown link detection
          - Code block isolation (ignoring links inside code fences)
          - Anchor normalization and resolution
  Tier 3: Cross-Document Consistency
          - Authority map alignment (docs/README.md, AGENTS.md)
          - Domain terminology and milestone synchronization
  Tier 4: Real-World Link Resolution
          - Exhaustive relative path resolution across all markdown files

Exit Code: 0 on clean pass, 1 if any verification errors exist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Known template placeholder links to exclude from broken link errors
TEMPLATE_PLACEHOLDER_SUBSTRINGS = [
    "path/to/doc1.md",
    "path/to/doc2.md",
    "path#anchor",
    "<canonical_dest_root>",
    "path/to/doc",
]

# Mandatory deprecation banner targets
REQUIRED_DEPRECATION_BANNER_FILES = [
    "docs/investigations/parity-job-monitoring-spec/design.md",
    "docs/investigations/parity-job-monitoring-spec/tasks.md",
    "docs/investigations/v22-pointer-corruption/INVESTIGATION_ARCHIVE.md",
    "docs/investigations/v22-pointer-corruption/KIRO_SPEC_ARCHIVE.md",
    "docs/investigations/exact-proof-findings-diary/EXACT_PROOF_FINDINGS.HISTORICAL.md",
    "docs/investigations/kiro-matlab-python-parity/design.md",
    "docs/plans/directory-restructure-plan.md",
]

# Mandatory publication and performance links
MANDATORY_PERFORMANCE_DOC = "PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md"
MANDATORY_PAPER_DOC = "MATLAB_PYTHON_TRANSLATION_PAPER.md"


def slugify_heading(heading: str) -> str:
    """Generate GitHub-compatible slug from heading text following GFM rules."""
    # Strip HTML tags
    h = re.sub(r"<[^>]+>", "", heading)
    # Strip markdown images and links: ![alt](url) -> alt, [text](url) -> text
    h = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", h)
    h = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", h)
    # Strip markdown styling (bold, italic, code, strikethrough)
    h = re.sub(r"[*_`~]", "", h)
    h = h.strip().lower()
    # Replace punctuation characters (preserve alphanumeric, whitespace, underscore, and hyphens)
    h = re.sub(r"[^\w\s-]", "", h)
    # Convert whitespace to hyphens (preserving double hyphens resulting from stripped punctuation)
    h = re.sub(r"\s", "-", h)
    return h


def extract_headings(content: str) -> set[str]:
    """Extract all heading anchor slugs and explicit HTML anchors from markdown content."""
    slugs = set()
    in_code_block = False
    for line in content.splitlines():
        trimmed = line.strip()
        if trimmed.startswith(("```", "~~~")):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Check for explicit HTML anchors: <a name="foo">, <a id="foo">, <div id="foo">
        for match in re.finditer(r'<[a-zA-Z0-9_-]+\s+[^>]*(?:id|name)=["\']([^"\']+)["\']', line):
            anchor_val = match.group(1).strip()
            slugs.add(anchor_val)
            slugs.add(slugify_heading(anchor_val))

        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if match:
            text = match.group(1).strip()
            slug = slugify_heading(text)
            if slug:
                slugs.add(slug)
                # Also add normalized slug variants (collapsed hyphens, stripped hyphens)
                slugs.add(re.sub(r"-+", "-", slug))
                slugs.add(slug.strip("-"))
                slugs.add(re.sub(r"-+", "-", slug).strip("-"))
    return slugs


def strip_code_blocks(content: str) -> list[tuple[int, str]]:
    """Return lines of markdown outside of code fences, with 1-based line numbers."""
    lines_out = []
    in_code_block = False
    for i, line in enumerate(content.splitlines(), start=1):
        trimmed = line.strip()
        if trimmed.startswith(("```", "~~~")):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            lines_out.append((i, line))
    return lines_out


def extract_links_from_line(line: str) -> list[tuple[str, str, int]]:
    """Extract (text, target, char_col) markdown links from a single line."""
    links = []
    # Match standard markdown inline links: [text](target) and ![alt](target)
    pattern = re.compile(r"!?\[([^\]]*)\]\(([^)]*)\)")
    for match in pattern.finditer(line):
        text = match.group(1).strip()
        target = match.group(2).strip()
        col = match.start() + 1
        links.append((text, target, col))
    return links


class DocsIntegrityVerifier:
    def __init__(self, root_dir: Path, strict: bool = False, verbose: bool = False):
        self.root = root_dir.resolve()
        self.strict = strict
        self.verbose = verbose

        self.all_md_files: list[Path] = []
        self.file_headings_cache: dict[Path, set[str]] = {}

        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.tier_results: dict[str, Any] = {}

        self.total_links_checked = 0
        self.total_external_links = 0

    def discover_files(self) -> None:
        """Find all relevant markdown files to audit."""
        exclude_dirs = {
            ".git",
            ".agents",
            "workspace",
            "__pycache__",
            ".pytest_cache",
            "venv",
            ".venv",
            "node_modules",
            "build",
            "dist",
            "egg-info",
        }

        md_files = []
        for root_p, dirs, files in os.walk(self.root):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith(".egg-info")]

            rel_root = Path(root_p).relative_to(self.root)
            # Only scan docs/, .claude/, root *.md, and tests/
            parts = rel_root.parts
            if parts and parts[0] not in ("docs", ".claude", "tests"):
                continue

            for f in files:
                if f.endswith(".md"):
                    full_p = Path(root_p) / f
                    md_files.append(full_p)

        self.all_md_files = sorted(md_files)

    def log_error(self, tier: str, check: str, file: Path, line: int | None, message: str) -> None:
        rel_file = str(file.relative_to(self.root)).replace("\\", "/")
        self.errors.append(
            {
                "tier": tier,
                "check": check,
                "file": rel_file,
                "line": line,
                "message": message,
            }
        )

    def log_warning(
        self, tier: str, check: str, file: Path, line: int | None, message: str
    ) -> None:
        rel_file = str(file.relative_to(self.root)).replace("\\", "/")
        self.warnings.append(
            {
                "tier": tier,
                "check": check,
                "file": rel_file,
                "line": line,
                "message": message,
            }
        )

    def run_tier1_authority_and_features(self) -> dict[str, Any]:
        """Tier 1: Feature Coverage & Authority Invariants."""
        checks: list[dict[str, Any]] = []

        # 1.1 ONE TRUTH Exclusivity & No Duplicates
        one_truth_files = []
        one_truth_pattern = re.compile(r"^##\s+ONE TRUTH", re.IGNORECASE)
        canonical_one_truth_rel = "docs/reference/core/EXACT_PROOF_FINDINGS.md"
        canonical_found = False

        for f in self.all_md_files:
            rel = str(f.relative_to(self.root)).replace("\\", "/")
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                self.log_error("tier1", "file_read", f, None, f"Failed to read file: {e}")
                continue

            for line_no, line in enumerate(content.splitlines(), start=1):
                if one_truth_pattern.match(line.strip()):
                    one_truth_files.append((rel, line_no, line.strip()))
                    if rel == canonical_one_truth_rel:
                        canonical_found = True

        duplicates = [item for item in one_truth_files if item[0] != canonical_one_truth_rel]

        if not canonical_found:
            self.log_error(
                "tier1",
                "one_truth_canonical",
                self.root / canonical_one_truth_rel,
                None,
                f"Canonical ONE TRUTH header '## ONE TRUTH' not found in {canonical_one_truth_rel}",
            )
            checks.append({"name": "canonical_one_truth_presence", "passed": False})
        else:
            checks.append({"name": "canonical_one_truth_presence", "passed": True})

        if duplicates:
            for rel, line_no, heading in duplicates:
                self.log_error(
                    "tier1",
                    "duplicate_one_truth",
                    self.root / rel,
                    line_no,
                    f"Duplicate '## ONE TRUTH' header found in non-canonical file: '{heading}'",
                )
            checks.append(
                {"name": "one_truth_no_duplicates", "passed": False, "duplicates": duplicates}
            )
        else:
            checks.append({"name": "one_truth_no_duplicates", "passed": True})

        # 1.2 Deprecation Banners in Historical Files
        banner_check_passed = True
        banner_keywords = [
            "deprecated",
            "historical archive",
            "historical specification",
            "archive plan",
            "archived plan",
        ]

        for req_rel in REQUIRED_DEPRECATION_BANNER_FILES:
            target_path = self.root / req_rel
            if not target_path.exists():
                self.log_error(
                    "tier1",
                    "banner_file_missing",
                    target_path,
                    None,
                    f"Required file {req_rel} does not exist",
                )
                banner_check_passed = False
                continue

            content = target_path.read_text(encoding="utf-8", errors="replace")
            first_lines = "\n".join(content.splitlines()[:15]).lower()
            has_banner = any(kw in first_lines for kw in banner_keywords)

            if not has_banner:
                self.log_error(
                    "tier1",
                    "missing_deprecation_banner",
                    target_path,
                    1,
                    f"Missing top deprecation/historical warning banner in {req_rel}",
                )
                banner_check_passed = False

        checks.append({"name": "deprecation_banners_present", "passed": banner_check_passed})

        # 1.3 Phase 2 Performance & Publication Integration Wiring
        wiring_targets = [
            ("docs/README.md", [MANDATORY_PERFORMANCE_DOC, MANDATORY_PAPER_DOC]),
            ("docs/ROADMAP.md", [MANDATORY_PERFORMANCE_DOC, MANDATORY_PAPER_DOC]),
            ("docs/TODO.md", [MANDATORY_PERFORMANCE_DOC, MANDATORY_PAPER_DOC]),
            ("AGENTS.md", [MANDATORY_PERFORMANCE_DOC, MANDATORY_PAPER_DOC]),
        ]
        wiring_passed = True

        for rel_doc, expected_links in wiring_targets:
            target_path = self.root / rel_doc
            if not target_path.exists():
                self.log_error(
                    "tier1",
                    "wiring_target_missing",
                    target_path,
                    None,
                    f"Wiring target {rel_doc} missing",
                )
                wiring_passed = False
                continue

            content = target_path.read_text(encoding="utf-8", errors="replace")
            for expected in expected_links:
                if expected not in content:
                    self.log_error(
                        "tier1",
                        "publication_wiring_missing",
                        target_path,
                        None,
                        f"Expected cross-reference to '{expected}' not found in {rel_doc}",
                    )
                    wiring_passed = False

        checks.append({"name": "publication_and_performance_wiring", "passed": wiring_passed})

        status = "PASSED" if all(c["passed"] for c in checks) else "FAILED"
        return {"status": status, "checks": checks}

    def run_tier2_boundary_cases(self) -> dict[str, Any]:
        """Tier 2: Boundary & Corner Cases (file:/// URIs, malformed/empty links)."""
        file_uri_violations = []
        malformed_link_violations = []

        file_uri_pattern = re.compile(r"file:///[a-zA-Z]:/[^\s\)\"\'>]+", re.IGNORECASE)

        for f in self.all_md_files:
            rel = str(f.relative_to(self.root)).replace("\\", "/")
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = strip_code_blocks(content)
            for line_no, line in lines:
                # Check for file:/// absolute URIs
                for match in file_uri_pattern.finditer(line):
                    uri = match.group(0)
                    self.log_error(
                        "tier2",
                        "file_uri_prohibited",
                        f,
                        line_no,
                        f"Prohibited absolute URI found: {uri}. Must use relative path.",
                    )
                    file_uri_violations.append({"file": rel, "line": line_no, "uri": uri})

                # Check for empty links: [text]() or []()
                links = extract_links_from_line(line)
                for text, target, _col in links:
                    if not target.strip():
                        self.log_error(
                            "tier2",
                            "empty_link_target",
                            f,
                            line_no,
                            f"Empty markdown link target: [{text}]()",
                        )
                        malformed_link_violations.append(
                            {"file": rel, "line": line_no, "text": text}
                        )

        passed = len(file_uri_violations) == 0 and len(malformed_link_violations) == 0
        return {
            "status": "PASSED" if passed else "FAILED",
            "file_uri_violations_count": len(file_uri_violations),
            "malformed_link_violations_count": len(malformed_link_violations),
        }

    def run_tier3_cross_doc_consistency(self) -> dict[str, Any]:
        """Tier 3: Cross-Document Consistency (authority maps, terminology, dashboards)."""
        checks: list[dict[str, Any]] = []

        # Check .claude/HANDOFF.md for frozen KPI violations
        handoff_path = self.root / ".claude/HANDOFF.md"
        handoff_passed = True
        if handoff_path.exists():
            content = handoff_path.read_text(encoding="utf-8", errors="replace")
            # Should not contain frozen KPI tables like "strands 10,722/10,722"
            if "10,722/10,722" in content:
                self.log_error(
                    "tier3",
                    "handoff_frozen_kpi",
                    handoff_path,
                    None,
                    "HANDOFF.md contains frozen KPI counts (10,722/10,722). Live counts belong in ONE TRUTH only.",
                )
                handoff_passed = False
        checks.append({"name": "handoff_operator_commands_exclusivity", "passed": handoff_passed})

        # Check docs/README.md Authority Map
        readme_path = self.root / "docs/README.md"
        readme_passed = True
        if readme_path.exists():
            content = readme_path.read_text(encoding="utf-8", errors="replace")
            if (
                "Documentation authority map" not in content
                and "Documentation Authority Map" not in content
            ):
                self.log_error(
                    "tier3",
                    "readme_authority_map_missing",
                    readme_path,
                    None,
                    "docs/README.md missing canonical Documentation Authority Map section",
                )
                readme_passed = False
        checks.append({"name": "readme_authority_map_presence", "passed": readme_passed})

        status = "PASSED" if all(c["passed"] for c in checks) else "FAILED"
        return {"status": status, "checks": checks}

    def get_file_headings(self, path: Path) -> set[str]:
        """Get cached heading slugs for a markdown file."""
        if path not in self.file_headings_cache:
            if path.exists() and path.is_file():
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    self.file_headings_cache[path] = extract_headings(content)
                except Exception:
                    self.file_headings_cache[path] = set()
            else:
                self.file_headings_cache[path] = set()
        return self.file_headings_cache[path]

    def run_tier4_link_resolution(self) -> dict[str, Any]:
        """Tier 4: Exhaustive Real-World Relative Link & Anchor Resolution."""
        broken_links: list[dict[str, Any]] = []
        broken_anchors: list[dict[str, Any]] = []

        for f in self.all_md_files:
            rel_source = str(f.relative_to(self.root)).replace("\\", "/")
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = strip_code_blocks(content)
            for line_no, line in lines:
                links = extract_links_from_line(line)
                for _text, raw_target, _col in links:
                    self.total_links_checked += 1
                    target = raw_target.strip()

                    # Check for template placeholders
                    if any(ph in target for ph in TEMPLATE_PLACEHOLDER_SUBSTRINGS):
                        continue

                    # External protocols
                    if target.startswith(("http://", "https://", "mailto:", "ftp://", "irc://")):
                        self.total_external_links += 1
                        continue

                    # file:/// URIs are handled in Tier 2
                    if target.lower().startswith("file:///"):
                        continue

                    # Split path and anchor
                    parsed = urllib.parse.urlparse(target)
                    path_part = urllib.parse.unquote(parsed.path)
                    anchor_part = urllib.parse.unquote(parsed.fragment)

                    # Intra-document anchor: #some-anchor
                    if not path_part:
                        if anchor_part:
                            valid_anchors = self.get_file_headings(f)
                            slug_clean = slugify_heading(anchor_part)
                            if slug_clean not in valid_anchors and anchor_part not in valid_anchors:
                                self.log_warning(
                                    "tier4",
                                    "broken_intra_anchor",
                                    f,
                                    line_no,
                                    f"Intra-doc anchor '#{anchor_part}' not found in {rel_source}",
                                )
                                broken_anchors.append(
                                    {
                                        "file": rel_source,
                                        "line": line_no,
                                        "target": target,
                                        "anchor": anchor_part,
                                    }
                                )
                        continue

                    # Resolve target path relative to source file's directory
                    try:
                        resolved_path = (f.parent / path_part).resolve()
                    except Exception as e:
                        self.log_error(
                            "tier4",
                            "invalid_path_syntax",
                            f,
                            line_no,
                            f"Invalid path syntax: '{target}' ({e})",
                        )
                        broken_links.append(
                            {
                                "file": rel_source,
                                "line": line_no,
                                "target": target,
                                "reason": f"Path resolve error: {e}",
                            }
                        )
                        continue

                    # Check existence
                    if not resolved_path.exists():
                        self.log_error(
                            "tier4",
                            "broken_relative_link",
                            f,
                            line_no,
                            f"Broken link: '{target}' does not resolve to an existing file/dir (resolved: {resolved_path})",
                        )
                        broken_links.append(
                            {
                                "file": rel_source,
                                "line": line_no,
                                "target": target,
                                "resolved": str(resolved_path),
                            }
                        )
                    else:
                        # If target exists and is markdown, and has anchor
                        if (
                            anchor_part
                            and resolved_path.is_file()
                            and resolved_path.suffix == ".md"
                        ):
                            target_anchors = self.get_file_headings(resolved_path)
                            slug_clean = slugify_heading(anchor_part)
                            if (
                                slug_clean not in target_anchors
                                and anchor_part not in target_anchors
                            ):
                                self.log_warning(
                                    "tier4",
                                    "broken_cross_anchor",
                                    f,
                                    line_no,
                                    f"Anchor '#{anchor_part}' not found in target file '{path_part}'",
                                )
                                broken_anchors.append(
                                    {
                                        "file": rel_source,
                                        "line": line_no,
                                        "target": target,
                                        "anchor": anchor_part,
                                    }
                                )

        passed = len(broken_links) == 0
        if self.strict:
            passed = passed and len(broken_anchors) == 0

        return {
            "status": "PASSED" if passed else "FAILED",
            "broken_links_count": len(broken_links),
            "broken_anchors_count": len(broken_anchors),
            "broken_links": broken_links,
            "broken_anchors": broken_anchors,
        }

    def verify_all(self, target_tier: str = "all") -> dict[str, Any]:
        """Execute verification across specified tiers."""
        self.discover_files()

        results: dict[str, Any] = {}

        if target_tier in ("1", "all"):
            results["tier1_authority_and_features"] = self.run_tier1_authority_and_features()

        if target_tier in ("2", "all"):
            results["tier2_boundary_cases"] = self.run_tier2_boundary_cases()

        if target_tier in ("3", "all"):
            results["tier3_cross_doc_consistency"] = self.run_tier3_cross_doc_consistency()

        if target_tier in ("4", "all"):
            results["tier4_link_resolution"] = self.run_tier4_link_resolution()

        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        verdict = (
            "PASSED" if total_errors == 0 and (not self.strict or total_warnings == 0) else "FAILED"
        )

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "root_directory": str(self.root),
            "total_files_scanned": len(self.all_md_files),
            "total_links_checked": self.total_links_checked,
            "total_external_links": self.total_external_links,
            "errors_count": total_errors,
            "warnings_count": total_warnings,
            "verdict": verdict,
        }

        return {
            "summary": summary,
            "tiers": results,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def print_cli_report(self, report: dict[str, Any]) -> None:
        """Print human-readable summary to console."""
        s = report["summary"]
        print("=" * 80)
        print("  SLAVV DOCUMENTATION INTEGRITY VERIFICATION REPORT")
        print("=" * 80)
        print(f"Timestamp:             {s['timestamp']}")
        print(f"Root Directory:        {s['root_directory']}")
        print(f"Files Scanned:         {s['total_files_scanned']}")
        print(f"Total Links Checked:   {s['total_links_checked']}")
        print(f"External Links:        {s['total_external_links']}")
        print(f"Errors Detected:       {s['errors_count']}")
        print(f"Warnings Detected:     {s['warnings_count']}")
        print("-" * 80)

        # Tier breakdown
        for tier_name, tier_data in report["tiers"].items():
            status = tier_data.get("status", "UNKNOWN")
            status_str = "[PASS]" if status == "PASSED" else "[FAIL]"
            print(f"  {status_str} {tier_name}")

        print("-" * 80)

        if self.errors:
            print("\nITEMIZED DEFECT LIST (ERRORS):")
            for i, err in enumerate(self.errors, start=1):
                loc = f"{err['file']}:{err['line']}" if err["line"] else err["file"]
                print(f"  {i}. [{err['tier']} / {err['check']}] {loc}")
                print(f"     Message: {err['message']}")

        if self.warnings and (self.verbose or self.strict):
            print("\nWARNINGS:")
            for i, w in enumerate(self.warnings, start=1):
                loc = f"{w['file']}:{w['line']}" if w["line"] else w["file"]
                print(f"  {i}. [{w['tier']} / {w['check']}] {loc}")
                print(f"     Message: {w['message']}")

        print("=" * 80)
        verdict_str = (
            ">>> VERDICT: PASSED <<<" if s["verdict"] == "PASSED" else ">>> VERDICT: FAILED <<<"
        )
        print(f"  {verdict_str}")
        print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify SLAVV documentation integrity across all 4 tiers.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root directory (default: parent of scripts/)",
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Specific tier to verify (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON to stdout instead of text summary",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Save structured JSON report to specified file path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit code 1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    verifier = DocsIntegrityVerifier(
        root_dir=args.root,
        strict=args.strict,
        verbose=args.verbose,
    )
    report = verifier.verify_all(target_tier=args.tier)

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        verifier.print_cli_report(report)

    return 0 if report["summary"]["verdict"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
