"""Unit tests for the documentation integrity verifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.docs.verify_integrity import (
    MANDATORY_PAPER_DOC,
    MANDATORY_PERFORMANCE_DOC,
    REQUIRED_DEPRECATION_BANNER_FILES,
    DocsIntegrityVerifier,
    extract_headings,
    extract_links_from_line,
    slugify_heading,
    strip_code_blocks,
)

if TYPE_CHECKING:
    from pathlib import Path

_BANNER = "> **DEPRECATED historical archive** — not live status.\n\n# Archive\n"
_WIRING = (
    "# Hub\n\n"
    "See PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md and "
    "MATLAB_PYTHON_TRANSLATION_PAPER.md.\n\n"
    "## Documentation authority map\n"
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_min_docs_tree(root: Path) -> None:
    """Create the smallest tree that satisfies verifier invariants."""
    _write(root / "AGENTS.md", _WIRING)
    _write(root / "docs" / "README.md", _WIRING + "See [hub](ROADMAP.md).\n")
    _write(root / "docs" / "ROADMAP.md", _WIRING)
    _write(root / "docs" / "TODO.md", _WIRING)
    _write(
        root / "docs" / "reference" / "core" / "EXACT_PROOF_FINDINGS.md",
        "# Findings\n\n## ONE TRUTH\n\nLive status lives here.\n",
    )
    _write(root / ".claude" / "HANDOFF.md", "# Operator brief\n\nNo frozen KPI table.\n")
    for rel in REQUIRED_DEPRECATION_BANNER_FILES:
        _write(root / rel, _BANNER)


def test_required_constants_name_catalog_and_banner_paths() -> None:
    assert MANDATORY_PERFORMANCE_DOC == "PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md"
    assert MANDATORY_PAPER_DOC == "MATLAB_PYTHON_TRANSLATION_PAPER.md"
    assert REQUIRED_DEPRECATION_BANNER_FILES
    assert all(path.endswith(".md") for path in REQUIRED_DEPRECATION_BANNER_FILES)


def test_slugify_and_heading_extraction_ignore_code_fences() -> None:
    assert slugify_heading("ONE TRUTH — Phase 1") == "one-truth--phase-1"
    content = "# Title\n\n```\n## Not A Heading\n```\n\n## Real Heading\n"
    headings = extract_headings(content)
    assert "title" in headings
    assert "real-heading" in headings
    assert "not-a-heading" not in headings


def test_extract_links_and_strip_code_blocks() -> None:
    line = "See [hub](../README.md#docs) and ![img](figures/a.png)."
    links = extract_links_from_line(line)
    assert [target for _text, target, _col in links] == [
        "../README.md#docs",
        "figures/a.png",
    ]
    kept = strip_code_blocks("a\n```\nb\n```\nc\n")
    texts = [text for _line_no, text in kept]
    assert "a" in texts
    assert "c" in texts
    assert "b" not in texts


def test_verifier_passes_seeded_tree_and_flags_broken_link(tmp_path: Path) -> None:
    _seed_min_docs_tree(tmp_path)
    _write(tmp_path / "docs" / "broken.md", "# Broken\n\nSee [missing](nope.md).\n")

    verifier = DocsIntegrityVerifier(root_dir=tmp_path)
    report = verifier.verify_all()

    assert report["tiers"]["tier1_authority_and_features"]["status"] == "PASSED"
    assert report["tiers"]["tier2_boundary_cases"]["status"] == "PASSED"
    assert report["tiers"]["tier3_cross_doc_consistency"]["status"] == "PASSED"
    assert report["summary"]["verdict"] == "FAILED"
    broken_files = {
        err["file"] for err in report["errors"] if err["check"] == "broken_relative_link"
    }
    assert "docs/broken.md" in broken_files


def test_verifier_passes_clean_seeded_tree(tmp_path: Path) -> None:
    _seed_min_docs_tree(tmp_path)

    verifier = DocsIntegrityVerifier(root_dir=tmp_path)
    report = verifier.verify_all()

    assert report["summary"]["verdict"] == "PASSED"
    assert report["summary"]["errors_count"] == 0
    assert report["tiers"]["tier4_link_resolution"]["broken_links_count"] == 0
