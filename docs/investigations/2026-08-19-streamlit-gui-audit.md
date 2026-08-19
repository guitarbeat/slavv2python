# Streamlit GUI audit (2026-08-19)

> **Product UI audit, not parity status.** Live MATLAB↔Python pass/fail stays in
> [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

## In short

**G1–G18 closed (2026-08-19 follow-ups).** Launch works (G1–G4). Copy, empty
states, viz controls, desktop-curator labeling, and optional Energy backends
are updated (G5–G18). Dashboard remains a Home section, not a separate nav item.

Original finding: `slavv-app` / `streamlit run …/app.py` died on a relative
import before any page rendered. Existing UI tests never launched the real
entry script, so they stayed green.

## Method

- Code map of `slavv_python/interface/streamlit/` (shell, views, services, state).
- Live launch: `python -m streamlit run slavv_python/interface/streamlit/app.py --server.port 8501 --server.headless true`.
- Browser: Edge headless screenshots of `http://127.0.0.1:8501` (desktop 1400×900 and 390×844). Both captured Streamlit’s empty skeleton; the script runner log has the crash.
- `AppTest.from_file(app.py)` reproduced the same `ImportError`.
- Official launcher resolves the same file: `slavv_python.interface.streamlit.launcher._resolve_web_app_path()` → `slavv_python/interface/streamlit/app.py`.
- Out of scope: CLI, TUI runner, exact-route writers.

## Live result

**Crash (repeated on every browser hit):**

```text
File slavv_python/interface/streamlit/app.py, line 31
    from .pages.analysis import show_analysis_page
ImportError: attempted relative import with no known parent package
```

Streamlit 1.58 runs `app.py` as a script (`_mpa_v1` / `page.run()`), so
relative imports are invalid. The same path is what `slavv-app` passes to
`python -m streamlit run`.

Even if that import were fixed, **`main()` is imported in `app.py` and never
called**. The shell that owns the sidebar (`shell.py` `PAGE_HANDLERS`) would
still not run. Streamlit also treats the sibling `pages/` folder as native
multipage scripts; those modules only *define* `show_*_page` and do not render
on import, so MPA pages would be empty.

`tests/ui/test_summary_dashboard.py::test_app_main_runs` calls `shell.main()`
in-process. `tests/unit/interface/test_streamlit_launcher.py` mocks
`subprocess.run` and never starts Streamlit. Neither catches this crash.

## Ranked findings

| ID | Sev | Page | Finding | Evidence | Suggested fix |
|----|-----|------|---------|----------|----------------|
| G1 | **P0 closed** | Launch | App does not start: relative imports in the Streamlit entry script | Live traceback; `AppTest.from_file` | Absolute `slavv_python.interface.streamlit…` imports in `app.py` |
| G2 | **P0 closed** | Launch | Entry script never calls `shell.main()`, so the product shell would not render after G1 | `app.py` imports `main` and ends at `__all__` | `if __name__ == "__main__": main()` |
| G3 | **P0 closed** | Nav | `interface/streamlit/pages/` collides with Streamlit’s reserved multipage `pages/` directory | Traceback path `_mpa_v1`; Streamlit 1.58 MPA | Renamed to `interface/streamlit/views/` |
| G4 | **P1 closed** | Tests | No test actually runs `streamlit run` / `slavv-app` against the real entry file | launcher test mocks subprocess; smoke test imports `shell.main` | `tests/ui/test_streamlit_entry.py` |
| G5 | **P1 closed** | Home | Documentation links are placeholders | `views/static.py` | Repo paths: TUTORIAL, docs hub, NEW_ENGINEER |
| G6 | **P1 closed** | Home / Processing | Copy ignored Paper Path vs Exact Route | Home steps + Processing captions/edge-method labels | Tracing Discovery vs Watershed Discovery named in the UI |
| G7 | **P1 closed** | Home | “Export Formats: 5+” vs in-app downloads | `static.py` | Metric is 4 in-app formats (VMV, CASX, CSV zip, HTML) |
| G8 | **P1 closed** | Visualization | Opacity and camera widgets unused | `_apply_figure_display` | Slider/camera applied to Plotly traces / scene camera |
| G9 | **P1 closed** | Visualization | Energy slice controls lived in the global sidebar | Display Options column | Slice axis/index sit with other viz controls |
| G10 | **P1 closed** | Visualization | Inline import of `record_share_event` | `visualization.py` | Module-level import |
| G11 | **P1 closed** | Curation | Desktop Qt/napari launched as if it were in-browser | `desktop_curator_available()` | Labeled desktop-only; disabled headless / via env flag |
| G12 | **P1 closed** | Curation | No-op parameters lookup; ASCII `[!]` / `[OK]` prefixes | `curation.py` | Plain Streamlit status text |
| G13 | **P1 closed** | Processing | `cupy_hessian` offered without CuPy | `available_public_energy_methods()` | Optional backends only if importable |
| G14 | **P1 closed** | Processing | Force-rerun used raw stage ids | `processing.py` | Human labels matching Pipeline Target |
| G15 | **P1 closed** | Empty states | Inconsistent missing-run copy | `empty_state.py` | Shared warnings that name Image Processing |
| G16 | **P2 closed** | Shell | Emoji nav labels | `shell.py` | Plain Home / Image Processing / Curation / … (dashboard stays on Home) |
| G17 | **P2 closed** | About | No Paper Path / Exact Route; no doc pointers | `views/static.py` | Two-product copy + repo doc paths |
| G18 | **P2 closed** | Analysis | MATLAB script names in user captions | `analysis.py` | User-language captions |

## Per-page notes (source; live shell never reached)

| Page | Empty path | With results | Notes |
|------|------------|--------------|-------|
| Home | Dashboard + welcome | N/A | Dashboard controls bind query params; dead doc links |
| Image Processing | “Please upload a TIFF file…” | Upload-only; no workspace dataset picker | Profile captions mention tracing; Advanced tab can override to watershed |
| ML Curation | `[!] No processing results…` | Interactive desktop GUI / automatic / ML | Requires vertices **and** edges |
| Visualization | “No processing results…” | 2D/3D/depth/strand/energy; exports if network present | Dead opacity/camera widgets |
| Analysis | “No processing results…” | Metrics + four tabs | Requires full Network stage |
| About | Always static | — | No nav from Home doc links |

## Test coverage gap

| Test | What it actually proves |
|------|-------------------------|
| `tests/ui/test_summary_dashboard.py` | Plotly dashboard figure; `shell.main()` import smoke |
| `tests/ui/test_visualization_*.py` | Visualizer / exports, not the Streamlit page |
| `tests/unit/interface/test_streamlit_launcher.py` | CLI argv plumbing with mocked `subprocess.run` |
| `tests/unit/interface/test_app_run_state.py` | `AppRunState` envelope |

Missing: one test that the **entry file Streamlit executes** imports and calls `main()` without exception.

## Recommended first fix slice

1. **G1+G2+G3+G4** so `slavv-app` actually paints the shell (absolute imports, call `main()`, stop colliding with Streamlit `pages/`, add an entry-file smoke test).
2. Then G8/G11/G15 (dead viz controls, desktop curator in a web page, empty-state copy).

No Streamlit product code was changed in this audit.
