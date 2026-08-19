# Streamlit GUI audit (2026-08-19)

> **Product UI audit, not parity status.** Live MATLAB↔Python pass/fail stays in
> [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

## In short

**G1–G4 closed (2026-08-19 follow-up).** `app.py` uses absolute imports, calls
`shell.main()`, and page modules live in `views/` (not Streamlit’s reserved
`pages/`). `tests/ui/test_streamlit_entry.py` loads the real entry file and
visits every sidebar page. Remaining items G5–G18 are still open.

Original finding: `slavv-app` / `streamlit run …/app.py` died on a relative
import before any page rendered. Existing UI tests never launched the real
entry script, so they stayed green.

## Method

- Code map of `slavv_python/interface/streamlit/` (shell, pages, shared state).
- Live launch: `python -m streamlit run slavv_python/interface/streamlit/app.py --server.port 8501 --server.headless true`.
- Browser: Edge headless screenshots of `http://127.0.0.1:8501` (desktop 1400×900 and 390×844). Both captured Streamlit’s empty skeleton; the script runner log has the crash.
- `AppTest.from_file(app.py)` reproduced the same `ImportError`.
- Official launcher resolves the same file: `streamlit_launcher._resolve_web_app_path()` → `slavv_python/interface/streamlit/app.py`.
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
| G5 | P1 | Home | Documentation links are placeholders (`[Algorithm Overview](#)` and siblings) | `pages/static.py` | Point at `docs/TUTORIAL.md` / `docs/README.md` or remove the list |
| G6 | P1 | Home | Onboarding copy does not match the two-product model (Paper Path tracing vs Exact Route Watershed Discovery) | Home “Edge Extraction - Tracing”; Processing still exposes both `tracing` and `watershed` | Name Paper Path vs Exact Route; default edge method should follow the selected profile |
| G7 | P1 | Home | “Export Formats: 5+” vs Visualization’s three downloads + share HTML | `static.py` vs `EXPORT_BUTTON_SPECS` | Count real buttons, or add MAT/JSON downloads |
| G8 | P1 | Visualization | Opacity slider and 3D camera-angle selectbox are never read or passed into `NetworkVisualizer` | `pages/visualization.py` | Wire them or remove the widgets |
| G9 | P1 | Visualization | Energy-field slice controls are stuffed into the global sidebar, unlike other viz options | same file, `st.sidebar.selectbox` | Keep slice controls in the page column |
| G10 | P1 | Visualization | `from ...shared_services.share_report import record_share_event` is an inline import | `visualization.py` ~253 | Module-level import (repo rule) |
| G11 | P1 | Curation | Interactive path launches a **desktop** Qt/napari window from the web app; Streamlit blocks until it closes | `pages/curation.py` | Label as desktop-only, disable in headless/server, or drop from the web flow |
| G12 | P1 | Curation | No-op `st.session_state["parameters"]`; mixed `[!]`, `[Curation]`, `[Launch]`, `[OK]` prefixes | `curation.py` | Delete the no-op; use Streamlit warnings without ASCII badges |
| G13 | P1 | Processing | `cupy_hessian` is offered in Energy method; will fail if CuPy is absent | `processing.py` energy_method options | Hide unavailable backends |
| G14 | P1 | Processing | “Force Recalculation From” uses raw stage ids (`energy`) next to “Pipeline Target” human labels | `processing.py` | Same label style as stop-after |
| G15 | P1 | Empty states | Curation vs Visualization/Analysis warn with different copy and `[!]` vs plain text | curation / visualization / analysis pages | One empty-state helper: “Process an image first (Image Processing)” |
| G16 | P2 | Shell | Sidebar `selectbox` navigation + emoji labels; dashboard is only on Home, not in the nav list | `shell.py`, `static.py` | `st.navigation` with Home, Processing, Curation, Visualization, Analysis, About; Dashboard as Home section or its own item |
| G17 | P2 | About | Credits say “Python Port” with no paper vs exact-route split; no link to docs | `static.py` `show_about_page` | Link NEW_ENGINEER / TUTORIAL |
| G18 | P2 | Analysis | MATLAB `SpecialOutput` / `area_histogram_plotter.m` names in user-facing captions | `analysis.py` | User language first; MATLAB names in help tooltips only |

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
