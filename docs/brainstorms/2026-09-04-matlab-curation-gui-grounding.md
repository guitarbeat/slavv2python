# Grounding: Python curation GUI <-> MATLAB curator screens/interaction

Extraction only. Generated 2026-09-04. Budget ~150 lines of quotes.

---

## Product / strategy intent

```33:35:STRATEGY.md
Keep Python scientifically equivalent to MATLAB: exact-route stretch (bit-equal Energy, then discrete) and paper-profile certification. Also make the Python curation GUI a MATLAB-familiar operator surface — same screens and interaction (layout, keybindings, review workflow) — so a MATLAB curator can work without relearning.

_Why it serves the approach:_ Without a frozen, proved baseline, speed and C++ work cannot be trusted. The MATLAB-familiar operator surface is part of that same trust: a curator should not have to relearn screens to use Python.
```

```45:45:STRATEGY.md
Additional real volumes (`neurovasc-db`) and later packaging, after the exact route is trusted. Curation GUI is not "later UX"; it lives under Trust.
```

---

## Docs (workflow, backends, GUI audit)

```22:38:docs/reference/workflow/MANUAL_CURATION_WORKFLOW.md
## Curate the result

Open **Curation** after an Edge Set is available. The default **MATLAB-style browser curator** follows the original two-stage sequence: Vertex Curation, then Edge Curation. It provides:

- calibrated X, Y, and Z projections with depth, logarithmic thickness, pan, zoom, and a physical scale bar;
- the original grayscale/cyan/red visual language, a field-of-view minimap, and intensity and Energy histograms;
- point, rectangle, polyline, and circular-complement toggling, plus local threshold, Sweep, Crop, Paint, Add Vertex, and Add Edge tools where their stage supports them;
- local undo/redo with keyboard shortcuts, and a versioned `.slavv-curation.json` session file for Save/Load;
- a final **Apply and rebuild Network** action after Edge Curation.
...
**Desktop manual review (MATLAB-style)** is a Python Qt/PyVista or experimental napari interface that opens in a separate local window when optional GUI dependencies and a display are available.
```

```40:48:docs/reference/workflow/MANUAL_CURATION_WORKFLOW.md
## Compare with the original MATLAB curator
...
The script opens the unmodified `external/Vectorization-Public/source/vertex_curator.m` against preserved MATLAB artifacts.
```

```13:44:docs/reference/backends/NAPARI_CURATOR.md
The current production curator still lives in
`slavv_python/visualization/interactive_curator.py`. The new prototype lives in
`slavv_python/visualization/napari_curator.py` ...
It does not yet attempt to recreate the full four-panel MATLAB-style UI from
the current Qt/PyVista implementation.
```

```63:64:docs/investigations/2026-08-19-streamlit-gui-audit.md
| G11 | **P1 closed** | Curation | Desktop Qt/napari launched as if it were in-browser | `desktop_curator_available()` | Labeled desktop-only; disabled headless / via env flag |
| G12 | **P1 closed** | Curation | No-op parameters lookup; ASCII `[!]` / `[OK]` prefixes | `curation.py` | Plain Streamlit status text |
```

ADR touch (curation artifacts, not GUI layout):

```121:121:docs/adr/0011-energy-float-certification-policy.md
- **Vertex energies are sourced from the raw `vertices.mat`** (true physical energy), since MATLAB curation overwrites `curated_vertices.mat` energies with a rank ramp.
```

---

## Streamlit surfaces (page + browser curator)

```62:93:slavv_python/interface/streamlit/views/curation.py
def show_ml_curation_page():
    """Display manual, automatic, and model-based curation workflows."""
...
    curation_type = st.radio(
        "Workflow",
        (
            "MATLAB-style browser curator",
            "Desktop manual review (MATLAB-style)",
            "Automatic filtering",
            "Model-assisted filtering",
        ),
```

```101:106:slavv_python/interface/streamlit/views/curation.py
    elif curation_type == "Desktop manual review (MATLAB-style)":
        st.info(
            "This Python desktop interface reproduces the four-panel MATLAB GCI layout: "
            "volume map, volume display, intensity histogram, and energy histogram. "
```

```1:1:slavv_python/interface/streamlit/components/matlab_curator/__init__.py
"""Python wrapper for the MATLAB-faithful browser curator."""
```

```45:51:slavv_python/interface/streamlit/components/matlab_curator/__init__.py
def matlab_curator(...):
    """Mount the browser curator and expose only commit-oriented triggers."""
```

```375:376:slavv_python/interface/streamlit/views/manual_curation.py
def render_browser_manual_curation(...):
    """Render the MATLAB-faithful two-stage curator and commit validated edits."""
```

```430:430:slavv_python/interface/streamlit/views/manual_curation.py
        st.caption("Vertex -> Edge -> Network")
```

```495:495:slavv_python/interface/streamlit/views/manual_curation.py
                curation_mode="MATLAB-faithful browser curator",
```

```30:46:slavv_python/interface/streamlit/state/manual_curation.py
@dataclass
class CurationSessionV1:
    """Versioned, replayable browser curation state."""
    stage: str = "vertices"
    vertex_truth / edge_truth / added_* / history / schema_version
```

---

## Browser frontend layout / tools / keybindings / review workflow

```2:9:slavv_python/interface/streamlit/components/matlab_curator/frontend/src/types.ts
export type Stage = "vertices" | "edges";
export type Tool = "view" | "toggle" | "add-vertex" | "add-edge" | "crop";
export type ToggleMethod = "rect" | "line" | "circle";
```

```132:144:slavv_python/interface/streamlit/components/matlab_curator/frontend/src/style.css
.mc-workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
}
.mc-display {
  display: grid;
  grid-template-rows: auto minmax(540px, 1fr) auto auto;
```

```279:282:slavv_python/interface/streamlit/components/matlab_curator/frontend/src/style.css
.mc-context {
  display: grid;
  grid-template-rows: 265px minmax(210px, 1fr) minmax(230px, 1fr);
```

```394:407:slavv_python/interface/streamlit/components/matlab_curator/frontend/src/App.tsx
KeyboardEvent: Escape -> tool view; Ctrl/Cmd+Z undo (Shift+Z redo); Ctrl/Cmd+Y redo
```

```714:720:.../App.tsx
h2: "Vertex Curator" | "Edge Curator"; stage buttons "1 · Vertices" / "2 · Edges"
```

```767:783:.../App.tsx
Tool ribbon: Undo, Redo, Crop|Paint, Sweep, Add Vertex, Add Edge (edges stage), Toggle, rect|line|circ. comp.
```

```839:842:.../App.tsx
vertices: "Continue to edges"; edges: "Apply and rebuild network"
```

---

## Desktop Qt/PyVista + napari

```1:10:slavv_python/visualization/interactive_curator.py
Achieves 1:1 feature parity with the MATLAB GCI:
  - 4-panel layout (Volume Map, Volume Display, Intensity Histogram, Energy Histogram)
  - Depth/Thickness sliders; Blue/Red toggle; Sweep; energy threshold; Add Edge
```

```89:90:slavv_python/visualization/interactive_curator.py
    def _init_ui(self):
        """Build the 4-panel layout that mirrors the MATLAB GCI."""
```

```1:6:slavv_python/visualization/napari_curator.py
Experimental napari-based curator prototype. Same high-level contract as desktop curator.
```

```26:42:slavv_python/interface/streamlit/services/curation.py
run_interactive_curator backends: qt -> interactive_curator.run_curator; napari -> run_curator_napari
```

---

## MATLAB reference path (this checkout)

```1:5:scripts/curation/launch_matlab_curator_sample.m
% opens unmodified Vectorization-Public vertex_curator against y_junction_32
```

```65:75:scripts/curation/launch_matlab_curator_sample.m
vertex_curator(energies, space_subscripts, scale_subscripts, display_radii, microns_per_pixel, original_path, curation_path, energy_path, intensity_limits, energy_range);
```

Checkout status:
- `external/Vectorization-Public/source/vertex_curator.m` — File not found
- `external/Vectorization-Public/source/edge_curator.m` — Glob 0 files
- Conversation-start git status lists `D` for those MATLAB curator sources under the submodule
- No ADR dedicated to curation GUI layout/keybindings (ADR 0011 only notes curated_vertices rank ramp)

## Adjacent non-GUI curation
- `slavv_python/analytics/curation/automated.py`, `machine_learning*.py`
- Route: `slavv_python/interface/streamlit/routes/curation.py` -> `show_ml_curation_page()`
