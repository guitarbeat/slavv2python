# napari Curator Prototype

This note documents the experimental `napari`-based curator prototype that now
exists alongside the maintained Qt/PyVista curator.

The current production curator still lives in
`source/slavv/visualization/interactive_curator.py`. The new prototype lives in
`source/slavv/visualization/napari_curator.py` and is intended as a lower-code
surface for image + point + path review.

## Why Add It

- The existing curator works, but it is a custom GUI stack assembled from
  PyQt5, PyVista, and PyQtGraph.
- `napari` already provides native volume/image viewing and annotation layers
  for points and paths.
- That makes it a strong candidate for future curator maintenance if manual QA
  grows.

## Current Scope

This is a prototype, not a full replacement.

Today the `napari` curator supports:

- viewing the energy volume in a desktop viewer
- reviewing vertices as a points layer
- reviewing edges as a paths layer
- toggling selected vertices and edges between accepted/rejected states
- applying a simple vertex-energy threshold
- adding a straight-line edge from two selected vertices
- hiding rejected items visually
- returning the same curated `(vertices, edges)` result shape as the current
  curator after the viewer closes

It does not yet attempt to recreate the full four-panel MATLAB-style UI from
the current Qt/PyVista implementation.

## How To Launch It

The Streamlit app now exposes an interactive curator backend selector with:

- `Qt/PyVista (default)`
- `napari (experimental)`

The Qt/PyVista path remains the default. Choose the `napari` option before
launching the interactive curator to open the prototype instead.

## Installation

Install the optional dependency with:

```powershell
pip install -e ".[napari]"
```

If the prototype is selected without `napari` installed, SLAVV raises a runtime
error with install guidance instead of silently falling back.

## Maintenance Direction

This prototype is meant to answer one question first: does `napari` make manual
curation simpler to extend than the current hand-built Qt/PyVista surface?

If the answer is yes, the next steps should focus on:

- matching more of the current toggle/sweep workflow
- improving edge editing beyond straight-line insertion
- deciding whether the web app should default to the `napari` backend when it
  is available
