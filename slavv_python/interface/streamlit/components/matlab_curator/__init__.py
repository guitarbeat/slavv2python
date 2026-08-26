"""Python wrapper for the MATLAB-faithful browser curator."""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
import streamlit as st

_BUFFER_DTYPES = {
    "displayVolume": np.dtype("uint8"),
    "cornerstoneVolume": np.dtype("uint8"),
    "energyVolume": np.dtype("<f4"),
    "scaleVolume": np.dtype("<i2"),
}


def _pack_payload(data: dict[str, Any], session: dict[str, Any]) -> bytes:
    """Pack JSON metadata and aligned typed buffers into one v2 bytes payload."""
    metadata = dict(data)
    descriptors: dict[str, dict[str, int | str]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, dtype in _BUFFER_DTYPES.items():
        value = metadata.pop(name, np.empty(0, dtype=dtype))
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype)).reshape(-1)
        chunk = array.tobytes()
        descriptors[name] = {
            "dtype": dtype.str,
            "offset": offset,
            "length": len(chunk),
        }
        chunks.append(chunk)
        offset += len(chunk)
    header = json.dumps(
        {"data": metadata, "session": session, "buffers": descriptors},
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return struct.pack("<I", len(header)) + header + b"".join(chunks)


def matlab_curator(
    *,
    data: dict[str, Any],
    session: dict[str, Any],
    key: str,
):
    """Mount the browser curator and expose only commit-oriented triggers."""
    # Registration is intentionally lazy. Importing the Streamlit app in a
    # unit-test process has no Runtime-backed component manager, while a real
    # Streamlit/AppTest runtime has already discovered this package manifest.
    component = st.components.v2.component(
        "slavv_python.matlab_curator",
        js="index-*.js",
        css="style-*.css",
    )
    return component(
        key=key,
        data=_pack_payload(data, session),
        default={},
        on_apply_change=lambda: None,
        on_save_change=lambda: None,
        on_load_change=lambda: None,
    )


__all__ = ["matlab_curator"]
