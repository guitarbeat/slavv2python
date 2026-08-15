#!/usr/bin/env python
"""Python 3.7 MATLAB Engine worker for stretch Energy float math.

Must run under Python 3.7 (R2019a Engine ABI). Do not import slavv_python.
Protocol: one JSON object per stdin line; one JSON object per stdout line.
"""

from __future__ import print_function

import json
import os
import sys
import traceback

import numpy as np


def _fail(message):
    print(json.dumps({"ok": False, "error": message}), flush=True)


def _ok(payload=None):
    body = {"ok": True}
    if payload:
        body.update(payload)
    print(json.dumps(body), flush=True)


def _as_row(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr.tolist()


def main():
    try:
        import matlab
        import matlab.engine
    except Exception as exc:
        _fail("matlab.engine import failed: %s" % exc)
        return 2

    engine = None
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError as exc:
                _fail("invalid JSON: %s" % exc)
                continue
            op = msg.get("op")
            if op == "start":
                if engine is not None:
                    _ok()
                    continue
                try:
                    engine = matlab.engine.start_matlab()
                    engine.addpath(str(msg["vectorization_source"]), nargout=0)
                    engine.addpath(str(msg["helper_dir"]), nargout=0)
                except Exception as exc:
                    engine = None
                    _fail("start_matlab failed: %s" % exc)
                    continue
                _ok()
            elif op == "energy_filter":
                if engine is None:
                    _fail("engine not started")
                    continue
                try:
                    chunk = np.load(str(msg["chunk_npy"]))
                    chunk = np.asfortranarray(np.asarray(chunk, dtype=np.float64))
                    flat = [float(v) for v in np.ravel(chunk, order="F")]
                    size = [int(d) for d in chunk.shape]
                    try:
                        ml_chunk = matlab.double(flat, size=size)
                    except TypeError:
                        ml_chunk = matlab.double(flat)
                    microns = matlab.double(_as_row(msg["microns_per_pixel"]))
                    psf = matlab.double(_as_row(msg["pixels_per_sigma_psf"]))
                    energy = engine.stretch_energy_filter_v200(
                        ml_chunk,
                        str(msg["matching_kernel_string"]),
                        float(msg["radius"]),
                        float(msg["vessel_wall"]),
                        microns,
                        psf,
                        float(msg["y0"]),
                        float(msg["y1"]),
                        float(msg["x0"]),
                        float(msg["x1"]),
                        float(msg["z0"]),
                        float(msg["z1"]),
                        float(msg["gaussian_to_ideal_ratio"]),
                        float(msg["spherical_to_annular_ratio"]),
                        float(msg["scales_per_octave"]),
                        nargout=1,
                    )
                    y_len = int(msg["y1"]) - int(msg["y0"]) + 1
                    x_len = int(msg["x1"]) - int(msg["x0"]) + 1
                    z_len = int(msg["z1"]) - int(msg["z0"]) + 1
                    data = getattr(energy, "_data", None)
                    if data is not None:
                        out = np.frombuffer(data, dtype=np.float64).copy()
                        out = np.reshape(out, (y_len, x_len, z_len), order="F")
                    else:
                        out = np.asarray(energy, dtype=np.float64)
                    out_path = str(msg["out_npy"])
                    np.save(out_path, np.ascontiguousarray(out))
                except Exception as exc:
                    _fail("energy_filter failed: %s" % exc)
                    continue
                _ok({"out_npy": out_path})
            elif op == "quit":
                if engine is not None:
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = None
                _ok()
                return 0
            else:
                _fail("unknown op: %s" % op)
    except Exception:
        _fail(traceback.format_exc())
        return 1
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
