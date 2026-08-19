#!/usr/bin/env python
"""Python 3.7 MATLAB Engine worker for stretch Energy float math.

Must run under Python 3.7 (R2019a Engine ABI). Do not import slavv_python.
Protocol: one JSON object per stdin line; one JSON object per stdout line.
"""

# ruff: noqa: UP010, UP031, SIM105

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


def _to_matlab_double(matlab, array):
    arr = np.asfortranarray(np.asarray(array, dtype=np.float64))
    flat = [float(v) for v in np.ravel(arr, order="F")]
    size = [int(d) for d in arr.shape]
    try:
        return matlab.double(flat, size=size)
    except TypeError:
        return matlab.double(flat)


def _from_matlab_double(ml_array, shape):
    data = getattr(ml_array, "_data", None)
    if data is not None:
        out = np.frombuffer(data, dtype=np.float64).copy()
        return np.reshape(out, shape, order="F")
    return np.asarray(ml_array, dtype=np.float64)


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
                    ml_chunk = _to_matlab_double(matlab, chunk)
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
                    out = _from_matlab_double(energy, (y_len, x_len, z_len))
                    out_path = str(msg["out_npy"])
                    np.save(out_path, np.ascontiguousarray(out))
                except Exception as exc:
                    _fail("energy_filter failed: %s" % exc)
                    continue
                _ok({"out_npy": out_path})
            elif op == "energy_chunk":
                if engine is None:
                    _fail("engine not started")
                    continue
                try:
                    chunk = np.load(str(msg["chunk_npy"]))
                    ml_chunk = _to_matlab_double(matlab, chunk)
                    microns = matlab.double(_as_row(msg["microns_per_pixel"]))
                    psf = matlab.double(_as_row(msg["pixels_per_sigma_psf"]))
                    radii = matlab.double(_as_row(msg["radii"]))
                    energy, scale_idx = engine.stretch_energy_chunk_v202(
                        ml_chunk,
                        str(msg["matching_kernel_string"]),
                        radii,
                        float(msg["vessel_wall"]),
                        microns,
                        psf,
                        float(msg["y0"]),
                        float(msg["y1"]),
                        float(msg["x0"]),
                        float(msg["x1"]),
                        float(msg["z0"]),
                        float(msg["z1"]),
                        float(msg["y_offset"]),
                        float(msg["x_offset"]),
                        float(msg["z_offset"]),
                        float(msg["y_write_count"]),
                        float(msg["x_write_count"]),
                        float(msg["z_write_count"]),
                        float(msg["rf_y"]),
                        float(msg["rf_x"]),
                        float(msg["rf_z"]),
                        float(msg["gaussian_to_ideal_ratio"]),
                        float(msg["spherical_to_annular_ratio"]),
                        float(msg["scales_per_octave"]),
                        nargout=2,
                    )
                    y_len = int(msg["y_write_count"])
                    x_len = int(msg["x_write_count"])
                    z_len = int(msg["z_write_count"])
                    out_shape = (y_len, x_len, z_len)
                    energy_out = _from_matlab_double(energy, out_shape)
                    scale_out = _from_matlab_double(scale_idx, out_shape)
                    energy_path = str(msg["energy_npy"])
                    scale_path = str(msg["scale_npy"])
                    np.save(energy_path, np.ascontiguousarray(energy_out))
                    np.save(scale_path, np.ascontiguousarray(scale_out))
                except Exception as exc:
                    _fail("energy_chunk failed: %s" % exc)
                    continue
                _ok({"energy_npy": energy_path, "scale_npy": scale_path})
            elif op == "roundtrip":
                if engine is None:
                    _fail("engine not started")
                    continue
                try:
                    arr = np.load(str(msg["in_npy"]))
                    arr = np.asfortranarray(np.asarray(arr, dtype=np.float64))
                    ml_arr = _to_matlab_double(matlab, arr)
                    ml_out = engine.stretch_identity(ml_arr, nargout=1)
                    out_shape = tuple(int(d) for d in arr.shape)
                    out = _from_matlab_double(ml_out, out_shape)
                    out_path = str(msg["out_npy"])
                    np.save(out_path, np.ascontiguousarray(out))
                except Exception as exc:
                    _fail("roundtrip failed: %s" % exc)
                    continue
                _ok({"out_npy": out_path})
            elif op == "linspace_mesh":
                if engine is None:
                    _fail("engine not started")
                    continue
                try:
                    mesh = engine.stretch_linspace_1based(
                        float(msg["offset"]),
                        float(msg["rf"]),
                        float(msg["count"]),
                        nargout=1,
                    )
                    count = int(msg["count"])
                    out = _from_matlab_double(mesh, (count,))
                    out_path = str(msg["out_npy"])
                    np.save(out_path, np.ascontiguousarray(out.reshape(-1)))
                except Exception as exc:
                    _fail("linspace_mesh failed: %s" % exc)
                    continue
                _ok({"out_npy": out_path})
            elif op == "interp3":
                if engine is None:
                    _fail("engine not started")
                    continue
                try:
                    volume = np.load(str(msg["volume_npy"]))
                    mesh_x = np.load(str(msg["mesh_x_npy"]))
                    mesh_y = np.load(str(msg["mesh_y_npy"]))
                    mesh_z = np.load(str(msg["mesh_z_npy"]))
                    ml_volume = _to_matlab_double(matlab, volume)
                    ml_x = _to_matlab_double(matlab, mesh_x)
                    ml_y = _to_matlab_double(matlab, mesh_y)
                    ml_z = _to_matlab_double(matlab, mesh_z)
                    sampled = engine.stretch_interp3_probe(ml_volume, ml_x, ml_y, ml_z, nargout=1)
                    out_shape = tuple(int(d) for d in np.asarray(mesh_y).shape)
                    out = _from_matlab_double(sampled, out_shape)
                    out_path = str(msg["out_npy"])
                    np.save(out_path, np.ascontiguousarray(out))
                except Exception as exc:
                    _fail("interp3 failed: %s" % exc)
                    continue
                _ok({"out_npy": out_path})
            elif op == "get_energy_v202":
                if engine is None:
                    _fail("engine not started")
                    continue
                try:
                    radii = matlab.double(_as_row(msg["lumen_radius_in_microns_range"]))
                    microns = matlab.double(_as_row(msg["microns_per_voxel"]))
                    psf = matlab.double(_as_row(msg["pixels_per_sigma_psf"]))
                    elapsed = engine.stretch_get_energy_v202(
                        str(msg["matching_kernel_string"]),
                        radii,
                        float(msg["vessel_wall"]),
                        microns,
                        psf,
                        float(msg["max_voxels_per_node"]),
                        str(msg["data_directory"]),
                        str(msg["original_handle"]),
                        str(msg["energy_handle"]),
                        float(msg["gaussian_to_ideal_ratio"]),
                        float(msg["spherical_to_annular_ratio"]),
                        nargout=1,
                    )
                except Exception as exc:
                    _fail("get_energy_v202 failed: %s" % exc)
                    continue
                _ok({"elapsed_sec": float(elapsed)})
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
