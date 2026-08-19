"""One production-sized crop Energy chunk isolation (stretch, not unlock).

Maps a known mismatch voxel onto the v2 octave-chunk lattice and re-runs that
chunk via ``stretch_energy_chunk_v202``. Does **not** refactor nested
``_process_chunk`` (parity-sensitive). Indexing is copied from
``matlab_get_energy_v202_chunked`` (~494-579 extract, ~708-714 write merge).

Isolation only: never emit ``stretch_complete`` or an Energy unlock.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.analytics.parity.proof.stretch import (
    INTERPRET_INCOMPLETE_INFRA,
    StretchStatus,
)
from slavv_python.pipeline.energy import matlab_energy_filter_v200 as native_hessian
from slavv_python.pipeline.energy.matlab_engine_backend import MatlabEngineInfraError
from slavv_python.pipeline.energy.matlab_engine_host import (
    MatlabEnginePy37Worker,
    MatlabEngineSession,
    energy_chunk_v202_from_spatial,
)
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    _matlab_coarse_local_slices,
    get_chunking_lattice_v190,
    get_starts_and_counts_v200,
)

logger = logging.getLogger(__name__)

DEFAULT_MISMATCH_VOXEL_ZYX: tuple[int, int, int] = (13, 0, 0)
DEFAULT_WINNER_SCALE = 43

INTERPRET_HELPER_ORACLE = (
    "Re-run == v2 and both ≠ oracle → helper/oracle "
    "(lattice or filter/interp3 vs original MATLAB batch), packaging OK."
)
INTERPRET_PACKAGING = "Re-run ≠ v2 → packaging/merge/resume bug (infra), still not unlock."
INTERPRET_OTHER_CHUNKS = (
    "Re-run == oracle on that window → residual is other chunks / merge; do not declare unlock."
)
INTERPRET_WINDOW_MATCHES_ALL = (
    "Re-run == v2 == oracle on this surface; residual is other chunks / merge. "
    "Do not declare unlock."
)

_STATUS_BLOCKED = StretchStatus.BLOCKED_FLOAT_PATH.value


@dataclass(frozen=True)
class OctaveChunkLattice:
    """v2 octave lattice: MATLAB ``(Y, X, Z)`` chunking + Fortran unravel."""

    octave: int
    scale_indices_at_octave: tuple[int, ...]
    rf_zyx: tuple[int, int, int]
    lattice_dimensions_yxz: tuple[int, int, int]
    number_of_chunks: int
    prev_scales_count: int
    y_read_starts: np.ndarray
    x_read_starts: np.ndarray
    z_read_starts: np.ndarray
    y_read_counts: np.ndarray
    x_read_counts: np.ndarray
    z_read_counts: np.ndarray
    y_write_starts: np.ndarray
    x_write_starts: np.ndarray
    z_write_starts: np.ndarray
    y_write_counts: np.ndarray
    x_write_counts: np.ndarray
    z_write_counts: np.ndarray
    y_offsets: np.ndarray
    x_offsets: np.ndarray
    z_offsets: np.ndarray
    microns_per_pixel_matlab: np.ndarray
    pixels_per_sigma_psf: np.ndarray
    lumen_radius_microns: np.ndarray


@dataclass(frozen=True)
class ChunkLatticeHit:
    """Write-window hit for one voxel on one octave lattice."""

    chunk_index: int
    octave: int
    winner_scale: int
    lattice_dimensions_yxz: tuple[int, int, int]
    lattice_indices_yxz: tuple[int, int, int]
    number_of_chunks: int
    write_start_zyx: tuple[int, int, int]
    write_count_zyx: tuple[int, int, int]
    rf_zyx: tuple[int, int, int]
    scale_indices_at_octave: tuple[int, ...]
    prev_scales_count: int

    @property
    def write_slices_zyx(self) -> tuple[slice, slice, slice]:
        z0, y0, x0 = self.write_start_zyx
        dz, dy, dx = self.write_count_zyx
        return slice(z0, z0 + dz), slice(y0, y0 + dy), slice(x0, x0 + dx)


@dataclass(frozen=True)
class ThreeWayCompare:
    """``np.array_equal`` three-way on one named surface (window / owned / voxel)."""

    rerun_equals_v2: bool
    rerun_equals_oracle: bool
    v2_equals_oracle: bool
    interpretation: str
    n_voxels: int
    n_rerun_ne_v2: int
    n_rerun_ne_oracle: int
    n_v2_ne_oracle: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rerun_equals_v2": self.rerun_equals_v2,
            "rerun_equals_oracle": self.rerun_equals_oracle,
            "v2_equals_oracle": self.v2_equals_oracle,
            "interpretation": self.interpretation,
            "n_voxels": self.n_voxels,
            "n_rerun_ne_v2": self.n_rerun_ne_v2,
            "n_rerun_ne_oracle": self.n_rerun_ne_oracle,
            "n_v2_ne_oracle": self.n_v2_ne_oracle,
            "not_stretch_success": True,
        }


def octave_for_scale(config: dict[str, Any], winner_scale: int) -> int:
    """Return the consolidated octave that owns ``winner_scale`` (0-based)."""
    octave_at_scales = np.asarray(config["octave_at_scales"])
    if winner_scale < 0 or winner_scale >= int(octave_at_scales.size):
        raise ValueError(
            f"winner_scale {winner_scale} out of range for {int(octave_at_scales.size)} scales"
        )
    return int(octave_at_scales[winner_scale])


def build_octave_chunk_lattice(config: dict[str, Any], octave: int) -> OctaveChunkLattice:
    """Build the v2 ``get_chunking_lattice_v190`` lattice for one octave."""
    image_shape = np.asarray(config["image_shape"], dtype=float)
    if image_shape.size != 3:
        raise ValueError("config must include image_shape=(Z, Y, X)")
    octave_at_scales = np.asarray(config["octave_at_scales"])
    scale_indices_at_octave = np.where(octave_at_scales == octave)[0]
    if len(scale_indices_at_octave) == 0:
        raise ValueError(f"no scales at octave {octave}")

    microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
    pixels_per_sigma_PSF = np.asarray(config["pixels_per_sigma_PSF"], dtype=float)
    lumen_radius_microns = np.asarray(config["lumen_radius_microns"], dtype=float)
    rf = np.asarray(config["scale_resolution_factors"][scale_indices_at_octave[0]], dtype=float)
    matlab_image_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
    rf_matlab = np.array([rf[1], rf[2], rf[0]], dtype=float)
    largest_scale_idx = int(scale_indices_at_octave[-1])
    largest_pixels_per_radius = lumen_radius_microns[largest_scale_idx] / microns_per_voxel
    approx_size = np.round(matlab_image_shape / rf_matlab)
    microns_per_pixel = microns_per_voxel * rf
    microns_per_pixel_matlab = np.array(
        [microns_per_pixel[1], microns_per_pixel[2], microns_per_pixel[0]],
        dtype=float,
    )
    voxel_aspect_ratio = 1.0 / microns_per_pixel_matlab
    chunk_lattice_dimensions, number_of_chunks = get_chunking_lattice_v190(
        voxel_aspect_ratio,
        float(config["max_voxels"]),
        approx_size,
    )
    chunk_overlap_vector = np.ceil(
        6.0 * np.sqrt(pixels_per_sigma_PSF**2 + largest_pixels_per_radius**2)
    ).astype(np.int32)
    chunk_overlap_matlab = chunk_overlap_vector[[1, 2, 0]]
    res_starts_counts = get_starts_and_counts_v200(
        chunk_lattice_dimensions,
        chunk_overlap_matlab,
        matlab_image_shape,
        rf_matlab,
    )
    dims = tuple(int(v) for v in np.asarray(chunk_lattice_dimensions).tolist())
    return OctaveChunkLattice(
        octave=int(octave),
        scale_indices_at_octave=tuple(int(v) for v in scale_indices_at_octave.tolist()),
        rf_zyx=(int(rf[0]), int(rf[1]), int(rf[2])),
        lattice_dimensions_yxz=(dims[0], dims[1], dims[2]),
        number_of_chunks=int(number_of_chunks),
        prev_scales_count=int(np.sum(octave_at_scales < octave)),
        y_read_starts=res_starts_counts[0],
        x_read_starts=res_starts_counts[1],
        z_read_starts=res_starts_counts[2],
        y_read_counts=res_starts_counts[3],
        x_read_counts=res_starts_counts[4],
        z_read_counts=res_starts_counts[5],
        y_write_starts=res_starts_counts[6],
        x_write_starts=res_starts_counts[7],
        z_write_starts=res_starts_counts[8],
        y_write_counts=res_starts_counts[9],
        x_write_counts=res_starts_counts[10],
        z_write_counts=res_starts_counts[11],
        y_offsets=res_starts_counts[12],
        x_offsets=res_starts_counts[13],
        z_offsets=res_starts_counts[14],
        microns_per_pixel_matlab=microns_per_pixel_matlab,
        pixels_per_sigma_psf=pixels_per_sigma_PSF,
        lumen_radius_microns=lumen_radius_microns,
    )


def chunk_index_for_voxel_zyx(
    config: dict[str, Any],
    voxel_zyx: tuple[int, int, int],
    *,
    winner_scale: int = DEFAULT_WINNER_SCALE,
    octave: int | None = None,
    lattice: OctaveChunkLattice | None = None,
) -> ChunkLatticeHit:
    """Map a ZYX voxel to the Fortran-order chunk whose write window contains it."""
    resolved_octave = int(octave) if octave is not None else octave_for_scale(config, winner_scale)
    resolved_lattice = lattice or build_octave_chunk_lattice(config, resolved_octave)
    py_z, py_y, py_x = (int(voxel_zyx[0]), int(voxel_zyx[1]), int(voxel_zyx[2]))
    dims = resolved_lattice.lattice_dimensions_yxz
    for chunk_idx in range(resolved_lattice.number_of_chunks):
        y_idx, x_idx, z_idx = np.unravel_index(chunk_idx, dims, order="F")
        py_z_w_start = int(resolved_lattice.z_write_starts[z_idx]) - 1
        py_y_w_start = int(resolved_lattice.y_write_starts[y_idx]) - 1
        py_x_w_start = int(resolved_lattice.x_write_starts[x_idx]) - 1
        w_count_z = int(resolved_lattice.z_write_counts[z_idx])
        w_count_y = int(resolved_lattice.y_write_counts[y_idx])
        w_count_x = int(resolved_lattice.x_write_counts[x_idx])
        if (
            py_z_w_start <= py_z < py_z_w_start + w_count_z
            and py_y_w_start <= py_y < py_y_w_start + w_count_y
            and py_x_w_start <= py_x < py_x_w_start + w_count_x
        ):
            return ChunkLatticeHit(
                chunk_index=int(chunk_idx),
                octave=resolved_lattice.octave,
                winner_scale=int(winner_scale),
                lattice_dimensions_yxz=dims,
                lattice_indices_yxz=(int(y_idx), int(x_idx), int(z_idx)),
                number_of_chunks=resolved_lattice.number_of_chunks,
                write_start_zyx=(py_z_w_start, py_y_w_start, py_x_w_start),
                write_count_zyx=(w_count_z, w_count_y, w_count_x),
                rf_zyx=resolved_lattice.rf_zyx,
                scale_indices_at_octave=resolved_lattice.scale_indices_at_octave,
                prev_scales_count=resolved_lattice.prev_scales_count,
            )
    raise ValueError(
        f"voxel {voxel_zyx} is not covered by any write window at octave {resolved_octave}"
    )


def interpret_three_way(
    *,
    rerun_equals_v2: bool,
    rerun_equals_oracle: bool,
    v2_equals_oracle: bool,
) -> str:
    """Name helper/oracle vs packaging vs other-chunks. Never unlock."""
    if not rerun_equals_v2:
        return INTERPRET_PACKAGING
    if rerun_equals_oracle and v2_equals_oracle:
        return INTERPRET_WINDOW_MATCHES_ALL
    if rerun_equals_oracle:
        return INTERPRET_OTHER_CHUNKS
    if not v2_equals_oracle:
        return INTERPRET_HELPER_ORACLE
    return INTERPRET_PACKAGING


def compare_three_way(
    rerun: np.ndarray,
    v2: np.ndarray,
    oracle: np.ndarray,
) -> ThreeWayCompare:
    """Bit-compare three same-shaped arrays with ``np.array_equal``."""
    rerun_a = np.asarray(rerun)
    v2_a = np.asarray(v2)
    oracle_a = np.asarray(oracle)
    if rerun_a.shape != v2_a.shape or rerun_a.shape != oracle_a.shape:
        raise ValueError(
            f"three-way shape mismatch rerun={rerun_a.shape} v2={v2_a.shape} "
            f"oracle={oracle_a.shape}"
        )
    rerun_equals_v2 = bool(np.array_equal(rerun_a, v2_a))
    rerun_equals_oracle = bool(np.array_equal(rerun_a, oracle_a))
    v2_equals_oracle = bool(np.array_equal(v2_a, oracle_a))
    n_voxels = int(rerun_a.size)
    n_rerun_ne_v2 = int(np.count_nonzero(rerun_a != v2_a)) if n_voxels else 0
    n_rerun_ne_oracle = int(np.count_nonzero(rerun_a != oracle_a)) if n_voxels else 0
    n_v2_ne_oracle = int(np.count_nonzero(v2_a != oracle_a)) if n_voxels else 0
    return ThreeWayCompare(
        rerun_equals_v2=rerun_equals_v2,
        rerun_equals_oracle=rerun_equals_oracle,
        v2_equals_oracle=v2_equals_oracle,
        interpretation=interpret_three_way(
            rerun_equals_v2=rerun_equals_v2,
            rerun_equals_oracle=rerun_equals_oracle,
            v2_equals_oracle=v2_equals_oracle,
        ),
        n_voxels=n_voxels,
        n_rerun_ne_v2=n_rerun_ne_v2,
        n_rerun_ne_oracle=n_rerun_ne_oracle,
        n_v2_ne_oracle=n_v2_ne_oracle,
    )


def octave_owned_mask(
    v2_scales: np.ndarray, scale_indices_at_octave: tuple[int, ...]
) -> np.ndarray:
    """True where v2 ``scale_indices`` were won by this octave (min-merge survivors)."""
    result: np.ndarray = np.isin(
        np.asarray(v2_scales), np.asarray(scale_indices_at_octave, dtype=np.int16)
    )
    return result


def run_stretch_chunk_v202(
    image: np.ndarray,
    config: dict[str, Any],
    lattice: OctaveChunkLattice,
    chunk_index: int,
    session: MatlabEngineSession | MatlabEnginePy37Worker,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    """Re-run one chunk via ``energy_chunk_v202_from_spatial`` (engine body only).

    Indexing is a copy of nested ``_process_chunk`` extract + write-window slices.
    Does not merge into a master volume.
    """
    if session is None:
        raise MatlabEngineInfraError("incomplete_infra: MATLAB engine session missing")
    dims = lattice.lattice_dimensions_yxz
    y_idx, x_idx, z_idx = np.unravel_index(int(chunk_index), dims, order="F")

    py_z_start = int(lattice.z_read_starts[z_idx]) - 1
    py_y_start = int(lattice.y_read_starts[y_idx]) - 1
    py_x_start = int(lattice.x_read_starts[x_idx]) - 1
    py_z_count = int(lattice.z_read_counts[z_idx])
    py_y_count = int(lattice.y_read_counts[y_idx])
    py_x_count = int(lattice.x_read_counts[x_idx])

    stride_z, stride_y, stride_x = lattice.rf_zyx
    original_chunk_zyx = image[
        py_z_start : py_z_start + py_z_count : stride_z,
        py_y_start : py_y_start + py_y_count : stride_y,
        py_x_start : py_x_start + py_x_count : stride_x,
    ]
    original_chunk = np.transpose(original_chunk_zyx, (1, 2, 0)).copy(order="F")
    original_chunk = original_chunk.astype(np.float64, copy=False)
    padded_chunk = native_hessian._fourier_transform_input(original_chunk)
    padded_shape = padded_chunk.shape

    w_count_z = int(lattice.z_write_counts[z_idx])
    w_count_y = int(lattice.y_write_counts[y_idx])
    w_count_x = int(lattice.x_write_counts[x_idx])
    off_z = int(lattice.z_offsets[z_idx])
    off_y = int(lattice.y_offsets[y_idx])
    off_x = int(lattice.x_offsets[x_idx])
    y_local, x_local, z_local = _matlab_coarse_local_slices(
        offsets=(off_y, off_x, off_z),
        write_counts=(w_count_y, w_count_x, w_count_z),
        strides=(stride_y, stride_x, stride_z),
        padded_shape=padded_shape,
    )
    rf = np.asarray(lattice.rf_zyx, dtype=float)
    pixels_per_sigma_psf_at_oct = lattice.pixels_per_sigma_psf / rf
    scale_idx = np.asarray(lattice.scale_indices_at_octave, dtype=int)
    energy_yxz, scale_1based = energy_chunk_v202_from_spatial(
        session,
        original_chunk,
        matching_kernel_string=str(
            config.get("matching_kernel_string", "3D gaussian conv annular pulse")
        ),
        radii=np.asarray(lattice.lumen_radius_microns[scale_idx], dtype=np.float64),
        vessel_wall=float(config.get("vessel_wall_thickness_in_microns", 0.0)),
        microns_per_pixel=lattice.microns_per_pixel_matlab,
        pixels_per_sigma_psf=pixels_per_sigma_psf_at_oct[[1, 2, 0]],
        y0=int(y_local.start or 0) + 1,
        y1=int(y_local.stop),
        x0=int(x_local.start or 0) + 1,
        x1=int(x_local.stop),
        z0=int(z_local.start or 0) + 1,
        z1=int(z_local.stop),
        y_offset=off_y,
        x_offset=off_x,
        z_offset=off_z,
        y_write_count=w_count_y,
        x_write_count=w_count_x,
        z_write_count=w_count_z,
        rf_y=stride_y,
        rf_x=stride_x,
        rf_z=stride_z,
        gaussian_to_ideal_ratio=float(config["gaussian_to_ideal_ratio"]),
        spherical_to_annular_ratio=float(config["spherical_to_annular_ratio"]),
        scales_per_octave=float(config.get("scales_per_octave", 1.5)),
    )
    chunk_best_energy = np.asfortranarray(np.asarray(energy_yxz, dtype=np.float64))
    chunk_best_scale_sub_idx = np.asfortranarray(
        np.asarray(scale_1based, dtype=np.float64).astype(np.int16) - 1
    )
    chunk_energy_min = chunk_best_energy.transpose(2, 0, 1)
    chunk_scale_min = chunk_best_scale_sub_idx.transpose(2, 0, 1)
    chunk_scale_min[chunk_energy_min >= 0.0] = -1
    valid_scale = chunk_scale_min >= 0
    chunk_scale_min = chunk_scale_min.astype(np.int16)
    chunk_scale_min[valid_scale] += int(lattice.prev_scales_count)

    py_z_w_start = int(lattice.z_write_starts[z_idx]) - 1
    py_y_w_start = int(lattice.y_write_starts[y_idx]) - 1
    py_x_w_start = int(lattice.x_write_starts[x_idx]) - 1
    slice_z = slice(py_z_w_start, py_z_w_start + w_count_z)
    slice_y = slice(py_y_w_start, py_y_w_start + w_count_y)
    slice_x = slice(py_x_w_start, py_x_w_start + w_count_x)
    logger.info(
        "stretch one-chunk extract chunk=%s octave=%s write_zyx=%s",
        chunk_index,
        lattice.octave,
        (slice_z, slice_y, slice_x),
    )
    return chunk_energy_min, chunk_scale_min, (slice_z, slice_y, slice_x)


def hit_to_dict(hit: ChunkLatticeHit) -> dict[str, Any]:
    z0, y0, x0 = hit.write_start_zyx
    dz, dy, dx = hit.write_count_zyx
    return {
        "chunk_index": hit.chunk_index,
        "octave": hit.octave,
        "winner_scale": hit.winner_scale,
        "lattice_dimensions_yxz": list(hit.lattice_dimensions_yxz),
        "lattice_indices_yxz": list(hit.lattice_indices_yxz),
        "number_of_chunks": hit.number_of_chunks,
        "write_start_zyx": list(hit.write_start_zyx),
        "write_count_zyx": list(hit.write_count_zyx),
        "write_window_zyx": [[z0, z0 + dz], [y0, y0 + dy], [x0, x0 + dx]],
        "rf_zyx": list(hit.rf_zyx),
        "scale_indices_at_octave": list(hit.scale_indices_at_octave),
        "prev_scales_count": hit.prev_scales_count,
    }


def patch_stretch_status_extra(
    path: Path,
    extra: dict[str, Any],
    *,
    require_blocked_float_path: bool = True,
) -> dict[str, Any]:
    """Merge ``extra`` into dest ``stretch_status.json`` without changing ``status``."""
    status_path = Path(path)
    if not status_path.is_file():
        raise FileNotFoundError(f"stretch_status.json missing: {status_path}")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stretch_status.json is not an object: {status_path}")
    current = str(payload.get("status", ""))
    if require_blocked_float_path and current != _STATUS_BLOCKED:
        raise ValueError(
            f"refusing extra patch: status is {current!r}, expected {_STATUS_BLOCKED!r}"
        )
    merged = dict(payload.get("extra") or {})
    merged.update(extra)
    payload["extra"] = merged
    payload["status"] = current
    status_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("patched stretch_status extra only; status remains %s", current)
    return payload


__all__ = [
    "DEFAULT_MISMATCH_VOXEL_ZYX",
    "DEFAULT_WINNER_SCALE",
    "INTERPRET_HELPER_ORACLE",
    "INTERPRET_INCOMPLETE_INFRA",
    "INTERPRET_OTHER_CHUNKS",
    "INTERPRET_PACKAGING",
    "INTERPRET_WINDOW_MATCHES_ALL",
    "ChunkLatticeHit",
    "OctaveChunkLattice",
    "ThreeWayCompare",
    "build_octave_chunk_lattice",
    "chunk_index_for_voxel_zyx",
    "compare_three_way",
    "hit_to_dict",
    "interpret_three_way",
    "octave_for_scale",
    "octave_owned_mask",
    "patch_stretch_status_extra",
    "run_stretch_chunk_v202",
]
