"""MATLAB support: frontier heap and voxel-claim maps for watershed discovery.

Role: ``FrontierQueue`` / ``VoxelClaimMap`` used by ``matlab_get_edges_by_watershed.py``.
MATLAB lineage: ``get_edges_by_watershed.m`` shared-state helpers.
"""

from __future__ import annotations

import heapq
from typing import cast

import numpy as np
from scipy import sparse


def _matlab_global_watershed_border_locations(shape: tuple[int, int, int]) -> np.ndarray:
    """Return MATLAB-order linear indices for the image border at ``strel_apothem = 1``."""
    border_mask: np.ndarray = np.zeros(shape, dtype=bool)
    border_mask[0, :, :] = True
    border_mask[shape[0] - 1, :, :] = True
    border_mask[:, 0, :] = True
    border_mask[:, shape[1] - 1, :] = True
    border_mask[:, :, 0] = True
    border_mask[:, :, shape[2] - 1] = True
    return cast("np.ndarray", np.flatnonzero(border_mask.ravel(order="F")).astype(np.int64))


def _matlab_global_watershed_insert_available_location(
    available_locations: list[int],
    next_location: int,
    next_energy: float,
    energy_lookup: dict[int, float] | np.ndarray,
    seed_idx: int,
    is_current_location_clear: bool,
) -> tuple[list[int], bool]:
    """Insert ``next_location`` into MATLAB worst-to-best ``available_locations`` order."""
    is_clear = is_current_location_clear
    original = list(available_locations)

    target_energy = float(next_energy)

    if not original:
        return [int(next_location)], True

    if seed_idx == 1:
        if float(energy_lookup[int(original[0])]) <= target_energy:
            insert_at = 0
        else:
            insert_at = len(original)
            for idx in range(len(original) - 1, -1, -1):
                if float(energy_lookup[int(original[idx])]) > target_energy:
                    insert_at = idx + 1
                    break
    elif float(energy_lookup[int(original[-1])]) >= target_energy:
        insert_at = len(original) if is_current_location_clear else len(original) - 1
    else:
        insert_at = len(original)
        for idx, loc in enumerate(original):
            if float(energy_lookup[int(loc)]) < target_energy:
                insert_at = idx
                break

    if not is_current_location_clear:
        is_clear = True
        updated = [*original[:insert_at], int(next_location), *original[insert_at:-1]]
    else:
        updated = [*original[:insert_at], int(next_location), *original[insert_at:]]
    return updated, is_clear


def _matlab_global_watershed_reset_join_locations(
    available_locations: list[int],
    *,
    next_vertex_locations: np.ndarray,
    is_current_location_clear: bool,
) -> tuple[list[int], bool]:
    """Remove join targets from ``available_locations`` (MATLAB watershed join reset)."""
    is_clear = is_current_location_clear
    updated = list(available_locations)
    next_locations = set(np.asarray(next_vertex_locations, dtype=np.int64).tolist())
    locations_to_reset = sorted({int(loc) for loc in updated if int(loc) in next_locations})
    if not is_clear:
        if updated:
            # MATLAB builds ``locations_to_reset`` before clearing the current tail, then
            # removes the tail value from that reset list before popping ``end``.
            tail_location = int(updated[-1])
            locations_to_reset = [loc for loc in locations_to_reset if loc != tail_location]
            updated.pop()
        is_clear = True

    reset_indices: list[int] = []
    for location in locations_to_reset:
        for idx, available_location in enumerate(updated):
            if int(available_location) == int(location):
                reset_indices.append(idx)
                break

    for idx in sorted(set(reset_indices), reverse=True):
        del updated[idx]
    return updated, is_clear


def _claim_unowned_strel_arrays(
    *,
    current_vertex_index: int,
    current_scale_label: int,
    current_d_over_r: float,
    valid_linear: np.ndarray,
    strel_pointer_indices: np.ndarray,
    strel_r_over_R: np.ndarray,
    adjusted_energies: np.ndarray,
    vertex_index_map_flat: np.ndarray,
    pointer_map_flat: np.ndarray,
    energy_map_flat: np.ndarray,
    d_over_r_map_flat: np.ndarray,
    size_map_flat: np.ndarray,
    lut_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Claim unowned strel voxels into flat shared watershed maps."""
    if len(strel_pointer_indices) != len(valid_linear):
        raise AssertionError("Strel arrays must stay aligned")

    vertices_of_current_strel = np.asarray(vertex_index_map_flat[valid_linear], dtype=np.uint32)
    is_without_vertex = vertices_of_current_strel == 0

    if np.any(is_without_vertex):
        claim_linear = valid_linear[is_without_vertex]
        claim_pointers = np.asarray(strel_pointer_indices[is_without_vertex], dtype=np.uint64)
        if np.any(claim_pointers < 1) or np.any(claim_pointers > lut_size):
            raise AssertionError("invalid claim pointers")

        vertex_index_map_flat[claim_linear] = np.uint32(current_vertex_index)
        pointer_map_flat[claim_linear] = claim_pointers
        energy_map_flat[claim_linear] = adjusted_energies[is_without_vertex]
        d_over_r_map_flat[claim_linear] = (
            np.asarray(strel_r_over_R[is_without_vertex], dtype=np.float64) + current_d_over_r
        )
        size_map_flat[claim_linear] = np.int16(current_scale_label)

    return vertices_of_current_strel, is_without_vertex


class SortedFrontier:
    """Faithful port of MATLAB ``available_locations`` (stable worst-to-best sorted array)."""

    def __init__(self, initial_locations: list[int], energy_lookup: np.ndarray):
        self._energy_lookup = energy_lookup
        self._available: list[int] = list(initial_locations)
        self._is_current_location_clear = False

    def begin_seed_loop(self) -> None:
        """Reset per-iteration clear flag before MATLAB seed loop."""
        self._is_current_location_clear = False

    def __bool__(self) -> bool:
        return bool(self._available)

    def peek_best(self) -> int:
        if not self._available:
            raise KeyError("peek from an empty sorted frontier")
        return int(self._available[-1])

    def debug_snapshot(self, targets: set[int] | None = None, tail_count: int = 8) -> dict[str, object]:
        target_set = targets or set()
        return {
            "length": len(self._available),
            "is_current_location_clear": self._is_current_location_clear,
            "tail": [int(value) for value in self._available[-tail_count:]],
            "target_counts": {
                str(int(target)): sum(1 for value in self._available if int(value) == int(target))
                for target in sorted(target_set)
            },
        }

    def pop_best(self) -> int:
        if not self._available:
            raise KeyError("pop from an empty sorted frontier")
        return int(self._available.pop())

    def push(
        self,
        location: int,
        energy: float,
        seed_idx: int,
        *,
        current_linear: int | None = None,
    ) -> None:
        del current_linear
        self._available, self._is_current_location_clear = (
            _matlab_global_watershed_insert_available_location(
                self._available,
                next_location=int(location),
                next_energy=float(energy),
                energy_lookup=self._energy_lookup,
                seed_idx=int(seed_idx),
                is_current_location_clear=self._is_current_location_clear,
            )
        )

    def discard_current_location_if_not_clear(self, current_linear: int) -> None:
        del current_linear
        if self._is_current_location_clear:
            return
        if self._available:
            self._available.pop()
        self._is_current_location_clear = True

    def remove_first_occurrence(
        self,
        locations: list[int] | np.ndarray,
        *,
        current_linear: int | None = None,
    ) -> None:
        del current_linear
        self._available, self._is_current_location_clear = (
            _matlab_global_watershed_reset_join_locations(
                self._available,
                next_vertex_locations=np.asarray(locations, dtype=np.int64),
                is_current_location_clear=self._is_current_location_clear,
            )
        )


def build_watershed_frontier(
    backend: str,
    initial_locations: list[int],
    energy_lookup: np.ndarray,
) -> SortedFrontier | FrontierQueue:
    """Select the watershed frontier implementation (``sorted`` default, ``heap`` fallback)."""
    normalized = str(backend).strip().lower()
    if normalized in {"", "sorted", "array", "matlab"}:
        return SortedFrontier(initial_locations, energy_lookup)
    if normalized == "heap":
        return FrontierQueue(initial_locations, energy_lookup)
    raise ValueError(
        f"Unknown watershed frontier backend {backend!r}; expected 'sorted' or 'heap'."
    )


class FrontierQueue:
    """Legacy heap frontier (non-stable; retained for rollback and perf comparison)."""

    def __init__(self, initial_locations: list[int], energy_lookup: np.ndarray):
        self._energy_lookup = energy_lookup
        self._heap: list[list] = []
        self._entry_finder: dict[int, list] = {}
        self._is_current_location_clear = False

        # Seed tie-break: MATLAB pops available_locations from the end, so use pop rank
        # (not Fortran linear index) to preserve energy-rank flood order at -Inf seeds.
        n_initial = len(initial_locations)
        for order, loc in enumerate(initial_locations):
            loc_idx = int(loc)
            seed_rank = n_initial - order
            entry = [float(self._energy_lookup[loc_idx]), seed_rank, loc_idx, False]
            self._entry_finder[loc_idx] = entry
            self._heap.append(entry)

        heapq.heapify(self._heap)
        initial_locations.clear()

    def begin_seed_loop(self) -> None:
        """Reset per-iteration clear flag before MATLAB seed loop."""
        self._is_current_location_clear = False

    def __bool__(self) -> bool:
        while self._heap:
            entry = self._heap[0]
            if not entry[3] and self._entry_finder.get(entry[2]) is entry:
                return True
            heapq.heappop(self._heap)
        return False

    def _remove_location(self, linear_index: int) -> None:
        entry = self._entry_finder.get(int(linear_index))
        if entry is not None:
            entry[3] = True
            del self._entry_finder[int(linear_index)]

    def peek_best(self) -> int:
        while self._heap:
            entry = self._heap[0]
            if not entry[3] and self._entry_finder.get(entry[2]) is entry:
                return cast("int", entry[2])
            heapq.heappop(self._heap)
        raise KeyError("peek from an empty priority queue")

    def pop_best(self) -> int:
        while self._heap:
            entry = heapq.heappop(self._heap)
            _energy, _tie, loc, removed = entry
            if not removed and self._entry_finder.get(loc) is entry:
                del self._entry_finder[loc]
                return cast("int", loc)
        raise KeyError("pop from an empty priority queue")

    def push(
        self,
        location: int,
        energy: float,
        seed_idx: int,
        *,
        current_linear: int | None = None,
    ) -> None:
        """Adds a location to the heap with MATLAB-exact tie-breaking."""
        if not self._is_current_location_clear and current_linear is not None:
            self._remove_location(current_linear)
            self._is_current_location_clear = True

        _target_index = int(location)
        seed_rank = max(1, int(seed_idx))
        tie_break = (seed_rank - 1) * (1 << 31) + _target_index
        if _target_index in self._entry_finder:
            entry = self._entry_finder[_target_index]
            if energy < entry[0] or (energy == entry[0] and tie_break < entry[1]):
                entry[3] = True  # Mark old entry as deleted
            else:
                return

        entry = [float(energy), tie_break, _target_index, False]
        self._entry_finder[_target_index] = entry
        heapq.heappush(self._heap, entry)

    def discard_current_location_if_not_clear(self, current_linear: int) -> None:
        """MATLAB: when strel energy is not tolerated, pop tail if current not yet cleared."""
        if self._is_current_location_clear:
            return
        self._remove_location(current_linear)
        self._is_current_location_clear = True

    def remove_first_occurrence(
        self,
        locations: list[int] | np.ndarray,
        *,
        current_linear: int | None = None,
    ) -> None:
        """Removes the specified locations to handle watershed joins."""
        if not self._is_current_location_clear and current_linear is not None:
            self._remove_location(current_linear)
            self._is_current_location_clear = True

        locations_to_reset = set(np.asarray(locations, dtype=np.int64).tolist())
        for loc in locations_to_reset:
            entry = self._entry_finder.get(loc)
            if entry is not None:
                entry[3] = True
                del self._entry_finder[loc]


class VoxelClaimMap:
    """Encapsulates the flat state arrays and exposes domain-level atomic claiming."""

    def __init__(
        self, shape: tuple[int, int, int], vertex_positions: np.ndarray, energy_map_raw: np.ndarray
    ):
        self.shape = shape
        self.vertex_index_map: np.ndarray = np.zeros(shape, dtype=np.uint32, order="F")
        self.pointer_map: np.ndarray = np.zeros(shape, dtype=np.uint64, order="F")
        self.branch_order_map: np.ndarray = np.zeros(shape, dtype=np.uint8, order="F")
        self.d_over_r_map: np.ndarray = np.zeros(shape, dtype=np.float64, order="F")

        # MATLAB maintains two energy maps: energy_map_temp (static priority source)
        # and energy_map (dynamic penalized source for trace results).
        self.energy_map = np.array(energy_map_raw, dtype=np.float64, order="F", copy=True)
        self.energy_map_temp = np.array(energy_map_raw, dtype=np.float64, order="F", copy=True)
        self.vertex_energies_raw_flat = np.array(
            energy_map_raw, dtype=np.float64, order="F", copy=True
        ).ravel(order="F")

        self._initialize_vertices(vertex_positions)

        # Flat views for performance in the inner loop
        self.vertex_index_flat = self.vertex_index_map.ravel(order="F")
        self.pointer_flat = self.pointer_map.ravel(order="F")
        self.branch_order_flat = self.branch_order_map.ravel(order="F")
        self.d_over_r_flat = self.d_over_r_map.ravel(order="F")
        self.energy_flat = self.energy_map.ravel(order="F")
        self.energy_temp_flat = self.energy_map_temp.ravel(order="F")

    def _initialize_vertices(self, vertex_positions: np.ndarray):
        from slavv_python.utils.matlab_order import zyx_to_matlab_linear_indices

        physical_shape = (self.shape[2], self.shape[0], self.shape[1])
        self.vertex_locations = zyx_to_matlab_linear_indices(vertex_positions, physical_shape)
        self.number_of_vertices = len(self.vertex_locations)

        border_locations = _matlab_global_watershed_border_locations(self.shape)
        v_index_flat = self.vertex_index_map.ravel(order="F")
        temp_flat = self.energy_map_temp.ravel(order="F")

        for linear_index in border_locations:
            v_index_flat[int(linear_index)] = np.uint32(self.number_of_vertices + 1)

        self.vertex_energies: np.ndarray = np.empty((self.number_of_vertices,), dtype=np.float64)
        for vertex_offset, linear_index in enumerate(self.vertex_locations):
            idx = int(linear_index)
            v_index_flat[idx] = np.uint32(vertex_offset + 1)
            self.vertex_energies[vertex_offset] = self.vertex_energies_raw_flat[idx]
            temp_flat[idx] = np.float64(-np.inf)

        self.initial_locations = [int(loc) for loc in self.vertex_locations[::-1]]
        self.adjacency_matrix = sparse.identity(
            self.number_of_vertices + 1, format="lil", dtype=bool
        )

    def restore_vertex_energy(self, linear_index: int) -> float:
        """Read vertex energy; leave shared -Inf sentinel in place (MATLAB)."""
        current_energy = float(self.energy_temp_flat[linear_index])
        if current_energy == float("-inf"):
            current_energy = float(self.vertex_energies_raw_flat[linear_index])
        return current_energy

    def claim_unowned_strel(
        self,
        *,
        current_vertex_index: int,
        current_scale_label: int,
        current_d_over_r: float,
        valid_linear: np.ndarray,
        strel_pointer_indices: np.ndarray,
        strel_r_over_R: np.ndarray,
        adjusted_energies: np.ndarray,
        size_map_flat: np.ndarray,
        lut_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Claim unowned strel voxels into the shared maps."""
        return _claim_unowned_strel_arrays(
            current_vertex_index=current_vertex_index,
            current_scale_label=current_scale_label,
            current_d_over_r=current_d_over_r,
            valid_linear=valid_linear,
            strel_pointer_indices=strel_pointer_indices,
            strel_r_over_R=strel_r_over_R,
            adjusted_energies=adjusted_energies,
            vertex_index_map_flat=self.vertex_index_flat,
            pointer_map_flat=self.pointer_flat,
            energy_map_flat=self.energy_flat,
            d_over_r_map_flat=self.d_over_r_flat,
            size_map_flat=size_map_flat,
            lut_size=lut_size,
        )
