from __future__ import annotations

import heapq
import itertools
from typing import cast

import numpy as np
from scipy import sparse

from slavv_python.pipeline.edges.matlab_indexing import (
    _coord_to_matlab_linear_index,
    _matlab_linear_index_to_coord,
)


def _matlab_global_watershed_border_locations(shape: tuple[int, int, int]) -> np.ndarray:
    """Return MATLAB-order linear indices for the image border at ``strel_apothem = 1``."""
    border_mask = np.zeros(shape, dtype=bool)
    border_mask[0, :, :] = True
    border_mask[shape[0] - 1, :, :] = True
    border_mask[:, 0, :] = True
    border_mask[:, shape[1] - 1, :] = True
    border_mask[:, :, 0] = True
    border_mask[:, :, shape[2] - 1] = True
    return cast("np.ndarray", np.flatnonzero(border_mask.ravel(order="F")).astype(np.int64))


class FrontierQueue:
    """Encapsulates MATLAB's exact priority and tie-breaking rules for the watershed frontier.
    Optimized for O(log N) operations using heapq."""

    def __init__(self, initial_locations: list[int], energy_lookup: np.ndarray):
        self._energy_lookup = energy_lookup
        self._heap: list[list] = []
        self._entry_finder: dict[int, list] = {}
        self._counter = itertools.count(1)

        # Populate initial queue
        # Elements at the end of `initial_locations` must pop first (mimicking list.pop()).
        # We assign smaller (more negative) tie-breakers to elements at the end.
        for i, loc in enumerate(initial_locations):
            count = -(i + 1)
            entry = [float(self._energy_lookup[loc]), count, int(loc), False]
            self._entry_finder[int(loc)] = entry
            self._heap.append(entry)

        heapq.heapify(self._heap)

        # Mutate the initial_locations list to maintain exact backward compatibility
        # in case any caller is relying on the side effect of `self._queue = initial_locations`
        # emptying the list.
        initial_locations.clear()

    def __bool__(self) -> bool:
        while self._heap:
            entry = self._heap[0]
            if not entry[3] and self._entry_finder.get(entry[2]) is entry:
                return True
            heapq.heappop(self._heap)
        return False

    def peek_best(self) -> int:
        while self._heap:
            entry = self._heap[0]
            if not entry[3] and self._entry_finder.get(entry[2]) is entry:
                return cast(int, entry[2])
            heapq.heappop(self._heap)
        raise KeyError("peek from an empty priority queue")

    def pop_best(self) -> int:
        while self._heap:
            entry = heapq.heappop(self._heap)
            _energy, _tie, loc, removed = entry
            if not removed:
                if self._entry_finder.get(loc) is entry:
                    del self._entry_finder[loc]
                    return cast(int, loc)
        raise KeyError("pop from an empty priority queue")

    def push(self, location: int, energy: float, seed_idx: int) -> None:
        """Inserts a location into the frontier using MATLAB's exact priority rules."""
        target_energy = float(energy)
        _target_index = int(location)
        count = next(self._counter)

        if seed_idx == 1:
            # Seed 1: FIFO chronological. Older pops first -> smaller tie_breaker.
            tie_breaker = count
        else:
            # Seed >1: LIFO chronological. Newer pops first -> smaller tie_breaker.
            tie_breaker = -count

        entry = [target_energy, tie_breaker, _target_index, False]
        self._entry_finder[_target_index] = entry
        heapq.heappush(self._heap, entry)

    def remove_first_occurrence(self, locations: list[int] | np.ndarray) -> None:
        """Removes the specified locations to handle watershed joins."""
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
        self.vertex_index_map = np.zeros(shape, dtype=np.uint32, order="F")
        self.pointer_map = np.zeros(shape, dtype=np.uint64, order="F")
        self.branch_order_map = np.zeros(shape, dtype=np.uint8, order="F")
        self.d_over_r_map = np.zeros(shape, dtype=np.float64, order="F")

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
        vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(
            np.int32, copy=False
        )
        max_coord: np.ndarray = np.asarray(self.shape, dtype=np.int32) - 1
        vertex_coords = np.clip(vertex_coords, 0, max_coord)
        self.vertex_locations = np.asarray(
            [_coord_to_matlab_linear_index(coord, self.shape) for coord in vertex_coords],
            dtype=np.int64,
        )
        self.number_of_vertices = len(self.vertex_locations)

        border_locations = _matlab_global_watershed_border_locations(self.shape)
        for linear_index in border_locations:
            coord = _matlab_linear_index_to_coord(int(linear_index), self.shape)
            self.vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(
                self.number_of_vertices + 1
            )

        self.vertex_energies = np.empty((self.number_of_vertices,), dtype=np.float64)
        for vertex_offset, linear_index in enumerate(self.vertex_locations):
            coord = _matlab_linear_index_to_coord(int(linear_index), self.shape)
            self.vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(vertex_offset + 1)
            self.vertex_energies[vertex_offset] = np.float64(
                self.energy_map[coord[0], coord[1], coord[2]]
            )
            # vertex energies are encoded as -Inf to ensure their priority selection
            self.energy_map_temp[coord[0], coord[1], coord[2]] = np.float64(-np.inf)

        self.initial_locations = self.vertex_locations.astype(np.int64, copy=False)[::-1].tolist()
        self.adjacency_matrix = sparse.identity(
            self.number_of_vertices + 1, format="lil", dtype=bool
        )

    def restore_vertex_energy(self, linear_index: int) -> float:
        """Restores a vertex's true energy if it was marked as -Inf."""
        current_energy = float(self.energy_temp_flat[linear_index])
        if current_energy == float("-inf"):
            current_energy = float(self.vertex_energies_raw_flat[linear_index])
            self.energy_temp_flat[linear_index] = current_energy
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
        """Reveal one MATLAB strel into the shared maps, claiming only previously unowned voxels."""
        if len(strel_pointer_indices) != len(valid_linear):
            raise AssertionError("Strel arrays must stay aligned")

        vertices_of_current_strel = np.asarray(
            self.vertex_index_flat[valid_linear], dtype=np.uint32
        )
        is_without_vertex = vertices_of_current_strel == 0

        if np.any(is_without_vertex):
            claim_linear = valid_linear[is_without_vertex]
            claim_pointers = np.asarray(strel_pointer_indices[is_without_vertex], dtype=np.uint64)
            if np.any(claim_pointers < 1) or np.any(claim_pointers > lut_size):
                raise AssertionError("Invalid claim pointers")

            self.vertex_index_flat[claim_linear] = np.uint32(current_vertex_index)
            self.pointer_flat[claim_linear] = claim_pointers
            self.energy_flat[claim_linear] = adjusted_energies[is_without_vertex]
            self.d_over_r_flat[claim_linear] = (
                np.asarray(strel_r_over_R[is_without_vertex], dtype=np.float32) + current_d_over_r
            )
            size_map_flat[claim_linear] = np.int16(current_scale_label)

        return vertices_of_current_strel, is_without_vertex
