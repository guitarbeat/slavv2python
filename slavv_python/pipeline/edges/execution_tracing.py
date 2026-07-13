"""Execution tracing for watershed discovery algorithms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable


@runtime_checkable
class ExecutionTracer(Protocol):
    """Protocol for recording internal algorithm events during watershed discovery."""

    def on_iteration_start(
        self, iteration: int, current_linear: int, current_energy: float
    ) -> None: ...

    def on_seed_selected(
        self, seed_idx: int, selected_linear: int, selected_energy: float
    ) -> None: ...

    def on_join(
        self, start_vertex: int, end_vertex: int, half_1: list[int], half_2: list[int]
    ) -> None: ...

    def on_join_skipped(
        self,
        start_vertex: int,
        end_vertex: int,
        *,
        reason: str,
        iteration: int,
        current_linear: int,
    ) -> None: ...

    def on_strel_state(
        self,
        *,
        iteration: int,
        current_linear: int,
        current_vertex_index: int,
        current_scale_label: int,
        current_pointer_value: int,
        current_d_over_r: float,
        strel_linear: np.ndarray,
        strel_pointer_indices: np.ndarray,
        strel_r_over_R: np.ndarray,
        raw_energies: np.ndarray,
        adjusted_energies: np.ndarray,
        vertices_of_current_strel: np.ndarray,
        is_without_vertex: np.ndarray,
        pointer_values: np.ndarray,
        d_over_r_values: np.ndarray,
        size_values: np.ndarray,
    ) -> None: ...

    def on_frontier_state(
        self,
        *,
        iteration: int,
        label: str,
        current_linear: int,
        snapshot: dict[str, Any],
    ) -> None: ...

    def frontier_state_targets(self) -> set[int]: ...


class JsonExecutionTracer:
    """Records execution events to a JSONL file for offline comparison."""

    def __init__(
        self,
        output_path: Path | str,
        *,
        state_iterations: Iterable[int] | None = None,
        state_linear_targets: Iterable[int] | None = None,
        state_sample_limit: int = 12,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("")
        self._iteration = 0
        self._state_iterations = {int(value) for value in state_iterations or ()}
        self._state_linear_targets = {int(value) for value in state_linear_targets or ()}
        self._state_sample_limit = max(1, int(state_sample_limit))

    def _append(self, event_type: str, data: dict[str, Any]) -> None:
        payload = {"event": event_type, **data}
        with self.output_path.open("a") as f:
            f.write(json.dumps(payload, default=self._json_default) + "\n")

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def on_iteration_start(
        self, iteration: int, current_linear: int, current_energy: float
    ) -> None:
        self._iteration = int(iteration)
        self._append(
            "iteration_start",
            {
                "iteration": iteration,
                "current_linear": current_linear,
                "current_energy": current_energy,
            },
        )

    def on_seed_selected(self, seed_idx: int, selected_linear: int, selected_energy: float) -> None:
        self._append(
            "seed_selected",
            {
                "iteration": self._iteration,
                "seed_idx": seed_idx,
                "selected_linear": selected_linear,
                "selected_energy": selected_energy,
            },
        )

    def on_join(
        self, start_vertex: int, end_vertex: int, half_1: list[int], half_2: list[int]
    ) -> None:
        self._append(
            "join",
            {
                "start_vertex": start_vertex,
                "end_vertex": end_vertex,
                "half_1_len": len(half_1),
                "half_2_len": len(half_2),
                "half_1_sample": half_1[:5],
                "half_2_sample": half_2[:5],
            },
        )

    def on_join_skipped(
        self,
        start_vertex: int,
        end_vertex: int,
        *,
        reason: str,
        iteration: int,
        current_linear: int,
    ) -> None:
        self._append(
            "join_skipped",
            {
                "start_vertex": start_vertex,
                "end_vertex": end_vertex,
                "reason": reason,
                "iteration": iteration,
                "current_linear": current_linear,
            },
        )

    def on_strel_state(
        self,
        *,
        iteration: int,
        current_linear: int,
        current_vertex_index: int,
        current_scale_label: int,
        current_pointer_value: int,
        current_d_over_r: float,
        strel_linear: np.ndarray,
        strel_pointer_indices: np.ndarray,
        strel_r_over_R: np.ndarray,
        raw_energies: np.ndarray,
        adjusted_energies: np.ndarray,
        vertices_of_current_strel: np.ndarray,
        is_without_vertex: np.ndarray,
        pointer_values: np.ndarray,
        d_over_r_values: np.ndarray,
        size_values: np.ndarray,
    ) -> None:
        if not self._should_trace_strel_state(iteration, strel_linear):
            return

        strel_linear_arr = np.asarray(strel_linear, dtype=np.int64).reshape(-1)
        adjusted_arr = np.asarray(adjusted_energies, dtype=np.float64).reshape(-1)
        raw_arr = np.asarray(raw_energies, dtype=np.float64).reshape(-1)
        vertices_arr = np.asarray(vertices_of_current_strel, dtype=np.int64).reshape(-1)
        without_arr = np.asarray(is_without_vertex, dtype=bool).reshape(-1)
        pointer_arr = np.asarray(pointer_values, dtype=np.int64).reshape(-1)
        d_over_r_arr = np.asarray(d_over_r_values, dtype=np.float64).reshape(-1)
        size_arr = np.asarray(size_values, dtype=np.int64).reshape(-1)
        strel_pointer_arr = np.asarray(strel_pointer_indices, dtype=np.int64).reshape(-1)
        strel_r_arr = np.asarray(strel_r_over_R, dtype=np.float64).reshape(-1)

        finite_adjusted = np.nan_to_num(
            adjusted_arr,
            nan=np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )
        order = np.lexsort((strel_linear_arr, finite_adjusted))
        sample_indices = order[: self._state_sample_limit]

        target_indices = [
            int(index)
            for index, linear in enumerate(strel_linear_arr)
            if int(linear) in self._state_linear_targets
        ]

        self._append(
            "strel_state",
            {
                "iteration": int(iteration),
                "current_linear": int(current_linear),
                "current_vertex_index": int(current_vertex_index),
                "current_scale_label": int(current_scale_label),
                "current_pointer_value": int(current_pointer_value),
                "current_d_over_r": float(current_d_over_r),
                "top_adjusted": self._strel_rows(
                    sample_indices,
                    strel_linear_arr=strel_linear_arr,
                    strel_pointer_arr=strel_pointer_arr,
                    strel_r_arr=strel_r_arr,
                    raw_arr=raw_arr,
                    adjusted_arr=adjusted_arr,
                    vertices_arr=vertices_arr,
                    without_arr=without_arr,
                    pointer_arr=pointer_arr,
                    d_over_r_arr=d_over_r_arr,
                    size_arr=size_arr,
                ),
                "targets": self._strel_rows(
                    target_indices,
                    strel_linear_arr=strel_linear_arr,
                    strel_pointer_arr=strel_pointer_arr,
                    strel_r_arr=strel_r_arr,
                    raw_arr=raw_arr,
                    adjusted_arr=adjusted_arr,
                    vertices_arr=vertices_arr,
                    without_arr=without_arr,
                    pointer_arr=pointer_arr,
                    d_over_r_arr=d_over_r_arr,
                    size_arr=size_arr,
                ),
            },
        )

    def on_frontier_state(
        self,
        *,
        iteration: int,
        label: str,
        current_linear: int,
        snapshot: dict[str, Any],
    ) -> None:
        if not self._should_trace_frontier_state(iteration, snapshot):
            return
        self._append(
            "frontier_state",
            {
                "iteration": int(iteration),
                "label": str(label),
                "current_linear": int(current_linear),
                "snapshot": snapshot,
            },
        )

    def frontier_state_targets(self) -> set[int]:
        return set(self._state_linear_targets)

    def _should_trace_strel_state(self, iteration: int, strel_linear: np.ndarray) -> bool:
        if int(iteration) in self._state_iterations:
            return True
        if not self._state_linear_targets:
            return False
        linear_values = set(np.asarray(strel_linear, dtype=np.int64).reshape(-1).tolist())
        return bool(linear_values & self._state_linear_targets)

    def _should_trace_frontier_state(self, iteration: int, snapshot: dict[str, Any]) -> bool:
        if int(iteration) in self._state_iterations:
            return True
        if not self._state_linear_targets:
            return False
        target_counts = snapshot.get("target_counts")
        if not isinstance(target_counts, dict):
            return False
        return any(int(count) > 0 for count in target_counts.values())

    @staticmethod
    def _strel_rows(
        indices: Iterable[int],
        *,
        strel_linear_arr: np.ndarray,
        strel_pointer_arr: np.ndarray,
        strel_r_arr: np.ndarray,
        raw_arr: np.ndarray,
        adjusted_arr: np.ndarray,
        vertices_arr: np.ndarray,
        without_arr: np.ndarray,
        pointer_arr: np.ndarray,
        d_over_r_arr: np.ndarray,
        size_arr: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index in indices:
            idx = int(index)
            rows.append(
                {
                    "strel_index": idx,
                    "linear": int(strel_linear_arr[idx]),
                    "strel_pointer": int(strel_pointer_arr[idx]),
                    "strel_r_over_R": float(strel_r_arr[idx]),
                    "raw_energy": float(raw_arr[idx]),
                    "adjusted_energy": float(adjusted_arr[idx]),
                    "vertex_index_before_claim": int(vertices_arr[idx]),
                    "is_without_vertex": bool(without_arr[idx]),
                    "pointer_before_claim": int(pointer_arr[idx]),
                    "d_over_r_before_claim": float(d_over_r_arr[idx]),
                    "size_before_claim": int(size_arr[idx]),
                }
            )
        return rows


class NullExecutionTracer:
    """No-op tracer for production use."""

    def on_iteration_start(
        self, iteration: int, current_linear: int, current_energy: float
    ) -> None:
        pass

    def on_seed_selected(self, seed_idx: int, selected_linear: int, selected_energy: float) -> None:
        pass

    def on_join(
        self, start_vertex: int, end_vertex: int, half_1: list[int], half_2: list[int]
    ) -> None:
        pass

    def on_join_skipped(
        self,
        start_vertex: int,
        end_vertex: int,
        *,
        reason: str,
        iteration: int,
        current_linear: int,
    ) -> None:
        pass

    def on_strel_state(
        self,
        *,
        iteration: int,
        current_linear: int,
        current_vertex_index: int,
        current_scale_label: int,
        current_pointer_value: int,
        current_d_over_r: float,
        strel_linear: np.ndarray,
        strel_pointer_indices: np.ndarray,
        strel_r_over_R: np.ndarray,
        raw_energies: np.ndarray,
        adjusted_energies: np.ndarray,
        vertices_of_current_strel: np.ndarray,
        is_without_vertex: np.ndarray,
        pointer_values: np.ndarray,
        d_over_r_values: np.ndarray,
        size_values: np.ndarray,
    ) -> None:
        pass

    def on_frontier_state(
        self,
        *,
        iteration: int,
        label: str,
        current_linear: int,
        snapshot: dict[str, Any],
    ) -> None:
        pass

    def frontier_state_targets(self) -> set[int]:
        return set()
