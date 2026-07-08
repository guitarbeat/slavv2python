"""Execution tracing for watershed discovery algorithms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import Protocol, runtime_checkable


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


class JsonExecutionTracer:
    """Records execution events to a JSONL file for offline comparison."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("")
        self._iteration = 0

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
