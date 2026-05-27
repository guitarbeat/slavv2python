"""Living state of a pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from slavv_python.schema.results import (
    EdgeSet,
    EnergyResult,
    NetworkResult,
    PipelineResult,
    VertexSet,
)


@dataclass
class RunState:
    """
    Mutable container for pipeline results during execution.
    Provides typed access and centralized state locality.
    """

    parameters: dict[str, Any]
    energy_data: EnergyResult | None = None
    vertices: VertexSet | None = None
    edges: EdgeSet | None = None
    network: NetworkResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_pipeline_result(self) -> PipelineResult:
        """Convert the living state into a final immutable PipelineResult."""
        return PipelineResult(
            parameters=self.parameters,
            energy_data=self.energy_data,
            vertices=self.vertices,
            edges=self.edges,
            network=self.network,
            extra=self.extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the living state into a legacy dictionary payload."""
        return cast("dict[str, Any]", self.to_pipeline_result().to_dict())

    def set_result(self, key: str, payload: Any) -> None:
        """Set a result by key, updating the corresponding typed field if known."""
        if key == "energy_data" and (payload is None or isinstance(payload, EnergyResult)):
            self.energy_data = payload
        elif key == "vertices" and (payload is None or isinstance(payload, VertexSet)):
            self.vertices = payload
        elif key == "edges" and (payload is None or isinstance(payload, EdgeSet)):
            self.edges = payload
        elif key == "network" and (payload is None or isinstance(payload, NetworkResult)):
            self.network = payload
        else:
            self.extra[key] = payload
