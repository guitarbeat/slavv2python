"""Typed application run envelope for UI and curation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from slavv_python.schema.results import (
    EdgeSet,
    PipelineResult,
    VertexSet,
    normalize_pipeline_result,
)


@dataclass
class AppRunState(Mapping[str, Any]):
    """Holds typed pipeline results plus optional UI session metadata."""

    pipeline: PipelineResult
    image_shape: tuple[int, ...] | None = None
    dataset_name: str | None = None
    run_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: AppRunState | PipelineResult | Mapping[str, Any]) -> AppRunState:
        if isinstance(value, AppRunState):
            return value
        if isinstance(value, PipelineResult):
            return cls(pipeline=value)
        return cls(pipeline=normalize_pipeline_result(dict(value)))

    def with_pipeline(self, pipeline: PipelineResult) -> AppRunState:
        return AppRunState(
            pipeline=pipeline,
            image_shape=self.image_shape,
            dataset_name=self.dataset_name,
            run_dir=self.run_dir,
            extra=dict(self.extra),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.pipeline.to_dict()

    def __getitem__(self, key: str) -> Any:
        return self.pipeline.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.pipeline.to_dict())

    def __len__(self) -> int:
        return len(self.pipeline.to_dict())


def get_app_run(session: Mapping[str, Any], *, key: str = "processing_results") -> AppRunState:
    """Load an AppRunState from a Streamlit-like session mapping."""
    if key not in session:
        raise KeyError(f"session is missing {key!r}")
    return AppRunState.from_value(session[key])


def counts_from_app_run(app_run: AppRunState) -> dict[str, int]:
    """Return vertex/edge/network counts for curation dashboards."""
    pipeline = app_run.pipeline
    return {
        "Vertices": len(pipeline.vertices.positions) if pipeline.vertices is not None else 0,
        "Edges": len(pipeline.edges.traces) if pipeline.edges is not None else 0,
        "Strands": len(pipeline.network.strands) if pipeline.network is not None else 0,
        "Bifurcations": len(pipeline.network.bifurcations) if pipeline.network is not None else 0,
    }


def rebuild_network_for_curation(
    app_run: AppRunState,
    curated_vertices: Mapping[str, Any],
    curated_edges: Mapping[str, Any],
) -> AppRunState:
    """Rebuild the network after curation while preserving typed stage payloads."""
    from slavv_python.processing.stages.network.manager import NetworkManager

    typed_vertices = VertexSet.from_dict(dict(curated_vertices))
    typed_edges = EdgeSet.from_dict(dict(curated_edges))
    rebuilt_network = NetworkManager.run(
        typed_edges,
        typed_vertices,
        app_run.pipeline.parameters,
    )
    return app_run.with_pipeline(
        PipelineResult(
            parameters=dict(app_run.pipeline.parameters),
            energy_data=app_run.pipeline.energy_data,
            vertices=typed_vertices,
            edges=typed_edges,
            network=rebuilt_network,
            extra=dict(app_run.pipeline.extra),
        )
    )


__all__ = [
    "AppRunState",
    "counts_from_app_run",
    "get_app_run",
    "rebuild_network_for_curation",
]
