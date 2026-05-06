from __future__ import annotations

from source.workflows import (
    PipelineStageStep,
    PreparedPipelineRun,
    finalize_pipeline_results,
    stage_artifacts,
)
from source.workflows.pipeline import (
    PipelineStageStep as GroupedPipelineStageStep,
)
from source.workflows.pipeline import PreparedPipelineRun as GroupedPreparedPipelineRun
from source.workflows.pipeline import (
    finalize_pipeline_results as grouped_finalize_pipeline_results,
)
from source.workflows.pipeline import stage_artifacts as grouped_stage_artifacts
from source.workflows.pipeline.artifacts import stage_artifacts as flat_stage_artifacts
from source.workflows.pipeline.execution import PipelineStageStep as FlatPipelineStageStep
from source.workflows.pipeline.results import (
    finalize_pipeline_results as flat_finalize_pipeline_results,
)
from source.workflows.pipeline.session import (
    PreparedPipelineRun as AliasPreparedPipelineRun,
)
from source.workflows.pipeline_setup import (
    PreparedPipelineRun as FlatPreparedPipelineRun,
)


def test_grouped_workflow_import_surfaces_resolve_consistently():
    assert PipelineStageStep is GroupedPipelineStageStep is FlatPipelineStageStep
    assert PreparedPipelineRun is GroupedPreparedPipelineRun
    assert PreparedPipelineRun is AliasPreparedPipelineRun is FlatPreparedPipelineRun
    assert finalize_pipeline_results is grouped_finalize_pipeline_results
    assert finalize_pipeline_results is flat_finalize_pipeline_results
    assert stage_artifacts is grouped_stage_artifacts is flat_stage_artifacts
