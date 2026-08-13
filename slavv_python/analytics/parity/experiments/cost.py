"""Cheap-loop gate for a Parity Experiment."""

from __future__ import annotations

from enum import Enum


class HypothesisKind(Enum):
    """What the experiment is trying to falsify."""

    RANKING = "ranking"
    ARTIFACT_CLASS = "artifact_class"
    PAIR_SET = "pair_set"
    GENERATION = "generation"
    OWNERSHIP = "ownership"


class ExperimentCost(Enum):
    """Allowed cost ladder, cheapest first."""

    UNIT = "unit"
    CROP_PAIR_SET = "crop_pair_set"
    NO_WRITER_RESELECT = "no_writer_reselect"
    FULL_WRITER = "full_writer"


class CheapLoopError(ValueError):
    """Raised when the requested cost is too expensive for the hypothesis."""


_CHEAP_ONLY = frozenset(
    {
        HypothesisKind.RANKING,
        HypothesisKind.ARTIFACT_CLASS,
        HypothesisKind.PAIR_SET,
    }
)


def require_cheap_loop(*, hypothesis_kind: HypothesisKind, requested_cost: ExperimentCost) -> None:
    """Refuse a full Edges writer when a cheaper adapter can falsify.

    Ranking, artifact-class, and pair-set questions stay on unit / crop /
    no-writer re-selection. Generation and ownership may request a writer.
    """
    if hypothesis_kind in _CHEAP_ONLY and requested_cost is ExperimentCost.FULL_WRITER:
        raise CheapLoopError(
            f"{hypothesis_kind.value} hypothesis cannot request a full writer; "
            "use unit, crop pair-set, or no-writer re-selection"
        )
