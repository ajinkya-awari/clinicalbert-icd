"""Validation-only deterministic selection of one global threshold."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from clinicalbert_icd.evaluation.metrics import (
    _micro_f1,
    _probability,
    _validated_matrices,
)


@dataclass(frozen=True, slots=True)
class TaggedScores:
    """Labels and scores carrying an explicit partition tag."""

    partition: str
    y_true: Sequence[Sequence[int]]
    scores: Sequence[Sequence[float]]


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """Aggregate validation evidence for a deterministic threshold choice."""

    threshold: float
    objective: str
    objective_value: float
    tie_break: str
    candidate_scores: tuple[tuple[float, float], ...]


def select_global_threshold(
    validation_scores: TaggedScores,
    candidates: Iterable[float],
    objective: str,
    tie_break: str,
) -> ThresholdSelection:
    """Select a threshold from validation-tagged data without accepting test input."""

    if not isinstance(validation_scores, TaggedScores):
        raise ValueError("validation_scores_invalid")
    if validation_scores.partition != "validation":
        raise ValueError("validation_partition_required")
    if objective != "micro_f1":
        raise ValueError("objective_not_supported")
    if tie_break not in {"lowest_threshold", "highest_threshold"}:
        raise ValueError("tie_break_not_supported")

    truth, scores = _validated_matrices(
        validation_scores.y_true,
        validation_scores.scores,
        reason="score_matrix_invalid",
    )
    candidate_values = _validated_candidates(candidates)
    scored = []
    for candidate in candidate_values:
        predictions = tuple(
            tuple(1 if score >= candidate else 0 for score in row) for row in scores
        )
        scored.append((candidate, _micro_f1(truth, predictions)))

    best_value = max(value for _, value in scored)
    tied = tuple(threshold for threshold, value in scored if value == best_value)
    selected = min(tied) if tie_break == "lowest_threshold" else max(tied)
    return ThresholdSelection(
        threshold=selected,
        objective=objective,
        objective_value=best_value,
        tie_break=tie_break,
        candidate_scores=tuple(scored),
    )


def _validated_candidates(candidates: Iterable[float]) -> tuple[float, ...]:
    try:
        raw_values = tuple(candidates)
    except TypeError:
        raise ValueError("threshold_candidates_invalid") from None
    try:
        values = tuple(
            _probability(item, reason="threshold_candidates_invalid")
            for item in raw_values
        )
    except ValueError:
        raise ValueError("threshold_candidates_invalid") from None
    if not values or len(values) != len(set(values)):
        raise ValueError("threshold_candidates_invalid")
    return tuple(sorted(values))
