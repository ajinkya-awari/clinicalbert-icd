"""Deterministic aggregate multi-label metrics over in-memory numeric matrices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class MetricsResult:
    """Aggregate-only metric values; no row-level predictions are retained."""

    micro_f1: float
    macro_f1: float
    precision_at_k: tuple[tuple[int, float], ...]
    per_label_support: tuple[int, ...]
    sample_count: int
    label_count: int


def multilabel_metrics(
    y_true: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    threshold: float,
    ks: Iterable[int],
    short_label_policy: str,
) -> MetricsResult:
    """Compute F1, ranked descending-score P@K, and label support."""

    truth, score_rows = _validated_matrices(
        y_true, scores, reason="metric_matrix_invalid"
    )
    threshold_value = _probability(threshold, reason="threshold_invalid")
    k_values = _validated_ks(ks)
    if short_label_policy not in {"available", "k"}:
        raise ValueError("short_label_policy_invalid")

    predictions = tuple(
        tuple(1 if score >= threshold_value else 0 for score in row)
        for row in score_rows
    )
    micro_f1 = _micro_f1(truth, predictions)
    label_count = len(truth[0])
    label_f1 = []
    support = []
    for label_index in range(label_count):
        label_truth = tuple(row[label_index] for row in truth)
        label_predictions = tuple(row[label_index] for row in predictions)
        label_f1.append(_binary_f1(label_truth, label_predictions))
        support.append(sum(label_truth))

    precision_values = tuple(
        (
            k,
            _ranked_precision_at_k(
                truth,
                score_rows,
                k=k,
                short_label_policy=short_label_policy,
            ),
        )
        for k in k_values
    )
    return MetricsResult(
        micro_f1=micro_f1,
        macro_f1=sum(label_f1) / label_count,
        precision_at_k=precision_values,
        per_label_support=tuple(support),
        sample_count=len(truth),
        label_count=label_count,
    )


def _validated_matrices(
    y_true: object,
    scores: object,
    *,
    reason: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    try:
        truth = tuple(tuple(row) for row in y_true)  # type: ignore[union-attr]
        score_rows = tuple(tuple(row) for row in scores)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        raise ValueError(reason) from None

    if not truth or len(truth) != len(score_rows) or not truth[0]:
        raise ValueError(reason)
    label_count = len(truth[0])
    if any(len(row) != label_count for row in truth + score_rows):
        raise ValueError(reason)
    if any(type(item) is not int or item not in (0, 1) for row in truth for item in row):
        raise ValueError(reason)
    normalized_scores: list[tuple[float, ...]] = []
    for row in score_rows:
        normalized_row = []
        for item in row:
            normalized_row.append(_probability(item, reason=reason))
        normalized_scores.append(tuple(normalized_row))
    return (
        tuple(tuple(int(item) for item in row) for row in truth),
        tuple(normalized_scores),
    )


def _probability(value: object, *, reason: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(reason)
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise ValueError(reason) from None
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(reason)
    return normalized


def _validated_ks(ks: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(ks)
    except TypeError:
        raise ValueError("precision_k_invalid") from None
    if (
        not values
        or len(values) != len(set(values))
        or any(type(value) is not int or value <= 0 for value in values)
    ):
        raise ValueError("precision_k_invalid")
    return values


def _micro_f1(
    truth: tuple[tuple[int, ...], ...],
    predictions: tuple[tuple[int, ...], ...],
) -> float:
    flat_truth = tuple(item for row in truth for item in row)
    flat_predictions = tuple(item for row in predictions for item in row)
    return _binary_f1(flat_truth, flat_predictions)


def _binary_f1(truth: Sequence[int], predictions: Sequence[int]) -> float:
    true_positive = sum(
        expected == 1 and predicted == 1
        for expected, predicted in zip(truth, predictions, strict=True)
    )
    false_positive = sum(
        expected == 0 and predicted == 1
        for expected, predicted in zip(truth, predictions, strict=True)
    )
    false_negative = sum(
        expected == 1 and predicted == 0
        for expected, predicted in zip(truth, predictions, strict=True)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2 * true_positive) / denominator


def _ranked_precision_at_k(
    truth: tuple[tuple[int, ...], ...],
    scores: tuple[tuple[float, ...], ...],
    *,
    k: int,
    short_label_policy: str,
) -> float:
    label_count = len(truth[0])
    available = min(k, label_count)
    denominator = available if short_label_policy == "available" else k
    sample_values = []
    for expected, score_row in zip(truth, scores, strict=True):
        ranked = sorted(range(label_count), key=lambda index: (-score_row[index], index))
        relevant = sum(expected[index] for index in ranked[:available])
        sample_values.append(relevant / denominator)
    return sum(sample_values) / len(sample_values)
