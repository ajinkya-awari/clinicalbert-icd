"""A data-free one-use state transition for final test evaluation."""

from __future__ import annotations

from collections.abc import Callable

from clinicalbert_icd.evaluation.metrics import MetricsResult


class UntouchedTestSession:
    """Permit exactly one evaluator invocation without storing test data."""

    __slots__ = ("_used",)

    def __init__(self) -> None:
        self._used = False

    @property
    def used(self) -> bool:
        """Return whether the one-use transition has been consumed."""

        return self._used

    def evaluate_once(self, evaluator: Callable[[], MetricsResult]) -> MetricsResult:
        """Consume the session before invoking the supplied aggregate evaluator."""

        if self._used:
            raise RuntimeError("test_session_already_used")
        self._used = True
        if not callable(evaluator):
            raise TypeError("test_evaluator_invalid")
        result = evaluator()
        if not isinstance(result, MetricsResult):
            raise TypeError("test_evaluator_result_invalid")
        return result
