"""State-transition tests for one-use untouched-test evaluation."""

from __future__ import annotations

import unittest

from clinicalbert_icd.evaluation.metrics import MetricsResult
from clinicalbert_icd.evaluation.test_session import UntouchedTestSession


def synthetic_result() -> MetricsResult:
    return MetricsResult(
        micro_f1=0.5,
        macro_f1=0.4,
        precision_at_k=((5, 0.25),),
        per_label_support=(1, 2),
        sample_count=2,
        label_count=2,
    )


class UntouchedTestSessionTests(unittest.TestCase):
    def test_evaluator_runs_exactly_once(self) -> None:
        session = UntouchedTestSession()
        calls = 0

        def evaluator() -> MetricsResult:
            nonlocal calls
            calls += 1
            return synthetic_result()

        result = session.evaluate_once(evaluator)

        self.assertEqual(result, synthetic_result())
        self.assertEqual(calls, 1)
        self.assertTrue(session.used)
        with self.assertRaisesRegex(RuntimeError, "test_session_already_used"):
            session.evaluate_once(evaluator)
        self.assertEqual(calls, 1)

    def test_failed_evaluator_still_consumes_session(self) -> None:
        session = UntouchedTestSession()

        def failing_evaluator() -> MetricsResult:
            raise RuntimeError("synthetic_evaluator_failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic_evaluator_failure"):
            session.evaluate_once(failing_evaluator)
        self.assertTrue(session.used)
        with self.assertRaisesRegex(RuntimeError, "test_session_already_used"):
            session.evaluate_once(synthetic_result)

    def test_non_metric_result_is_rejected_after_consuming_session(self) -> None:
        session = UntouchedTestSession()

        with self.assertRaisesRegex(TypeError, "test_evaluator_result_invalid"):
            session.evaluate_once(lambda: object())
        self.assertTrue(session.used)


if __name__ == "__main__":
    unittest.main()
