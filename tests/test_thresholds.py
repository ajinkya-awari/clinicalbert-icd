"""Synthetic tests for validation-only global-threshold selection."""

from __future__ import annotations

import unittest

from clinicalbert_icd.evaluation.thresholds import (
    TaggedScores,
    select_global_threshold,
)


def validation_batch() -> TaggedScores:
    return TaggedScores(
        partition="validation",
        y_true=((1,), (0,)),
        scores=((0.6,), (0.1,)),
    )


class GlobalThresholdTests(unittest.TestCase):
    def test_only_validation_tagged_scores_are_accepted(self) -> None:
        test_batch = TaggedScores(
            partition="test",
            y_true=((1,),),
            scores=((0.9,),),
        )

        with self.assertRaisesRegex(ValueError, "validation_partition_required"):
            select_global_threshold(
                test_batch,
                candidates=(0.25, 0.5),
                objective="micro_f1",
                tie_break="lowest_threshold",
            )

    def test_micro_f1_selection_records_each_candidate(self) -> None:
        selection = select_global_threshold(
            validation_batch(),
            candidates=(0.25, 0.5, 0.75),
            objective="micro_f1",
            tie_break="lowest_threshold",
        )

        self.assertEqual(selection.threshold, 0.25)
        self.assertEqual(selection.objective, "micro_f1")
        self.assertEqual(selection.objective_value, 1.0)
        self.assertEqual(
            selection.candidate_scores,
            ((0.25, 1.0), (0.5, 1.0), (0.75, 0.0)),
        )

    def test_tie_break_is_explicit_and_deterministic(self) -> None:
        lowest = select_global_threshold(
            validation_batch(),
            candidates=(0.5, 0.25),
            objective="micro_f1",
            tie_break="lowest_threshold",
        )
        highest = select_global_threshold(
            validation_batch(),
            candidates=(0.25, 0.5),
            objective="micro_f1",
            tie_break="highest_threshold",
        )

        self.assertEqual(lowest.threshold, 0.25)
        self.assertEqual(highest.threshold, 0.5)

    def test_unsupported_objective_or_tie_break_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "objective_not_supported"):
            select_global_threshold(
                validation_batch(),
                candidates=(0.5,),
                objective="accuracy",
                tie_break="lowest_threshold",
            )
        with self.assertRaisesRegex(ValueError, "tie_break_not_supported"):
            select_global_threshold(
                validation_batch(),
                candidates=(0.5,),
                objective="micro_f1",
                tie_break="first_seen",
            )

    def test_invalid_shapes_scores_or_candidates_are_rejected(self) -> None:
        malformed = TaggedScores(
            partition="validation",
            y_true=((1, 0),),
            scores=((0.5,),),
        )

        with self.assertRaisesRegex(ValueError, "score_matrix_invalid"):
            select_global_threshold(
                malformed,
                candidates=(0.5,),
                objective="micro_f1",
                tie_break="lowest_threshold",
            )
        with self.assertRaisesRegex(ValueError, "threshold_candidates_invalid"):
            select_global_threshold(
                validation_batch(),
                candidates=(),
                objective="micro_f1",
                tie_break="lowest_threshold",
            )

    def test_huge_integers_and_infinity_raise_named_validation_errors(self) -> None:
        huge = 10**10000
        for invalid_score in (huge, float("inf"), float("-inf")):
            batch = TaggedScores(
                partition="validation",
                y_true=((1,),),
                scores=((invalid_score,),),
            )
            with self.subTest(score=type(invalid_score).__name__):
                with self.assertRaisesRegex(ValueError, "score_matrix_invalid"):
                    select_global_threshold(
                        batch,
                        candidates=(0.5,),
                        objective="micro_f1",
                        tie_break="lowest_threshold",
                    )
        for invalid_candidate in (huge, float("inf"), float("-inf")):
            with self.subTest(candidate=type(invalid_candidate).__name__):
                with self.assertRaisesRegex(ValueError, "threshold_candidates_invalid"):
                    select_global_threshold(
                        validation_batch(),
                        candidates=(invalid_candidate,),
                        objective="micro_f1",
                        tie_break="lowest_threshold",
                    )


if __name__ == "__main__":
    unittest.main()
