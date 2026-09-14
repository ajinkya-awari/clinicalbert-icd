"""Known literal cases for deterministic multi-label metrics."""

from __future__ import annotations

import unittest

from clinicalbert_icd.evaluation.metrics import multilabel_metrics


Y_TRUE = ((1, 0, 1), (0, 1, 0))
SCORES = ((0.9, 0.8, 0.7), (0.2, 0.6, 0.1))


class MultilabelMetricTests(unittest.TestCase):
    def test_known_micro_macro_support_and_ranked_precision_values(self) -> None:
        result = multilabel_metrics(
            Y_TRUE,
            SCORES,
            threshold=0.5,
            ks=(1, 5),
            short_label_policy="available",
        )

        self.assertAlmostEqual(result.micro_f1, 6 / 7)
        self.assertAlmostEqual(result.macro_f1, 8 / 9)
        self.assertEqual(result.per_label_support, (1, 1, 1))
        self.assertEqual(result.precision_at_k[0], (1, 1.0))
        self.assertAlmostEqual(result.precision_at_k[1][1], 0.5)
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.label_count, 3)

    def test_short_label_denominator_policy_is_explicit(self) -> None:
        available = multilabel_metrics(
            Y_TRUE,
            SCORES,
            threshold=0.5,
            ks=(5,),
            short_label_policy="available",
        )
        fixed_k = multilabel_metrics(
            Y_TRUE,
            SCORES,
            threshold=0.5,
            ks=(5,),
            short_label_policy="k",
        )

        self.assertAlmostEqual(available.precision_at_k[0][1], 0.5)
        self.assertAlmostEqual(fixed_k.precision_at_k[0][1], 0.3)

    def test_rank_ties_use_label_index_for_determinism(self) -> None:
        result = multilabel_metrics(
            ((0, 1),),
            ((0.5, 0.5),),
            threshold=0.75,
            ks=(1,),
            short_label_policy="available",
        )

        self.assertEqual(result.precision_at_k, ((1, 0.0),))

    def test_zero_division_is_stable_when_no_positive_labels_exist(self) -> None:
        result = multilabel_metrics(
            ((0, 0), (0, 0)),
            ((0.1, 0.2), (0.3, 0.4)),
            threshold=0.9,
            ks=(1,),
            short_label_policy="available",
        )

        self.assertEqual(result.micro_f1, 0.0)
        self.assertEqual(result.macro_f1, 0.0)
        self.assertEqual(result.per_label_support, (0, 0))

    def test_invalid_matrix_threshold_k_or_policy_is_rejected(self) -> None:
        cases = (
            (((1,),), ((0.5, 0.4),), 0.5, (1,), "available", "metric_matrix_invalid"),
            (((1,),), ((0.5,),), 1.5, (1,), "available", "threshold_invalid"),
            (((1,),), ((0.5,),), 0.5, (0,), "available", "precision_k_invalid"),
            (((1,),), ((0.5,),), 0.5, (1,), "implicit", "short_label_policy_invalid"),
        )
        for y_true, scores, threshold, ks, policy, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    multilabel_metrics(
                        y_true,
                        scores,
                        threshold=threshold,
                        ks=ks,
                        short_label_policy=policy,
                    )

    def test_huge_integers_and_infinity_raise_named_validation_errors(self) -> None:
        huge = 10**10000
        for invalid_score in (huge, float("inf"), float("-inf")):
            with self.subTest(score=type(invalid_score).__name__):
                with self.assertRaisesRegex(ValueError, "metric_matrix_invalid"):
                    multilabel_metrics(
                        ((1,),),
                        ((invalid_score,),),
                        threshold=0.5,
                        ks=(1,),
                        short_label_policy="available",
                    )
        for invalid_threshold in (huge, float("inf"), float("-inf")):
            with self.subTest(threshold=type(invalid_threshold).__name__):
                with self.assertRaisesRegex(ValueError, "threshold_invalid"):
                    multilabel_metrics(
                        ((1,),),
                        ((0.5,),),
                        threshold=invalid_threshold,
                        ks=(1,),
                        short_label_policy="available",
                    )


if __name__ == "__main__":
    unittest.main()
