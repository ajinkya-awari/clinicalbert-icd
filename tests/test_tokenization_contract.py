from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, asdict

from clinicalbert_icd.data.contracts import (
    TokenizationAudit,
    TokenizedBatch,
    TokenizerConfig,
)
from clinicalbert_icd.data.tokenize import tokenize_batch
from tests.fixtures.fake_tokenizer import FakeTokenizer


def _config(
    *,
    revision: str = "test-v1",
    maximum_length: int = 5,
    truncation_side: str = "right",
    padding: str = "max_length",
) -> TokenizerConfig:
    return TokenizerConfig(
        identifier="synthetic-whitespace-tokenizer",
        revision=revision,
        maximum_length=maximum_length,
        truncation_side=truncation_side,
        padding=padding,
    )


class OfflineTokenizationContractTests(unittest.TestCase):
    def test_below_limit_accounts_for_special_tokens_and_padding(self) -> None:
        batch = tokenize_batch(("one two",), FakeTokenizer(), _config())

        self.assertEqual(batch.input_ids, ((101, 11, 12, 102, 0),))
        self.assertEqual(batch.attention_mask, ((1, 1, 1, 1, 0),))
        self.assertEqual(batch.audit.truncated_example_count, 0)
        self.assertEqual(batch.audit.total_content_tokens_before, 2)
        self.assertEqual(batch.audit.total_content_tokens_after, 2)
        self.assertEqual(batch.audit.special_tokens_per_example, 2)

    def test_exact_limit_is_not_reported_as_truncated(self) -> None:
        batch = tokenize_batch(("one two three",), FakeTokenizer(), _config())

        self.assertEqual(batch.input_ids, ((101, 11, 12, 13, 102),))
        self.assertEqual(batch.attention_mask, ((1, 1, 1, 1, 1),))
        self.assertEqual(batch.audit.truncated_example_count, 0)

    def test_above_limit_right_truncation_keeps_leading_content(self) -> None:
        batch = tokenize_batch(
            ("one two three four five",), FakeTokenizer(), _config()
        )

        self.assertEqual(batch.input_ids, ((101, 11, 12, 13, 102),))
        self.assertEqual(batch.audit.truncated_example_count, 1)
        self.assertEqual(batch.audit.total_content_tokens_before, 5)
        self.assertEqual(batch.audit.total_content_tokens_after, 3)

    def test_above_limit_left_truncation_keeps_trailing_content(self) -> None:
        batch = tokenize_batch(
            ("one two three four five",),
            FakeTokenizer(),
            _config(truncation_side="left"),
        )

        self.assertEqual(batch.input_ids, ((101, 13, 14, 15, 102),))
        self.assertEqual(batch.audit.truncated_example_count, 1)

    def test_missing_or_mismatched_tokenizer_revision_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "revision"):
            _config(revision="")
        with self.assertRaisesRegex(ValueError, "revision"):
            tokenize_batch(
                ("one",), FakeTokenizer(revision="other-revision"), _config()
            )

    def test_revision_contract_allows_only_explicit_fixture_or_content_addresses(self) -> None:
        valid = (
            ("synthetic-whitespace-tokenizer", "test-v1"),
            ("synthetic-whitespace-tokenizer", "test:fixture-v2"),
            ("future-real-tokenizer", "commit:" + "a" * 40),
            ("future-real-tokenizer", "sha256:" + "b" * 64),
        )
        for identifier, revision in valid:
            with self.subTest(identifier=identifier, revision=revision):
                TokenizerConfig(identifier, revision, 8, "right", "max_length")

        invalid = (
            ("future-real-tokenizer", "main"),
            ("future-real-tokenizer", "latest"),
            ("future-real-tokenizer", "release-branch"),
            ("future-real-tokenizer", "commit:main"),
            ("future-real-tokenizer", "commit:" + "a" * 39),
            ("future-real-tokenizer", "sha256:" + "b" * 63),
            ("future-real-tokenizer", "test-v1"),
            ("future-real-tokenizer", "test:fixture-v2"),
        )
        for identifier, revision in invalid:
            with self.subTest(identifier=identifier, revision=revision):
                with self.assertRaisesRegex(ValueError, "revision"):
                    TokenizerConfig(identifier, revision, 8, "right", "max_length")

    def test_maximum_length_must_leave_room_for_special_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "special"):
            tokenize_batch(("one",), FakeTokenizer(), _config(maximum_length=1))

    def test_padding_policy_is_explicit_and_do_not_pad_preserves_length(self) -> None:
        batch = tokenize_batch(
            ("one",), FakeTokenizer(), _config(padding="do_not_pad")
        )

        self.assertEqual(batch.input_ids, ((101, 11, 102),))
        self.assertEqual(batch.attention_mask, ((1, 1, 1),))

    def test_empty_tokenization_input_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            tokenize_batch((), FakeTokenizer(), _config())

    def test_direct_tokenized_batch_rejects_bad_masks_or_audit_counts(self) -> None:
        audit = TokenizationAudit(
            tokenizer_identifier="synthetic-whitespace-tokenizer",
            tokenizer_revision="test-v1",
            maximum_length=5,
            truncation_side="right",
            padding="max_length",
            example_count=1,
            truncated_example_count=0,
            total_content_tokens_before=1,
            total_content_tokens_after=1,
            special_tokens_per_example=2,
        )
        bad_audit = TokenizationAudit(
            tokenizer_identifier="synthetic-whitespace-tokenizer",
            tokenizer_revision="test-v1",
            maximum_length=5,
            truncation_side="right",
            padding="max_length",
            example_count=2,
            truncated_example_count=0,
            total_content_tokens_before=1,
            total_content_tokens_after=1,
            special_tokens_per_example=2,
        )

        with self.assertRaisesRegex(ValueError, "mask"):
            TokenizedBatch(
                input_ids=((101, 11, 102, 0, 0),),
                attention_mask=((1, 1),),
                audit=audit,
            )
        with self.assertRaisesRegex(ValueError, "audit"):
            TokenizedBatch(
                input_ids=((101, 11, 102, 0, 0),),
                attention_mask=((1, 1, 1, 0, 0),),
                audit=bad_audit,
            )

    def test_config_batch_and_audit_are_immutable_and_audit_has_no_text(self) -> None:
        config = _config()
        batch = tokenize_batch(("one two", "three four five six"), FakeTokenizer(), config)

        with self.assertRaises(FrozenInstanceError):
            config.maximum_length = 99  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            batch.audit.truncated_example_count = 0  # type: ignore[misc]

        self.assertEqual(
            set(asdict(batch.audit)),
            {
                "tokenizer_identifier",
                "tokenizer_revision",
                "maximum_length",
                "truncation_side",
                "padding",
                "example_count",
                "truncated_example_count",
                "total_content_tokens_before",
                "total_content_tokens_after",
                "special_tokens_per_example",
            },
        )
        self.assertNotIn("one two", repr(batch.audit))
        self.assertNotIn("three four", repr(batch.audit))

    def test_invalid_policy_values_and_non_text_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            _config(truncation_side="middle")
        with self.assertRaises(ValueError):
            _config(padding="implicit")
        with self.assertRaisesRegex(ValueError, "text"):
            tokenize_batch(("one", 7), FakeTokenizer(), _config())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
