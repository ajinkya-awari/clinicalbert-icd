"""Synthetic tests for the exact five-system registry contract."""

from __future__ import annotations

import unittest

from clinicalbert_icd.models.protocol import ModelSpec
from clinicalbert_icd.models.registry import validate_five_system_registry


def valid_specs() -> tuple[ModelSpec, ...]:
    shared = {
        "label_contract_digest": "sha256:synthetic-label-contract",
        "output_contract": "multilabel_scores_v1",
    }
    return (
        ModelSpec(
            logical_system="rule",
            family="rule",
            adaptation="rule",
            model_reference="synthetic-rule-map",
            model_revision="sha256:synthetic-rule-revision",
            tokenizer_reference=None,
            tokenizer_revision=None,
            **shared,
        ),
        ModelSpec(
            logical_system="domain_lora",
            family="domain",
            adaptation="lora",
            model_reference="synthetic-domain-model",
            model_revision="commit:" + "a" * 40,
            tokenizer_reference="synthetic-domain-tokenizer",
            tokenizer_revision="commit:" + "b" * 40,
            **shared,
        ),
        ModelSpec(
            logical_system="domain_full",
            family="domain",
            adaptation="full",
            model_reference="synthetic-domain-model",
            model_revision="commit:" + "a" * 40,
            tokenizer_reference="synthetic-domain-tokenizer",
            tokenizer_revision="commit:" + "b" * 40,
            **shared,
        ),
        ModelSpec(
            logical_system="general_lora",
            family="general",
            adaptation="lora",
            model_reference="synthetic-general-model",
            model_revision="commit:" + "c" * 40,
            tokenizer_reference="synthetic-general-tokenizer",
            tokenizer_revision="commit:" + "d" * 40,
            **shared,
        ),
        ModelSpec(
            logical_system="general_full",
            family="general",
            adaptation="full",
            model_reference="synthetic-general-model",
            model_revision="commit:" + "c" * 40,
            tokenizer_reference="synthetic-general-tokenizer",
            tokenizer_revision="commit:" + "d" * 40,
            **shared,
        ),
    )


class FiveSystemRegistryTests(unittest.TestCase):
    def test_exact_registry_passes_with_shared_contracts(self) -> None:
        audit = validate_five_system_registry(valid_specs())

        self.assertTrue(audit.allowed)
        self.assertEqual(audit.reasons, ())
        self.assertEqual(audit.system_count, 5)
        self.assertEqual(
            audit.logical_systems,
            ("domain_full", "domain_lora", "general_full", "general_lora", "rule"),
        )

    def test_missing_or_duplicate_system_fails(self) -> None:
        specs = valid_specs()

        missing = validate_five_system_registry(specs[:-1])
        duplicate = validate_five_system_registry(specs[:-1] + (specs[0],))

        self.assertFalse(missing.allowed)
        self.assertIn("exact_system_set_required", missing.reasons)
        self.assertFalse(duplicate.allowed)
        self.assertIn("duplicate_logical_system", duplicate.reasons)

    def test_neural_specs_require_immutable_model_and_tokenizer_revisions(self) -> None:
        specs = list(valid_specs())
        specs[1] = ModelSpec(
            **{
                **specs[1].as_mapping(),
                "model_revision": "main",
                "tokenizer_revision": "latest",
            }
        )

        audit = validate_five_system_registry(specs)

        self.assertFalse(audit.allowed)
        self.assertIn("neural_revision_not_immutable", audit.reasons)

    def test_arbitrary_or_unhashable_revision_is_not_treated_as_immutable(self) -> None:
        for revision in ("refs/heads/main", "release-branch", ["unhashable"]):
            with self.subTest(revision=revision):
                specs = list(valid_specs())
                specs[1] = ModelSpec(
                    **{
                        **specs[1].as_mapping(),
                        "model_revision": revision,
                    }
                )

                audit = validate_five_system_registry(specs)

                self.assertFalse(audit.allowed)
                expected_reason = (
                    "neural_revision_not_immutable"
                    if isinstance(revision, str)
                    else "registry_invalid"
                )
                self.assertIn(expected_reason, audit.reasons)

    def test_commit_revision_requires_a_40_to_64_character_hex_digest(self) -> None:
        for malformed in (
            "commit:",
            "commit:short",
            "commit:" + "g" * 40,
            "commit:" + "a" * 39,
            "commit:" + "a" * 65,
        ):
            with self.subTest(malformed=malformed):
                specs = list(valid_specs())
                specs[1] = ModelSpec(
                    **{**specs[1].as_mapping(), "model_revision": malformed}
                )
                audit = validate_five_system_registry(specs)
                self.assertFalse(audit.allowed)
                self.assertIn("neural_revision_not_immutable", audit.reasons)

    def test_sha256_revision_requires_exactly_64_hex_characters(self) -> None:
        specs = list(valid_specs())
        specs[1] = ModelSpec(
            **{
                **specs[1].as_mapping(),
                "model_revision": "sha256:" + "e" * 64,
            }
        )
        specs[2] = ModelSpec(
            **{
                **specs[2].as_mapping(),
                "model_revision": "sha256:" + "e" * 64,
            }
        )

        self.assertTrue(validate_five_system_registry(specs).allowed)

    def test_all_systems_must_share_label_and_output_contract_metadata(self) -> None:
        specs = list(valid_specs())
        specs[4] = ModelSpec(
            **{
                **specs[4].as_mapping(),
                "label_contract_digest": "sha256:different-label-contract",
                "output_contract": "different-output-contract",
            }
        )

        audit = validate_five_system_registry(specs)

        self.assertFalse(audit.allowed)
        self.assertIn("label_contract_mismatch", audit.reasons)
        self.assertIn("output_contract_mismatch", audit.reasons)

    def test_logical_system_family_and_adaptation_are_fixed(self) -> None:
        specs = list(valid_specs())
        specs[2] = ModelSpec(
            **{
                **specs[2].as_mapping(),
                "family": "general",
                "adaptation": "lora",
            }
        )

        audit = validate_five_system_registry(specs)

        self.assertFalse(audit.allowed)
        self.assertIn("logical_system_definition_invalid", audit.reasons)

    def test_lora_and_full_entries_share_the_same_family_revision(self) -> None:
        specs = list(valid_specs())
        specs[2] = ModelSpec(
            **{
                **specs[2].as_mapping(),
                "model_revision": "commit:" + "e" * 40,
            }
        )

        audit = validate_five_system_registry(specs)

        self.assertFalse(audit.allowed)
        self.assertIn("family_revision_mismatch", audit.reasons)

    def test_rule_system_forbids_model_tokenizer_metadata(self) -> None:
        for field in ("tokenizer_reference", "tokenizer_revision"):
            with self.subTest(field=field):
                specs = list(valid_specs())
                specs[0] = ModelSpec(
                    **{**specs[0].as_mapping(), field: "synthetic-not-allowed"}
                )
                audit = validate_five_system_registry(specs)
                self.assertFalse(audit.allowed)
                self.assertIn("rule_tokenizer_forbidden", audit.reasons)

    def test_malformed_spec_fields_fail_closed_without_type_error(self) -> None:
        fields = tuple(valid_specs()[1].as_mapping())
        for field in fields:
            with self.subTest(field=field):
                specs = list(valid_specs())
                specs[1] = ModelSpec(
                    **{**specs[1].as_mapping(), field: ["unhashable"]}
                )

                audit = validate_five_system_registry(specs)

                self.assertFalse(audit.allowed)


if __name__ == "__main__":
    unittest.main()
