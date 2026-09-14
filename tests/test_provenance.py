"""Synthetic tests for canonical aggregate-only run provenance."""

from __future__ import annotations

import unittest

from clinicalbert_icd.evaluation.provenance import RunProvenance


def provenance_mapping(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "fixture_only": True,
        "command": ["python", "-m", "unittest", "tests.test_provenance"],
        "versions": {"python": "3.11.synthetic"},
        "seeds": {"synthetic_seed": 8408, "split_seed": 8408},
        "hashes": {
            "config": "sha256:" + "a" * 64,
            "label_contract": "sha256:" + "b" * 64,
            "split_manifest": "sha256:" + "c" * 64,
        },
        "device": {"kind": "cpu", "environment": "local-offline-synthetic"},
        "artifact_paths": [
            "artifacts/synthetic/provenance.json",
            "artifacts/synthetic/aggregate-metrics.json",
        ],
    }
    value.update(overrides)
    return value


class RunProvenanceTests(unittest.TestCase):
    def test_required_aggregate_provenance_is_canonical_and_digestible(self) -> None:
        provenance = RunProvenance.from_mapping(provenance_mapping())

        self.assertTrue(provenance.fixture_only)
        self.assertEqual(provenance.seeds, (("split_seed", 8408), ("synthetic_seed", 8408)))
        self.assertRegex(provenance.digest(), r"^sha256:[0-9a-f]{64}$")

    def test_mapping_order_does_not_change_digest(self) -> None:
        first = provenance_mapping()
        second = dict(reversed(tuple(first.items())))

        self.assertEqual(
            RunProvenance.from_mapping(first).digest(),
            RunProvenance.from_mapping(second).digest(),
        )

    def test_each_required_field_is_enforced(self) -> None:
        for field in (
            "command",
            "versions",
            "seeds",
            "hashes",
            "device",
            "artifact_paths",
        ):
            with self.subTest(field=field):
                value = provenance_mapping()
                del value[field]
                with self.assertRaisesRegex(ValueError, "missing_provenance_fields"):
                    RunProvenance.from_mapping(value)

    def test_run_results_or_unknown_fields_are_rejected(self) -> None:
        value = provenance_mapping(metrics={"micro_f1": 0.99})

        with self.assertRaisesRegex(ValueError, "unexpected_provenance_fields"):
            RunProvenance.from_mapping(value)

    def test_sensitive_content_and_nonlocal_artifact_paths_are_rejected(self) -> None:
        sensitive = provenance_mapping(
            command=["python", "SYNTHETIC_NOTE_TEXT_CANARY_DO_NOT_USE"]
        )
        external_path = provenance_mapping(
            artifact_paths=["https://example.invalid/external.json"]
        )

        with self.assertRaisesRegex(ValueError, "sensitive_provenance_content"):
            RunProvenance.from_mapping(sensitive)
        with self.assertRaisesRegex(ValueError, "artifact_paths_invalid"):
            RunProvenance.from_mapping(external_path)

    def test_fixture_only_and_sha256_hash_contracts_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixture_only_required"):
            RunProvenance.from_mapping(provenance_mapping(fixture_only=False))
        with self.assertRaisesRegex(ValueError, "hashes_invalid"):
            RunProvenance.from_mapping(
                provenance_mapping(hashes={"config": "latest"})
            )

    def test_restricted_or_external_command_forms_are_rejected(self) -> None:
        blocked_commands = (
            ("curl", "https://example.invalid"),
            ("wget", "https://example.invalid"),
            ("Invoke-WebRequest", "https://example.invalid"),
            ("git", "push"),
            ("kaggle", "kernels", "push"),
            ("wandb", "login"),
            ("huggingface-cli", "upload"),
            ("hf", "upload"),
            ("pip", "install", "synthetic-package"),
            ("python", "download.py"),
            ("python", "train.py"),
            ("deploy", "synthetic"),
            ("publish", "synthetic"),
            ("email", "synthetic"),
            ("Get-Content", "C:external.txt"),
        )
        for command in blocked_commands:
            with self.subTest(command=command):
                with self.assertRaisesRegex(ValueError, "command_not_allowed"):
                    RunProvenance.from_mapping(provenance_mapping(command=command))

    def test_local_unittest_compile_and_read_only_evidence_commands_are_allowed(self) -> None:
        allowed_commands = (
            ("python", "-m", "unittest", "tests.test_provenance", "-v"),
            ("python", "-m", "compileall", "-q", "src", "tests"),
            ("rg", "--files"),
            ("Get-Content", "config/synthetic.json"),
            ("Get-FileHash", "config/synthetic.json"),
        )
        for command in allowed_commands:
            with self.subTest(command=command):
                provenance = RunProvenance.from_mapping(
                    provenance_mapping(command=command)
                )
                self.assertEqual(provenance.command, command)

    def test_direct_constructor_enforces_the_same_invariants(self) -> None:
        valid = provenance_mapping()
        direct = RunProvenance(**valid)
        mapped = RunProvenance.from_mapping(valid)

        self.assertEqual(direct, mapped)
        self.assertEqual(direct.digest(), mapped.digest())
        invalid_values = (
            {**valid, "command": ("git", "push")},
            {**valid, "hashes": {"config": "latest"}},
            {**valid, "device": {"kind": "gpu", "environment": "local"}},
            {**valid, "artifact_paths": ("C:/private/value.json",)},
            {
                **valid,
                "command": ("python", "-m", "unittest", "SYNTHETIC_NOTE_TEXT_CANARY"),
            },
        )
        for invalid in invalid_values:
            with self.subTest(invalid_field=next(key for key in invalid if invalid[key] != valid.get(key))):
                with self.assertRaises(ValueError):
                    RunProvenance(**invalid)

    def test_direct_constructor_accepts_its_canonical_tuple_field_types(self) -> None:
        direct = RunProvenance(
            schema_version=1,
            fixture_only=True,
            command=("python", "-m", "unittest", "tests.test_provenance"),
            versions=(("python", "3.11.synthetic"),),
            seeds=(("synthetic_seed", 8408), ("split_seed", 8408)),
            hashes=(("config", "sha256:" + "a" * 64),),
            device=(
                ("kind", "cpu"),
                ("environment", "local-offline-synthetic"),
            ),
            artifact_paths=("artifacts/synthetic/provenance.json",),
        )

        self.assertEqual(
            direct.device,
            (("environment", "local-offline-synthetic"), ("kind", "cpu")),
        )
        self.assertRegex(direct.digest(), r"^sha256:[0-9a-f]{64}$")

    def test_synthetic_provenance_rejects_gpu_or_non_synthetic_artifact_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_scope_invalid"):
            RunProvenance.from_mapping(
                provenance_mapping(device={"kind": "gpu", "environment": "local"})
            )
        for path in (
            "checkpoints/synthetic-model.pt",
            "C:/private/synthetic.json",
            "artifacts/synthetic/model.safetensors",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "artifact_paths_invalid"):
                    RunProvenance.from_mapping(
                        provenance_mapping(artifact_paths=[path])
                    )


if __name__ == "__main__":
    unittest.main()
