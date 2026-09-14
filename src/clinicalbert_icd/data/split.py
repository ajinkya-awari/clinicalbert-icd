"""Deterministic, subject-grouped partitioning for admission examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from .contracts import AdmissionExample, SplitAssignment, SplitAudit, SplitConfig


_PARTITION_ORDER = ("train", "validation", "test")


def _subject_score(subject_id: str, seed: int) -> float:
    payload = f"{seed}:{subject_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return value / float(1 << 256)


def _partition_name(subject_id: str, config: SplitConfig) -> str:
    score = _subject_score(subject_id, config.seed)
    if score < config.train_ratio:
        return "train"
    if score < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


def split_by_subject(
    examples: tuple[AdmissionExample, ...] | list[AdmissionExample], config: SplitConfig
) -> SplitAssignment:
    """Assign every admission for one subject using the same SHA-256 bucket."""

    values = tuple(examples)
    if not values:
        raise ValueError("split examples cannot be empty")
    if any(not isinstance(example, AdmissionExample) for example in values):
        raise ValueError("split examples are invalid")
    partitions: dict[str, list[AdmissionExample]] = {
        name: [] for name in _PARTITION_ORDER
    }
    admission_owners: dict[str, str] = {}
    for example in values:
        if example.hadm_id in admission_owners:
            existing_owner = admission_owners[example.hadm_id]
            if existing_owner != example.subject_id:
                raise ValueError(
                    f"conflicting admission ownership for {example.hadm_id}"
                )
            raise ValueError(f"duplicate admission before split: {example.hadm_id}")
        admission_owners[example.hadm_id] = example.subject_id
        partition = _partition_name(example.subject_id, config)
        partitions[partition].append(replace(example, partition=partition))

    for values in partitions.values():
        values.sort(key=lambda example: (example.subject_id, example.hadm_id))

    subject_counts = tuple(
        (name, len({example.subject_id for example in partitions[name]}))
        for name in _PARTITION_ORDER
    )
    admission_counts = tuple(
        (name, len(partitions[name])) for name in _PARTITION_ORDER
    )
    assignment = SplitAssignment(
        train=tuple(partitions["train"]),
        validation=tuple(partitions["validation"]),
        test=tuple(partitions["test"]),
        audit=SplitAudit(
            algorithm=config.algorithm,
            seed=config.seed,
            ratios=config.ratios,
            subject_counts=subject_counts,
            admission_counts=admission_counts,
        ),
    )
    assert_subject_disjoint(assignment)
    return assignment


def assert_subject_disjoint(assignment: SplitAssignment) -> None:
    """Reject any subject appearing in more than one partition."""

    subjects = {
        "train": {example.subject_id for example in assignment.train},
        "validation": {example.subject_id for example in assignment.validation},
        "test": {example.subject_id for example in assignment.test},
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        if subjects[left] & subjects[right]:
            raise ValueError(f"subject overlap between {left} and {right}")


def partition_manifest_digest(assignment: SplitAssignment) -> str:
    """Hash a canonical aggregate-only public partition manifest."""

    assert_subject_disjoint(assignment)
    payload = json.dumps(
        asdict(assignment.audit), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
