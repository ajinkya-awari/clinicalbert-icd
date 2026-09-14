"""Training-only label vocabulary and deterministic multi-label encoding."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .contracts import (
    AdmissionExample,
    EncodedLabels,
    LabelConfig,
    LabelContract,
    LabelFitAudit,
    LabelTransformAudit,
    _label_vocabulary_digest,
)
from .prepare import parse_native_icd9


def fit_training_label_contract(
    examples: Iterable[AdmissionExample], config: LabelConfig
) -> LabelContract:
    """Fit a stable vocabulary only when every example is training-tagged."""

    training_examples = tuple(examples)
    if any(example.partition != "train" for example in training_examples):
        raise ValueError("label vocabulary fitting accepts training examples only")
    admission_ids = tuple(example.hadm_id for example in training_examples)
    if len(admission_ids) != len(set(admission_ids)):
        raise ValueError("duplicate_training_admission")

    support: Counter[str] = Counter()
    for example in training_examples:
        labels = {parse_native_icd9(label) for label in example.labels}
        support.update(labels)

    vocabulary = tuple(
        sorted(
            label
            for label, count in support.items()
            if count >= config.minimum_support
        )
    )
    if not vocabulary:
        raise ValueError("training label selection produced an empty vocabulary")

    return LabelContract(
        selection=config.selection,
        minimum_support=config.minimum_support,
        vocabulary=vocabulary,
        vocabulary_hash=_label_vocabulary_digest(vocabulary),
        audit=LabelFitAudit(
            training_example_count=len(training_examples),
            vocabulary_size=len(vocabulary),
        ),
    )


def transform_labels(
    examples: Iterable[AdmissionExample], contract: LabelContract
) -> EncodedLabels:
    """Encode labels with the frozen training vocabulary and audit unknowns."""

    values = tuple(examples)
    vocabulary_index = {
        label: index for index, label in enumerate(contract.vocabulary)
    }
    rows: list[tuple[int, ...]] = []
    unknown_label_count = 0
    examples_with_unknown = 0

    for example in values:
        labels = {parse_native_icd9(label) for label in example.labels}
        unknown = labels.difference(vocabulary_index)
        if unknown:
            examples_with_unknown += 1
            unknown_label_count += len(unknown)
        row = [0] * len(contract.vocabulary)
        for label in labels.intersection(vocabulary_index):
            row[vocabulary_index[label]] = 1
        rows.append(tuple(row))

    return EncodedLabels(
        rows=tuple(rows),
        vocabulary_hash=contract.vocabulary_hash,
        audit=LabelTransformAudit(
            example_count=len(values),
            unknown_label_count=unknown_label_count,
            examples_with_unknown_labels=examples_with_unknown,
        ),
    )
