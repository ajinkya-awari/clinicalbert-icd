"""Trainability audits over injected synthetic parameter records only."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """One observed parameter without any framework dependency."""

    name: str
    trainable: bool
    count: int


@dataclass(frozen=True, slots=True)
class TrainabilityAudit:
    """Aggregate parameter evidence with stable guard reasons."""

    allowed: bool
    reasons: tuple[str, ...]
    total_parameter_count: int
    trainable_parameter_count: int
    adapter_parameter_count: int
    classifier_parameter_count: int
    matched_target_modules: tuple[str, ...]
    unexpected_trainable_parameters: tuple[str, ...]


def audit_lora_trainability(
    parameters: Iterable[ParameterRecord],
    target_suffixes: Iterable[str],
    classifier_path: str,
) -> TrainabilityAudit:
    """Require trainable target adapters and head while freezing all other base parameters."""

    items, records_valid = _parameter_records(parameters)
    suffixes, config_valid = _target_suffixes(target_suffixes, classifier_path)
    reasons: list[str] = []
    if not records_valid:
        reasons.append("parameter_records_invalid")
    if not config_valid:
        reasons.append("trainability_config_invalid")

    classifier = tuple(item for item in items if _under_path(item.name, classifier_path))
    adapter_items: list[tuple[ParameterRecord, str]] = []
    matched_modules: set[str] = set()
    unmatched_adapter = False
    for item in items:
        module = _adapter_module(item.name)
        if module is None:
            continue
        adapter_items.append((item, module))
        if item.trainable and any(_module_has_suffix(module, suffix) for suffix in suffixes):
            matched_modules.add(module)
        elif item.trainable:
            unmatched_adapter = True

    trainable_adapters = tuple(item for item, _ in adapter_items if item.trainable)
    frozen_target_adapters = tuple(
        item
        for item, module in adapter_items
        if not item.trainable
        and any(_module_has_suffix(module, suffix) for suffix in suffixes)
    )
    matched_suffixes = {
        suffix
        for suffix in suffixes
        if any(_module_has_suffix(module, suffix) for module in matched_modules)
    }
    unexpected = tuple(
        sorted(
            item.name
            for item in items
            if item.trainable
            and _adapter_module(item.name) is None
            and not _under_path(item.name, classifier_path)
        )
    )

    if not trainable_adapters:
        reasons.append("target_adapters_missing")
    if frozen_target_adapters:
        reasons.append("adapter_not_fully_trainable")
    if suffixes and matched_suffixes != set(suffixes):
        reasons.append("target_suffix_unmatched")
    if unmatched_adapter:
        reasons.append("unexpected_adapter_target")
    if not classifier:
        reasons.append("classifier_missing")
    elif any(not item.trainable for item in classifier):
        reasons.append("classifier_not_fully_trainable")
    if unexpected:
        reasons.append("unexpected_trainable_base")

    total = sum(item.count for item in items if _valid_count(item.count))
    trainable = sum(
        item.count for item in items if item.trainable and _valid_count(item.count)
    )
    if items and trainable >= total:
        reasons.append("lora_scope_not_bounded")

    return _audit(
        reasons,
        items,
        adapter_parameter_count=sum(
            item.count for item in trainable_adapters if _valid_count(item.count)
        ),
        classifier_parameter_count=sum(
            item.count for item in classifier if item.trainable and _valid_count(item.count)
        ),
        matched_target_modules=tuple(sorted(matched_modules)),
        unexpected_trainable_parameters=unexpected,
    )


def audit_full_trainability(
    parameters: Iterable[ParameterRecord], classifier_path: str
) -> TrainabilityAudit:
    """Require every observed parameter, including the classifier, to be trainable."""

    items, records_valid = _parameter_records(parameters)
    reasons: list[str] = []
    if not records_valid:
        reasons.append("parameter_records_invalid")
    if not isinstance(classifier_path, str) or not classifier_path.strip():
        reasons.append("trainability_config_invalid")

    classifier = tuple(item for item in items if _under_path(item.name, classifier_path))
    if not classifier:
        reasons.append("classifier_missing")
    elif any(not item.trainable for item in classifier):
        reasons.append("classifier_not_fully_trainable")
    if any(not item.trainable for item in items):
        reasons.append("full_parameters_not_trainable")

    return _audit(
        reasons,
        items,
        adapter_parameter_count=0,
        classifier_parameter_count=sum(
            item.count for item in classifier if item.trainable and _valid_count(item.count)
        ),
        matched_target_modules=(),
        unexpected_trainable_parameters=(),
    )


def _parameter_records(
    parameters: Iterable[ParameterRecord],
) -> tuple[tuple[ParameterRecord, ...], bool]:
    try:
        items = tuple(parameters)
    except TypeError:
        return (), False
    valid_items = tuple(
        item
        for item in items
        if isinstance(item, ParameterRecord)
        and isinstance(item.name, str)
        and bool(item.name.strip())
        and type(item.trainable) is bool
        and _valid_count(item.count)
    )
    names = [item.name for item in valid_items]
    valid = (
        bool(items)
        and len(valid_items) == len(items)
        and len(names) == len(set(names))
    )
    return valid_items, valid


def _target_suffixes(
    target_suffixes: Iterable[str], classifier_path: object
) -> tuple[tuple[str, ...], bool]:
    try:
        raw_suffixes = tuple(target_suffixes)
    except TypeError:
        return (), False
    suffixes = tuple(
        item.strip()
        for item in raw_suffixes
        if isinstance(item, str) and bool(item.strip())
    )
    valid = (
        bool(suffixes)
        and len(suffixes) == len(raw_suffixes)
        and len(suffixes) == len(set(suffixes))
        and isinstance(classifier_path, str)
        and bool(classifier_path.strip())
    )
    return suffixes, valid


def _valid_count(value: object) -> bool:
    return type(value) is int and value > 0


def _under_path(name: str, path: object) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    normalized = path.strip()
    return name == normalized or name.startswith(normalized + ".")


def _adapter_module(name: str) -> str | None:
    marker = ".lora_"
    if marker not in name:
        return None
    return name.split(marker, 1)[0]


def _module_has_suffix(module: str, suffix: str) -> bool:
    return module == suffix or module.endswith("." + suffix)


def _audit(
    reasons: list[str],
    items: tuple[ParameterRecord, ...],
    *,
    adapter_parameter_count: int,
    classifier_parameter_count: int,
    matched_target_modules: tuple[str, ...],
    unexpected_trainable_parameters: tuple[str, ...],
) -> TrainabilityAudit:
    unique_reasons = tuple(dict.fromkeys(reasons))
    total = sum(item.count for item in items if _valid_count(item.count))
    trainable = sum(
        item.count for item in items if item.trainable and _valid_count(item.count)
    )
    return TrainabilityAudit(
        allowed=not unique_reasons,
        reasons=unique_reasons,
        total_parameter_count=total,
        trainable_parameter_count=trainable,
        adapter_parameter_count=adapter_parameter_count,
        classifier_parameter_count=classifier_parameter_count,
        matched_target_modules=matched_target_modules,
        unexpected_trainable_parameters=unexpected_trainable_parameters,
    )
