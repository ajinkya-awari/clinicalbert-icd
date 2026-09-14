"""Canonical synthetic provenance with no result or clinical-content fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re


_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "fixture_only",
        "command",
        "versions",
        "seeds",
        "hashes",
        "device",
        "artifact_paths",
    }
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FORBIDDEN_ARTIFACT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)
_SENSITIVE_PATTERNS = (
    re.compile(r"SYNTHETIC_NOTE_TEXT_CANARY", re.IGNORECASE),
    re.compile(r"SYNTHETIC_SECRET_CANARY", re.IGNORECASE),
    re.compile(r"\b(?:NOTEEVENTS|DIAGNOSES_ICD)\b", re.IGNORECASE),
    re.compile(r"\b(?:hadm|patient|subject)[_-]?ids?\b", re.IGNORECASE),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9_-]{8,})\b"),
)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Immutable local synthetic run provenance suitable for canonical hashing."""

    schema_version: int
    fixture_only: bool
    command: tuple[str, ...]
    versions: tuple[tuple[str, str], ...]
    seeds: tuple[tuple[str, int], ...]
    hashes: tuple[tuple[str, str], ...]
    device: tuple[tuple[str, str | int | bool], ...]
    artifact_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        """Apply the same fail-closed normalization to every construction path."""

        normalized = _validated_components(
            {
                "schema_version": self.schema_version,
                "fixture_only": self.fixture_only,
                "command": self.command,
                "versions": self.versions,
                "seeds": self.seeds,
                "hashes": self.hashes,
                "device": self.device,
                "artifact_paths": self.artifact_paths,
            }
        )
        for field, value in normalized.items():
            object.__setattr__(self, field, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RunProvenance":
        """Validate exact local synthetic provenance fields without fabricating results."""

        if not isinstance(value, Mapping):
            raise ValueError("provenance_must_be_mapping")
        if any(field not in value for field in _REQUIRED_FIELDS):
            raise ValueError("missing_provenance_fields")
        if set(value) != set(_REQUIRED_FIELDS):
            raise ValueError("unexpected_provenance_fields")
        return cls(**value)  # type: ignore[arg-type]

    def digest(self) -> str:
        """Return a SHA-256 digest of canonical JSON serialization."""

        payload = {
            "artifact_paths": list(self.artifact_paths),
            "command": list(self.command),
            "device": dict(self.device),
            "fixture_only": self.fixture_only,
            "hashes": dict(self.hashes),
            "schema_version": self.schema_version,
            "seeds": dict(self.seeds),
            "versions": dict(self.versions),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _string_sequence(value: object, *, reason: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(reason)
    items = tuple(value)
    if not items or any(not _required_string(item) for item in items):
        raise ValueError(reason)
    return tuple(item.strip() for item in items if isinstance(item, str))


def _validated_components(value: Mapping[str, object]) -> dict[str, object]:
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported_provenance_schema")
    if value["fixture_only"] is not True:
        raise ValueError("fixture_only_required")

    command = _string_sequence(value["command"], reason="command_invalid")
    if any(
        pattern.search(item)
        for item in command
        for pattern in _SENSITIVE_PATTERNS
    ):
        raise ValueError("sensitive_provenance_content")
    if not _command_allowed(command):
        raise ValueError("command_not_allowed")
    versions = _string_mapping(value["versions"], reason="versions_invalid")
    seeds = _seed_mapping(value["seeds"])
    hashes = _hash_mapping(value["hashes"])
    device = _device_mapping(value["device"])
    if dict(device) != {
        "environment": "local-offline-synthetic",
        "kind": "cpu",
    }:
        raise ValueError("device_scope_invalid")
    artifact_paths = tuple(sorted(_artifact_paths(value["artifact_paths"])))

    all_strings = command + artifact_paths
    all_strings += tuple(item for pair in versions + hashes for item in pair)
    all_strings += tuple(key for key, _ in seeds + device)
    all_strings += tuple(item for _, item in device if isinstance(item, str))
    if any(
        pattern.search(item)
        for item in all_strings
        for pattern in _SENSITIVE_PATTERNS
    ):
        raise ValueError("sensitive_provenance_content")

    return {
        "schema_version": 1,
        "fixture_only": True,
        "command": command,
        "versions": versions,
        "seeds": seeds,
        "hashes": hashes,
        "device": device,
        "artifact_paths": artifact_paths,
    }


def _command_allowed(command: tuple[str, ...]) -> bool:
    executable = command[0].casefold().removesuffix(".exe")
    if any(any(character in item for character in ";|&`") for item in command):
        return False
    if executable in {"python", "python3"}:
        return (
            len(command) >= 3
            and command[1] == "-m"
            and command[2] in {"unittest", "compileall"}
            and _local_evidence_arguments(command[3:])
        )
    if executable in {"rg", "get-content", "get-filehash", "test-path"}:
        return _local_evidence_arguments(command[1:]) and not any(
            item.casefold().startswith("--pre") for item in command[1:]
        )
    return False


def _local_evidence_arguments(arguments: Sequence[str]) -> bool:
    for item in arguments:
        lowered = item.casefold()
        if (
            "://" in item
            or "\\" in item
            or ":" in item
            or item.startswith("/")
            or ".." in item.split("/")
            or any(segment in {"data", "private", "secrets", "checkpoints", "weights"}
                   for segment in lowered.split("/"))
        ):
            return False
    return True


def _string_mapping(
    value: object, *, reason: str
) -> tuple[tuple[str, str], ...]:
    items = _mapping_items(value, reason=reason)
    if any(not _required_string(key) or not _required_string(item) for key, item in items):
        raise ValueError(reason)
    normalized = tuple((key.strip(), item.strip()) for key, item in items)
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(reason)
    return tuple(sorted(normalized))


def _seed_mapping(value: object) -> tuple[tuple[str, int], ...]:
    items = _mapping_items(value, reason="seeds_invalid")
    if any(
        not _required_string(key) or type(item) is not int or item < 0
        for key, item in items
    ):
        raise ValueError("seeds_invalid")
    normalized = tuple((key.strip(), item) for key, item in items)
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("seeds_invalid")
    return tuple(sorted(normalized))


def _hash_mapping(value: object) -> tuple[tuple[str, str], ...]:
    items = _mapping_items(value, reason="hashes_invalid")
    if any(
        not _required_string(key)
        or not isinstance(item, str)
        or _SHA256.fullmatch(item) is None
        for key, item in items
    ):
        raise ValueError("hashes_invalid")
    normalized = tuple((key.strip(), item) for key, item in items)
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("hashes_invalid")
    return tuple(sorted(normalized))


def _device_mapping(
    value: object,
) -> tuple[tuple[str, str | int | bool], ...]:
    items = _mapping_items(value, reason="device_invalid")
    if any(
        not _required_string(key)
        or type(item) not in (str, int, bool)
        or (isinstance(item, str) and not item.strip())
        for key, item in items
    ):
        raise ValueError("device_invalid")
    normalized = tuple(
        (key.strip(), item.strip() if isinstance(item, str) else item)
        for key, item in items
    )
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("device_invalid")
    return tuple(sorted(normalized))


def _mapping_items(
    value: object, *, reason: str
) -> tuple[tuple[object, object], ...]:
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs: list[tuple[object, object]] = []
        for item in value:
            if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
                raise ValueError(reason)
            pair = tuple(item)
            if len(pair) != 2:
                raise ValueError(reason)
            pairs.append((pair[0], pair[1]))
        items = tuple(pairs)
    else:
        raise ValueError(reason)
    if not items:
        raise ValueError(reason)
    return items


def _artifact_paths(value: object) -> tuple[str, ...]:
    items = _string_sequence(value, reason="artifact_paths_invalid")
    for item in items:
        lowered = item.casefold()
        if (
            "\\" in item
            or ":" in item
            or "://" in item
            or item.startswith("/")
            or any(part in {"", ".", ".."} for part in item.split("/"))
            or not lowered.startswith("artifacts/synthetic/")
            or lowered.endswith(_FORBIDDEN_ARTIFACT_SUFFIXES)
        ):
            raise ValueError("artifact_paths_invalid")
    return items


def _required_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
