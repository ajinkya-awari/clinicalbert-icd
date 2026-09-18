"""Small deterministic tokenizer double for offline contract tests only."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class FakeTokenizer:
    identifier: str = "synthetic-whitespace-tokenizer"
    revision: str = "test-v1"
    prefix_token_ids: tuple[int, ...] = (101,)
    suffix_token_ids: tuple[int, ...] = (102,)
    pad_token_id: int = 0

    _TOKEN_IDS: ClassVar[MappingProxyType[str, int]] = MappingProxyType(
        {
            "one": 11,
            "two": 12,
            "three": 13,
            "four": 14,
            "five": 15,
            "six": 16,
        }
    )

    def encode_content(self, text: str) -> tuple[int, ...]:
        return tuple(self._TOKEN_IDS.get(token, 99) for token in text.split())

