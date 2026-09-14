"""Offline tokenization contract over an injected tokenizer object."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .contracts import TokenizationAudit, TokenizedBatch, TokenizerConfig


class InjectedTokenizer(Protocol):
    identifier: str
    revision: str
    prefix_special_token_ids: tuple[int, ...]
    suffix_special_token_ids: tuple[int, ...]
    pad_token_id: int

    def encode_content(self, text: str) -> tuple[int, ...]: ...


def _integer_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"tokenizer {name} must be a tuple of integer token IDs")
    return value


def tokenize_batch(
    texts: Iterable[str], tokenizer: InjectedTokenizer, config: TokenizerConfig
) -> TokenizedBatch:
    """Tokenize locally while enforcing revision, length, and audit contracts."""

    tokenizer_identifier = getattr(tokenizer, "identifier", None)
    tokenizer_revision = getattr(tokenizer, "revision", None)
    try:
        TokenizerConfig(
            identifier=tokenizer_identifier,
            revision=tokenizer_revision,
            maximum_length=config.maximum_length,
            truncation_side=config.truncation_side,
            padding=config.padding,
        )
    except ValueError:
        raise ValueError("injected tokenizer identifier or revision is invalid") from None
    if tokenizer_identifier != config.identifier:
        raise ValueError("tokenizer identifier does not match the explicit config")
    if tokenizer_revision != config.revision:
        raise ValueError("tokenizer revision does not match the explicit config")

    prefix = _integer_tuple(
        "prefix_special_token_ids",
        getattr(tokenizer, "prefix_special_token_ids", None),
    )
    suffix = _integer_tuple(
        "suffix_special_token_ids",
        getattr(tokenizer, "suffix_special_token_ids", None),
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool):
        raise ValueError("tokenizer pad_token_id must be an integer")
    special_token_count = len(prefix) + len(suffix)
    if config.maximum_length < special_token_count:
        raise ValueError("maximum_length leaves insufficient room for special tokens")

    values = tuple(texts)
    if not values:
        raise ValueError("tokenization input cannot be empty")
    input_rows: list[tuple[int, ...]] = []
    mask_rows: list[tuple[int, ...]] = []
    truncated_count = 0
    tokens_before = 0
    tokens_after = 0
    content_capacity = config.maximum_length - special_token_count

    for text in values:
        if not isinstance(text, str):
            raise ValueError("every tokenization input must be text")
        content = tuple(tokenizer.encode_content(text))
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in content):
            raise ValueError("tokenizer content IDs must be integers")
        tokens_before += len(content)
        if len(content) > content_capacity:
            truncated_count += 1
            if content_capacity == 0:
                content = ()
            elif config.truncation_side == "right":
                content = content[:content_capacity]
            else:
                content = content[-content_capacity:]
        tokens_after += len(content)

        token_ids = prefix + content + suffix
        attention_mask = (1,) * len(token_ids)
        if config.padding == "max_length":
            padding_count = config.maximum_length - len(token_ids)
            token_ids += (pad_token_id,) * padding_count
            attention_mask += (0,) * padding_count
        input_rows.append(token_ids)
        mask_rows.append(attention_mask)

    return TokenizedBatch(
        input_ids=tuple(input_rows),
        attention_mask=tuple(mask_rows),
        audit=TokenizationAudit(
            tokenizer_identifier=config.identifier,
            tokenizer_revision=config.revision,
            maximum_length=config.maximum_length,
            truncation_side=config.truncation_side,
            padding=config.padding,
            example_count=len(values),
            truncated_example_count=truncated_count,
            total_content_tokens_before=tokens_before,
            total_content_tokens_after=tokens_after,
            special_tokens_per_example=special_token_count,
        ),
    )
