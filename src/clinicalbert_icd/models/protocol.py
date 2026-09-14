"""Pure model metadata contracts for the offline five-system registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Declarative model-system metadata; this class never loads a model."""

    logical_system: str
    family: str
    adaptation: str
    model_reference: str | None
    model_revision: str | None
    tokenizer_reference: str | None
    tokenizer_revision: str | None
    label_contract_digest: str
    output_contract: str

    def as_mapping(self) -> dict[str, object]:
        """Return a stable field mapping for synthetic tests and provenance."""

        return {
            "logical_system": self.logical_system,
            "family": self.family,
            "adaptation": self.adaptation,
            "model_reference": self.model_reference,
            "model_revision": self.model_revision,
            "tokenizer_reference": self.tokenizer_reference,
            "tokenizer_revision": self.tokenizer_revision,
            "label_contract_digest": self.label_contract_digest,
            "output_contract": self.output_contract,
        }
