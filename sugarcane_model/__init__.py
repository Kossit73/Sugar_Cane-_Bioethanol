"""Cassava-style modular architecture for the Sugar Cane Bioethanol model."""

from .financial_model import SugarcaneBioethanolModel
from .inputs import (
    SCENARIOS,
    SugarcaneBioethanolInputs,
    default_input_page,
    input_from_payload,
    input_values,
)

__all__ = [
    "SCENARIOS",
    "SugarcaneBioethanolInputs",
    "SugarcaneBioethanolModel",
    "default_input_page",
    "input_from_payload",
    "input_values",
]
