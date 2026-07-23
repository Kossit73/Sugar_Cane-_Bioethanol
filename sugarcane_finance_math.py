"""Small NumPy-only replacements for the finance functions used by the model."""
from __future__ import annotations

import math

import numpy as np


class _FinanceFunctions:
    """Compatibility surface for the two numpy-financial functions we need."""

    @staticmethod
    def npv(rate: float, cashflows: np.ndarray) -> float:
        values = np.asarray(cashflows, dtype=float)
        periods = np.arange(values.size, dtype=float)
        return float(np.sum(values / np.power(1.0 + rate, periods)))

    @staticmethod
    def irr(cashflows: np.ndarray) -> float:
        values = np.asarray(cashflows, dtype=float)
        if values.size < 2 or not np.any(values < 0) or not np.any(values > 0):
            return math.nan

        periods = np.arange(values.size, dtype=float)

        def present_value(rate: float) -> float:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = float(np.sum(values / np.power(1.0 + rate, periods)))
            return result

        lower = -0.9999
        upper = 1.0
        lower_value = present_value(lower)
        upper_value = present_value(upper)

        while np.signbit(lower_value) == np.signbit(upper_value) and upper < 1_000_000.0:
            upper *= 2.0
            upper_value = present_value(upper)

        if math.isnan(lower_value) or math.isnan(upper_value):
            return math.nan
        if np.signbit(lower_value) == np.signbit(upper_value):
            return math.nan

        for _ in range(200):
            midpoint = (lower + upper) / 2.0
            midpoint_value = present_value(midpoint)
            if abs(midpoint_value) < 1e-7:
                return midpoint
            if np.signbit(midpoint_value) == np.signbit(lower_value):
                lower = midpoint
                lower_value = midpoint_value
            else:
                upper = midpoint

        return (lower + upper) / 2.0


npf = _FinanceFunctions()
