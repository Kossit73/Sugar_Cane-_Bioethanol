"""Scenario helpers matching the Cassava model's scenario boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from .financial_model import SugarcaneBioethanolModel


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


def scenario_comparison(
    model: SugarcaneBioethanolModel,
    scenarios: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in model.build_all_scenarios(scenarios).items():
        metrics = result["metrics"]
        rows.append(
            {
                "Scenario": name,
                "Farm Share": metrics.get("farm_share"),
                "Total CAPEX": metrics.get("total_capex"),
                "Project NPV": metrics.get("project_npv"),
                "Project IRR": metrics.get("project_irr"),
                "Equity IRR": metrics.get("equity_irr"),
                "Minimum DSCR": metrics.get("min_dscr"),
                "Payback (years)": metrics.get("payback_years"),
                "Status": metrics.get("model_status"),
            }
        )
    return pd.DataFrame(rows)
