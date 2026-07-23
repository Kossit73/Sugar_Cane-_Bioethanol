"""Annual driver schedules for time-varying Sugar Cane assumptions."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .inputs import validate_farm_planning_values


SCHEDULE_DEFINITIONS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "other_assumptions",
            {
                "label": "Other Assumptions",
                "section": "other_assumptions",
                "fields": [
                    "plant_availability",
                    "process_loss",
                    "startup_ramp_year_1",
                    "startup_ramp_year_2",
                    "price_escalation_rate",
                    "cost_inflation_rate",
                    "minimum_dscr_target",
                ],
            },
        ),
        (
            "cycle_planning",
            {
                "label": "Cycle Planning",
                "section": "cycle_planning",
                "fields": [
                    "crop_cycle_months",
                    "establishment_months",
                    "harvest_window_months",
                    "ratoon_cycles",
                    "replant_share_per_cycle",
                ],
            },
        ),
        (
            "farm_planning",
            {
                "label": "Farm Planning",
                "section": "farm_planning",
                "fields": [
                    "total_land_hectares",
                    "arable_land_hectares",
                    "planned_cultivated_hectares",
                    "irrigation_capacity_hectares",
                    "planned_irrigated_hectares",
                    "hectares_harvested",
                ],
            },
        ),
        (
            "farming",
            {
                "label": "Farming",
                "section": "farming",
                "fields": [
                    "sugarcane_yield_tonnes_per_hectare",
                    "harvest_recovery",
                    "farm_opex_per_tonne",
                    "farm_overhead_per_year",
                    "internal_transfer_price_per_tonne",
                ],
            },
        ),
        (
            "sourcing",
            {
                "label": "Sourcing",
                "section": "sourcing",
                "fields": [
                    "cane_purchase_price_per_tonne",
                    "contracted_purchase_share",
                    "contract_discount",
                    "logistics_cost_per_tonne",
                    "supplier_loss_rate",
                ],
            },
        ),
        (
            "processing_routing",
            {
                "label": "Processing & Production Routing",
                "section": "processing_routing",
                "fields": [
                    "annual_cane_capacity_tonnes",
                    "ethanol_litres_per_tonne",
                    "sugar_tonnes_per_tonne",
                    "raw_bagasse_tonnes_per_tonne",
                    "bagasse_to_electricity_share",
                    "bagasse_to_animal_feed_share",
                    "bagasse_to_sale_share",
                    "electricity_mwh_per_tonne_bagasse",
                    "internal_electricity_mwh_per_tonne_cane",
                    "animal_feed_conversion_rate",
                ],
            },
        ),
        (
            "commercialization",
            {
                "label": "Commercialization",
                "section": "commercialization",
                "fields": [
                    "ethanol_price_per_litre",
                    "sugar_price_per_tonne",
                    "electricity_tariff_per_mwh",
                    "bagasse_price_per_tonne",
                    "animal_feed_price_per_tonne",
                    "ethanol_sales_capture",
                    "sugar_sales_capture",
                    "power_sales_capture",
                    "coproduct_sales_capture",
                ],
            },
        ),
        (
            "costs",
            {
                "label": "Costs",
                "section": "costs",
                "fields": [
                    "processing_variable_cost_per_tonne_cane",
                    "ethanol_variable_cost_per_litre",
                    "sugar_variable_cost_per_tonne",
                    "electricity_variable_cost_per_mwh",
                    "bagasse_handling_cost_per_tonne",
                    "animal_feed_variable_cost_per_tonne",
                    "fixed_processing_opex_per_year",
                    "commercial_and_admin_cost_per_year",
                ],
            },
        ),
        (
            "working_capital",
            {
                "label": "Working Capital",
                "section": "working_capital",
                "fields": ["receivable_days", "inventory_days", "payable_days"],
            },
        ),
        (
            "financing",
            {
                "label": "Financing Draw & Rate",
                "section": "financing",
                "fields": ["debt_ratio", "interest_rate"],
            },
        ),
    ]
)


def _model_values(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper()
    return model.dict()


def default_schedule_row(inputs: Any, schedule_key: str) -> dict[str, Any]:
    """Materialize the scalar landing-page values as the first driver row."""

    config = SCHEDULE_DEFINITIONS[schedule_key]
    values = _model_values(getattr(inputs, config["section"]))
    return {
        "Year": int(inputs.global_assumptions.start_year),
        **{field: values[field] for field in config["fields"]},
    }


def schedule_rows(inputs: Any, schedule_key: str) -> list[dict[str, Any]]:
    """Return sorted explicit rows, falling back to one scalar-derived row."""

    configured = getattr(inputs, "yearly_schedules", {}).get(schedule_key, [])
    rows = [dict(row) for row in configured if isinstance(row, dict)]
    if not rows:
        rows = [default_schedule_row(inputs, schedule_key)]
    return sorted(rows, key=lambda row: int(row.get("Year", inputs.global_assumptions.start_year)))


def parameter_series(
    inputs: Any,
    schedule_key: str,
    field: str,
    index: Iterable[Any],
    *,
    dtype: type = float,
) -> np.ndarray:
    """Resolve one scheduled field to the latest effective value for each period."""

    config = SCHEDULE_DEFINITIONS[schedule_key]
    if field not in config["fields"]:
        raise KeyError(f"{field!r} is not scheduled by {schedule_key!r}")
    base = getattr(getattr(inputs, config["section"]), field)
    rows = schedule_rows(inputs, schedule_key)
    overrides: list[tuple[int, Any]] = []
    for row in rows:
        if field in row and row[field] is not None:
            overrides.append((int(row["Year"]), row[field]))
    overrides.sort(key=lambda item: item[0])

    dates = pd.DatetimeIndex(index)
    resolved: list[Any] = []
    for year in dates.year:
        value = base
        for effective_year, candidate in overrides:
            if effective_year > int(year):
                break
            value = candidate
        resolved.append(value)
    return np.asarray(resolved, dtype=dtype)


def escalation_factor(rates: Iterable[float]) -> np.ndarray:
    """Compound a time-varying annual rate monthly, anchored at 1.0."""

    annual_rates = np.asarray(list(rates), dtype=float)
    if len(annual_rates) == 0:
        return annual_rates
    if np.any(annual_rates <= -1.0):
        raise ValueError("Scheduled escalation rates must be greater than -100%.")
    factors = np.ones(len(annual_rates), dtype=float)
    for index in range(1, len(annual_rates)):
        factors[index] = factors[index - 1] * (1.0 + annual_rates[index]) ** (1.0 / 12.0)
    return factors


def validate_driver_schedules(inputs: Any) -> list[str]:
    """Validate schedule keys, years, uniqueness, and section-level field constraints."""

    errors: list[str] = []
    schedules = getattr(inputs, "yearly_schedules", {})
    unknown = sorted(set(schedules) - set(SCHEDULE_DEFINITIONS))
    if unknown:
        errors.append(f"Unknown yearly schedules: {', '.join(unknown)}")

    start_year = int(inputs.global_assumptions.start_year)
    end_year = int(inputs.global_assumptions.end_year)
    for key, rows in schedules.items():
        if key not in SCHEDULE_DEFINITIONS:
            continue
        config = SCHEDULE_DEFINITIONS[key]
        section = getattr(inputs, config["section"])
        section_type = type(section)
        base_values = _model_values(section)
        seen_years: set[int] = set()
        for row_number, row in enumerate(rows, start=1):
            try:
                year = int(row.get("Year"))
            except (TypeError, ValueError):
                errors.append(f"{config['label']} row {row_number}: Year must be an integer.")
                continue
            if year < start_year or year > end_year:
                errors.append(
                    f"{config['label']} row {row_number}: Year {year} is outside {start_year}-{end_year}."
                )
            if year in seen_years:
                errors.append(f"{config['label']}: Year {year} appears more than once.")
            seen_years.add(year)
            candidate = dict(base_values)
            candidate.update({field: row.get(field, candidate[field]) for field in config["fields"]})
            try:
                validator = getattr(section_type, "model_validate", None)
                if callable(validator):
                    validator(candidate)
                else:
                    section_type.parse_obj(candidate)
            except Exception as exc:
                errors.append(f"{config['label']} row {row_number}: {exc}")
                continue
            if key == "cycle_planning" and (
                int(candidate["establishment_months"])
                + int(candidate["harvest_window_months"])
                > int(candidate["crop_cycle_months"])
            ):
                errors.append(
                    f"{config['label']} row {row_number}: establishment and harvest windows exceed the crop cycle."
                )
            if key == "farm_planning":
                errors.extend(
                    f"{config['label']} row {row_number}: {message}"
                    for message in validate_farm_planning_values(candidate)
                )
            if key == "processing_routing":
                bagasse_share = sum(float(candidate[field]) for field in (
                    "bagasse_to_electricity_share", "bagasse_to_animal_feed_share", "bagasse_to_sale_share"
                ))
                if abs(bagasse_share - 1.0) > 1e-9:
                    errors.append(f"{config['label']} row {row_number}: bagasse routing shares must sum to 1.00.")
    return errors
