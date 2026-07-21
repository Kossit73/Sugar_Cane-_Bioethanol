from __future__ import annotations

import copy

import pytest

from sugarcane_model import SugarcaneBioethanolModel, default_input_page
from sugarcane_model.driver_schedules import (
    default_schedule_row,
    parameter_series,
    validate_driver_schedules,
)
from sugarcane_model.schedule_workspace import (
    ROW_ID,
    add_capex_row,
    add_year_row,
    propagate_capex,
    propagate_yearly,
    remove_row,
    row_label,
    strip_row_ids,
    update_row,
    with_row_ids,
)


def test_row_ids_are_stable_and_capex_selector_is_business_readable() -> None:
    rows = with_row_ids(
        [
            {
                "item": "Fermentation & Distillation",
                "component": "Bioethanol",
                "amount": 14_000_000.0,
            }
        ],
        "capex",
    )
    assert rows[0][ROW_ID] == "capex-1"
    assert row_label(rows[0], 1, capex=True) == (
        "1. Fermentation & Distillation — Bioethanol"
    )
    assert with_row_ids(rows, "capex")[0][ROW_ID] == "capex-1"
    assert ROW_ID not in strip_row_ids(rows)[0]


def test_add_edit_and_remove_target_the_selected_schedule_row() -> None:
    inputs = default_input_page()
    first = with_row_ids([default_schedule_row(inputs, "farming")], "farming")
    rows, added_id = add_year_row(
        first,
        first[0][ROW_ID],
        start_year=inputs.global_assumptions.start_year,
        end_year=inputs.global_assumptions.end_year,
        prefix="farming",
    )
    assert len(rows) == 2
    assert next(row for row in rows if row[ROW_ID] == added_id)["Year"] == 2026

    rows = update_row(rows, added_id, {"farm_opex_per_tonne": 24.0})
    assert next(row for row in rows if row[ROW_ID] == added_id)["farm_opex_per_tonne"] == 24.0
    assert next(row for row in rows if row[ROW_ID] != added_id)["farm_opex_per_tonne"] == 18.0

    rows = remove_row(rows, added_id)
    assert [row[ROW_ID] for row in rows] == ["farming-1"]


def test_yearly_increment_propagates_compounded_values_through_horizon() -> None:
    inputs = default_input_page()
    rows = with_row_ids(
        [default_schedule_row(inputs, "commercialization")],
        "commercialization",
    )
    propagated = propagate_yearly(
        rows,
        rows[0][ROW_ID],
        value_fields=["ethanol_price_per_litre"],
        annual_rate=0.10,
        end_year=2027,
        prefix="commercialization",
    )
    by_year = {row["Year"]: row for row in propagated}
    assert by_year[2025]["ethanol_price_per_litre"] == pytest.approx(0.70)
    assert by_year[2026]["ethanol_price_per_litre"] == pytest.approx(0.77)
    assert by_year[2027]["ethanol_price_per_litre"] == pytest.approx(0.847)
    assert len({row[ROW_ID] for row in propagated}) == 3


def test_capex_add_remove_and_propagate_preserve_valid_item_rows() -> None:
    inputs = default_input_page()
    rows = with_row_ids(
        [inputs.capex.items[0].dict()],
        "capex",
    )
    rows, copied_id = add_capex_row(rows, rows[0][ROW_ID])
    assert next(row for row in rows if row[ROW_ID] == copied_id)["item"].startswith("Copy of")
    rows = remove_row(rows, copied_id)
    propagated = propagate_capex(
        rows,
        rows[0][ROW_ID],
        annual_rate=0.05,
        end_year=2027,
    )
    assert [row["start_month"] for row in propagated] == ["2025-01", "2026-01", "2027-01"]
    assert propagated[-1]["amount"] == pytest.approx(5_000_000.0 * 1.05**2)


def test_driver_schedule_is_consumed_by_monthly_and_financial_schedules() -> None:
    inputs = copy.deepcopy(default_input_page())
    processing_2025 = default_schedule_row(inputs, "processing_routing")
    processing_2027 = dict(processing_2025)
    processing_2027["Year"] = 2027
    processing_2027["annual_cane_capacity_tonnes"] = 150_000.0
    inputs.yearly_schedules["processing_routing"] = [processing_2025, processing_2027]

    result = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")
    january_2027 = result["sourcing"].monthly.loc["2027-01-01", "TotalCaneTonnes"]
    assert january_2027 == pytest.approx(150_000.0 * 0.90 / 12.0)
    assert result["processing"].monthly.loc["2027-01-01", "BioethanolLitres"] > (
        result["processing"].monthly.loc["2026-01-01", "BioethanolLitres"]
    )
    assert result["financials"].income_annual.loc[2027, "Revenue"] > (
        result["financials"].income_annual.loc[2026, "Revenue"]
    )


def test_parameter_series_forward_fills_and_validation_rejects_bad_rows() -> None:
    inputs = default_input_page()
    row = default_schedule_row(inputs, "working_capital")
    later = dict(row)
    later["Year"] = 2027
    later["receivable_days"] = 45.0
    inputs.yearly_schedules["working_capital"] = [row, later]
    dates = result_dates = SugarcaneBioethanolModel(inputs).build()["working_capital"].monthly.index
    values = parameter_series(inputs, "working_capital", "receivable_days", result_dates)
    assert values[0] == pytest.approx(30.0)
    assert values[dates.get_loc("2027-01-01")] == pytest.approx(45.0)

    duplicate = dict(later)
    inputs.yearly_schedules["working_capital"].append(duplicate)
    assert any("appears more than once" in error for error in validate_driver_schedules(inputs))
