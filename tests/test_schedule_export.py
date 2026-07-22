from __future__ import annotations

from io import BytesIO

import pandas as pd

import pytest
from sugarcane_model import SugarcaneBioethanolModel, default_input_page
from sugarcane_model.driver_schedules import default_schedule_row
from sugarcane_model.exporter import build_excel_report


def test_excel_export_writes_yearly_driver_schedules_as_tables() -> None:
    inputs = default_input_page()
    first = default_schedule_row(inputs, "farming")
    second = dict(first)
    second["Year"] = 2026
    second["sugarcane_yield_tonnes_per_hectare"] = 75.0
    inputs.yearly_schedules["farming"] = [first, second]

    report = build_excel_report(inputs, {"scenario": "HYBRID"}, {})
    workbook = pd.ExcelFile(BytesIO(report))
    assert "Farming Drivers" in workbook.sheet_names

    schedule = pd.read_excel(BytesIO(report), sheet_name="Farming Drivers")
    assert schedule["Year"].tolist() == [2025, 2026]
    assert schedule["sugarcane_yield_tonnes_per_hectare"].tolist() == [70.0, 75.0]


def test_excel_export_includes_default_farm_planning_inputs_and_derived_schedule() -> None:
    inputs = default_input_page()
    result = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")
    report = build_excel_report(
        inputs,
        result["metrics"],
        {"Farm Planning Annual": result["farm_plan"].annual},
    )
    workbook = pd.ExcelFile(BytesIO(report))

    assert "Farm Planning Drivers" in workbook.sheet_names
    assert "Farm Planning Annual" in workbook.sheet_names
    drivers = pd.read_excel(BytesIO(report), sheet_name="Farm Planning Drivers")
    assert drivers["Year"].tolist() == [2025]
    assert drivers.loc[0, "planned_cultivated_hectares"] == 700.0
    annual = pd.read_excel(BytesIO(report), sheet_name="Farm Planning Annual")
    assert annual["Year"].tolist() == list(range(2025, 2036))
    assert annual.loc[0, "RainFedHa"] == pytest.approx(200.0)
    assert annual.loc[0, "FallowReserveHa"] == pytest.approx(100.0)
