from __future__ import annotations

from io import BytesIO

import pandas as pd

from sugarcane_model import default_input_page
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
