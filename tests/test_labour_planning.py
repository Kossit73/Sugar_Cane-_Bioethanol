from __future__ import annotations

import copy
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from sugarcane_model import (
    LabourPlanItem,
    SugarcaneBioethanolModel,
    default_input_page,
)
from sugarcane_model.exporter import build_excel_report


def _build(inputs=None):
    model_inputs = inputs or default_input_page()
    return SugarcaneBioethanolModel(model_inputs, "HYBRID").build("HYBRID")


def _simple_role(**updates) -> LabourPlanItem:
    values = {
        "role_id": "TEST-ROLE",
        "cost_centre": "Test cost centre",
        "department": "Test department",
        "position": "Test position",
        "worker_type": "Permanent",
        "component": "Farming",
        "start_month": "2025-01",
        "end_month": "2026-01",
        "headcount_fte": 1.0,
        "number_of_shifts": 1,
        "monthly_wage_per_fte": 1_000.0,
        "annual_salary_escalation": 0.10,
        "allocation_driver": "Direct",
        "productivity_driver": "Cultivated hectares",
    }
    values.update(updates)
    return LabourPlanItem(**values)


def test_default_labour_schedule_has_monthly_role_detail_and_reconciles() -> None:
    inputs = default_input_page()
    result = _build(inputs)
    labour = result["labour"]

    assert len(labour.item_schedule) == len(inputs.labour.items)
    assert labour.item_schedule["role_id"].is_unique
    assert {
        "Year",
        "Month",
        "CostCentre",
        "Department",
        "Position",
        "WorkerType",
        "HeadcountFTE",
        "NumberOfShifts",
        "MonthlyWagePerFTE",
        "OvertimeCost",
        "BenefitsCost",
        "StatutoryContributions",
        "TrainingCost",
        "PPECost",
        "TransportCost",
        "AccommodationCost",
        "TotalLabourCost",
        "ProductivityDriver",
        "ProductivityPerFTE",
    }.issubset(labour.monthly_detail.columns)

    np.testing.assert_allclose(
        result["costs"].monthly["LabourCost"],
        labour.monthly["TotalLabourCost"],
    )
    allocated = labour.allocation_monthly.groupby("Date")["AllocatedLabourCost"].sum()
    np.testing.assert_allclose(
        allocated.reindex(labour.monthly.index, fill_value=0.0),
        labour.monthly["TotalLabourCost"],
        atol=1e-6,
    )
    assert result["financials"].income_annual["LabourCosts"].sum() == pytest.approx(
        labour.monthly["TotalLabourCost"].sum()
    )
    checks = result["financials"].checks.set_index("Check")
    assert checks.loc["Labour cost allocations reconcile", "Status"] == "OK"


def test_shared_role_is_entered_once_and_custom_allocation_is_transparent() -> None:
    inputs = default_input_page()
    inputs.labour.items = [
        _simple_role(
            role_id="SHARED-LAB",
            component="Shared Plant",
            allocation_driver="Custom product share",
            bioethanol_share=0.40,
            sugar_share=0.20,
            electricity_share=0.10,
            bagasse_share=0.10,
            animal_feed_share=0.20,
        )
    ]
    result = _build(inputs)
    labour = result["labour"]

    assert labour.item_schedule["role_id"].tolist() == ["SHARED-LAB"]
    allocation = labour.allocation_annual.groupby("Component")["AllocatedLabourCost"].sum()
    shares = allocation / allocation.sum()
    expected = {
        "Bioethanol": 0.40,
        "Sugar": 0.20,
        "Electricity Generation": 0.10,
        "Bagasse": 0.10,
        "Animal Feed": 0.20,
    }
    for component, share in expected.items():
        assert shares[component] == pytest.approx(share)
    assert result["financials"].component_annual["LabourCost"].sum() == pytest.approx(
        labour.monthly["TotalLabourCost"].sum()
    )


def test_salary_escalation_and_labour_cost_are_integrated_once() -> None:
    no_labour_inputs = default_input_page()
    no_labour_inputs.labour.items = []
    no_labour = _build(no_labour_inputs)

    labour_inputs = copy.deepcopy(no_labour_inputs)
    labour_inputs.labour.items = [_simple_role()]
    with_labour = _build(labour_inputs)
    detail = with_labour["labour"].monthly_detail.set_index("Date")

    assert detail.loc[pd.Timestamp("2025-01-01"), "MonthlyWagePerFTE"] == pytest.approx(1_000.0)
    assert detail.loc[pd.Timestamp("2026-01-01"), "MonthlyWagePerFTE"] == pytest.approx(1_100.0)
    total_labour = with_labour["labour"].monthly["TotalLabourCost"].sum()
    opex_delta = (
        with_labour["costs"].monthly["EconomicOperatingCost"].sum()
        - no_labour["costs"].monthly["EconomicOperatingCost"].sum()
    )
    assert total_labour == pytest.approx(13_100.0)
    assert opex_delta == pytest.approx(total_labour)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([_simple_role(), _simple_role(position="Duplicate")], "must be unique"),
        (
            [_simple_role(component="Shared Plant", allocation_driver="Direct")],
            "requires a product allocation driver",
        ),
        (
            [
                _simple_role(
                    component="Shared Plant",
                    allocation_driver="Custom product share",
                    bioethanol_share=0.50,
                )
            ],
            "must sum to 100%",
        ),
    ],
)
def test_invalid_labour_plans_are_rejected(items, message: str) -> None:
    inputs = default_input_page()
    inputs.labour.items = items
    with pytest.raises(ValueError, match=message):
        _build(inputs)


def test_excel_export_includes_labour_inputs_details_and_allocations() -> None:
    inputs = default_input_page()
    result = _build(inputs)
    labour = result["labour"]
    report = build_excel_report(
        inputs,
        result["metrics"],
        {
            "Labour Monthly Summary": labour.monthly,
            "Labour Annual Summary": labour.annual,
            "Labour Monthly Detail": labour.monthly_detail,
            "Labour Annual Detail": labour.annual_detail,
            "Labour Allocation Monthly": labour.allocation_monthly,
            "Labour Allocation Annual": labour.allocation_annual,
        },
    )
    workbook = pd.ExcelFile(BytesIO(report))

    expected_sheets = {
        "Labour",
        "Labour Monthly Summary",
        "Labour Annual Summary",
        "Labour Monthly Detail",
        "Labour Annual Detail",
        "Labour Allocation Monthly",
        "Labour Allocation Annual",
    }
    assert expected_sheets.issubset(set(workbook.sheet_names))
    labour_inputs = pd.read_excel(BytesIO(report), sheet_name="Labour")
    assert labour_inputs["role_id"].is_unique
    assert set(labour_inputs["component"]) >= {"Farming", "Shared Plant"}
