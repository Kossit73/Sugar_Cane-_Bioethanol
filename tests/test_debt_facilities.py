from __future__ import annotations

import copy
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from sugarcane_model import (
    DebtFacilityAssumptions,
    SugarcaneBioethanolModel,
    default_input_page,
)
from sugarcane_model.exporter import build_excel_report


def _facility(
    name: str,
    amount: float,
    *,
    interest_rate: float = 0.08,
    tenor_years: int = 6,
    grace_years: int = 1,
    amortization_type: str = "straight",
    capitalize_idc: bool = True,
) -> DebtFacilityAssumptions:
    return DebtFacilityAssumptions(
        name=name,
        amount=amount,
        interest_rate=interest_rate,
        tenor_years=tenor_years,
        grace_years=grace_years,
        amortization_type=amortization_type,
        capitalize_idc=capitalize_idc,
    )


def test_multiple_fixed_facilities_keep_separate_terms_and_reduce_equity() -> None:
    inputs = default_input_page()
    inputs.financing.additional_debt_facilities = [
        _facility(
            "Development Facility",
            2_400_000.0,
            interest_rate=0.07,
            tenor_years=6,
            amortization_type="annuity",
            capitalize_idc=False,
        ),
        _facility(
            "Local Bank Facility",
            1_000_000.0,
            interest_rate=0.12,
            tenor_years=4,
            grace_years=0,
        ),
    ]

    result = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")
    debt = result["debt"]
    summary = debt.summary.set_index("Facility")

    assert list(summary.index) == [
        "Senior Debt",
        "Development Facility",
        "Local Bank Facility",
    ]
    assert summary.loc["Development Facility", "InitialDraw"] == pytest.approx(
        2_400_000.0
    )
    assert summary.loc["Development Facility", "InterestRate"] == pytest.approx(0.07)
    assert summary.loc["Development Facility", "TenorYears"] == 6
    assert summary.loc["Development Facility", "Amortization"] == "annuity"
    assert bool(summary.loc["Development Facility", "CapitalizeIDC"]) is False

    draws_by_facility = debt.facility_monthly.groupby("Facility")["Draw"].sum()
    assert draws_by_facility["Development Facility"] == pytest.approx(2_400_000.0)
    assert draws_by_facility["Local Bank Facility"] == pytest.approx(1_000_000.0)
    np.testing.assert_allclose(
        debt.monthly["Draw"],
        debt.facility_monthly.groupby(level="Date")["Draw"].sum(),
    )
    assert result["metrics"]["senior_debt_draw"] == pytest.approx(21_000_000.0)
    assert result["metrics"]["additional_debt_draw"] == pytest.approx(3_400_000.0)
    assert result["metrics"]["total_debt_draw"] == pytest.approx(24_400_000.0)
    assert result["metrics"]["total_equity_contribution"] == pytest.approx(
        10_600_000.0
    )
    checks = result["financials"].checks.set_index("Check")
    assert checks.loc["Debt roll-forward", "Status"] == "OK"
    assert checks.loc["Balance sheet balances", "Status"] == "OK"


def test_fixed_amount_is_unchanged_when_capex_changes() -> None:
    inputs = default_input_page()
    inputs.financing.additional_debt_facilities = [
        _facility("Fixed DFI Loan", 2_400_000.0)
    ]
    base = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")

    changed_inputs = copy.deepcopy(inputs)
    changed_inputs.capex.items[2].amount += 5_000_000.0
    changed = SugarcaneBioethanolModel(changed_inputs, "HYBRID").build("HYBRID")

    base_summary = base["debt"].summary.set_index("Facility")
    changed_summary = changed["debt"].summary.set_index("Facility")
    assert base_summary.loc["Fixed DFI Loan", "InitialDraw"] == pytest.approx(
        2_400_000.0
    )
    assert changed_summary.loc["Fixed DFI Loan", "InitialDraw"] == pytest.approx(
        2_400_000.0
    )
    assert changed_summary.loc["Senior Debt", "InitialDraw"] > base_summary.loc[
        "Senior Debt", "InitialDraw"
    ]


def test_invalid_terms_duplicate_names_and_overfunding_are_rejected() -> None:
    invalid_terms = default_input_page()
    invalid_terms.financing.additional_debt_facilities = [
        _facility("Invalid Loan", 2_400_000.0, tenor_years=2, grace_years=2)
    ]
    with pytest.raises(ValueError, match="must exceed grace period"):
        SugarcaneBioethanolModel(invalid_terms, "HYBRID").build("HYBRID")

    duplicate_names = default_input_page()
    duplicate_names.financing.additional_debt_facilities = [
        _facility("DFI Loan", 2_000_000.0),
        _facility("dfi loan", 400_000.0),
    ]
    with pytest.raises(ValueError, match="names must be unique"):
        SugarcaneBioethanolModel(duplicate_names, "HYBRID").build("HYBRID")

    overfunded = default_input_page()
    overfunded.financing.additional_debt_facilities = [
        _facility("Oversized Loan", 20_000_000.0)
    ]
    with pytest.raises(ValueError, match="exceed scenario CAPEX"):
        SugarcaneBioethanolModel(overfunded, "HYBRID").build("HYBRID")


def test_excel_export_contains_debt_facility_inputs_and_summary() -> None:
    inputs = default_input_page()
    inputs.financing.additional_debt_facilities = [
        _facility("Exported Loan", 2_400_000.0)
    ]
    result = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")
    report = build_excel_report(
        inputs,
        result["metrics"],
        {"Debt Facilities Summary": result["debt"].summary},
    )

    workbook = pd.ExcelFile(BytesIO(report))
    assert "Additional Debt Inputs" in workbook.sheet_names
    assert "Debt Facilities Summary" in workbook.sheet_names
    additional_inputs = pd.read_excel(workbook, sheet_name="Additional Debt Inputs")
    assert additional_inputs.loc[0, "name"] == "Exported Loan"
    assert additional_inputs.loc[0, "amount"] == pytest.approx(2_400_000.0)
