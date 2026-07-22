from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from sugarcane_model import SugarcaneBioethanolModel, default_input_page
from sugarcane_model.exporter import build_excel_report


def _build(inputs=None):
    model_inputs = inputs or default_input_page()
    return model_inputs, SugarcaneBioethanolModel(
        model_inputs, "HYBRID"
    ).build("HYBRID")


def test_cod_gates_operations_revenue_and_depreciation() -> None:
    inputs, result = _build()
    cod = pd.Timestamp("2026-01-01")
    before_cod = result["construction"].monthly.index < cod

    assert result["construction"].monthly.loc[cod, "Phase"] == "Operating"
    assert result["sourcing"].monthly.loc[before_cod, "TotalCaneTonnes"].eq(0.0).all()
    assert result["commercialization"].monthly.loc[before_cod, "ExternalRevenue"].eq(0.0).all()
    assert result["capex"].monthly.loc[before_cod, "TotalDepreciation"].eq(0.0).all()

    delayed_inputs = default_input_page()
    delayed_inputs.construction.construction_delay_months = 6
    _, delayed = _build(delayed_inputs)
    effective_cod = pd.Timestamp("2026-07-01")
    delayed_pre_cod = delayed["construction"].monthly.index < effective_cod
    assert delayed["metrics"]["effective_cod"] == "2026-07"
    assert delayed["sourcing"].monthly.loc[delayed_pre_cod, "TotalCaneTonnes"].eq(0.0).all()

    senior = delayed["debt"].summary.set_index("Facility").loc["Senior Debt"]
    assert senior["RepaymentStart"] == pd.Timestamp("2027-07-01")
    assert senior["ContractualMaturity"] == pd.Timestamp("2034-06-01")


def test_dscr_sculpting_caps_senior_debt_and_builds_covenants() -> None:
    fixed_inputs, fixed = _build()
    sculpted_inputs = default_input_page()
    sculpted_inputs.financing.debt_sizing_mode = "dscr_sculpted"
    _, sculpted = _build(sculpted_inputs)

    fixed_senior = fixed["metrics"]["senior_debt_draw"]
    sculpted_senior = sculpted["metrics"]["senior_debt_draw"]
    assert sculpted_senior <= fixed_senior
    assert sculpted["metrics"]["min_dscr"] >= (
        sculpted_inputs.other_assumptions.minimum_dscr_target - 0.05
    )
    summary = sculpted["debt"].summary.set_index("Facility")
    assert summary.loc["Senior Debt", "SizingMode"] == "dscr_sculpted"
    assert summary.loc["Senior Debt", "Amortization"] == "sculpted"
    assert {"DSCR", "LLCR", "PLCR"}.issubset(
        sculpted["financials"].covenant_schedule.columns
    )
    assert sculpted["metrics"]["covenant_period"] == "annual"
    assert fixed_inputs.financing.debt_sizing_mode == "fixed_ratio"


def test_liquidity_waterfall_and_hard_bankability_failures() -> None:
    _, result = _build()
    liquidity = result["financials"].liquidity_monthly

    assert liquidity["PreCODFunding"].sum() > 0.0
    assert liquidity["InitialLiquidityFunding"].sum() > 0.0
    assert liquidity["DSRABalance"].max() > 0.0
    assert liquidity["MaintenanceReserveBalance"].max() > 0.0
    assert result["metrics"]["maximum_funding_shortfall"] == pytest.approx(0.0)
    assert result["metrics"]["calculation_status"] == "OK"
    assert result["metrics"]["bankability_status"] == "FAIL"

    stressed_inputs = default_input_page()
    stressed_inputs.liquidity.working_capital_facility_limit = 0.0
    stressed_inputs.commercialization.ethanol_price_per_litre = 0.0
    stressed_inputs.commercialization.sugar_price_per_tonne = 0.0
    stressed_inputs.commercialization.electricity_tariff_per_mwh = 0.0
    stressed_inputs.commercialization.bagasse_price_per_tonne = 0.0
    stressed_inputs.commercialization.animal_feed_price_per_tonne = 0.0
    _, stressed = _build(stressed_inputs)
    checks = stressed["financials"].checks.set_index("Check")
    assert checks.loc["No unresolved funding shortfall", "Status"] == "FAIL"
    assert checks.loc["Minimum unrestricted cash is maintained", "Status"] == "FAIL"
    assert stressed["metrics"]["maximum_funding_shortfall"] > 0.0


def test_long_tenor_is_not_compressed_into_the_model_horizon() -> None:
    inputs = default_input_page()
    inputs.global_assumptions.end_year = 2030
    inputs.financing.tenor_years = 20
    _, result = _build(inputs)

    senior = result["debt"].summary.set_index("Facility").loc["Senior Debt"]
    assert senior["ContractualMaturity"] == pd.Timestamp("2045-12-01")
    assert senior["EndingBalance"] > 0.0
    assert result["debt"].model_horizon_covers_tail is False
    checks = result["financials"].checks.set_index("Check")
    assert checks.loc["Projection covers debt maturity plus tail", "Status"] == "FAIL"
    assert checks.loc["Debt repays by model end", "Status"] == "FAIL"


def test_excel_export_includes_bankability_schedules() -> None:
    inputs, result = _build()
    financials = result["financials"]
    report = build_excel_report(
        inputs,
        result["metrics"],
        {
            "Construction Annual": result["construction"].annual,
            "Covenant Schedule": financials.covenant_schedule,
            "Liquidity Annual": financials.liquidity_annual,
            "Checks": financials.checks,
        },
    )

    workbook = pd.ExcelFile(BytesIO(report))
    assert "Construction" in workbook.sheet_names
    assert "Liquidity" in workbook.sheet_names
    assert "Construction Annual" in workbook.sheet_names
    assert "Covenant Schedule" in workbook.sheet_names
    assert "Liquidity Annual" in workbook.sheet_names
