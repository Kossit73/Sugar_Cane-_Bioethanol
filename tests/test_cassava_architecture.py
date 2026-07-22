from __future__ import annotations

import copy

import numpy as np
import pytest

from sugarcane_model import SugarcaneBioethanolModel, default_input_page


EXPECTED_COMPONENTS = {
    "Farming",
    "Bioethanol",
    "Sugar",
    "Electricity Generation",
    "Bagasse",
    "Animal Feed",
}


def _build(scenario: str = "HYBRID"):
    inputs = copy.deepcopy(default_input_page())
    inputs.global_assumptions.scenario = scenario
    return inputs, SugarcaneBioethanolModel(inputs, scenario).build(scenario)


def test_grouped_inputs_match_the_cassava_architecture() -> None:
    inputs = default_input_page()
    assert list(inputs.grouped_sections()) == [
        "Global assumptions",
        "Other assumptions",
        "Capex",
        "Cycle planning",
        "Farm planning",
        "Farming",
        "Sourcing",
        "Processing & production routing",
        "Commercialization",
        "Costs",
        "Working capital",
        "Financing",
    ]


def test_farm_buy_and_hybrid_scenarios_route_cane_and_capex() -> None:
    _, farm = _build("FARM_ONLY")
    _, hybrid = _build("HYBRID")
    _, buy = _build("BUY_ONLY")
    assert farm["sourcing"].annual["PurchasedCaneTonnes"].sum() == 0
    assert hybrid["sourcing"].annual["FarmCaneTonnes"].sum() > 0
    assert hybrid["sourcing"].annual["PurchasedCaneTonnes"].sum() > 0
    assert buy["sourcing"].annual["FarmCaneTonnes"].sum() == 0
    assert farm["metrics"]["farm_share"] == pytest.approx(1.0)
    assert hybrid["metrics"]["farm_share"] == pytest.approx(0.5)
    assert buy["metrics"]["farm_share"] == pytest.approx(0.0)
    assert farm["metrics"]["farm_share_target"] == 1.0
    assert hybrid["metrics"]["farm_share_target"] == 0.5
    assert buy["metrics"]["farm_share_target"] == 0.0
    assert farm["metrics"]["total_capex"] == pytest.approx(37_500_000.0)
    assert hybrid["metrics"]["total_capex"] == pytest.approx(35_000_000.0)
    assert buy["metrics"]["total_capex"] == pytest.approx(32_500_000.0)


def test_processing_routes_bagasse_and_generates_five_saleable_outputs() -> None:
    _, result = _build()
    routing = result["processing"].monthly
    routed = routing[[
        "BagasseToElectricityTonnes",
        "BagasseToAnimalFeedTonnes",
        "BagasseForSaleTonnes",
    ]].sum(axis=1)
    np.testing.assert_allclose(routing["RawBagasseTonnes"], routed)
    np.testing.assert_allclose(
        routing["GrossElectricityMWh"],
        routing["InternalElectricityMWh"] + routing["ExportElectricityMWh"],
    )
    commercial = result["commercialization"].monthly
    for column in (
        "BioethanolRevenue",
        "SugarRevenue",
        "ElectricityRevenue",
        "BagasseRevenue",
        "AnimalFeedRevenue",
    ):
        assert commercial[column].sum() > 0


def test_component_financials_reconcile_to_consolidated_statements() -> None:
    _, result = _build()
    financials = result["financials"]
    assert set(financials.component_annual["Component"]) == EXPECTED_COMPONENTS
    np.testing.assert_allclose(
        financials.reconciliation["RevenueDifference"], 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        financials.reconciliation["EBITDADifference"], 0.0, atol=1e-6
    )
    checks = financials.checks.set_index("Check")
    assert checks.loc["Debt roll-forward", "Status"] == "OK"
    assert checks.loc["Balance sheet balances", "Status"] == "OK"
    assert checks.loc["Component consolidation reconciles", "Status"] == "OK"
    assert checks.loc["Minimum DSCR meets target", "Status"] == "WARN"
    assert financials.metrics["model_status"] == "OK"


def test_planning_start_controls_pre_operational_months() -> None:
    inputs = default_input_page()
    inputs.global_assumptions.planning_start = "2026-04"
    result = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")
    cycle = result["cycle_plan"].monthly
    sourcing = result["sourcing"].monthly
    pre_operational = cycle.index < "2026-04-01"
    assert cycle.loc[pre_operational, "Phase"].eq("Pre-operational").all()
    assert sourcing.loc[pre_operational, "TotalCaneTonnes"].eq(0.0).all()
