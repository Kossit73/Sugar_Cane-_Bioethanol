from __future__ import annotations

import copy

import numpy as np
import pytest

from sugarcane_model import SugarcaneBioethanolModel, default_input_page
from sugarcane_model.driver_schedules import (
    default_schedule_row,
    validate_driver_schedules,
)


def test_farm_planning_schedule_has_one_derived_row_per_year() -> None:
    inputs = default_input_page()
    result = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")
    annual = result["farm_plan"].annual

    assert annual.index.tolist() == list(range(2025, 2036))
    assert annual.columns.tolist() == [
        "TotalLandHa",
        "ArableCultivableLandHa",
        "PlannedCultivatedHa",
        "IrrigationCapacityHa",
        "PlannedIrrigatedHa",
        "RainFedHa",
        "FallowReserveHa",
        "HectaresHarvested",
        "HectaresReplanted",
        "FarmCaneAvailableTonnes",
    ]
    first = annual.loc[2025]
    assert first["TotalLandHa"] == pytest.approx(1_000.0)
    assert first["ArableCultivableLandHa"] == pytest.approx(800.0)
    assert first["PlannedCultivatedHa"] == pytest.approx(700.0)
    assert first["IrrigationCapacityHa"] == pytest.approx(600.0)
    assert first["PlannedIrrigatedHa"] == pytest.approx(500.0)
    assert first["RainFedHa"] == pytest.approx(200.0)
    assert first["FallowReserveHa"] == pytest.approx(100.0)
    assert first["HectaresReplanted"] == pytest.approx(
        first["HectaresHarvested"] * 0.20
    )
    assert first["FarmCaneAvailableTonnes"] == pytest.approx(
        first["HectaresHarvested"] * 70.0 * 0.98
    )


def test_yearly_land_plan_drives_farm_cane_and_purchase_requirement() -> None:
    inputs = default_input_page()
    first = default_schedule_row(inputs, "farm_planning")
    first["hectares_harvested"] = 300.0
    later = dict(first)
    later["Year"] = 2027
    later["hectares_harvested"] = 600.0
    inputs.yearly_schedules["farm_planning"] = [first, later]

    result = SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")
    farm = result["farming"].annual
    sourcing = result["sourcing"].annual

    assert farm.loc[2026, "FarmCaneAvailableTonnes"] == pytest.approx(
        300.0 * 70.0 * 0.98
    )
    assert farm.loc[2027, "FarmCaneAvailableTonnes"] == pytest.approx(
        600.0 * 70.0 * 0.98
    )
    assert (
        sourcing.loc[2027, "PurchasedCaneTonnes"]
        < sourcing.loc[2026, "PurchasedCaneTonnes"]
    )
    np.testing.assert_allclose(
        sourcing["PurchasedCaneTonnes"],
        np.maximum(
            sourcing["ProcessingCaneTargetTonnes"]
            - sourcing["FarmCaneProcessedTonnes"],
            0.0,
        ),
    )
    np.testing.assert_allclose(
        sourcing["TotalCaneTonnes"],
        sourcing["FarmCaneProcessedTonnes"] + sourcing["PurchasedCaneTonnes"],
    )


def test_hybrid_farm_share_is_a_target_not_a_production_override() -> None:
    low_target = default_input_page()
    high_target = copy.deepcopy(low_target)
    low_target.global_assumptions.hybrid_farm_share = 0.10
    high_target.global_assumptions.hybrid_farm_share = 0.90

    low = SugarcaneBioethanolModel(low_target, "HYBRID").build("HYBRID")
    high = SugarcaneBioethanolModel(high_target, "HYBRID").build("HYBRID")

    np.testing.assert_allclose(
        low["farming"].monthly["FarmCaneProcessedTonnes"],
        high["farming"].monthly["FarmCaneProcessedTonnes"],
    )
    np.testing.assert_allclose(
        low["sourcing"].monthly["PurchasedCaneTonnes"],
        high["sourcing"].monthly["PurchasedCaneTonnes"],
    )
    assert low["metrics"]["farm_share_target"] == pytest.approx(0.10)
    assert high["metrics"]["farm_share_target"] == pytest.approx(0.90)
    assert low["metrics"]["farm_share"] == pytest.approx(high["metrics"]["farm_share"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("arable_land_hectares", 1_100.0, "Arable/cultivable land"),
        ("planned_cultivated_hectares", 900.0, "Planned cultivated hectares"),
        ("planned_irrigated_hectares", 750.0, "planned cultivated hectares"),
        ("irrigation_capacity_hectares", 400.0, "irrigation capacity"),
        ("hectares_harvested", 750.0, "Hectares harvested"),
    ],
)
def test_land_capacity_validation_rejects_invalid_base_plan(
    field: str, value: float, message: str
) -> None:
    inputs = default_input_page()
    setattr(inputs.farm_planning, field, value)

    with pytest.raises(ValueError, match=message):
        SugarcaneBioethanolModel(inputs, "HYBRID").build("HYBRID")


def test_yearly_farm_plan_validation_reuses_the_land_rules() -> None:
    inputs = default_input_page()
    row = default_schedule_row(inputs, "farm_planning")
    row["planned_cultivated_hectares"] = row["arable_land_hectares"] + 1.0
    inputs.yearly_schedules["farm_planning"] = [row]

    errors = validate_driver_schedules(inputs)

    assert any("Planned cultivated hectares" in error for error in errors)


def test_farm_only_underutilizes_when_harvest_is_short_and_never_purchases() -> None:
    inputs = default_input_page()
    inputs.farm_planning.hectares_harvested = 100.0

    result = SugarcaneBioethanolModel(inputs, "FARM_ONLY").build("FARM_ONLY")
    sourcing = result["sourcing"].annual

    assert sourcing["PurchasedCaneTonnes"].sum() == pytest.approx(0.0)
    assert sourcing["FeedstockShortfallTonnes"].sum() > 0.0
    np.testing.assert_allclose(
        sourcing["TotalCaneTonnes"],
        sourcing["FarmCaneProcessedTonnes"],
    )


def test_land_and_feedstock_checks_are_visible_and_reconcile() -> None:
    result = SugarcaneBioethanolModel(default_input_page(), "HYBRID").build("HYBRID")
    checks = result["financials"].checks.set_index("Check")
    expected = [
        "Cultivated land is within arable/cultivable land",
        "Arable/cultivable land is within total land",
        "Irrigated land is within cultivated land",
        "Irrigated land is within irrigation capacity",
        "Farm cane processed is within harvested cane available",
        "Purchased cane reconciles the feedstock shortage",
    ]

    assert checks.loc[expected, "Status"].eq("OK").all()
