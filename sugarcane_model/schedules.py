"""Operating and financial schedule builders for Sugar Cane Bioethanol."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from sugarcane_finance_math import npf
from .driver_schedules import escalation_factor, parameter_series

from .inputs import CapexItem, SugarcaneBioethanolInputs


COMPONENTS: tuple[str, ...] = (
    "Farming",
    "Bioethanol",
    "Sugar",
    "Electricity Generation",
    "Bagasse",
    "Animal Feed",
)
PRODUCT_COMPONENTS: tuple[str, ...] = COMPONENTS[1:]
REVENUE_COLUMNS: dict[str, str] = {
    "Bioethanol": "BioethanolRevenue",
    "Sugar": "SugarRevenue",
    "Electricity Generation": "ElectricityRevenue",
    "Bagasse": "BagasseRevenue",
    "Animal Feed": "AnimalFeedRevenue",
}


@dataclass
class ScheduleOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame


@dataclass
class CapexOutput(ScheduleOutput):
    item_schedule: pd.DataFrame
    total_capex: float


@dataclass
class DebtOutput(ScheduleOutput):
    summary: pd.DataFrame
    facility_monthly: pd.DataFrame
    facility_annual: pd.DataFrame


@dataclass
class FinancialOutput:
    income_monthly: pd.DataFrame
    income_annual: pd.DataFrame
    cashflow_monthly: pd.DataFrame
    cashflow_annual: pd.DataFrame
    balance_monthly: pd.DataFrame
    balance_annual: pd.DataFrame
    component_monthly: pd.DataFrame
    component_annual: pd.DataFrame
    reconciliation: pd.DataFrame
    checks: pd.DataFrame
    metrics: dict[str, Any]


def _annual_sum(frame: pd.DataFrame) -> pd.DataFrame:
    annual = frame.groupby(frame.index.year).sum(numeric_only=True)
    annual.index.name = "Year"
    return annual


def _annual_last(frame: pd.DataFrame) -> pd.DataFrame:
    annual = frame.groupby(frame.index.year).last()
    annual.index.name = "Year"
    return annual




def _scenario_farm_share_target(inputs: SugarcaneBioethanolInputs, scenario: str) -> float:
    scenario_name = scenario.upper()
    if scenario_name == "FARM_ONLY":
        return 1.0
    if scenario_name == "BUY_ONLY":
        return 0.0
    return float(inputs.global_assumptions.hybrid_farm_share)


def _depreciation(spend: np.ndarray, life_months: int) -> np.ndarray:
    result = np.zeros(len(spend), dtype=float)
    life = max(1, int(life_months))
    for month, amount in enumerate(spend):
        first = month + 1
        last = min(len(spend), first + life)
        if amount > 0 and first < last:
            result[first:last] += amount / life
    return result


def _tax_with_nol(ebt: np.ndarray, rate: float) -> np.ndarray:
    tax = np.zeros(len(ebt), dtype=float)
    losses = 0.0
    for index, value in enumerate(ebt):
        if value <= 0:
            losses += -float(value)
            continue
        loss_used = min(losses, float(value))
        losses -= loss_used
        tax[index] = (float(value) - loss_used) * rate
    return tax


def build_timeline(inputs: SugarcaneBioethanolInputs) -> pd.DatetimeIndex:
    global_inputs = inputs.global_assumptions
    return pd.date_range(
        f"{global_inputs.start_year}-01-01",
        f"{global_inputs.end_year}-12-01",
        freq="MS",
    )


def compute_cycle_plan(inputs: SugarcaneBioethanolInputs) -> ScheduleOutput:
    dates = build_timeline(inputs)
    planning_start = pd.Period(inputs.global_assumptions.planning_start, freq="M").to_timestamp()
    active = np.asarray(dates >= planning_start)
    operating_month = np.maximum(
        0,
        (dates.year - planning_start.year) * 12
        + dates.month
        - planning_start.month,
    ).to_numpy(dtype=int)
    crop_cycle = parameter_series(
        inputs, "cycle_planning", "crop_cycle_months", dates, dtype=int
    )
    establishment = parameter_series(
        inputs, "cycle_planning", "establishment_months", dates, dtype=int
    )
    harvest_window = parameter_series(
        inputs, "cycle_planning", "harvest_window_months", dates, dtype=int
    )
    replant_share = parameter_series(
        inputs, "cycle_planning", "replant_share_per_cycle", dates
    )
    cycle_month = operating_month % crop_cycle + 1
    phases = np.where(
        active,
        np.where(
            cycle_month <= establishment,
            "Establishment",
            np.where(
                cycle_month > crop_cycle - harvest_window,
                "Harvest",
                "Growing",
            ),
        ),
        "Pre-operational",
    )
    operating_year = operating_month // 12
    ramp_year_1 = parameter_series(
        inputs, "other_assumptions", "startup_ramp_year_1", dates
    )
    ramp_year_2 = parameter_series(
        inputs, "other_assumptions", "startup_ramp_year_2", dates
    )
    ramp = np.where(
        active,
        np.where(operating_year == 0, ramp_year_1, np.where(operating_year == 1, ramp_year_2, 1.0)),
        0.0,
    )
    monthly = pd.DataFrame(
        {
            "CycleNumber": np.where(active, operating_month // crop_cycle + 1, 0),
            "CycleMonth": np.where(active, cycle_month, 0),
            "Phase": phases,
            "RampFactor": ramp,
            "ReplantShare": np.where(active & (cycle_month == 1), replant_share, 0.0),
        },
        index=dates,
    )
    annual = monthly.groupby(monthly.index.year).agg(
        Cycles=("CycleNumber", "max"),
        AverageRamp=("RampFactor", "mean"),
        ReplantEvents=("ReplantShare", lambda values: int((values > 0).sum())),
    )
    annual.index.name = "Year"
    return ScheduleOutput(monthly=monthly, annual=annual)


def compute_farm_planning_schedule(
    inputs: SugarcaneBioethanolInputs,
    cycle_plan: ScheduleOutput,
) -> ScheduleOutput:
    """Build the land-led annual plan and its calculated monthly deployment."""

    dates = cycle_plan.monthly.index

    def values(field: str) -> np.ndarray:
        return parameter_series(inputs, "farm_planning", field, dates)

    total_land = values("total_land_hectares")
    arable_land = values("arable_land_hectares")
    cultivated = values("planned_cultivated_hectares")
    irrigation_capacity = values("irrigation_capacity_hectares")
    irrigated = values("planned_irrigated_hectares")
    annual_harvested = values("hectares_harvested")
    active = cycle_plan.monthly["Phase"].ne("Pre-operational").to_numpy()
    active_counts = (
        pd.Series(active.astype(float), index=dates)
        .groupby(dates.year)
        .transform("sum")
        .to_numpy()
    )
    harvest_weights = np.divide(
        active.astype(float),
        active_counts,
        out=np.zeros(len(dates), dtype=float),
        where=active_counts > 0,
    )
    hectares_harvested = annual_harvested * harvest_weights

    replant_events = cycle_plan.monthly["ReplantShare"].to_numpy(dtype=float)
    hectares_replanted = np.zeros(len(dates), dtype=float)
    for year in sorted(set(dates.year)):
        year_mask = np.asarray(dates.year == year)
        event_total = float(replant_events[year_mask].sum())
        if event_total <= 0.0:
            continue
        planned_replant = min(
            float(cultivated[year_mask][-1]),
            float(annual_harvested[year_mask][-1]) * event_total,
        )
        hectares_replanted[year_mask] = (
            planned_replant * replant_events[year_mask] / event_total
        )

    cane_yield = parameter_series(
        inputs, "farming", "sugarcane_yield_tonnes_per_hectare", dates
    )
    harvest_recovery = parameter_series(inputs, "farming", "harvest_recovery", dates)
    monthly = pd.DataFrame(
        {
            "TotalLandHa": total_land,
            "ArableCultivableLandHa": arable_land,
            "PlannedCultivatedHa": cultivated,
            "IrrigationCapacityHa": irrigation_capacity,
            "PlannedIrrigatedHa": irrigated,
            "RainFedHa": np.maximum(cultivated - irrigated, 0.0),
            "FallowReserveHa": np.maximum(arable_land - cultivated, 0.0),
            "HectaresHarvested": hectares_harvested,
            "HectaresReplanted": hectares_replanted,
            "FarmCaneAvailableTonnes": (
                hectares_harvested * cane_yield * harvest_recovery
            ),
        },
        index=dates,
    )
    stock_columns = [
        "TotalLandHa",
        "ArableCultivableLandHa",
        "PlannedCultivatedHa",
        "IrrigationCapacityHa",
        "PlannedIrrigatedHa",
        "RainFedHa",
        "FallowReserveHa",
    ]
    flow_columns = [
        "HectaresHarvested",
        "HectaresReplanted",
        "FarmCaneAvailableTonnes",
    ]
    annual = monthly[stock_columns].groupby(dates.year).last()
    annual[flow_columns] = monthly[flow_columns].groupby(dates.year).sum()
    annual.index.name = "Year"
    return ScheduleOutput(monthly=monthly, annual=annual)


def compute_farming_schedule(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    cycle_plan: ScheduleOutput,
    farm_plan: ScheduleOutput,
) -> ScheduleOutput:
    dates = cycle_plan.monthly.index
    farm_share_target = _scenario_farm_share_target(inputs, scenario)
    cost_rates = parameter_series(
        inputs, "other_assumptions", "cost_inflation_rate", dates
    )
    cost_factor = escalation_factor(cost_rates)
    capacity = parameter_series(
        inputs, "processing_routing", "annual_cane_capacity_tonnes", dates
    )
    availability = parameter_series(
        inputs, "other_assumptions", "plant_availability", dates
    )
    farm_opex = parameter_series(inputs, "farming", "farm_opex_per_tonne", dates)
    annual_overhead = parameter_series(inputs, "farming", "farm_overhead_per_year", dates)
    base_transfer_price = parameter_series(
        inputs, "farming", "internal_transfer_price_per_tonne", dates
    )
    processing_target = (
        capacity * availability * cycle_plan.monthly["RampFactor"].to_numpy() / 12.0
    )
    farm_available = farm_plan.monthly["FarmCaneAvailableTonnes"].to_numpy()
    farm_operating = (
        scenario.upper() != "BUY_ONLY"
    ) & cycle_plan.monthly["Phase"].ne("Pre-operational").to_numpy()
    farm_available = np.where(farm_operating, farm_available, 0.0)
    farm_cane_processed = np.minimum(farm_available, processing_target)
    unused_farm_cane = np.maximum(farm_available - farm_cane_processed, 0.0)
    farm_cost = farm_available * farm_opex * cost_factor
    farm_overhead = annual_overhead / 12.0 * cost_factor * farm_operating.astype(float)
    transfer_price = base_transfer_price * cost_factor
    monthly = pd.DataFrame(
        {
            "FarmShareTarget": farm_share_target,
            "ProcessingCaneTargetTonnes": processing_target,
            "FarmCaneAvailableTonnes": farm_available,
            "FarmCaneProcessedTonnes": farm_cane_processed,
            "FarmCaneTonnes": farm_cane_processed,
            "UnusedFarmCaneTonnes": unused_farm_cane,
            "FarmAreaHarvestedHa": np.where(
                farm_operating,
                farm_plan.monthly["HectaresHarvested"].to_numpy(),
                0.0,
            ),
            "FarmAreaReplantedHa": np.where(
                farm_operating,
                farm_plan.monthly["HectaresReplanted"].to_numpy(),
                0.0,
            ),
            "FarmDirectCost": farm_cost,
            "FarmOverhead": farm_overhead,
            "FarmOperatingCost": farm_cost + farm_overhead,
            "InternalTransferPrice": transfer_price,
            "InternalCaneRevenue": farm_cane_processed * transfer_price,
        },
        index=dates,
    )
    monthly["FarmShareOfProcessingTarget"] = np.divide(
        farm_cane_processed,
        processing_target,
        out=np.zeros(len(dates), dtype=float),
        where=processing_target > 0,
    )
    rate_columns = {
        "FarmShareTarget",
        "FarmShareOfProcessingTarget",
        "InternalTransferPrice",
    }
    flow_columns = [column for column in monthly.columns if column not in rate_columns]
    annual = monthly[flow_columns].groupby(dates.year).sum()
    annual["FarmShareTarget"] = monthly["FarmShareTarget"].groupby(dates.year).last()
    annual["FarmShareOfProcessingTarget"] = np.divide(
        annual["FarmCaneProcessedTonnes"],
        annual["ProcessingCaneTargetTonnes"],
        out=np.zeros(len(annual), dtype=float),
        where=annual["ProcessingCaneTargetTonnes"].to_numpy() > 0,
    )
    annual["InternalTransferPrice"] = (
        monthly["InternalTransferPrice"].groupby(dates.year).mean()
    )
    annual.index.name = "Year"
    return ScheduleOutput(monthly=monthly, annual=annual)


def compute_sourcing_schedule(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    farming: ScheduleOutput,
    cycle_plan: ScheduleOutput,
) -> ScheduleOutput:
    dates = farming.monthly.index
    capacity = parameter_series(
        inputs, "processing_routing", "annual_cane_capacity_tonnes", dates
    )
    availability = parameter_series(
        inputs, "other_assumptions", "plant_availability", dates
    )
    processing_target = (
        capacity * availability * cycle_plan.monthly["RampFactor"].to_numpy() / 12.0
    )
    farm_cane_processed = farming.monthly["FarmCaneProcessedTonnes"].to_numpy()
    if scenario.upper() == "FARM_ONLY":
        purchased_cane = np.zeros(len(dates), dtype=float)
    else:
        purchased_cane = np.maximum(0.0, processing_target - farm_cane_processed)
    total_cane = farm_cane_processed + purchased_cane
    feedstock_shortfall = np.maximum(processing_target - total_cane, 0.0)
    farm_share_target = _scenario_farm_share_target(inputs, scenario)
    supplier_loss = parameter_series(inputs, "sourcing", "supplier_loss_rate", dates)
    supplier_gross = purchased_cane / np.maximum(1e-9, 1.0 - supplier_loss)
    cost_rates = parameter_series(
        inputs, "other_assumptions", "cost_inflation_rate", dates
    )
    cost_factor = escalation_factor(cost_rates)
    spot_price = parameter_series(
        inputs, "sourcing", "cane_purchase_price_per_tonne", dates
    )
    contract_share = parameter_series(
        inputs, "sourcing", "contracted_purchase_share", dates
    )
    contract_discount = parameter_series(inputs, "sourcing", "contract_discount", dates)
    logistics_rate = parameter_series(
        inputs, "sourcing", "logistics_cost_per_tonne", dates
    )
    contracted_price = spot_price * (1.0 - contract_discount)
    weighted_price = contract_share * contracted_price + (1.0 - contract_share) * spot_price
    purchase_price = weighted_price * cost_factor
    purchase_cost = supplier_gross * purchase_price
    logistics = purchased_cane * logistics_rate * cost_factor
    farm_transfer = farming.monthly["InternalCaneRevenue"].to_numpy()
    monthly = pd.DataFrame(
        {
            "FarmShareTarget": farm_share_target,
            "ActualFarmShare": np.divide(
                farm_cane_processed,
                total_cane,
                out=np.zeros(len(dates), dtype=float),
                where=total_cane > 0,
            ),
            "ProcessingCaneTargetTonnes": processing_target,
            "TotalCaneTonnes": total_cane,
            "FeedstockShortfallTonnes": feedstock_shortfall,
            "FarmCaneAvailableTonnes": farming.monthly[
                "FarmCaneAvailableTonnes"
            ].to_numpy(),
            "FarmCaneProcessedTonnes": farm_cane_processed,
            "FarmCaneTonnes": farm_cane_processed,
            "PurchasedCaneTonnes": purchased_cane,
            "SupplierGrossTonnes": supplier_gross,
            "EffectivePurchasePrice": purchase_price,
            "PurchasedCaneCost": purchase_cost,
            "InboundLogisticsCost": logistics,
            "FarmTransferCost": farm_transfer,
            "TotalFeedstockTransferCost": farm_transfer + purchase_cost + logistics,
        },
        index=dates,
    )
    rate_columns = {"FarmShareTarget", "ActualFarmShare", "EffectivePurchasePrice"}
    flow_columns = [column for column in monthly.columns if column not in rate_columns]
    annual = monthly[flow_columns].groupby(dates.year).sum()
    annual["FarmShareTarget"] = monthly["FarmShareTarget"].groupby(dates.year).last()
    annual["ActualFarmShare"] = np.divide(
        annual["FarmCaneProcessedTonnes"],
        annual["TotalCaneTonnes"],
        out=np.zeros(len(annual), dtype=float),
        where=annual["TotalCaneTonnes"].to_numpy() > 0,
    )
    annual["EffectivePurchasePrice"] = (
        monthly["EffectivePurchasePrice"].groupby(dates.year).mean()
    )
    annual.index.name = "Year"
    return ScheduleOutput(monthly=monthly, annual=annual)


def compute_processing_routing(
    inputs: SugarcaneBioethanolInputs,
    sourcing: ScheduleOutput,
) -> ScheduleOutput:
    dates = sourcing.monthly.index
    cane = sourcing.monthly["TotalCaneTonnes"].to_numpy()
    process_loss = parameter_series(inputs, "other_assumptions", "process_loss", dates)
    ethanol_yield = parameter_series(
        inputs, "processing_routing", "ethanol_litres_per_tonne", dates
    )
    sugar_yield = parameter_series(
        inputs, "processing_routing", "sugar_tonnes_per_tonne", dates
    )
    bagasse_yield = parameter_series(
        inputs, "processing_routing", "raw_bagasse_tonnes_per_tonne", dates
    )
    power_share = parameter_series(
        inputs, "processing_routing", "bagasse_to_electricity_share", dates
    )
    feed_share = parameter_series(
        inputs, "processing_routing", "bagasse_to_animal_feed_share", dates
    )
    sale_share = parameter_series(
        inputs, "processing_routing", "bagasse_to_sale_share", dates
    )
    electricity_yield = parameter_series(
        inputs, "processing_routing", "electricity_mwh_per_tonne_bagasse", dates
    )
    internal_use = parameter_series(
        inputs, "processing_routing", "internal_electricity_mwh_per_tonne_cane", dates
    )
    feed_conversion = parameter_series(
        inputs, "processing_routing", "animal_feed_conversion_rate", dates
    )
    useful_cane = cane * (1.0 - process_loss)
    ethanol = useful_cane * ethanol_yield
    sugar = useful_cane * sugar_yield
    raw_bagasse = useful_cane * bagasse_yield
    electricity_bagasse = raw_bagasse * power_share
    feed_bagasse = raw_bagasse * feed_share
    sale_bagasse = raw_bagasse * sale_share
    gross_power = electricity_bagasse * electricity_yield
    internal_power = np.minimum(gross_power, cane * internal_use)
    exported_power = np.maximum(0.0, gross_power - internal_power)
    animal_feed = feed_bagasse * feed_conversion
    monthly = pd.DataFrame(
        {
            "CaneReceivedTonnes": cane, "UsefulCaneTonnes": useful_cane,
            "BioethanolLitres": ethanol, "SugarTonnes": sugar,
            "RawBagasseTonnes": raw_bagasse,
            "BagasseToElectricityTonnes": electricity_bagasse,
            "BagasseToAnimalFeedTonnes": feed_bagasse,
            "BagasseForSaleTonnes": sale_bagasse, "GrossElectricityMWh": gross_power,
            "InternalElectricityMWh": internal_power, "ExportElectricityMWh": exported_power,
            "AnimalFeedTonnes": animal_feed,
        }, index=dates,
    )
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_commercialization(
    inputs: SugarcaneBioethanolInputs,
    processing: ScheduleOutput,
) -> ScheduleOutput:
    dates = processing.monthly.index
    price_rates = parameter_series(
        inputs, "other_assumptions", "price_escalation_rate", dates
    )
    price_factor = escalation_factor(price_rates)

    def values(field: str) -> np.ndarray:
        return parameter_series(inputs, "commercialization", field, dates)

    ethanol_capture = values("ethanol_sales_capture")
    sugar_capture = values("sugar_sales_capture")
    power_capture = values("power_sales_capture")
    coproduct_capture = values("coproduct_sales_capture")
    ethanol_price = values("ethanol_price_per_litre") * price_factor
    sugar_price = values("sugar_price_per_tonne") * price_factor
    electricity_price = values("electricity_tariff_per_mwh") * price_factor
    bagasse_price = values("bagasse_price_per_tonne") * price_factor
    animal_feed_price = values("animal_feed_price_per_tonne") * price_factor

    monthly = pd.DataFrame(index=dates)
    monthly["BioethanolSalesLitres"] = processing.monthly["BioethanolLitres"] * ethanol_capture
    monthly["SugarSalesTonnes"] = processing.monthly["SugarTonnes"] * sugar_capture
    monthly["ElectricitySalesMWh"] = processing.monthly["ExportElectricityMWh"] * power_capture
    monthly["BagasseSalesTonnes"] = processing.monthly["BagasseForSaleTonnes"] * coproduct_capture
    monthly["AnimalFeedSalesTonnes"] = processing.monthly["AnimalFeedTonnes"] * coproduct_capture
    monthly["BioethanolRevenue"] = monthly["BioethanolSalesLitres"] * ethanol_price
    monthly["SugarRevenue"] = monthly["SugarSalesTonnes"] * sugar_price
    monthly["ElectricityRevenue"] = monthly["ElectricitySalesMWh"] * electricity_price
    monthly["BagasseRevenue"] = monthly["BagasseSalesTonnes"] * bagasse_price
    monthly["AnimalFeedRevenue"] = monthly["AnimalFeedSalesTonnes"] * animal_feed_price
    monthly["ExternalRevenue"] = monthly[list(REVENUE_COLUMNS.values())].sum(axis=1)
    monthly["ElectricityBagasseTransferCost"] = (
        processing.monthly["BagasseToElectricityTonnes"] * bagasse_price
    )
    monthly["AnimalFeedBagasseTransferCost"] = (
        processing.monthly["BagasseToAnimalFeedTonnes"] * bagasse_price
    )
    monthly["InternalBagasseTransferRevenue"] = (
        monthly["ElectricityBagasseTransferCost"] + monthly["AnimalFeedBagasseTransferCost"]
    )
    monthly["InternalElectricityTransferRevenue"] = (
        processing.monthly["InternalElectricityMWh"] * electricity_price
    )
    monthly["TotalInternalTransferRevenue"] = (
        monthly["InternalBagasseTransferRevenue"] + monthly["InternalElectricityTransferRevenue"]
    )
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_cost_schedule(
    inputs: SugarcaneBioethanolInputs,
    farming: ScheduleOutput,
    sourcing: ScheduleOutput,
    processing: ScheduleOutput,
) -> ScheduleOutput:
    dates = processing.monthly.index
    cost_rates = parameter_series(
        inputs, "other_assumptions", "cost_inflation_rate", dates
    )
    factor = escalation_factor(cost_rates)

    def values(field: str) -> np.ndarray:
        return parameter_series(inputs, "costs", field, dates) * factor

    monthly = pd.DataFrame(index=dates)
    monthly["FarmingOperatingCost"] = farming.monthly["FarmOperatingCost"]
    monthly["PurchasedCaneCost"] = sourcing.monthly["PurchasedCaneCost"]
    monthly["InboundLogisticsCost"] = sourcing.monthly["InboundLogisticsCost"]
    monthly["ProcessingVariableCost"] = (
        processing.monthly["CaneReceivedTonnes"] * values("processing_variable_cost_per_tonne_cane")
    )
    monthly["BioethanolVariableCost"] = (
        processing.monthly["BioethanolLitres"] * values("ethanol_variable_cost_per_litre")
    )
    monthly["SugarVariableCost"] = processing.monthly["SugarTonnes"] * values("sugar_variable_cost_per_tonne")
    monthly["ElectricityVariableCost"] = (
        processing.monthly["GrossElectricityMWh"] * values("electricity_variable_cost_per_mwh")
    )
    monthly["BagasseHandlingCost"] = (
        processing.monthly["BagasseForSaleTonnes"] * values("bagasse_handling_cost_per_tonne")
    )
    monthly["AnimalFeedVariableCost"] = (
        processing.monthly["AnimalFeedTonnes"] * values("animal_feed_variable_cost_per_tonne")
    )
    monthly["FixedProcessingOpex"] = values("fixed_processing_opex_per_year") / 12.0
    monthly["CommercialAndAdminCost"] = values("commercial_and_admin_cost_per_year") / 12.0
    monthly["ProductSpecificVariableCost"] = monthly[
        ["BioethanolVariableCost", "SugarVariableCost", "ElectricityVariableCost",
         "BagasseHandlingCost", "AnimalFeedVariableCost"]
    ].sum(axis=1)
    monthly["EconomicOperatingCost"] = monthly[
        ["FarmingOperatingCost", "PurchasedCaneCost", "InboundLogisticsCost",
         "ProcessingVariableCost", "ProductSpecificVariableCost",
         "FixedProcessingOpex", "CommercialAndAdminCost"]
    ].sum(axis=1)
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_capex_schedule(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    sourcing: ScheduleOutput,
) -> CapexOutput:
    dates = build_timeline(inputs)
    total_processed_cane = float(sourcing.monthly["TotalCaneTonnes"].sum())
    farm_capex_share = (
        float(sourcing.monthly["FarmCaneProcessedTonnes"].sum())
        / total_processed_cane
        if total_processed_cane > 0.0
        else 0.0
    )
    components = sorted({item.component for item in inputs.capex.items} | {"Shared Processing"})
    capex_by_component = {component: np.zeros(len(dates)) for component in components}
    depreciation_by_component = {component: np.zeros(len(dates)) for component in components}
    rows: list[dict[str, Any]] = []

    for item in inputs.capex.items:
        amount = float(item.amount) * (
            farm_capex_share if item.is_farm_capex else 1.0
        )
        start = pd.Period(item.start_month, freq="M").to_timestamp()
        if start < dates[0]:
            start = dates[0]
        if start > dates[-1]:
            continue
        start_index = int(np.searchsorted(dates.values, start.to_datetime64()))
        spend_months = min(item.spend_months, len(dates) - start_index)
        spend = np.zeros(len(dates))
        if spend_months > 0:
            spend[start_index : start_index + spend_months] = amount / spend_months
        capex_by_component[item.component] += spend
        depreciation_by_component[item.component] += _depreciation(spend, item.life_years * 12)
        rows.append(
            {
                "Item": item.item,
                "Component": item.component,
                "ScenarioAmount": amount,
                "StartMonth": item.start_month,
                "SpendMonths": item.spend_months,
                "LifeYears": item.life_years,
                "IsFarmCapex": item.is_farm_capex,
            }
        )

    monthly = pd.DataFrame(index=dates)
    for component in components:
        monthly[f"Capex::{component}"] = capex_by_component[component]
        monthly[f"Depreciation::{component}"] = depreciation_by_component[component]
    monthly["TotalCapex"] = monthly.filter(like="Capex::").sum(axis=1)
    monthly["TotalDepreciation"] = monthly.filter(like="Depreciation::").sum(axis=1)
    return CapexOutput(
        monthly=monthly,
        annual=_annual_sum(monthly),
        item_schedule=pd.DataFrame(rows),
        total_capex=float(monthly["TotalCapex"].sum()),
    )


def _build_facility_schedule(
    dates: pd.DatetimeIndex,
    draws: np.ndarray,
    *,
    annual_interest_rates: np.ndarray,
    tenor_years: int,
    grace_years: int,
    amortization_type: str,
    capitalize_idc: bool,
) -> pd.DataFrame:
    """Build one loan schedule; every facility uses this shared engine."""
    months = len(dates)
    monthly_rates = np.asarray(annual_interest_rates, dtype=float) / 12.0
    grace = min(grace_years * 12, months)
    maturity = min(tenor_years * 12, months)
    opening = np.zeros(months)
    interest = np.zeros(months)
    idc = np.zeros(months)
    cash_interest = np.zeros(months)
    principal = np.zeros(months)
    closing = np.zeros(months)
    balance = 0.0

    for month in range(months):
        monthly_rate = monthly_rates[month]
        opening[month] = balance
        interest[month] = (balance + 0.5 * draws[month]) * monthly_rate
        if month < grace and capitalize_idc:
            idc[month] = interest[month]
        else:
            cash_interest[month] = interest[month]
        available = balance + draws[month] + idc[month]
        if month >= grace and month < maturity:
            remaining = max(1, maturity - month)
            if amortization_type == "annuity" and monthly_rate > 0:
                payment = available * monthly_rate / (
                    1.0 - (1.0 + monthly_rate) ** (-remaining)
                )
                scheduled = max(0.0, payment - cash_interest[month])
            else:
                scheduled = available / remaining
            principal[month] = min(available, scheduled)
        balance = max(0.0, available - principal[month])
        closing[month] = balance

    return pd.DataFrame(
        {
            "OpeningBalance": opening,
            "Draw": draws,
            "Interest": interest,
            "CapitalizedInterest": idc,
            "CashInterest": cash_interest,
            "Principal": principal,
            "DebtService": cash_interest + principal,
            "ClosingBalance": closing,
        },
        index=dates,
    )


def _annualize_debt_schedule(monthly: pd.DataFrame) -> pd.DataFrame:
    annual = _annual_sum(monthly.drop(columns=["OpeningBalance", "ClosingBalance"]))
    annual["OpeningBalance"] = monthly.groupby(monthly.index.year)[
        "OpeningBalance"
    ].first()
    annual["ClosingBalance"] = monthly.groupby(monthly.index.year)[
        "ClosingBalance"
    ].last()
    return annual[
        [
            "OpeningBalance",
            "Draw",
            "CapitalizedInterest",
            "CashInterest",
            "Principal",
            "DebtService",
            "ClosingBalance",
        ]
    ]


def compute_debt_schedule(
    inputs: SugarcaneBioethanolInputs,
    capex: CapexOutput,
) -> DebtOutput:
    dates = capex.monthly.index
    financing = inputs.financing
    debt_ratios = parameter_series(inputs, "financing", "debt_ratio", dates)
    senior_interest_rates = parameter_series(
        inputs, "financing", "interest_rate", dates
    )
    capex_spend = capex.monthly["TotalCapex"].to_numpy(dtype=float)
    total_capex = float(capex_spend.sum())
    if total_capex <= 0:
        raise ValueError("Total CAPEX must be positive before debt can be scheduled.")

    senior_draws = capex_spend * debt_ratios
    fixed_commitment = sum(
        facility.amount for facility in financing.additional_debt_facilities
    )
    total_commitment = float(senior_draws.sum()) + fixed_commitment
    tolerance = max(1.0, total_capex * 1e-8)
    if total_commitment > total_capex + tolerance:
        raise ValueError(
            "Combined senior and additional debt commitments "
            f"(USD {total_commitment:,.0f}) exceed scenario CAPEX "
            f"(USD {total_capex:,.0f})."
        )

    facility_specs: list[dict[str, Any]] = [
        {
            "name": "Senior Debt",
            "facility_type": "Senior (% of CAPEX)",
            "debt_ratio": float(
                np.average(
                    debt_ratios,
                    weights=np.maximum(capex.monthly["TotalCapex"], 1e-9),
                )
            ),
            "draws": senior_draws,
            "interest_rates": senior_interest_rates,
            "summary_interest_rate": float(np.mean(senior_interest_rates)),
            "tenor_years": financing.tenor_years,
            "grace_years": financing.grace_years,
            "amortization_type": financing.amortization_type,
            "capitalize_idc": financing.capitalize_idc,
        }
    ]
    capex_profile = capex_spend / total_capex
    for facility in financing.additional_debt_facilities:
        facility_specs.append(
            {
                "name": facility.name.strip(),
                "facility_type": "Additional (fixed amount)",
                "debt_ratio": np.nan,
                "draws": capex_profile * facility.amount,
                "interest_rates": np.full(len(dates), facility.interest_rate),
                "summary_interest_rate": facility.interest_rate,
                "tenor_years": facility.tenor_years,
                "grace_years": facility.grace_years,
                "amortization_type": facility.amortization_type,
                "capitalize_idc": facility.capitalize_idc,
            }
        )

    schedules: list[pd.DataFrame] = []
    monthly_details: list[pd.DataFrame] = []
    annual_details: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in facility_specs:
        schedule = _build_facility_schedule(
            dates,
            np.asarray(spec["draws"], dtype=float),
            annual_interest_rates=np.asarray(
                spec["interest_rates"], dtype=float
            ),
            tenor_years=int(spec["tenor_years"]),
            grace_years=int(spec["grace_years"]),
            amortization_type=str(spec["amortization_type"]),
            capitalize_idc=bool(spec["capitalize_idc"]),
        )
        schedules.append(schedule)
        monthly_detail = schedule.copy()
        monthly_detail.insert(0, "Facility", spec["name"])
        monthly_details.append(monthly_detail)
        annual_detail = _annualize_debt_schedule(schedule)
        annual_detail.insert(0, "Facility", spec["name"])
        annual_details.append(annual_detail)
        summary_rows.append(
            {
                "Facility": spec["name"],
                "FacilityType": spec["facility_type"],
                "DebtRatio": spec["debt_ratio"],
                "InterestRate": spec["summary_interest_rate"],
                "TenorYears": spec["tenor_years"],
                "GraceYears": spec["grace_years"],
                "Amortization": spec["amortization_type"],
                "CapitalizeIDC": spec["capitalize_idc"],
                "InitialDraw": float(schedule["Draw"].sum()),
                "CapitalizedInterest": float(
                    schedule["CapitalizedInterest"].sum()
                ),
                "EndingBalance": float(schedule["ClosingBalance"].iloc[-1]),
            }
        )

    monthly = schedules[0].copy()
    for schedule in schedules[1:]:
        monthly = monthly.add(schedule, fill_value=0.0)
    annual = _annualize_debt_schedule(monthly)
    facility_monthly = pd.concat(monthly_details)
    facility_monthly.index.name = "Date"
    facility_annual = pd.concat(annual_details)
    facility_annual.index.name = "Year"
    return DebtOutput(
        monthly=monthly,
        annual=annual,
        summary=pd.DataFrame(summary_rows),
        facility_monthly=facility_monthly,
        facility_annual=facility_annual,
    )


def compute_working_capital(
    inputs: SugarcaneBioethanolInputs,
    commercial: ScheduleOutput,
    costs: ScheduleOutput,
) -> ScheduleOutput:
    dates = commercial.monthly.index
    receivable_days = parameter_series(inputs, "working_capital", "receivable_days", dates)
    inventory_days = parameter_series(inputs, "working_capital", "inventory_days", dates)
    payable_days = parameter_series(inputs, "working_capital", "payable_days", dates)
    days_per_month = 365.0 / 12.0
    revenue = commercial.monthly["ExternalRevenue"]
    cash_cost = costs.monthly["EconomicOperatingCost"]
    eligible_payables = costs.monthly[
        ["PurchasedCaneCost", "InboundLogisticsCost", "ProcessingVariableCost",
         "ProductSpecificVariableCost"]
    ].sum(axis=1)
    monthly = pd.DataFrame(index=dates)
    monthly["AccountsReceivable"] = revenue * receivable_days / days_per_month
    monthly["Inventory"] = cash_cost * inventory_days / days_per_month
    monthly["AccountsPayable"] = eligible_payables * payable_days / days_per_month
    monthly["NetWorkingCapital"] = (
        monthly["AccountsReceivable"] + monthly["Inventory"] - monthly["AccountsPayable"]
    )
    monthly["DeltaNWC"] = monthly["NetWorkingCapital"].diff().fillna(monthly["NetWorkingCapital"])
    annual = _annual_last(monthly.drop(columns=["DeltaNWC"]))
    annual["DeltaNWC"] = monthly.groupby(monthly.index.year)["DeltaNWC"].sum()
    return ScheduleOutput(monthly=monthly, annual=annual)


def _product_revenue_shares(commercial: ScheduleOutput) -> pd.DataFrame:
    revenue = commercial.monthly[list(REVENUE_COLUMNS.values())].copy()
    total = revenue.sum(axis=1).replace(0.0, np.nan)
    shares = revenue.div(total, axis=0).fillna(1.0 / len(PRODUCT_COMPONENTS))
    shares.columns = PRODUCT_COMPONENTS
    return shares


def _component_capex_and_depreciation(
    capex: CapexOutput,
    commercial: ScheduleOutput,
    idc_depreciation: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shares = _product_revenue_shares(commercial)
    component_capex = pd.DataFrame(0.0, index=capex.monthly.index, columns=COMPONENTS)
    component_dep = component_capex.copy()
    shared_capex = capex.monthly.get("Capex::Shared Processing", 0.0)
    shared_dep = capex.monthly.get("Depreciation::Shared Processing", 0.0)
    for component in COMPONENTS:
        component_capex[component] = capex.monthly.get(f"Capex::{component}", 0.0)
        component_dep[component] = capex.monthly.get(f"Depreciation::{component}", 0.0)
        if component in PRODUCT_COMPONENTS:
            component_capex[component] += shared_capex * shares[component]
            component_dep[component] += shared_dep * shares[component]
    capex_weights = component_capex.cumsum().div(
        component_capex.cumsum().sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(0.0)
    component_dep = component_dep.add(capex_weights.mul(idc_depreciation, axis=0), fill_value=0.0)
    return component_capex, component_dep


def _annualized_irr(cashflows: np.ndarray) -> float | None:
    if not np.any(cashflows < 0) or not np.any(cashflows > 0):
        return None
    monthly = float(npf.irr(cashflows))
    if not math.isfinite(monthly) or monthly <= -1:
        return None
    annual = (1.0 + monthly) ** 12 - 1.0
    return float(annual) if math.isfinite(annual) else None


def _npv(cashflows: np.ndarray, annual_rate: float) -> float:
    monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    return float(npf.npv(monthly_rate, cashflows))


def _payback(cashflows: np.ndarray) -> float | None:
    cumulative = np.cumsum(cashflows)
    recovered = np.flatnonzero(cumulative >= 0)
    if not np.any(cumulative < 0) or len(recovered) == 0:
        return None
    return float((recovered[0] + 1) / 12.0)


def compute_financials(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    farming: ScheduleOutput,
    farm_plan: ScheduleOutput,
    sourcing: ScheduleOutput,
    processing: ScheduleOutput,
    commercial: ScheduleOutput,
    costs: ScheduleOutput,
    capex: CapexOutput,
    debt: DebtOutput,
    working_capital: ScheduleOutput,
) -> FinancialOutput:
    dates = commercial.monthly.index
    idc = debt.monthly["CapitalizedInterest"].to_numpy()
    weighted_life = int(
        round(
            np.average(
                [item.life_years for item in inputs.capex.items],
                weights=[max(item.amount, 1.0) for item in inputs.capex.items],
            )
        )
    )
    idc_depreciation = _depreciation(idc, weighted_life * 12)
    depreciation = capex.monthly["TotalDepreciation"].to_numpy() + idc_depreciation
    revenue = commercial.monthly["ExternalRevenue"].to_numpy()
    operating_cost = costs.monthly["EconomicOperatingCost"].to_numpy()
    ebitda = revenue - operating_cost
    ebit = ebitda - depreciation
    cash_interest = debt.monthly["CashInterest"].to_numpy()
    tax = _tax_with_nol(ebit - cash_interest, inputs.global_assumptions.corporate_tax_rate)
    net_income = ebit - cash_interest - tax
    delta_nwc = working_capital.monthly["DeltaNWC"].to_numpy()
    capex_spend = capex.monthly["TotalCapex"].to_numpy()
    debt_draw = debt.monthly["Draw"].to_numpy()
    principal = debt.monthly["Principal"].to_numpy()
    debt_service = debt.monthly["DebtService"].to_numpy()
    equity_contribution = np.maximum(0.0, capex_spend - debt_draw)
    cfads = ebitda - tax - delta_nwc
    project_fcf = ebitda - tax - delta_nwc - capex_spend
    equity_fcf = -equity_contribution + cfads - cash_interest - principal
    dscr = np.divide(
        cfads,
        debt_service,
        out=np.full(len(dates), np.nan),
        where=debt_service > 1e-9,
    )

    income_monthly = pd.DataFrame(
        {
            "Revenue": revenue,
            "OperatingCosts": operating_cost,
            "EBITDA": ebitda,
            "Depreciation": depreciation,
            "EBIT": ebit,
            "Interest": cash_interest,
            "Tax": tax,
            "NetIncome": net_income,
        },
        index=dates,
    )
    cashflow_monthly = pd.DataFrame(
        {
            "EBITDA": ebitda,
            "Tax": tax,
            "DeltaNWC": delta_nwc,
            "CFADS": cfads,
            "Capex": capex_spend,
            "DebtDraw": debt_draw,
            "CapitalizedInterest": idc,
            "CashInterest": cash_interest,
            "Principal": principal,
            "DebtService": debt_service,
            "EquityContribution": equity_contribution,
            "ProjectFCF": project_fcf,
            "EquityFCF": equity_fcf,
            "DSCR": dscr,
        },
        index=dates,
    )

    cash_change = (
        ebitda
        - tax
        - delta_nwc
        - capex_spend
        + debt_draw
        + equity_contribution
        - cash_interest
        - principal
    )
    cash = np.cumsum(cash_change)
    gross_ppe = np.cumsum(capex_spend + idc)
    accumulated_depreciation = np.cumsum(depreciation)
    net_ppe = gross_ppe - accumulated_depreciation
    paid_in_capital = np.cumsum(equity_contribution)
    retained_earnings = np.cumsum(net_income)
    ar = working_capital.monthly["AccountsReceivable"].to_numpy()
    inventory = working_capital.monthly["Inventory"].to_numpy()
    ap = working_capital.monthly["AccountsPayable"].to_numpy()
    debt_balance = debt.monthly["ClosingBalance"].to_numpy()
    assets = cash + ar + inventory + net_ppe
    liabilities = ap + debt_balance
    equity = paid_in_capital + retained_earnings
    balance_check = assets - liabilities - equity
    balance_monthly = pd.DataFrame(
        {
            "Cash": cash,
            "AccountsReceivable": ar,
            "Inventory": inventory,
            "GrossPPE": gross_ppe,
            "AccumulatedDepreciation": accumulated_depreciation,
            "NetPPE": net_ppe,
            "TotalAssets": assets,
            "AccountsPayable": ap,
            "Debt": debt_balance,
            "TotalLiabilities": liabilities,
            "PaidInCapital": paid_in_capital,
            "RetainedEarnings": retained_earnings,
            "TotalEquity": equity,
            "BalanceCheck": balance_check,
        },
        index=dates,
    )

    shares = _product_revenue_shares(commercial)
    component_capex, component_dep = _component_capex_and_depreciation(
        capex, commercial, idc_depreciation
    )
    component_weight = component_capex.cumsum().div(
        component_capex.cumsum().sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(0.0)
    component_interest = component_weight.mul(cash_interest, axis=0)
    component_principal = component_weight.mul(principal, axis=0)
    component_delta_nwc = shares.mul(delta_nwc, axis=0)
    direct_product_costs = {
        "Bioethanol": costs.monthly["BioethanolVariableCost"],
        "Sugar": costs.monthly["SugarVariableCost"],
        "Electricity Generation": costs.monthly["ElectricityVariableCost"],
        "Bagasse": costs.monthly["BagasseHandlingCost"],
        "Animal Feed": costs.monthly["AnimalFeedVariableCost"],
    }
    power_consumers = ["Bioethanol", "Sugar", "Bagasse", "Animal Feed"]
    power_shares = shares[power_consumers].div(
        shares[power_consumers].sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(1.0 / len(power_consumers))
    internal_power_cost = commercial.monthly["InternalElectricityTransferRevenue"]
    direct_product_costs["Electricity Generation"] += commercial.monthly["ElectricityBagasseTransferCost"]
    direct_product_costs["Animal Feed"] += commercial.monthly["AnimalFeedBagasseTransferCost"]
    feedstock_transfer = sourcing.monthly["TotalFeedstockTransferCost"]
    shared_opex = costs.monthly[
        ["ProcessingVariableCost", "FixedProcessingOpex", "CommercialAndAdminCost"]
    ].sum(axis=1)

    component_frames: list[pd.DataFrame] = []
    for component in COMPONENTS:
        frame = pd.DataFrame(index=dates)
        frame["Component"] = component
        if component == "Farming":
            frame["Revenue"] = farming.monthly["InternalCaneRevenue"]
            frame["DirectCosts"] = farming.monthly["FarmOperatingCost"]
            frame["SharedOpex"] = 0.0
            frame["DeltaNWC"] = 0.0
        else:
            frame["Revenue"] = commercial.monthly[REVENUE_COLUMNS[component]]
            if component == "Electricity Generation":
                frame["Revenue"] += commercial.monthly["InternalElectricityTransferRevenue"]
            elif component == "Bagasse":
                frame["Revenue"] += commercial.monthly["InternalBagasseTransferRevenue"]
            frame["DirectCosts"] = direct_product_costs[component] + feedstock_transfer * shares[component]
            if component in power_consumers:
                frame["DirectCosts"] += internal_power_cost * power_shares[component]

            frame["SharedOpex"] = shared_opex * shares[component]
            frame["DeltaNWC"] = component_delta_nwc[component]
        frame["EBITDA"] = frame["Revenue"] - frame["DirectCosts"] - frame["SharedOpex"]
        frame["Depreciation"] = component_dep[component]
        frame["EBIT"] = frame["EBITDA"] - frame["Depreciation"]
        frame["Interest"] = component_interest[component]
        frame["EBT"] = frame["EBIT"] - frame["Interest"]
        frame["Tax"] = 0.0
        frame["NetIncome"] = frame["EBT"]
        frame["Capex"] = component_capex[component]
        frame["Principal"] = component_principal[component]
        component_frames.append(frame)

    component_monthly = pd.concat(component_frames).reset_index(names="Date")
    positive_ebt = component_monthly.pivot(index="Date", columns="Component", values="EBT").clip(lower=0.0)
    tax_weights = positive_ebt.div(positive_ebt.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    for component in COMPONENTS:
        mask = component_monthly["Component"] == component
        allocated_tax = tax_weights[component].to_numpy() * tax
        component_monthly.loc[mask, "Tax"] = allocated_tax
        component_monthly.loc[mask, "NetIncome"] = (
            component_monthly.loc[mask, "EBT"].to_numpy() - allocated_tax
        )
    component_monthly["OperatingCashFlow"] = (
        component_monthly["NetIncome"]
        + component_monthly["Depreciation"]
        - component_monthly["DeltaNWC"]
    )
    component_monthly["FreeCashFlow"] = (
        component_monthly["OperatingCashFlow"]
        - component_monthly["Capex"]
        - component_monthly["Principal"]
    )
    component_monthly["Year"] = component_monthly["Date"].dt.year
    component_annual = component_monthly.groupby(["Year", "Component"], as_index=False).sum(
        numeric_only=True
    )

    annual_component_totals = component_annual.groupby("Year", as_index=False)[
        ["Revenue", "DirectCosts", "SharedOpex", "EBITDA", "Depreciation", "EBIT", "Interest", "Tax", "NetIncome"]
    ].sum()
    farm_internal = farming.monthly["InternalCaneRevenue"].groupby(dates.year).sum()
    power_internal = commercial.monthly["InternalElectricityTransferRevenue"].groupby(dates.year).sum()
    bagasse_internal = commercial.monthly["InternalBagasseTransferRevenue"].groupby(dates.year).sum()
    internal_revenue = farm_internal + power_internal + bagasse_internal
    income_annual = _annual_sum(income_monthly)
    reconciliation = annual_component_totals.copy()
    reconciliation["InternalRevenueElimination"] = reconciliation["Year"].map(internal_revenue)
    reconciliation["InternalCostElimination"] = reconciliation["Year"].map(internal_revenue)
    reconciliation["ConsolidatedRevenue"] = income_annual["Revenue"].to_numpy()
    reconciliation["ConsolidatedEBITDA"] = income_annual["EBITDA"].to_numpy()
    reconciliation["RevenueDifference"] = (
        reconciliation["Revenue"]
        - reconciliation["InternalRevenueElimination"]
        - reconciliation["ConsolidatedRevenue"]
    )
    reconciliation["EBITDADifference"] = (
        reconciliation["EBITDA"] - reconciliation["ConsolidatedEBITDA"]
    )

    annual_cashflow = _annual_sum(cashflow_monthly.drop(columns=["DSCR"]))
    min_dscr_by_year = cashflow_monthly.groupby(cashflow_monthly.index.year)["DSCR"].min()
    annual_cashflow["MinDSCR"] = min_dscr_by_year
    annual_balance = _annual_last(balance_monthly)

    debt_error = debt.monthly["ClosingBalance"].to_numpy() - (
        debt.monthly["OpeningBalance"].to_numpy()
        + debt.monthly["Draw"].to_numpy()
        + debt.monthly["CapitalizedInterest"].to_numpy()
        - debt.monthly["Principal"].to_numpy()
    )
    bagasse_routing_error = processing.monthly["RawBagasseTonnes"].to_numpy() - (
        processing.monthly["BagasseToElectricityTonnes"].to_numpy()
        + processing.monthly["BagasseToAnimalFeedTonnes"].to_numpy()
        + processing.monthly["BagasseForSaleTonnes"].to_numpy()
    )
    total_capex = capex.total_capex
    tolerance = max(1.0, total_capex * 1e-8)
    max_balance_error = float(np.max(np.abs(balance_check)))
    max_reconciliation_error = float(
        max(
            reconciliation["RevenueDifference"].abs().max(),
            reconciliation["EBITDADifference"].abs().max(),
        )
    )
    active_dscr = dscr[np.isfinite(dscr)]
    minimum_dscr = float(np.min(active_dscr)) if len(active_dscr) else None
    dscr_target = float(
        np.max(parameter_series(inputs, "other_assumptions", "minimum_dscr_target", dates))
    )
    plan_monthly = farm_plan.monthly
    cultivated_excess = float(
        np.max(
            np.maximum(
                plan_monthly["PlannedCultivatedHa"].to_numpy()
                - plan_monthly["ArableCultivableLandHa"].to_numpy(),
                0.0,
            )
        )
    )
    arable_excess = float(
        np.max(
            np.maximum(
                plan_monthly["ArableCultivableLandHa"].to_numpy()
                - plan_monthly["TotalLandHa"].to_numpy(),
                0.0,
            )
        )
    )
    irrigated_cultivated_excess = float(
        np.max(
            np.maximum(
                plan_monthly["PlannedIrrigatedHa"].to_numpy()
                - plan_monthly["PlannedCultivatedHa"].to_numpy(),
                0.0,
            )
        )
    )
    irrigated_capacity_excess = float(
        np.max(
            np.maximum(
                plan_monthly["PlannedIrrigatedHa"].to_numpy()
                - plan_monthly["IrrigationCapacityHa"].to_numpy(),
                0.0,
            )
        )
    )
    farm_cane_processed = farming.monthly["FarmCaneProcessedTonnes"].to_numpy()
    farm_cane_available = farming.monthly["FarmCaneAvailableTonnes"].to_numpy()
    farm_cane_excess = float(
        np.max(np.maximum(farm_cane_processed - farm_cane_available, 0.0))
    )
    processing_target = sourcing.monthly["ProcessingCaneTargetTonnes"].to_numpy()
    expected_purchases = (
        np.zeros(len(dates), dtype=float)
        if scenario.upper() == "FARM_ONLY"
        else np.maximum(processing_target - farm_cane_processed, 0.0)
    )
    purchase_reconciliation_error = float(
        np.max(
            np.abs(
                sourcing.monthly["PurchasedCaneTonnes"].to_numpy()
                - expected_purchases
            )
        )
    )
    total_processed_cane = float(sourcing.monthly["TotalCaneTonnes"].sum())
    total_farm_cane = float(farming.monthly["FarmCaneProcessedTonnes"].sum())
    actual_farm_share = (
        total_farm_cane / total_processed_cane if total_processed_cane > 0 else 0.0
    )
    farm_share_target = _scenario_farm_share_target(inputs, scenario)
    farm_share_gap = abs(actual_farm_share - farm_share_target)
    land_tolerance = 1e-6
    farm_share_tolerance = 0.05


    checks = pd.DataFrame(
        [
            {
                "Check": "Cultivated land is within arable/cultivable land",
                "Actual": cultivated_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if cultivated_excess <= land_tolerance else "FAIL",
            },
            {
                "Check": "Arable/cultivable land is within total land",
                "Actual": arable_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if arable_excess <= land_tolerance else "FAIL",
            },
            {
                "Check": "Irrigated land is within cultivated land",
                "Actual": irrigated_cultivated_excess,
                "Tolerance": land_tolerance,
                "Status": (
                    "OK"
                    if irrigated_cultivated_excess <= land_tolerance
                    else "FAIL"
                ),
            },
            {
                "Check": "Irrigated land is within irrigation capacity",
                "Actual": irrigated_capacity_excess,
                "Tolerance": land_tolerance,
                "Status": (
                    "OK"
                    if irrigated_capacity_excess <= land_tolerance
                    else "FAIL"
                ),
            },
            {
                "Check": "Farm cane processed is within harvested cane available",
                "Actual": farm_cane_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if farm_cane_excess <= land_tolerance else "FAIL",
            },
            {
                "Check": "Purchased cane reconciles the feedstock shortage",
                "Actual": purchase_reconciliation_error,
                "Tolerance": land_tolerance,
                "Status": (
                    "OK"
                    if purchase_reconciliation_error <= land_tolerance
                    else "FAIL"
                ),
            },
            {
                "Check": "Actual farm share meets the scenario target",
                "Actual": actual_farm_share,
                "Tolerance": farm_share_target,
                "Status": "OK" if farm_share_gap <= farm_share_tolerance else "WARN",
            },
            {
                "Check": "Bagasse routing sums to raw bagasse",
                "Actual": float(np.max(np.abs(bagasse_routing_error))),
                "Tolerance": tolerance,
                "Status": "OK" if np.max(np.abs(bagasse_routing_error)) <= tolerance else "FAIL",
            },
            {
                "Check": "Debt roll-forward",
                "Actual": float(np.max(np.abs(debt_error))),
                "Tolerance": tolerance,
                "Status": "OK" if np.max(np.abs(debt_error)) <= tolerance else "FAIL",
            },
            {
                "Check": "Balance sheet balances",
                "Actual": max_balance_error,
                "Tolerance": tolerance,
                "Status": "OK" if max_balance_error <= tolerance else "FAIL",
            },
            {
                "Check": "Component consolidation reconciles",
                "Actual": max_reconciliation_error,
                "Tolerance": tolerance,
                "Status": "OK" if max_reconciliation_error <= tolerance else "FAIL",
            },
            {
                "Check": "Debt repays by model end",
                "Actual": float(debt_balance[-1]),
                "Tolerance": tolerance,
                "Status": "OK" if debt_balance[-1] <= tolerance else "FAIL",
            },
            {
                "Check": "Minimum DSCR meets target",
                "Actual": minimum_dscr,
                "Tolerance": dscr_target,
                "Status": "OK" if minimum_dscr is None or minimum_dscr >= dscr_target else "WARN",
            },
        ]
    )

    final_year_fcf = float(annual_cashflow.iloc[-1]["ProjectFCF"])
    terminal_value = max(
        0.0,
        final_year_fcf
        * (1.0 + inputs.global_assumptions.terminal_growth_rate)
        / (
            inputs.global_assumptions.discount_rate
            - inputs.global_assumptions.terminal_growth_rate
        ),
    )
    project_returns = project_fcf.copy()
    equity_returns = equity_fcf.copy()
    project_returns[-1] += terminal_value
    equity_returns[-1] += max(0.0, terminal_value - debt_balance[-1])
    metrics: dict[str, Any] = {
        "scenario": scenario,
        "farm_share": actual_farm_share,
        "farm_share_target": farm_share_target,
        "farm_share_gap": farm_share_gap,
        "total_farm_cane_available": float(
            farming.monthly["FarmCaneAvailableTonnes"].sum()
        ),
        "total_farm_cane_processed": total_farm_cane,
        "total_purchased_cane": float(
            sourcing.monthly["PurchasedCaneTonnes"].sum()
        ),
        "total_unused_farm_cane": float(
            farming.monthly["UnusedFarmCaneTonnes"].sum()
        ),
        "total_land_hectares": float(plan_monthly["TotalLandHa"].max()),
        "arable_land_hectares": float(
            plan_monthly["ArableCultivableLandHa"].max()
        ),
        "planned_cultivated_hectares": float(
            plan_monthly["PlannedCultivatedHa"].max()
        ),
        "planned_irrigated_hectares": float(
            plan_monthly["PlannedIrrigatedHa"].max()
        ),
        "total_capex": total_capex,
        "project_npv": _npv(project_returns, inputs.global_assumptions.discount_rate),
        "project_irr": _annualized_irr(project_returns),
        "equity_npv": _npv(equity_returns, inputs.global_assumptions.discount_rate),
        "equity_irr": _annualized_irr(equity_returns),
        "payback_years": _payback(project_fcf),
        "min_dscr": minimum_dscr,
        "average_dscr": float(np.mean(active_dscr)) if len(active_dscr) else None,
        "ending_debt": float(debt_balance[-1]),
        "senior_debt_draw": float(debt.summary.iloc[0]["InitialDraw"]),
        "additional_debt_draw": float(
            debt.summary.iloc[1:]["InitialDraw"].sum()
        ),
        "total_debt_draw": float(debt.summary["InitialDraw"].sum()),
        "total_equity_contribution": float(equity_contribution.sum()),
        "terminal_value": terminal_value,
        "model_status": "CHECK" if (checks["Status"] == "FAIL").any() else "OK",
    }
    metrics["investor_npv"] = metrics["equity_npv"] * inputs.global_assumptions.investor_share
    metrics["owner_npv"] = metrics["equity_npv"] * (
        1.0 - inputs.global_assumptions.investor_share
    )

    return FinancialOutput(
        income_monthly=income_monthly,
        income_annual=income_annual,
        cashflow_monthly=cashflow_monthly,
        cashflow_annual=annual_cashflow,
        balance_monthly=balance_monthly,
        balance_annual=annual_balance,
        component_monthly=component_monthly,
        component_annual=component_annual,
        reconciliation=reconciliation,
        checks=checks,
        metrics=metrics,
    )
