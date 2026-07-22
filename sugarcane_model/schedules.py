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
class LabourOutput(ScheduleOutput):
    item_schedule: pd.DataFrame
    monthly_detail: pd.DataFrame
    annual_detail: pd.DataFrame
    allocation_monthly: pd.DataFrame
    allocation_annual: pd.DataFrame


@dataclass
class CapexOutput(ScheduleOutput):
    item_schedule: pd.DataFrame
    total_capex: float


@dataclass
class DebtOutput(ScheduleOutput):
    summary: pd.DataFrame
    facility_monthly: pd.DataFrame
    facility_annual: pd.DataFrame
    contractual_maturity: pd.Timestamp
    repayment_start: pd.Timestamp
    model_horizon_covers_tail: bool


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
    covenant_schedule: pd.DataFrame
    liquidity_monthly: pd.DataFrame
    liquidity_annual: pd.DataFrame
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


def _depreciation(
    spend: np.ndarray,
    life_months: int,
    *,
    in_service_index: int = 0,
) -> np.ndarray:
    result = np.zeros(len(spend), dtype=float)
    life = max(1, int(life_months))
    for month, amount in enumerate(spend):
        first = max(month + 1, in_service_index)
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


def effective_cod_date(inputs: SugarcaneBioethanolInputs) -> pd.Timestamp:
    """Return the delayed commercial-operations date as a month-start timestamp."""

    scheduled = pd.Period(inputs.construction.scheduled_cod, freq="M")
    return (scheduled + inputs.construction.construction_delay_months).to_timestamp()


def compute_construction_schedule(
    inputs: SugarcaneBioethanolInputs,
) -> ScheduleOutput:
    """Build the single source of truth for construction, COD, and operating ramp."""

    dates = build_timeline(inputs)
    scheduled_cod = pd.Period(
        inputs.construction.scheduled_cod, freq="M"
    ).to_timestamp()
    effective_cod = effective_cod_date(inputs)
    planning_start = pd.Period(
        inputs.global_assumptions.planning_start, freq="M"
    ).to_timestamp()
    operating_start = max(effective_cod, planning_start)
    commissioning_start = (
        pd.Period(effective_cod, freq="M")
        - inputs.construction.commissioning_months
    ).to_timestamp()
    operational = np.asarray(dates >= operating_start)
    months_from_cod = (
        (dates.year - operating_start.year) * 12
        + dates.month
        - operating_start.month
    ).to_numpy(dtype=int)
    operating_year = np.maximum(months_from_cod, 0) // 12
    ramp_year_1 = parameter_series(
        inputs, "other_assumptions", "startup_ramp_year_1", dates
    )
    ramp_year_2 = parameter_series(
        inputs, "other_assumptions", "startup_ramp_year_2", dates
    )
    ramp = np.where(
        operational,
        np.where(
            operating_year == 0,
            ramp_year_1,
            np.where(operating_year == 1, ramp_year_2, 1.0),
        ),
        0.0,
    )
    phase = np.where(
        operational,
        "Operating",
        np.where(dates >= commissioning_start, "Commissioning", "Construction"),
    )
    monthly = pd.DataFrame(
        {
            "ScheduledCOD": scheduled_cod,
            "EffectiveCOD": effective_cod,
            "OperatingStart": operating_start,
            "ConstructionDelayMonths": inputs.construction.construction_delay_months,
            "CommissioningMonths": inputs.construction.commissioning_months,
            "Phase": phase,
            "OperationalFlag": operational.astype(int),
            "OperatingMonth": np.where(operational, months_from_cod + 1, 0),
            "RampFactor": ramp,
        },
        index=dates,
    )
    annual = monthly.groupby(dates.year).agg(
        ScheduledCOD=("ScheduledCOD", "last"),
        EffectiveCOD=("EffectiveCOD", "last"),
        OperatingStart=("OperatingStart", "last"),
        ConstructionDelayMonths=("ConstructionDelayMonths", "last"),
        CommissioningMonths=("CommissioningMonths", "last"),
        OperatingMonths=("OperationalFlag", "sum"),
        AverageRamp=("RampFactor", "mean"),
        ClosingPhase=("Phase", "last"),
    )
    annual.index.name = "Year"
    return ScheduleOutput(monthly=monthly, annual=annual)


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
    monthly = pd.DataFrame(
        {
            "CycleNumber": np.where(active, operating_month // crop_cycle + 1, 0),
            "CycleMonth": np.where(active, cycle_month, 0),
            "Phase": phases,
            "ReplantShare": np.where(active & (cycle_month == 1), replant_share, 0.0),
        },
        index=dates,
    )
    annual = monthly.groupby(monthly.index.year).agg(
        Cycles=("CycleNumber", "max"),
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
    construction: ScheduleOutput,
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
        capacity * availability * construction.monthly["RampFactor"].to_numpy() / 12.0
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
    construction: ScheduleOutput,
) -> ScheduleOutput:
    dates = farming.monthly.index
    capacity = parameter_series(
        inputs, "processing_routing", "annual_cane_capacity_tonnes", dates
    )
    availability = parameter_series(
        inputs, "other_assumptions", "plant_availability", dates
    )
    processing_target = (
        capacity * availability * construction.monthly["RampFactor"].to_numpy() / 12.0
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


def compute_labour_schedule(
    inputs: SugarcaneBioethanolInputs,
    farm_plan: ScheduleOutput,
    farming: ScheduleOutput,
    processing: ScheduleOutput,
    commercial: ScheduleOutput,
) -> LabourOutput:
    """Expand consolidated labour roles and allocate each cost exactly once."""

    dates = processing.monthly.index
    revenue_shares = _product_revenue_shares(commercial)
    productivity_series: dict[str, pd.Series] = {
        "None": pd.Series(0.0, index=dates),
        "Cultivated hectares": farm_plan.monthly["PlannedCultivatedHa"],
        "Harvested hectares": farm_plan.monthly["HectaresHarvested"],
        "Farm cane tonnes": farming.monthly["FarmCaneAvailableTonnes"],
        "Cane processed tonnes": processing.monthly["CaneReceivedTonnes"],
        "Bioethanol litres": processing.monthly["BioethanolLitres"],
        "Sugar tonnes": processing.monthly["SugarTonnes"],
        "Electricity MWh": processing.monthly["GrossElectricityMWh"],
        "Bagasse tonnes": processing.monthly["RawBagasseTonnes"],
        "Animal feed tonnes": processing.monthly["AnimalFeedTonnes"],
    }
    custom_share_fields = {
        "Bioethanol": "bioethanol_share",
        "Sugar": "sugar_share",
        "Electricity Generation": "electricity_share",
        "Bagasse": "bagasse_share",
        "Animal Feed": "animal_feed_share",
    }
    cost_columns = [
        "BaseSalary",
        "OvertimeCost",
        "BenefitsCost",
        "StatutoryContributions",
        "TrainingCost",
        "PPECost",
        "TransportCost",
        "AccommodationCost",
        "TotalLabourCost",
    ]
    detail_frames: list[pd.DataFrame] = []
    allocation_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []

    for item in inputs.labour.items:
        dumper = getattr(item, "model_dump", None)
        item_values = dumper() if callable(dumper) else item.dict()
        item_rows.append(item_values)
        start = pd.Period(item.start_month, freq="M").to_timestamp()
        end_value = (item.end_month or "").strip()
        end = (
            pd.Period(end_value, freq="M").to_timestamp()
            if end_value
            else dates[-1]
        )
        active = np.asarray((dates >= start) & (dates <= end))
        positions = np.flatnonzero(active)
        if not len(positions):
            continue

        months_since_start = np.maximum(
            0,
            (dates.year - start.year) * 12 + dates.month - start.month,
        ).to_numpy(dtype=int)
        escalation_factor = np.power(
            1.0 + item.annual_salary_escalation,
            months_since_start // 12,
        )
        headcount = np.where(active, item.headcount_fte, 0.0)
        base_salary = (
            headcount * item.monthly_wage_per_fte * escalation_factor
        )
        overtime = base_salary * item.overtime_rate
        cash_pay = base_salary + overtime
        benefits = cash_pay * item.benefits_rate
        statutory = cash_pay * item.statutory_contribution_rate
        training = headcount * item.training_cost_per_fte_year / 12.0
        ppe = headcount * item.ppe_cost_per_fte_year / 12.0
        transport = headcount * item.transport_cost_per_fte_month
        accommodation = headcount * item.accommodation_cost_per_fte_month
        total = (
            cash_pay
            + benefits
            + statutory
            + training
            + ppe
            + transport
            + accommodation
        )
        productivity = productivity_series[item.productivity_driver].to_numpy(
            dtype=float
        )
        detail_frames.append(
            pd.DataFrame(
                {
                    "Date": dates[positions],
                    "Year": dates[positions].year,
                    "Month": dates[positions].month,
                    "RoleID": item.role_id,
                    "CostCentre": item.cost_centre,
                    "Department": item.department,
                    "Position": item.position,
                    "WorkerType": item.worker_type,
                    "SourceComponent": item.component,
                    "AllocationDriver": item.allocation_driver,
                    "ProductivityDriver": item.productivity_driver,
                    "HeadcountFTE": headcount[positions],
                    "NumberOfShifts": item.number_of_shifts,
                    "FTEPerShift": headcount[positions] / item.number_of_shifts,
                    "MonthlyWagePerFTE": (
                        item.monthly_wage_per_fte
                        * escalation_factor[positions]
                    ),
                    "SalaryEscalationFactor": escalation_factor[positions],
                    "BaseSalary": base_salary[positions],
                    "OvertimeCost": overtime[positions],
                    "BenefitsCost": benefits[positions],
                    "StatutoryContributions": statutory[positions],
                    "TrainingCost": training[positions],
                    "PPECost": ppe[positions],
                    "TransportCost": transport[positions],
                    "AccommodationCost": accommodation[positions],
                    "TotalLabourCost": total[positions],
                    "ProductivityVolume": productivity[positions],
                    "ProductivityPerFTE": np.divide(
                        productivity[positions],
                        headcount[positions],
                        out=np.zeros(len(positions), dtype=float),
                        where=headcount[positions] > 0,
                    ),
                }
            )
        )

        for position in positions:
            date = dates[position]
            if item.allocation_driver == "Direct":
                weights = {item.component: 1.0}
            elif item.allocation_driver == "Product revenue share":
                weights = {
                    component: float(revenue_shares.loc[date, component])
                    for component in PRODUCT_COMPONENTS
                }
            elif item.allocation_driver == "Equal product share":
                weights = {
                    component: 1.0 / len(PRODUCT_COMPONENTS)
                    for component in PRODUCT_COMPONENTS
                }
            else:
                weights = {
                    component: float(getattr(item, field))
                    for component, field in custom_share_fields.items()
                }
            for component, share in weights.items():
                allocation_rows.append(
                    {
                        "Date": date,
                        "Year": date.year,
                        "RoleID": item.role_id,
                        "CostCentre": item.cost_centre,
                        "Department": item.department,
                        "Position": item.position,
                        "AllocationDriver": item.allocation_driver,
                        "Component": component,
                        "AllocationShare": share,
                        "AllocatedLabourCost": total[position] * share,
                    }
                )

    item_schedule = pd.DataFrame(item_rows)
    if detail_frames:
        monthly_detail = pd.concat(detail_frames, ignore_index=True)
    else:
        monthly_detail = pd.DataFrame(
            columns=[
                "Date", "Year", "Month", "RoleID", "CostCentre", "Department",
                "Position", "WorkerType", "SourceComponent", "AllocationDriver",
                "ProductivityDriver", "HeadcountFTE", "NumberOfShifts",
                "FTEPerShift", "MonthlyWagePerFTE", "SalaryEscalationFactor",
                *cost_columns, "ProductivityVolume", "ProductivityPerFTE",
            ]
        )

    monthly = pd.DataFrame(0.0, index=dates, columns=["HeadcountFTE", *cost_columns])
    monthly.index.name = "Date"
    monthly["ActiveRoles"] = 0
    if not monthly_detail.empty:
        grouped = monthly_detail.groupby("Date")
        monthly.loc[:, ["HeadcountFTE", *cost_columns]] = (
            grouped[["HeadcountFTE", *cost_columns]]
            .sum()
            .reindex(dates, fill_value=0.0)
        )
        monthly["ActiveRoles"] = (
            grouped["RoleID"].nunique().reindex(dates, fill_value=0).astype(int)
        )

    annual = monthly[cost_columns].groupby(monthly.index.year).sum()
    annual["AverageHeadcountFTE"] = monthly["HeadcountFTE"].groupby(
        monthly.index.year
    ).mean()
    annual["PeakHeadcountFTE"] = monthly["HeadcountFTE"].groupby(
        monthly.index.year
    ).max()
    annual["PeakActiveRoles"] = monthly["ActiveRoles"].groupby(
        monthly.index.year
    ).max()
    annual.index.name = "Year"

    if monthly_detail.empty:
        annual_detail = pd.DataFrame()
    else:
        annual_detail = (
            monthly_detail.groupby(
                [
                    "Year", "RoleID", "CostCentre", "Department", "Position",
                    "WorkerType", "SourceComponent", "AllocationDriver",
                    "ProductivityDriver",
                ],
                as_index=False,
            )
            .agg(
                MonthsActive=("Date", "nunique"),
                AverageHeadcountFTE=("HeadcountFTE", "mean"),
                PeakHeadcountFTE=("HeadcountFTE", "max"),
                NumberOfShifts=("NumberOfShifts", "max"),
                BaseSalary=("BaseSalary", "sum"),
                OvertimeCost=("OvertimeCost", "sum"),
                BenefitsCost=("BenefitsCost", "sum"),
                StatutoryContributions=("StatutoryContributions", "sum"),
                TrainingCost=("TrainingCost", "sum"),
                PPECost=("PPECost", "sum"),
                TransportCost=("TransportCost", "sum"),
                AccommodationCost=("AccommodationCost", "sum"),
                TotalLabourCost=("TotalLabourCost", "sum"),
                ProductivityVolume=("ProductivityVolume", "sum"),
            )
        )
        annual_detail["ProductivityPerAverageFTE"] = np.divide(
            annual_detail["ProductivityVolume"],
            annual_detail["AverageHeadcountFTE"],
            out=np.zeros(len(annual_detail), dtype=float),
            where=annual_detail["AverageHeadcountFTE"] > 0,
        )

    allocation_monthly = pd.DataFrame(allocation_rows)
    if allocation_monthly.empty:
        allocation_monthly = pd.DataFrame(
            columns=[
                "Date", "Year", "RoleID", "CostCentre", "Department", "Position",
                "AllocationDriver", "Component", "AllocationShare",
                "AllocatedLabourCost",
            ]
        )
        allocation_annual = pd.DataFrame(
            columns=["Year", "Component", "AllocatedLabourCost", "ShareOfAnnualLabourCost"]
        )
    else:
        allocation_annual = (
            allocation_monthly.groupby(["Year", "Component"], as_index=False)[
                "AllocatedLabourCost"
            ].sum()
        )
        year_total = allocation_annual.groupby("Year")[
            "AllocatedLabourCost"
        ].transform("sum")
        allocation_annual["ShareOfAnnualLabourCost"] = np.divide(
            allocation_annual["AllocatedLabourCost"],
            year_total,
            out=np.zeros(len(allocation_annual), dtype=float),
            where=year_total > 0,
        )

    return LabourOutput(
        monthly=monthly,
        annual=annual,
        item_schedule=item_schedule,
        monthly_detail=monthly_detail,
        annual_detail=annual_detail,
        allocation_monthly=allocation_monthly,
        allocation_annual=allocation_annual,
    )


def compute_cost_schedule(
    inputs: SugarcaneBioethanolInputs,
    farming: ScheduleOutput,
    sourcing: ScheduleOutput,
    processing: ScheduleOutput,
    labour: LabourOutput,
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
    monthly["LabourCost"] = labour.monthly["TotalLabourCost"]
    monthly["EconomicOperatingCost"] = monthly[
        ["FarmingOperatingCost", "PurchasedCaneCost", "InboundLogisticsCost",
         "ProcessingVariableCost", "ProductSpecificVariableCost",
         "FixedProcessingOpex", "CommercialAndAdminCost", "LabourCost"]
    ].sum(axis=1)
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_capex_schedule(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    sourcing: ScheduleOutput,
    construction: ScheduleOutput,
) -> CapexOutput:
    dates = construction.monthly.index
    effective_cod = effective_cod_date(inputs)
    cod_index = int(np.searchsorted(dates.values, effective_cod.to_datetime64()))
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
        depreciation_by_component[item.component] += _depreciation(
            spend, item.life_years * 12, in_service_index=cod_index
        )
        rows.append(
            {
                "Item": item.item,
                "Component": item.component,
                "ScenarioAmount": amount,
                "StartMonth": item.start_month,
                "SpendMonths": item.spend_months,
                "LifeYears": item.life_years,
                "IsFarmCapex": item.is_farm_capex,
                "InServiceMonth": max(effective_cod, start + pd.offsets.MonthBegin(item.spend_months)),
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


def _month_distance(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def _contractual_debt_dates(
    cod: pd.Timestamp,
    *,
    tenor_years: int,
    grace_years: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    cod_period = pd.Period(cod, freq="M")
    repayment_start = (cod_period + grace_years * 12).to_timestamp()
    maturity = (cod_period + tenor_years * 12 - 1).to_timestamp()
    return repayment_start, maturity


def _covenant_period_keys(
    dates: pd.DatetimeIndex,
    period: str,
) -> np.ndarray:
    if period == "monthly":
        return np.asarray([date.strftime("%Y-%m") for date in dates], dtype=object)
    if period == "quarterly":
        return np.asarray(
            [f"{date.year}-Q{((date.month - 1) // 3) + 1}" for date in dates],
            dtype=object,
        )
    if period == "semiannual":
        return np.asarray(
            [f"{date.year}-H{1 if date.month <= 6 else 2}" for date in dates],
            dtype=object,
        )
    return np.asarray([str(date.year) for date in dates], dtype=object)


def _period_end_mask(keys: np.ndarray) -> np.ndarray:
    return np.asarray(
        [index == len(keys) - 1 or keys[index + 1] != key for index, key in enumerate(keys)],
        dtype=bool,
    )


def _build_facility_schedule(
    dates: pd.DatetimeIndex,
    draws: np.ndarray,
    *,
    cod: pd.Timestamp,
    annual_interest_rates: np.ndarray,
    tenor_years: int,
    grace_years: int,
    amortization_type: str,
    capitalize_idc: bool,
    covenant_keys: np.ndarray | None = None,
    debt_service_capacity: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build one COD-anchored loan without compressing tenor to the model horizon."""

    months = len(dates)
    monthly_rates = np.asarray(annual_interest_rates, dtype=float) / 12.0
    repayment_start, maturity = _contractual_debt_dates(
        cod, tenor_years=tenor_years, grace_years=grace_years
    )
    opening = np.zeros(months)
    interest = np.zeros(months)
    idc = np.zeros(months)
    cash_interest = np.zeros(months)
    principal = np.zeros(months)
    closing = np.zeros(months)
    capacity = (
        np.asarray(debt_service_capacity, dtype=float)
        if debt_service_capacity is not None
        else None
    )
    keys = (
        np.asarray(covenant_keys, dtype=object)
        if covenant_keys is not None
        else np.asarray([date.strftime("%Y-%m") for date in dates], dtype=object)
    )
    period_end = _period_end_mask(keys)
    period_interest = 0.0
    balance = 0.0

    for month, date in enumerate(dates):
        if month > 0 and keys[month] != keys[month - 1]:
            period_interest = 0.0
        monthly_rate = monthly_rates[month]
        opening[month] = balance
        interest[month] = (balance + 0.5 * draws[month]) * monthly_rate
        if date < cod and capitalize_idc:
            idc[month] = interest[month]
        else:
            cash_interest[month] = interest[month]
            period_interest += cash_interest[month]
        available = balance + draws[month] + idc[month]

        within_repayment = repayment_start <= date <= maturity
        if within_repayment and capacity is not None and period_end[month]:
            scheduled = max(0.0, capacity[month] - period_interest)
            principal[month] = min(available, scheduled)
        elif within_repayment and capacity is None:
            remaining = max(1, _month_distance(date, maturity) + 1)
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


def compute_sizing_cfads(
    inputs: SugarcaneBioethanolInputs,
    commercial: ScheduleOutput,
    costs: ScheduleOutput,
    capex: CapexOutput,
    working_capital: ScheduleOutput,
) -> pd.Series:
    """Conservative pre-financing CFADS used only for debt sizing."""

    dates = commercial.monthly.index
    revenue = commercial.monthly[list(REVENUE_COLUMNS.values())].sum(axis=1).to_numpy()
    ebitda = revenue - costs.monthly["EconomicOperatingCost"].to_numpy()
    depreciation = capex.monthly["TotalDepreciation"].to_numpy()
    tax = _tax_with_nol(
        ebitda - depreciation,
        inputs.global_assumptions.corporate_tax_rate,
    )
    nwc = working_capital.monthly["NetWorkingCapital"].to_numpy()
    delta_nwc = np.diff(nwc, prepend=0.0)
    return pd.Series(ebitda - tax - delta_nwc, index=dates, name="SizingCFADS")


def _sculpting_capacity(
    dates: pd.DatetimeIndex,
    sizing_cfads: pd.Series,
    other_debt_service: pd.Series,
    *,
    covenant_period: str,
    dscr_target: float,
    cash_sweep_percent: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = _covenant_period_keys(dates, covenant_period)
    period_end = _period_end_mask(keys)
    capacity = np.zeros(len(dates))
    cfads_values = sizing_cfads.reindex(dates, fill_value=0.0).to_numpy(dtype=float)
    other_values = other_debt_service.reindex(dates, fill_value=0.0).to_numpy(dtype=float)
    for key in dict.fromkeys(keys):
        mask = keys == key
        last = int(np.flatnonzero(mask)[-1])
        period_cfads = max(0.0, float(cfads_values[mask].sum()))
        other_service = max(0.0, float(other_values[mask].sum()))
        target_capacity = max(0.0, period_cfads / max(dscr_target, 1e-9) - other_service)
        excess_after_target = max(0.0, period_cfads - other_service - target_capacity)
        capacity[last] = target_capacity + cash_sweep_percent * excess_after_target
    capacity[~period_end] = 0.0
    return keys, capacity


def compute_debt_schedule(
    inputs: SugarcaneBioethanolInputs,
    capex: CapexOutput,
    construction: ScheduleOutput,
    sizing_cfads: pd.Series,
) -> DebtOutput:
    dates = capex.monthly.index
    financing = inputs.financing
    cod = effective_cod_date(inputs)
    debt_ratios = parameter_series(inputs, "financing", "debt_ratio", dates)
    senior_interest_rates = parameter_series(inputs, "financing", "interest_rate", dates)
    capex_spend = capex.monthly["TotalCapex"].to_numpy(dtype=float)
    total_capex = float(capex_spend.sum())
    if total_capex <= 0:
        raise ValueError("Total CAPEX must be positive before debt can be scheduled.")

    capex_profile = capex_spend / total_capex
    fixed_commitment = sum(facility.amount for facility in financing.additional_debt_facilities)
    tolerance = max(1.0, total_capex * 1e-8)
    if fixed_commitment > total_capex + tolerance:
        raise ValueError(
            "Additional debt commitments "
            f"(USD {fixed_commitment:,.0f}) exceed scenario CAPEX "
            f"(USD {total_capex:,.0f})."
        )

    additional_specs: list[dict[str, Any]] = []
    additional_schedules: list[pd.DataFrame] = []
    for facility in financing.additional_debt_facilities:
        spec = {
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
            "sizing_mode": "fixed_amount",
        }
        additional_specs.append(spec)
        additional_schedules.append(
            _build_facility_schedule(
                dates,
                np.asarray(spec["draws"], dtype=float),
                cod=cod,
                annual_interest_rates=np.asarray(spec["interest_rates"], dtype=float),
                tenor_years=int(spec["tenor_years"]),
                grace_years=int(spec["grace_years"]),
                amortization_type=str(spec["amortization_type"]),
                capitalize_idc=bool(spec["capitalize_idc"]),
            )
        )

    other_service = pd.Series(0.0, index=dates)
    for schedule in additional_schedules:
        other_service = other_service.add(schedule["DebtService"], fill_value=0.0)

    maximum_senior_commitment = min(
        float(np.sum(capex_spend * debt_ratios)),
        max(0.0, total_capex - fixed_commitment),
    )
    senior_sizing_mode = financing.debt_sizing_mode
    if senior_sizing_mode == "dscr_sculpted":
        dscr_target = float(
            np.max(parameter_series(inputs, "other_assumptions", "minimum_dscr_target", dates))
        )
        covenant_keys, service_capacity = _sculpting_capacity(
            dates,
            sizing_cfads,
            other_service,
            covenant_period=financing.covenant_period,
            dscr_target=dscr_target,
            cash_sweep_percent=financing.cash_sweep_percent,
        )

        def candidate_schedule(commitment: float) -> pd.DataFrame:
            return _build_facility_schedule(
                dates,
                capex_profile * commitment,
                cod=cod,
                annual_interest_rates=senior_interest_rates,
                tenor_years=financing.tenor_years,
                grace_years=financing.grace_years,
                amortization_type="straight",
                capitalize_idc=financing.capitalize_idc,
                covenant_keys=covenant_keys,
                debt_service_capacity=service_capacity,
            )

        maximum_schedule = candidate_schedule(maximum_senior_commitment)
        if maximum_schedule["ClosingBalance"].iloc[-1] <= tolerance:
            senior_commitment = maximum_senior_commitment
            senior_schedule = maximum_schedule
        else:
            low = 0.0
            high = maximum_senior_commitment
            senior_schedule = candidate_schedule(0.0)
            for _ in range(60):
                midpoint = (low + high) / 2.0
                candidate = candidate_schedule(midpoint)
                if candidate["ClosingBalance"].iloc[-1] <= tolerance:
                    low = midpoint
                    senior_schedule = candidate
                else:
                    high = midpoint
            senior_commitment = low
    else:
        senior_draws = capex_spend * debt_ratios
        senior_commitment = float(senior_draws.sum())
        if senior_commitment + fixed_commitment > total_capex + tolerance:
            raise ValueError(
                "Combined senior and additional debt commitments "
                f"(USD {senior_commitment + fixed_commitment:,.0f}) exceed "
                f"scenario CAPEX (USD {total_capex:,.0f})."
            )
        senior_schedule = _build_facility_schedule(
            dates,
            senior_draws,
            cod=cod,
            annual_interest_rates=senior_interest_rates,
            tenor_years=financing.tenor_years,
            grace_years=financing.grace_years,
            amortization_type=financing.amortization_type,
            capitalize_idc=financing.capitalize_idc,
        )

    senior_spec = {
        "name": "Senior Debt",
        "facility_type": "Senior (% of CAPEX)",
        "debt_ratio": senior_commitment / total_capex,
        "draws": senior_schedule["Draw"].to_numpy(),
        "interest_rates": senior_interest_rates,
        "summary_interest_rate": float(np.mean(senior_interest_rates)),
        "tenor_years": financing.tenor_years,
        "grace_years": financing.grace_years,
        "amortization_type": (
            "sculpted" if senior_sizing_mode == "dscr_sculpted" else financing.amortization_type
        ),
        "capitalize_idc": financing.capitalize_idc,
        "sizing_mode": senior_sizing_mode,
    }
    facility_specs = [senior_spec, *additional_specs]
    schedules = [senior_schedule, *additional_schedules]

    monthly_details: list[pd.DataFrame] = []
    annual_details: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    repayment_dates: list[pd.Timestamp] = []
    maturity_dates: list[pd.Timestamp] = []
    for spec, schedule in zip(facility_specs, schedules):
        repayment_start, maturity = _contractual_debt_dates(
            cod,
            tenor_years=int(spec["tenor_years"]),
            grace_years=int(spec["grace_years"]),
        )
        repayment_dates.append(repayment_start)
        maturity_dates.append(maturity)
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
                "SizingMode": spec["sizing_mode"],
                "DebtRatio": spec["debt_ratio"],
                "InterestRate": spec["summary_interest_rate"],
                "TenorYears": spec["tenor_years"],
                "GraceYears": spec["grace_years"],
                "RepaymentStart": repayment_start,
                "ContractualMaturity": maturity,
                "Amortization": spec["amortization_type"],
                "CapitalizeIDC": spec["capitalize_idc"],
                "InitialDraw": float(schedule["Draw"].sum()),
                "CapitalizedInterest": float(schedule["CapitalizedInterest"].sum()),
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
    contractual_maturity = max(maturity_dates)
    repayment_start = min(repayment_dates)
    required_tail_end = (
        pd.Period(contractual_maturity, freq="M") + financing.debt_tail_months
    ).to_timestamp()
    return DebtOutput(
        monthly=monthly,
        annual=annual,
        summary=pd.DataFrame(summary_rows),
        facility_monthly=facility_monthly,
        facility_annual=facility_annual,
        contractual_maturity=contractual_maturity,
        repayment_start=repayment_start,
        model_horizon_covers_tail=bool(dates[-1] >= required_tail_end),
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


def _forward_sum(values: np.ndarray, months: int) -> np.ndarray:
    if months <= 0:
        return np.zeros(len(values))
    return np.asarray(
        [float(values[index : index + months].sum()) for index in range(len(values))]
    )


def _build_liquidity_schedule(
    inputs: SugarcaneBioethanolInputs,
    dates: pd.DatetimeIndex,
    *,
    cfads: np.ndarray,
    capex_spend: np.ndarray,
    debt_draw: np.ndarray,
    term_cash_interest: np.ndarray,
    term_principal: np.ndarray,
    total_capex: float,
) -> pd.DataFrame:
    """Apply the reserve and liquidity waterfall month by month."""

    liquidity = inputs.liquidity
    cod = effective_cod_date(inputs)
    operational = np.asarray(dates >= cod)
    minimum_cash_target = operational.astype(float) * liquidity.minimum_cash_balance
    term_debt_service = term_cash_interest + term_principal
    dsra_target = _forward_sum(term_debt_service, liquidity.dsra_months)
    dsra_target = np.where(operational, dsra_target, 0.0)
    maintenance_target = np.where(
        operational,
        total_capex * liquidity.maintenance_reserve_rate,
        0.0,
    )
    base_equity = np.maximum(0.0, capex_spend - debt_draw)

    columns = {
        name: np.zeros(len(dates))
        for name in (
            "OpeningCash",
            "MinimumCashTarget",
            "BaseEquityContribution",
            "InitialLiquidityFunding",
            "PreCODFunding",
            "SponsorSupport",
            "EquityContribution",
            "DSRATarget",
            "DSRAContribution",
            "DSRARelease",
            "DSRABalance",
            "MaintenanceReserveTarget",
            "MaintenanceReserveContribution",
            "MaintenanceReserveRelease",
            "MaintenanceReserveBalance",
            "RestrictedCash",
            "WorkingCapitalFacilityOpening",
            "WorkingCapitalFacilityDraw",
            "WorkingCapitalFacilityRepayment",
            "WorkingCapitalFacilityInterest",
            "WorkingCapitalFacilityClosing",
            "FundingShortfall",
            "EndingCash",
            "TermDebtService",
            "TotalDebtService",
        )
    }
    cash_balance = 0.0
    dsra_balance = 0.0
    maintenance_balance = 0.0
    facility_balance = 0.0
    sponsor_used = 0.0
    cod_index = int(np.searchsorted(dates.values, cod.to_datetime64()))

    for month in range(len(dates)):
        columns["OpeningCash"][month] = cash_balance
        columns["MinimumCashTarget"][month] = minimum_cash_target[month]
        columns["BaseEquityContribution"][month] = base_equity[month]
        columns["DSRATarget"][month] = dsra_target[month]
        columns["MaintenanceReserveTarget"][month] = maintenance_target[month]
        columns["WorkingCapitalFacilityOpening"][month] = facility_balance
        facility_interest = (
            facility_balance
            * liquidity.working_capital_facility_interest_rate
            / 12.0
        )
        columns["WorkingCapitalFacilityInterest"][month] = facility_interest
        columns["TermDebtService"][month] = term_debt_service[month]

        initial_funding = 0.0
        if month == cod_index:
            initial_funding = (
                minimum_cash_target[month]
                + dsra_target[month]
                + maintenance_target[month]
            )
        columns["InitialLiquidityFunding"][month] = initial_funding
        cash_available = (
            cash_balance
            + cfads[month]
            - capex_spend[month]
            + debt_draw[month]
            + base_equity[month]
            + initial_funding
            - term_debt_service[month]
            - facility_interest
        )

        pre_cod_funding = 0.0
        if not operational[month] and cash_available < 0.0:
            pre_cod_funding = -cash_available
            cash_available = 0.0
        columns["PreCODFunding"][month] = pre_cod_funding

        required_cash = minimum_cash_target[month]
        if cash_available < required_cash and dsra_balance > 0.0:
            release = min(required_cash - cash_available, dsra_balance)
            dsra_balance -= release
            cash_available += release
            columns["DSRARelease"][month] = release

        if dsra_balance > dsra_target[month]:
            release = dsra_balance - dsra_target[month]
            dsra_balance -= release
            cash_available += release
            columns["DSRARelease"][month] += release
        dsra_gap = max(0.0, dsra_target[month] - dsra_balance)
        dsra_contribution = min(dsra_gap, max(0.0, cash_available - required_cash))
        dsra_balance += dsra_contribution
        cash_available -= dsra_contribution
        columns["DSRAContribution"][month] = dsra_contribution

        if maintenance_balance > maintenance_target[month]:
            release = maintenance_balance - maintenance_target[month]
            maintenance_balance -= release
            cash_available += release
            columns["MaintenanceReserveRelease"][month] = release
        maintenance_gap = max(0.0, maintenance_target[month] - maintenance_balance)
        maintenance_contribution = min(
            maintenance_gap,
            max(0.0, cash_available - required_cash),
        )
        maintenance_balance += maintenance_contribution
        cash_available -= maintenance_contribution
        columns["MaintenanceReserveContribution"][month] = maintenance_contribution

        if operational[month] and cash_available < required_cash:
            facility_draw = min(
                required_cash - cash_available,
                max(0.0, liquidity.working_capital_facility_limit - facility_balance),
            )
            facility_balance += facility_draw
            cash_available += facility_draw
            columns["WorkingCapitalFacilityDraw"][month] = facility_draw

        if cash_available > required_cash and facility_balance > 0.0:
            facility_repayment = min(
                cash_available - required_cash,
                facility_balance,
            )
            facility_balance -= facility_repayment
            cash_available -= facility_repayment
            columns["WorkingCapitalFacilityRepayment"][month] = facility_repayment

        if cash_available < required_cash:
            sponsor_draw = min(
                required_cash - cash_available,
                max(0.0, liquidity.sponsor_support_limit - sponsor_used),
            )
            sponsor_used += sponsor_draw
            cash_available += sponsor_draw
            columns["SponsorSupport"][month] = sponsor_draw

        columns["FundingShortfall"][month] = max(
            0.0, required_cash - cash_available
        )
        columns["EquityContribution"][month] = (
            base_equity[month]
            + initial_funding
            + pre_cod_funding
            + columns["SponsorSupport"][month]
        )
        columns["DSRABalance"][month] = dsra_balance
        columns["MaintenanceReserveBalance"][month] = maintenance_balance
        columns["RestrictedCash"][month] = dsra_balance + maintenance_balance
        columns["WorkingCapitalFacilityClosing"][month] = facility_balance
        columns["TotalDebtService"][month] = (
            term_debt_service[month]
            + facility_interest
            + columns["WorkingCapitalFacilityRepayment"][month]
        )
        columns["EndingCash"][month] = cash_available
        cash_balance = cash_available

    return pd.DataFrame(columns, index=dates)


def _annualize_liquidity(monthly: pd.DataFrame) -> pd.DataFrame:
    stock_columns = [
        "MinimumCashTarget",
        "DSRATarget",
        "DSRABalance",
        "MaintenanceReserveTarget",
        "MaintenanceReserveBalance",
        "RestrictedCash",
        "WorkingCapitalFacilityClosing",
        "FundingShortfall",
        "EndingCash",
    ]
    flow_columns = [column for column in monthly.columns if column not in stock_columns]
    annual = monthly[flow_columns].groupby(monthly.index.year).sum()
    annual[stock_columns] = monthly[stock_columns].groupby(monthly.index.year).last()
    annual.index.name = "Year"
    return annual


def _build_covenant_schedule(
    inputs: SugarcaneBioethanolInputs,
    dates: pd.DatetimeIndex,
    *,
    cfads: np.ndarray,
    debt_service: np.ndarray,
    opening_debt: np.ndarray,
    closing_debt: np.ndarray,
    contractual_maturity: pd.Timestamp,
) -> pd.DataFrame:
    keys = _covenant_period_keys(dates, inputs.financing.covenant_period)
    discount_rate = max(0.0, inputs.financing.interest_rate) / 12.0
    rows: list[dict[str, Any]] = []
    for key in dict.fromkeys(keys):
        mask = keys == key
        positions = np.flatnonzero(mask)
        first = int(positions[0])
        last = int(positions[-1])
        period_cfads = float(cfads[mask].sum())
        period_service = float(debt_service[mask].sum())
        dscr = period_cfads / period_service if period_service > 1e-9 else np.nan
        debt_opening = float(opening_debt[first])
        debt_closing = float(closing_debt[last])
        loan_life_end = min(
            int(np.searchsorted(dates.values, contractual_maturity.to_datetime64(), side="right")),
            len(dates),
        )
        loan_cfads = cfads[first:loan_life_end]
        project_cfads = cfads[first:]
        llcr = (
            float(npf.npv(discount_rate, loan_cfads)) / debt_opening
            if debt_opening > 1e-9 and len(loan_cfads)
            else np.nan
        )
        plcr = (
            float(npf.npv(discount_rate, project_cfads)) / debt_opening
            if debt_opening > 1e-9 and len(project_cfads)
            else np.nan
        )
        rows.append(
            {
                "CovenantPeriod": key,
                "StartDate": dates[first],
                "EndDate": dates[last],
                "CFADS": period_cfads,
                "DebtService": period_service,
                "DSCR": dscr,
                "OpeningDebt": debt_opening,
                "ClosingDebt": debt_closing,
                "LLCR": llcr,
                "PLCR": plcr,
            }
        )
    return pd.DataFrame(rows)


def compute_financials(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    construction: ScheduleOutput,
    farming: ScheduleOutput,
    farm_plan: ScheduleOutput,
    sourcing: ScheduleOutput,
    processing: ScheduleOutput,
    commercial: ScheduleOutput,
    costs: ScheduleOutput,
    labour: LabourOutput,
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
    cod_index = int(
        np.searchsorted(dates.values, effective_cod_date(inputs).to_datetime64())
    )
    idc_depreciation = _depreciation(
        idc, weighted_life * 12, in_service_index=cod_index
    )
    depreciation = capex.monthly["TotalDepreciation"].to_numpy() + idc_depreciation
    revenue = commercial.monthly["ExternalRevenue"].to_numpy()
    operating_cost = costs.monthly["EconomicOperatingCost"].to_numpy()
    ebitda = revenue - operating_cost
    ebit = ebitda - depreciation
    term_cash_interest = debt.monthly["CashInterest"].to_numpy()
    delta_nwc = working_capital.monthly["DeltaNWC"].to_numpy()
    capex_spend = capex.monthly["TotalCapex"].to_numpy()
    debt_draw = debt.monthly["Draw"].to_numpy()
    principal = debt.monthly["Principal"].to_numpy()

    liquidity_monthly: pd.DataFrame | None = None
    facility_interest = np.zeros(len(dates))
    for _ in range(4):
        tax = _tax_with_nol(
            ebit - term_cash_interest - facility_interest,
            inputs.global_assumptions.corporate_tax_rate,
        )
        cfads = ebitda - tax - delta_nwc
        liquidity_monthly = _build_liquidity_schedule(
            inputs,
            dates,
            cfads=cfads,
            capex_spend=capex_spend,
            debt_draw=debt_draw,
            term_cash_interest=term_cash_interest,
            term_principal=principal,
            total_capex=capex.total_capex,
        )
        updated_interest = liquidity_monthly[
            "WorkingCapitalFacilityInterest"
        ].to_numpy()
        if np.allclose(updated_interest, facility_interest, atol=0.01):
            facility_interest = updated_interest
            break
        facility_interest = updated_interest
    assert liquidity_monthly is not None

    total_cash_interest = term_cash_interest + facility_interest
    tax = _tax_with_nol(
        ebit - total_cash_interest,
        inputs.global_assumptions.corporate_tax_rate,
    )
    net_income = ebit - total_cash_interest - tax
    cfads = ebitda - tax - delta_nwc
    liquidity_monthly = _build_liquidity_schedule(
        inputs,
        dates,
        cfads=cfads,
        capex_spend=capex_spend,
        debt_draw=debt_draw,
        term_cash_interest=term_cash_interest,
        term_principal=principal,
        total_capex=capex.total_capex,
    )
    facility_interest = liquidity_monthly[
        "WorkingCapitalFacilityInterest"
    ].to_numpy()
    total_cash_interest = term_cash_interest + facility_interest
    total_debt_service = liquidity_monthly["TotalDebtService"].to_numpy()
    equity_contribution = liquidity_monthly["EquityContribution"].to_numpy()
    project_fcf = ebitda - tax - delta_nwc - capex_spend
    equity_fcf = -equity_contribution
    monthly_dscr = np.divide(
        cfads,
        total_debt_service,
        out=np.full(len(dates), np.nan),
        where=total_debt_service > 1e-9,
    )

    income_monthly = pd.DataFrame(
        {
            "Revenue": revenue,
            "NonLabourOperatingCosts": operating_cost - labour.monthly["TotalLabourCost"].to_numpy(),
            "LabourCosts": labour.monthly["TotalLabourCost"].to_numpy(),
            "OperatingCosts": operating_cost,
            "EBITDA": ebitda,
            "Depreciation": depreciation,
            "EBIT": ebit,
            "TermDebtInterest": term_cash_interest,
            "LiquidityFacilityInterest": facility_interest,
            "Interest": total_cash_interest,
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
            "CashInterest": total_cash_interest,
            "Principal": principal,
            "DebtService": total_debt_service,
            "EquityContribution": equity_contribution,
            "ProjectFCF": project_fcf,
            "EquityFCF": equity_fcf,
            "DSCR": monthly_dscr,
            "EndingCash": liquidity_monthly["EndingCash"].to_numpy(),
            "RestrictedCash": liquidity_monthly["RestrictedCash"].to_numpy(),
            "FundingShortfall": liquidity_monthly["FundingShortfall"].to_numpy(),
            "WorkingCapitalFacilityDraw": liquidity_monthly[
                "WorkingCapitalFacilityDraw"
            ].to_numpy(),
            "WorkingCapitalFacilityRepayment": liquidity_monthly[
                "WorkingCapitalFacilityRepayment"
            ].to_numpy(),
        },
        index=dates,
    )

    cash = liquidity_monthly["EndingCash"].to_numpy()
    restricted_cash = liquidity_monthly["RestrictedCash"].to_numpy()
    gross_ppe = np.cumsum(capex_spend + idc)
    accumulated_depreciation = np.cumsum(depreciation)
    net_ppe = gross_ppe - accumulated_depreciation
    paid_in_capital = np.cumsum(equity_contribution)
    retained_earnings = np.cumsum(net_income)
    ar = working_capital.monthly["AccountsReceivable"].to_numpy()
    inventory = working_capital.monthly["Inventory"].to_numpy()
    ap = working_capital.monthly["AccountsPayable"].to_numpy()
    term_debt_balance = debt.monthly["ClosingBalance"].to_numpy()
    liquidity_debt_balance = liquidity_monthly[
        "WorkingCapitalFacilityClosing"
    ].to_numpy()
    debt_balance = term_debt_balance + liquidity_debt_balance
    assets = cash + restricted_cash + ar + inventory + net_ppe
    liabilities = ap + debt_balance
    equity = paid_in_capital + retained_earnings
    balance_check = assets - liabilities - equity
    balance_monthly = pd.DataFrame(
        {
            "Cash": cash,
            "RestrictedCash": restricted_cash,
            "AccountsReceivable": ar,
            "Inventory": inventory,
            "GrossPPE": gross_ppe,
            "AccumulatedDepreciation": accumulated_depreciation,
            "NetPPE": net_ppe,
            "TotalAssets": assets,
            "AccountsPayable": ap,
            "TermDebt": term_debt_balance,
            "WorkingCapitalFacility": liquidity_debt_balance,
            "Debt": debt_balance,
            "TotalLiabilities": liabilities,
            "PaidInCapital": paid_in_capital,
            "RetainedEarnings": retained_earnings,
            "TotalEquity": equity,
            "BalanceCheck": balance_check,
        },
        index=dates,
    )
    covenant_schedule = _build_covenant_schedule(
        inputs,
        dates,
        cfads=cfads,
        debt_service=total_debt_service,
        opening_debt=(
            debt.monthly["OpeningBalance"].to_numpy()
            + liquidity_monthly["WorkingCapitalFacilityOpening"].to_numpy()
        ),
        closing_debt=debt_balance,
        contractual_maturity=debt.contractual_maturity,
    )
    liquidity_annual = _annualize_liquidity(liquidity_monthly)

    shares = _product_revenue_shares(commercial)
    component_capex, component_dep = _component_capex_and_depreciation(
        capex, commercial, idc_depreciation
    )
    component_weight = component_capex.cumsum().div(
        component_capex.cumsum().sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(0.0)
    component_interest = component_weight.mul(total_cash_interest, axis=0)
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

    if labour.allocation_monthly.empty:
        labour_by_component = pd.DataFrame(
            0.0, index=dates, columns=COMPONENTS
        )
    else:
        labour_by_component = labour.allocation_monthly.pivot_table(
            index="Date",
            columns="Component",
            values="AllocatedLabourCost",
            aggfunc="sum",
            fill_value=0.0,
        )
        labour_by_component = labour_by_component.reindex(
            index=dates, columns=COMPONENTS, fill_value=0.0
        ).fillna(0.0)
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
        frame["LabourCost"] = labour_by_component[component]
        frame["EBITDA"] = (
            frame["Revenue"]
            - frame["DirectCosts"]
            - frame["SharedOpex"]
            - frame["LabourCost"]
        )
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
        [
            "Revenue", "DirectCosts", "SharedOpex", "LabourCost", "EBITDA",
            "Depreciation", "EBIT", "Interest", "Tax", "NetIncome",
        ]
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
    annual_cashflow["DSCR"] = np.divide(
        annual_cashflow["CFADS"],
        annual_cashflow["DebtService"],
        out=np.full(len(annual_cashflow), np.nan),
        where=annual_cashflow["DebtService"].to_numpy() > 1e-9,
    )
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
    labour_allocation_error = float(
        np.max(
            np.abs(
                labour.monthly["TotalLabourCost"] - labour_by_component.sum(axis=1)
            )
        )
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
    active_monthly_dscr = monthly_dscr[np.isfinite(monthly_dscr)]
    covenant_dscr = covenant_schedule["DSCR"].dropna().to_numpy(dtype=float)
    active_llcr = covenant_schedule["LLCR"].dropna().to_numpy(dtype=float)
    active_plcr = covenant_schedule["PLCR"].dropna().to_numpy(dtype=float)
    minimum_dscr = float(np.min(covenant_dscr)) if len(covenant_dscr) else None
    minimum_monthly_dscr = (
        float(np.min(active_monthly_dscr)) if len(active_monthly_dscr) else None
    )
    minimum_llcr = float(np.min(active_llcr)) if len(active_llcr) else None
    minimum_plcr = float(np.min(active_plcr)) if len(active_plcr) else None
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


    cod = effective_cod_date(inputs)
    pre_cod = np.asarray(dates < cod)
    pre_cod_processing = float(
        np.max(np.abs(sourcing.monthly.loc[pre_cod, "TotalCaneTonnes"]))
        if pre_cod.any()
        else 0.0
    )
    pre_cod_revenue = float(
        np.max(np.abs(revenue[pre_cod])) if pre_cod.any() else 0.0
    )
    pre_cod_depreciation = float(
        np.max(np.abs(depreciation[pre_cod])) if pre_cod.any() else 0.0
    )
    max_funding_shortfall = float(
        liquidity_monthly["FundingShortfall"].max()
    )
    cash_headroom = (
        liquidity_monthly["EndingCash"]
        - liquidity_monthly["MinimumCashTarget"]
    )
    minimum_cash_headroom = float(cash_headroom.min())
    max_dsra_gap = float(
        (
            liquidity_monthly["DSRATarget"]
            - liquidity_monthly["DSRABalance"]
        ).clip(lower=0.0).max()
    )
    max_maintenance_gap = float(
        (
            liquidity_monthly["MaintenanceReserveTarget"]
            - liquidity_monthly["MaintenanceReserveBalance"]
        ).clip(lower=0.0).max()
    )
    max_facility_excess = float(
        (
            liquidity_monthly["WorkingCapitalFacilityClosing"]
            - inputs.liquidity.working_capital_facility_limit
        ).clip(lower=0.0).max()
    )
    llcr_target = inputs.financing.minimum_llcr_target
    plcr_target = inputs.financing.minimum_plcr_target
    contingency_requirement = total_capex * inputs.construction.contingency_rate

    checks = pd.DataFrame(
        [
            {
                "Category": "Integrity",
                "Check": "Cultivated land is within arable/cultivable land",
                "Actual": cultivated_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if cultivated_excess <= land_tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Arable/cultivable land is within total land",
                "Actual": arable_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if arable_excess <= land_tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Irrigated land is within cultivated land",
                "Actual": irrigated_cultivated_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if irrigated_cultivated_excess <= land_tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Irrigated land is within irrigation capacity",
                "Actual": irrigated_capacity_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if irrigated_capacity_excess <= land_tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Farm cane processed is within harvested cane available",
                "Actual": farm_cane_excess,
                "Tolerance": land_tolerance,
                "Status": "OK" if farm_cane_excess <= land_tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Purchased cane reconciles the feedstock shortage",
                "Actual": purchase_reconciliation_error,
                "Tolerance": land_tolerance,
                "Status": "OK" if purchase_reconciliation_error <= land_tolerance else "FAIL",
            },
            {
                "Category": "Target",
                "Check": "Actual farm share meets the scenario target",
                "Actual": actual_farm_share,
                "Tolerance": farm_share_target,
                "Status": "OK" if farm_share_gap <= farm_share_tolerance else "WARN",
            },
            {
                "Category": "Integrity",
                "Check": "Bagasse routing sums to raw bagasse",
                "Actual": float(np.max(np.abs(bagasse_routing_error))),
                "Tolerance": tolerance,
                "Status": "OK" if np.max(np.abs(bagasse_routing_error)) <= tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Labour cost allocations reconcile",
                "Actual": labour_allocation_error,
                "Tolerance": tolerance,
                "Status": "OK" if labour_allocation_error <= tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Debt roll-forward",
                "Actual": float(np.max(np.abs(debt_error))),
                "Tolerance": tolerance,
                "Status": "OK" if np.max(np.abs(debt_error)) <= tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Balance sheet balances",
                "Actual": max_balance_error,
                "Tolerance": tolerance,
                "Status": "OK" if max_balance_error <= tolerance else "FAIL",
            },
            {
                "Category": "Integrity",
                "Check": "Component consolidation reconciles",
                "Actual": max_reconciliation_error,
                "Tolerance": tolerance,
                "Status": "OK" if max_reconciliation_error <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "No processing before effective COD",
                "Actual": pre_cod_processing,
                "Tolerance": tolerance,
                "Status": "OK" if pre_cod_processing <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "No revenue before effective COD",
                "Actual": pre_cod_revenue,
                "Tolerance": tolerance,
                "Status": "OK" if pre_cod_revenue <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "No depreciation before effective COD",
                "Actual": pre_cod_depreciation,
                "Tolerance": tolerance,
                "Status": "OK" if pre_cod_depreciation <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Debt repays by model end",
                "Actual": float(debt_balance[-1]),
                "Tolerance": tolerance,
                "Status": "OK" if debt_balance[-1] <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Projection covers debt maturity plus tail",
                "Actual": int(debt.model_horizon_covers_tail),
                "Tolerance": 1,
                "Status": "OK" if debt.model_horizon_covers_tail else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Minimum covenant-period DSCR meets target",
                "Actual": minimum_dscr,
                "Tolerance": dscr_target,
                "Status": "OK" if minimum_dscr is not None and minimum_dscr >= dscr_target else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Minimum LLCR meets target",
                "Actual": minimum_llcr,
                "Tolerance": llcr_target,
                "Status": "OK" if minimum_llcr is not None and minimum_llcr >= llcr_target else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Minimum PLCR meets target",
                "Actual": minimum_plcr,
                "Tolerance": plcr_target,
                "Status": "OK" if minimum_plcr is not None and minimum_plcr >= plcr_target else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "No unresolved funding shortfall",
                "Actual": max_funding_shortfall,
                "Tolerance": tolerance,
                "Status": "OK" if max_funding_shortfall <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Minimum unrestricted cash is maintained",
                "Actual": minimum_cash_headroom,
                "Tolerance": 0.0,
                "Status": "OK" if minimum_cash_headroom >= -tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "DSRA is fully funded",
                "Actual": max_dsra_gap,
                "Tolerance": tolerance,
                "Status": "OK" if max_dsra_gap <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Maintenance reserve is fully funded",
                "Actual": max_maintenance_gap,
                "Tolerance": tolerance,
                "Status": "OK" if max_maintenance_gap <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Working-capital facility remains within limit",
                "Actual": max_facility_excess,
                "Tolerance": tolerance,
                "Status": "OK" if max_facility_excess <= tolerance else "FAIL",
            },
            {
                "Category": "Bankability",
                "Check": "Construction contingency is budgeted",
                "Actual": contingency_requirement,
                "Tolerance": 0.0,
                "Status": "OK" if contingency_requirement > 0.0 else "FAIL",
            },
        ]
    )
    calculation_status = (
        "CHECK"
        if (
            (checks["Category"] == "Integrity")
            & (checks["Status"] == "FAIL")
        ).any()
        else "OK"
    )
    bankability_status = (
        "FAIL"
        if calculation_status != "OK"
        or (
            (checks["Category"] == "Bankability")
            & (checks["Status"] == "FAIL")
        ).any()
        else "PASS"
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
    equity_returns[-1] += max(
        0.0, terminal_value + cash[-1] + restricted_cash[-1] - debt_balance[-1]
    )
    metrics: dict[str, Any] = {
        "scenario": scenario,
        "scheduled_cod": inputs.construction.scheduled_cod,
        "effective_cod": cod.strftime("%Y-%m"),
        "construction_delay_months": inputs.construction.construction_delay_months,
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
        "total_labour_cost": float(labour.monthly["TotalLabourCost"].sum()),
        "peak_headcount_fte": float(labour.monthly["HeadcountFTE"].max()),
        "average_headcount_fte": float(labour.monthly["HeadcountFTE"].mean()),
        "labour_role_count": len(inputs.labour.items),
        "total_capex": total_capex,
        "construction_contingency_requirement": contingency_requirement,
        "project_npv": _npv(project_returns, inputs.global_assumptions.discount_rate),
        "project_irr": _annualized_irr(project_returns),
        "project_npv_without_terminal": _npv(
            project_fcf, inputs.global_assumptions.discount_rate
        ),
        "project_irr_without_terminal": _annualized_irr(project_fcf),
        "equity_npv": _npv(equity_returns, inputs.global_assumptions.discount_rate),
        "equity_irr": _annualized_irr(equity_returns),
        "payback_years": _payback(project_fcf),
        "covenant_period": inputs.financing.covenant_period,
        "min_dscr": minimum_dscr,
        "minimum_monthly_dscr": minimum_monthly_dscr,
        "average_dscr": (
            float(np.mean(covenant_dscr)) if len(covenant_dscr) else None
        ),
        "min_llcr": minimum_llcr,
        "min_plcr": minimum_plcr,
        "ending_debt": float(debt_balance[-1]),
        "ending_term_debt": float(term_debt_balance[-1]),
        "ending_working_capital_facility": float(liquidity_debt_balance[-1]),
        "senior_debt_draw": float(debt.summary.iloc[0]["InitialDraw"]),
        "additional_debt_draw": float(
            debt.summary.iloc[1:]["InitialDraw"].sum()
        ),
        "total_debt_draw": float(debt.summary["InitialDraw"].sum()),
        "total_liquidity_facility_draw": float(
            liquidity_monthly["WorkingCapitalFacilityDraw"].sum()
        ),
        "total_equity_contribution": float(equity_contribution.sum()),
        "total_equity_commitment": float(
            equity_contribution.sum() + contingency_requirement
        ),
        "minimum_cash_balance": float(cash[np.asarray(dates >= cod)].min()),
        "minimum_cash_headroom": minimum_cash_headroom,
        "maximum_funding_shortfall": max_funding_shortfall,
        "peak_dsra": float(liquidity_monthly["DSRABalance"].max()),
        "peak_maintenance_reserve": float(
            liquidity_monthly["MaintenanceReserveBalance"].max()
        ),
        "terminal_value": terminal_value,
        "calculation_status": calculation_status,
        "bankability_status": bankability_status,
        "model_status": (
            "CHECK"
            if calculation_status != "OK"
            else ("OK" if bankability_status == "PASS" else "BANKABILITY FAIL")
        ),
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
        covenant_schedule=covenant_schedule,
        liquidity_monthly=liquidity_monthly,
        liquidity_annual=liquidity_annual,
        checks=checks,
        metrics=metrics,
    )
