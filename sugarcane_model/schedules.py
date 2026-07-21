"""Operating and financial schedule builders for Sugar Cane Bioethanol."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from sugarcane_finance_math import npf

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


def _inflation(rate: float, month_index: np.ndarray) -> np.ndarray:
    return np.power(1.0 + rate, month_index / 12.0)


def _scenario_farm_share(inputs: SugarcaneBioethanolInputs, scenario: str) -> float:
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
    planning = inputs.cycle_planning
    planning_start = pd.Period(inputs.global_assumptions.planning_start, freq="M").to_timestamp()
    active = dates >= planning_start
    month_index = np.arange(len(dates))
    operating_month = np.maximum(0, (dates.year - planning_start.year) * 12 + dates.month - planning_start.month)
    cycle_month = operating_month % planning.crop_cycle_months + 1
    phases = np.where(
        active,
        np.where(
            cycle_month <= planning.establishment_months,
            "Establishment",
            np.where(
                cycle_month > planning.crop_cycle_months - planning.harvest_window_months,
                "Harvest",
                "Growing",
            ),
        ),
        "Pre-operational",
    )
    operating_year = operating_month // 12
    other = inputs.other_assumptions
    ramp = np.where(
        active,
        np.where(
            operating_year == 0,
            other.startup_ramp_year_1,
            np.where(operating_year == 1, other.startup_ramp_year_2, 1.0),
        ),
        0.0,
    )
    monthly = pd.DataFrame(
        {
            "CycleNumber": np.where(active, operating_month // planning.crop_cycle_months + 1, 0),
            "CycleMonth": np.where(active, cycle_month, 0),
            "Phase": phases,
            "RampFactor": ramp,
            "ReplantShare": np.where(active & (cycle_month == 1), planning.replant_share_per_cycle, 0.0),
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


def compute_farming_schedule(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    cycle_plan: ScheduleOutput,
) -> ScheduleOutput:
    dates = cycle_plan.monthly.index
    farm_share = _scenario_farm_share(inputs, scenario)
    other = inputs.other_assumptions
    routing = inputs.processing_routing
    farming = inputs.farming
    cost_factor = _inflation(other.cost_inflation_rate, np.arange(len(dates)))
    total_cane = (
        routing.annual_cane_capacity_tonnes
        * other.plant_availability
        * cycle_plan.monthly["RampFactor"].to_numpy()
        / 12.0
    )
    farm_cane = total_cane * farm_share
    harvested_area = farm_cane / (
        farming.sugarcane_yield_tonnes_per_hectare * farming.harvest_recovery
    )
    farm_cost = farm_cane * farming.farm_opex_per_tonne * cost_factor
    farm_overhead = (
        farming.farm_overhead_per_year / 12.0 * cost_factor * (1.0 if farm_share > 0 else 0.0)
    )
    transfer_price = farming.internal_transfer_price_per_tonne * cost_factor
    monthly = pd.DataFrame(
        {
            "FarmShare": farm_share,
            "FarmCaneTonnes": farm_cane,
            "FarmAreaHarvestedHa": harvested_area,
            "FarmAreaReplantedHa": harvested_area * cycle_plan.monthly["ReplantShare"].to_numpy(),
            "FarmDirectCost": farm_cost,
            "FarmOverhead": farm_overhead,
            "FarmOperatingCost": farm_cost + farm_overhead,
            "InternalTransferPrice": transfer_price,
            "InternalCaneRevenue": farm_cane * transfer_price,
        },
        index=dates,
    )
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_sourcing_schedule(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
    farming: ScheduleOutput,
    cycle_plan: ScheduleOutput,
) -> ScheduleOutput:
    dates = farming.monthly.index
    sourcing = inputs.sourcing
    other = inputs.other_assumptions
    routing = inputs.processing_routing
    total_cane = (
        routing.annual_cane_capacity_tonnes
        * other.plant_availability
        * cycle_plan.monthly["RampFactor"].to_numpy()
        / 12.0
    )
    purchased_cane = np.maximum(0.0, total_cane - farming.monthly["FarmCaneTonnes"].to_numpy())
    supplier_gross = purchased_cane / max(1e-9, 1.0 - sourcing.supplier_loss_rate)
    cost_factor = _inflation(other.cost_inflation_rate, np.arange(len(dates)))
    contracted_price = sourcing.cane_purchase_price_per_tonne * (1.0 - sourcing.contract_discount)
    weighted_price = (
        sourcing.contracted_purchase_share * contracted_price
        + (1.0 - sourcing.contracted_purchase_share) * sourcing.cane_purchase_price_per_tonne
    )
    purchase_price = weighted_price * cost_factor
    purchase_cost = supplier_gross * purchase_price
    logistics = purchased_cane * sourcing.logistics_cost_per_tonne * cost_factor
    farm_transfer = farming.monthly["InternalCaneRevenue"].to_numpy()
    monthly = pd.DataFrame(
        {
            "ScenarioFarmShare": _scenario_farm_share(inputs, scenario),
            "TotalCaneTonnes": total_cane,
            "FarmCaneTonnes": farming.monthly["FarmCaneTonnes"].to_numpy(),
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
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_processing_routing(
    inputs: SugarcaneBioethanolInputs,
    sourcing: ScheduleOutput,
) -> ScheduleOutput:
    route = inputs.processing_routing
    cane = sourcing.monthly["TotalCaneTonnes"].to_numpy()
    useful_cane = cane * (1.0 - inputs.other_assumptions.process_loss)
    ethanol = useful_cane * route.ethanol_litres_per_tonne
    sugar = useful_cane * route.sugar_tonnes_per_tonne
    raw_bagasse = useful_cane * route.raw_bagasse_tonnes_per_tonne
    electricity_bagasse = raw_bagasse * route.bagasse_to_electricity_share
    feed_bagasse = raw_bagasse * route.bagasse_to_animal_feed_share
    sale_bagasse = raw_bagasse * route.bagasse_to_sale_share
    gross_power = electricity_bagasse * route.electricity_mwh_per_tonne_bagasse
    internal_power = np.minimum(gross_power, cane * route.internal_electricity_mwh_per_tonne_cane)
    exported_power = np.maximum(0.0, gross_power - internal_power)
    animal_feed = feed_bagasse * route.animal_feed_conversion_rate
    monthly = pd.DataFrame(
        {
            "CaneReceivedTonnes": cane,
            "UsefulCaneTonnes": useful_cane,
            "BioethanolLitres": ethanol,
            "SugarTonnes": sugar,
            "RawBagasseTonnes": raw_bagasse,
            "BagasseToElectricityTonnes": electricity_bagasse,
            "BagasseToAnimalFeedTonnes": feed_bagasse,
            "BagasseForSaleTonnes": sale_bagasse,
            "GrossElectricityMWh": gross_power,
            "InternalElectricityMWh": internal_power,
            "ExportElectricityMWh": exported_power,
            "AnimalFeedTonnes": animal_feed,
        },
        index=sourcing.monthly.index,
    )
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_commercialization(
    inputs: SugarcaneBioethanolInputs,
    processing: ScheduleOutput,
) -> ScheduleOutput:
    commercial = inputs.commercialization
    price_factor = _inflation(
        inputs.other_assumptions.price_escalation_rate,
        np.arange(len(processing.monthly)),
    )
    monthly = pd.DataFrame(index=processing.monthly.index)
    monthly["BioethanolSalesLitres"] = (
        processing.monthly["BioethanolLitres"] * commercial.ethanol_sales_capture
    )
    monthly["SugarSalesTonnes"] = processing.monthly["SugarTonnes"] * commercial.sugar_sales_capture
    monthly["ElectricitySalesMWh"] = (
        processing.monthly["ExportElectricityMWh"] * commercial.power_sales_capture
    )
    monthly["BagasseSalesTonnes"] = (
        processing.monthly["BagasseForSaleTonnes"] * commercial.coproduct_sales_capture
    )
    monthly["AnimalFeedSalesTonnes"] = (
        processing.monthly["AnimalFeedTonnes"] * commercial.coproduct_sales_capture
    )
    monthly["BioethanolRevenue"] = (
        monthly["BioethanolSalesLitres"] * commercial.ethanol_price_per_litre * price_factor
    )
    monthly["SugarRevenue"] = monthly["SugarSalesTonnes"] * commercial.sugar_price_per_tonne * price_factor
    monthly["ElectricityRevenue"] = (
        monthly["ElectricitySalesMWh"] * commercial.electricity_tariff_per_mwh * price_factor
    )
    monthly["BagasseRevenue"] = monthly["BagasseSalesTonnes"] * commercial.bagasse_price_per_tonne * price_factor
    monthly["AnimalFeedRevenue"] = (
        monthly["AnimalFeedSalesTonnes"] * commercial.animal_feed_price_per_tonne * price_factor
    )
    monthly["ExternalRevenue"] = monthly[list(REVENUE_COLUMNS.values())].sum(axis=1)
    monthly["ElectricityBagasseTransferCost"] = (
        processing.monthly["BagasseToElectricityTonnes"]
        * commercial.bagasse_price_per_tonne
        * price_factor
    )
    monthly["AnimalFeedBagasseTransferCost"] = (
        processing.monthly["BagasseToAnimalFeedTonnes"]
        * commercial.bagasse_price_per_tonne
        * price_factor
    )
    monthly["InternalBagasseTransferRevenue"] = (
        monthly["ElectricityBagasseTransferCost"]
        + monthly["AnimalFeedBagasseTransferCost"]
    )
    monthly["InternalElectricityTransferRevenue"] = (
        processing.monthly["InternalElectricityMWh"]
        * commercial.electricity_tariff_per_mwh
        * price_factor
    )
    monthly["TotalInternalTransferRevenue"] = (
        monthly["InternalBagasseTransferRevenue"]
        + monthly["InternalElectricityTransferRevenue"]
    )
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_cost_schedule(
    inputs: SugarcaneBioethanolInputs,
    farming: ScheduleOutput,
    sourcing: ScheduleOutput,
    processing: ScheduleOutput,
) -> ScheduleOutput:
    cost = inputs.costs
    factor = _inflation(inputs.other_assumptions.cost_inflation_rate, np.arange(len(processing.monthly)))
    monthly = pd.DataFrame(index=processing.monthly.index)
    monthly["FarmingOperatingCost"] = farming.monthly["FarmOperatingCost"]
    monthly["PurchasedCaneCost"] = sourcing.monthly["PurchasedCaneCost"]
    monthly["InboundLogisticsCost"] = sourcing.monthly["InboundLogisticsCost"]
    monthly["ProcessingVariableCost"] = (
        processing.monthly["CaneReceivedTonnes"] * cost.processing_variable_cost_per_tonne_cane * factor
    )
    monthly["BioethanolVariableCost"] = (
        processing.monthly["BioethanolLitres"] * cost.ethanol_variable_cost_per_litre * factor
    )
    monthly["SugarVariableCost"] = processing.monthly["SugarTonnes"] * cost.sugar_variable_cost_per_tonne * factor
    monthly["ElectricityVariableCost"] = (
        processing.monthly["GrossElectricityMWh"] * cost.electricity_variable_cost_per_mwh * factor
    )
    monthly["BagasseHandlingCost"] = (
        processing.monthly["BagasseForSaleTonnes"] * cost.bagasse_handling_cost_per_tonne * factor
    )
    monthly["AnimalFeedVariableCost"] = (
        processing.monthly["AnimalFeedTonnes"] * cost.animal_feed_variable_cost_per_tonne * factor
    )
    monthly["FixedProcessingOpex"] = cost.fixed_processing_opex_per_year / 12.0 * factor
    monthly["CommercialAndAdminCost"] = cost.commercial_and_admin_cost_per_year / 12.0 * factor
    monthly["ProductSpecificVariableCost"] = monthly[
        [
            "BioethanolVariableCost",
            "SugarVariableCost",
            "ElectricityVariableCost",
            "BagasseHandlingCost",
            "AnimalFeedVariableCost",
        ]
    ].sum(axis=1)
    monthly["EconomicOperatingCost"] = monthly[
        [
            "FarmingOperatingCost",
            "PurchasedCaneCost",
            "InboundLogisticsCost",
            "ProcessingVariableCost",
            "ProductSpecificVariableCost",
            "FixedProcessingOpex",
            "CommercialAndAdminCost",
        ]
    ].sum(axis=1)
    return ScheduleOutput(monthly=monthly, annual=_annual_sum(monthly))


def compute_capex_schedule(
    inputs: SugarcaneBioethanolInputs,
    scenario: str,
) -> CapexOutput:
    dates = build_timeline(inputs)
    farm_share = _scenario_farm_share(inputs, scenario)
    components = sorted({item.component for item in inputs.capex.items} | {"Shared Processing"})
    capex_by_component = {component: np.zeros(len(dates)) for component in components}
    depreciation_by_component = {component: np.zeros(len(dates)) for component in components}
    rows: list[dict[str, Any]] = []

    for item in inputs.capex.items:
        amount = float(item.amount) * (farm_share if item.is_farm_capex else 1.0)
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


def compute_debt_schedule(
    inputs: SugarcaneBioethanolInputs,
    capex: CapexOutput,
) -> DebtOutput:
    dates = capex.monthly.index
    financing = inputs.financing
    draws = capex.monthly["TotalCapex"].to_numpy() * financing.debt_ratio
    months = len(dates)
    monthly_rate = financing.interest_rate / 12.0
    grace = min(financing.grace_years * 12, months)
    maturity = min(financing.tenor_years * 12, months)
    opening = np.zeros(months)
    interest = np.zeros(months)
    idc = np.zeros(months)
    cash_interest = np.zeros(months)
    principal = np.zeros(months)
    closing = np.zeros(months)
    balance = 0.0

    for month in range(months):
        opening[month] = balance
        interest[month] = (balance + 0.5 * draws[month]) * monthly_rate
        if month < grace and financing.capitalize_idc:
            idc[month] = interest[month]
        else:
            cash_interest[month] = interest[month]
        available = balance + draws[month] + idc[month]
        if month >= grace and month < maturity:
            remaining = max(1, maturity - month)
            if financing.amortization_type == "annuity" and monthly_rate > 0:
                payment = available * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-remaining))
                scheduled = max(0.0, payment - cash_interest[month])
            else:
                scheduled = available / remaining
            principal[month] = min(available, scheduled)
        balance = max(0.0, available - principal[month])
        closing[month] = balance

    monthly = pd.DataFrame(
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
    annual = _annual_sum(monthly.drop(columns=["OpeningBalance", "ClosingBalance"]))
    annual["OpeningBalance"] = monthly.groupby(monthly.index.year)["OpeningBalance"].first()
    annual["ClosingBalance"] = monthly.groupby(monthly.index.year)["ClosingBalance"].last()
    annual = annual[
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
    summary = pd.DataFrame(
        [
            {
                "DebtRatio": financing.debt_ratio,
                "InterestRate": financing.interest_rate,
                "TenorYears": financing.tenor_years,
                "GraceYears": financing.grace_years,
                "InitialDraw": float(draws.sum()),
                "CapitalizedInterest": float(idc.sum()),
                "EndingBalance": float(closing[-1]),
            }
        ]
    )
    return DebtOutput(monthly=monthly, annual=annual, summary=summary)


def compute_working_capital(
    inputs: SugarcaneBioethanolInputs,
    commercial: ScheduleOutput,
    costs: ScheduleOutput,
) -> ScheduleOutput:
    wc = inputs.working_capital
    days_per_month = 365.0 / 12.0
    revenue = commercial.monthly["ExternalRevenue"]
    cash_cost = costs.monthly["EconomicOperatingCost"]
    eligible_payables = costs.monthly[
        [
            "PurchasedCaneCost",
            "InboundLogisticsCost",
            "ProcessingVariableCost",
            "ProductSpecificVariableCost",
        ]
    ].sum(axis=1)
    monthly = pd.DataFrame(index=commercial.monthly.index)
    monthly["AccountsReceivable"] = revenue * wc.receivable_days / days_per_month
    monthly["Inventory"] = cash_cost * wc.inventory_days / days_per_month
    monthly["AccountsPayable"] = eligible_payables * wc.payable_days / days_per_month
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
    equity_contribution = capex_spend * (1.0 - inputs.financing.debt_ratio)
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
    checks = pd.DataFrame(
        [
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
                "Tolerance": inputs.other_assumptions.minimum_dscr_target,
                "Status": "OK" if minimum_dscr is None or minimum_dscr >= inputs.other_assumptions.minimum_dscr_target else "WARN",
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
        "farm_share": _scenario_farm_share(inputs, scenario),
        "total_capex": total_capex,
        "project_npv": _npv(project_returns, inputs.global_assumptions.discount_rate),
        "project_irr": _annualized_irr(project_returns),
        "equity_npv": _npv(equity_returns, inputs.global_assumptions.discount_rate),
        "equity_irr": _annualized_irr(equity_returns),
        "payback_years": _payback(project_fcf),
        "min_dscr": minimum_dscr,
        "average_dscr": float(np.mean(active_dscr)) if len(active_dscr) else None,
        "ending_debt": float(debt_balance[-1]),
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
