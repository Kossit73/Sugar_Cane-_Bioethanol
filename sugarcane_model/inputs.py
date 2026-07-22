"""Typed input landing page, grouped like the Cassava Ethanol model."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


ScenarioName = Literal["FARM_ONLY", "BUY_ONLY", "HYBRID"]
AmortizationType = Literal["straight", "annuity"]
SCENARIOS: tuple[str, ...] = ("FARM_ONLY", "BUY_ONLY", "HYBRID")


class GlobalAssumptions(BaseModel):
    start_year: int = Field(default=2025, ge=2020, le=2100)
    end_year: int = Field(default=2035, ge=2022, le=2120)
    planning_start: str = "2025-01"
    scenario: ScenarioName = "HYBRID"
    hybrid_farm_share: float = Field(default=0.50, ge=0.0, le=1.0)
    corporate_tax_rate: float = Field(default=0.28, ge=0.0, le=1.0)
    discount_rate: float = Field(default=0.12, gt=0.0, le=1.0)
    terminal_growth_rate: float = Field(default=0.02, ge=-0.50, le=0.25)
    investor_share: float = Field(default=0.60, ge=0.0, le=1.0)


class OtherAssumptions(BaseModel):
    plant_availability: float = Field(default=0.90, gt=0.0, le=1.0)
    process_loss: float = Field(default=0.02, ge=0.0, lt=1.0)
    startup_ramp_year_1: float = Field(default=0.70, gt=0.0, le=1.0)
    startup_ramp_year_2: float = Field(default=0.90, gt=0.0, le=1.0)
    price_escalation_rate: float = Field(default=0.02, ge=-0.50, le=1.0)
    cost_inflation_rate: float = Field(default=0.02, ge=-0.50, le=1.0)
    minimum_dscr_target: float = Field(default=1.20, ge=0.0, le=10.0)


class CapexItem(BaseModel):
    item: str
    component: str
    amount: float = Field(ge=0.0)
    start_month: str = "2025-01"
    spend_months: int = Field(default=12, ge=1, le=120)
    life_years: int = Field(default=10, ge=1, le=50)
    is_farm_capex: bool = False


def _default_capex_items() -> list[CapexItem]:
    return [
        CapexItem(
            item="Farm Development & Machinery",
            component="Farming",
            amount=5_000_000.0,
            life_years=10,
            is_farm_capex=True,
        ),
        CapexItem(
            item="Cane Reception & Milling",
            component="Shared Processing",
            amount=7_500_000.0,
            life_years=15,
        ),
        CapexItem(
            item="Fermentation & Distillation",
            component="Bioethanol",
            amount=14_000_000.0,
            life_years=15,
        ),
        CapexItem(
            item="Sugar Recovery",
            component="Sugar",
            amount=4_000_000.0,
            life_years=12,
        ),
        CapexItem(
            item="Bagasse Cogeneration",
            component="Electricity Generation",
            amount=4_000_000.0,
            life_years=15,
        ),
        CapexItem(
            item="Utilities, Feed & Bagasse Handling",
            component="Shared Processing",
            amount=3_000_000.0,
            life_years=10,
        ),
    ]


class CapexAssumptions(BaseModel):
    items: list[CapexItem] = Field(default_factory=_default_capex_items)


class CyclePlanningAssumptions(BaseModel):
    crop_cycle_months: int = Field(default=12, ge=6, le=36)
    establishment_months: int = Field(default=3, ge=1, le=12)
    harvest_window_months: int = Field(default=3, ge=1, le=12)
    ratoon_cycles: int = Field(default=4, ge=0, le=10)
    replant_share_per_cycle: float = Field(default=0.20, ge=0.0, le=1.0)


class FarmPlanningAssumptions(BaseModel):
    """Physical land envelope and annual cultivation deployment assumptions."""

    total_land_hectares: float = Field(default=1_000.0, gt=0.0)
    arable_land_hectares: float = Field(default=800.0, gt=0.0)
    planned_cultivated_hectares: float = Field(default=700.0, ge=0.0)
    irrigation_capacity_hectares: float = Field(default=600.0, ge=0.0)
    planned_irrigated_hectares: float = Field(default=500.0, ge=0.0)
    hectares_harvested: float = Field(default=632.122979, ge=0.0)


def validate_farm_planning_values(values: dict[str, Any]) -> list[str]:
    """Return the shared land-capacity validation messages for one annual plan."""

    total_land = float(values["total_land_hectares"])
    arable_land = float(values["arable_land_hectares"])
    cultivated = float(values["planned_cultivated_hectares"])
    irrigation_capacity = float(values["irrigation_capacity_hectares"])
    irrigated = float(values["planned_irrigated_hectares"])
    harvested = float(values["hectares_harvested"])
    errors: list[str] = []
    if arable_land > total_land:
        errors.append("Arable/cultivable land cannot exceed total land.")
    if cultivated > arable_land:
        errors.append("Planned cultivated hectares cannot exceed arable/cultivable land.")
    if irrigated > cultivated:
        errors.append("Planned irrigated hectares cannot exceed planned cultivated hectares.")
    if irrigated > irrigation_capacity:
        errors.append("Planned irrigated hectares cannot exceed irrigation capacity.")
    if harvested > cultivated:
        errors.append("Hectares harvested cannot exceed planned cultivated hectares.")
    return errors


class FarmingAssumptions(BaseModel):
    sugarcane_yield_tonnes_per_hectare: float = Field(default=70.0, gt=0.0)
    harvest_recovery: float = Field(default=0.98, gt=0.0, le=1.0)
    farm_opex_per_tonne: float = Field(default=18.0, ge=0.0)
    farm_overhead_per_year: float = Field(default=300_000.0, ge=0.0)
    internal_transfer_price_per_tonne: float = Field(default=65.0, ge=0.0)


class SourcingAssumptions(BaseModel):
    cane_purchase_price_per_tonne: float = Field(default=65.0, ge=0.0)
    contracted_purchase_share: float = Field(default=0.60, ge=0.0, le=1.0)
    contract_discount: float = Field(default=0.05, ge=0.0, le=0.80)
    logistics_cost_per_tonne: float = Field(default=5.0, ge=0.0)
    supplier_loss_rate: float = Field(default=0.01, ge=0.0, lt=1.0)


class ProcessingRoutingAssumptions(BaseModel):
    annual_cane_capacity_tonnes: float = Field(default=100_000.0, gt=0.0)
    ethanol_litres_per_tonne: float = Field(default=160.0, gt=0.0)
    sugar_tonnes_per_tonne: float = Field(default=0.10, ge=0.0, le=1.0)
    raw_bagasse_tonnes_per_tonne: float = Field(default=0.28, ge=0.0, le=1.0)
    bagasse_to_electricity_share: float = Field(default=0.55, ge=0.0, le=1.0)
    bagasse_to_animal_feed_share: float = Field(default=0.25, ge=0.0, le=1.0)
    bagasse_to_sale_share: float = Field(default=0.20, ge=0.0, le=1.0)
    electricity_mwh_per_tonne_bagasse: float = Field(default=0.45, ge=0.0)
    internal_electricity_mwh_per_tonne_cane: float = Field(default=0.04, ge=0.0)
    animal_feed_conversion_rate: float = Field(default=0.75, ge=0.0, le=1.0)


class CommercializationAssumptions(BaseModel):
    ethanol_price_per_litre: float = Field(default=0.70, ge=0.0)
    sugar_price_per_tonne: float = Field(default=450.0, ge=0.0)
    electricity_tariff_per_mwh: float = Field(default=80.0, ge=0.0)
    bagasse_price_per_tonne: float = Field(default=35.0, ge=0.0)
    animal_feed_price_per_tonne: float = Field(default=180.0, ge=0.0)
    ethanol_sales_capture: float = Field(default=1.0, ge=0.0, le=1.0)
    sugar_sales_capture: float = Field(default=1.0, ge=0.0, le=1.0)
    power_sales_capture: float = Field(default=1.0, ge=0.0, le=1.0)
    coproduct_sales_capture: float = Field(default=1.0, ge=0.0, le=1.0)


class CostAssumptions(BaseModel):
    processing_variable_cost_per_tonne_cane: float = Field(default=12.0, ge=0.0)
    ethanol_variable_cost_per_litre: float = Field(default=0.05, ge=0.0)
    sugar_variable_cost_per_tonne: float = Field(default=35.0, ge=0.0)
    electricity_variable_cost_per_mwh: float = Field(default=15.0, ge=0.0)
    bagasse_handling_cost_per_tonne: float = Field(default=4.0, ge=0.0)
    animal_feed_variable_cost_per_tonne: float = Field(default=25.0, ge=0.0)
    fixed_processing_opex_per_year: float = Field(default=2_000_000.0, ge=0.0)
    commercial_and_admin_cost_per_year: float = Field(default=500_000.0, ge=0.0)


class WorkingCapitalAssumptions(BaseModel):
    receivable_days: float = Field(default=30.0, ge=0.0, le=365.0)
    inventory_days: float = Field(default=20.0, ge=0.0, le=365.0)
    payable_days: float = Field(default=25.0, ge=0.0, le=365.0)


class DebtFacilityAssumptions(BaseModel):
    """Terms for one fixed-amount debt facility added to senior debt."""

    name: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0.0)
    interest_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    tenor_years: int = Field(default=8, ge=2, le=40)
    grace_years: int = Field(default=1, ge=0, le=10)
    amortization_type: AmortizationType = "straight"
    capitalize_idc: bool = True


class FinancingAssumptions(BaseModel):
    debt_ratio: float = Field(default=0.60, ge=0.0, le=0.95)
    interest_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    tenor_years: int = Field(default=8, ge=2, le=40)
    grace_years: int = Field(default=1, ge=0, le=10)
    amortization_type: AmortizationType = "straight"
    capitalize_idc: bool = True
    additional_debt_facilities: list[DebtFacilityAssumptions] = Field(
        default_factory=list
    )


class SugarcaneBioethanolInputs(BaseModel):
    """Input landing page whose sections mirror the Cassava architecture."""

    global_assumptions: GlobalAssumptions = Field(default_factory=GlobalAssumptions)
    other_assumptions: OtherAssumptions = Field(default_factory=OtherAssumptions)
    capex: CapexAssumptions = Field(default_factory=CapexAssumptions)
    cycle_planning: CyclePlanningAssumptions = Field(default_factory=CyclePlanningAssumptions)
    farm_planning: FarmPlanningAssumptions = Field(default_factory=FarmPlanningAssumptions)
    farming: FarmingAssumptions = Field(default_factory=FarmingAssumptions)
    sourcing: SourcingAssumptions = Field(default_factory=SourcingAssumptions)
    processing_routing: ProcessingRoutingAssumptions = Field(default_factory=ProcessingRoutingAssumptions)
    commercialization: CommercializationAssumptions = Field(default_factory=CommercializationAssumptions)
    costs: CostAssumptions = Field(default_factory=CostAssumptions)
    working_capital: WorkingCapitalAssumptions = Field(default_factory=WorkingCapitalAssumptions)
    financing: FinancingAssumptions = Field(default_factory=FinancingAssumptions)
    yearly_schedules: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    def grouped_sections(self) -> "OrderedDict[str, BaseModel]":
        return OrderedDict(
            [
                ("Global assumptions", self.global_assumptions),
                ("Other assumptions", self.other_assumptions),
                ("Capex", self.capex),
                ("Cycle planning", self.cycle_planning),
                ("Farm planning", self.farm_planning),
                ("Farming", self.farming),
                ("Sourcing", self.sourcing),
                ("Processing & production routing", self.processing_routing),
                ("Commercialization", self.commercialization),
                ("Costs", self.costs),
                ("Working capital", self.working_capital),
                ("Financing", self.financing),
            ]
        )

    def signature(self) -> str:
        payload = json.dumps(input_values(self), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def input_values(inputs: SugarcaneBioethanolInputs) -> dict[str, Any]:
    dumper = getattr(inputs, "model_dump", None)
    if callable(dumper):
        return dumper()
    return inputs.dict()


def input_from_payload(value: Any) -> SugarcaneBioethanolInputs:
    if isinstance(value, SugarcaneBioethanolInputs):
        return value
    validator = getattr(SugarcaneBioethanolInputs, "model_validate", None)
    if callable(validator):
        return validator(value)
    return SugarcaneBioethanolInputs.parse_obj(value)


def default_input_page() -> SugarcaneBioethanolInputs:
    return SugarcaneBioethanolInputs()
