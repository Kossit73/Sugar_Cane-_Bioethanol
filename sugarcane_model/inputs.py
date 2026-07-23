"""Typed input landing page, grouped like the Cassava Ethanol model."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


ScenarioName = Literal["FARM_ONLY", "BUY_ONLY", "HYBRID"]
AmortizationType = Literal["straight", "annuity"]
DebtSizingMode = Literal["fixed_ratio", "dscr_sculpted"]
CovenantPeriod = Literal["monthly", "quarterly", "semiannual", "annual"]
WorkerType = Literal["Permanent", "Seasonal", "Contract"]
LabourComponent = Literal[
    "Farming", "Bioethanol", "Sugar", "Electricity Generation",
    "Bagasse", "Animal Feed", "Shared Plant",
]
LabourAllocationDriver = Literal[
    "Direct", "Product revenue share", "Equal product share", "Custom product share",
]
ProductivityDriver = Literal[
    "None", "Cultivated hectares", "Harvested hectares", "Farm cane tonnes",
    "Cane processed tonnes", "Bioethanol litres", "Sugar tonnes",
    "Electricity MWh", "Bagasse tonnes", "Animal feed tonnes",
]
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


class ConstructionAssumptions(BaseModel):
    """Construction, commissioning, and commercial-operations timing."""

    scheduled_cod: str = "2026-01"
    construction_delay_months: int = Field(default=0, ge=0, le=60)
    commissioning_months: int = Field(default=2, ge=0, le=24)
    contingency_rate: float = Field(default=0.10, ge=0.0, le=1.0)


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


class LabourPlanItem(BaseModel):
    """One role entered once and expanded into the monthly labour schedule."""

    role_id: str = Field(min_length=1, max_length=40)
    cost_centre: str = Field(min_length=1, max_length=80)
    department: str = Field(min_length=1, max_length=80)
    position: str = Field(min_length=1, max_length=100)
    worker_type: WorkerType = "Permanent"
    component: LabourComponent = "Shared Plant"
    start_month: str = "2026-01"
    end_month: str | None = None
    headcount_fte: float = Field(default=1.0, ge=0.0)
    number_of_shifts: int = Field(default=1, ge=1, le=4)
    monthly_wage_per_fte: float = Field(default=1_000.0, ge=0.0)
    overtime_rate: float = Field(default=0.0, ge=0.0, le=2.0)
    benefits_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    statutory_contribution_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    training_cost_per_fte_year: float = Field(default=0.0, ge=0.0)
    ppe_cost_per_fte_year: float = Field(default=0.0, ge=0.0)
    transport_cost_per_fte_month: float = Field(default=0.0, ge=0.0)
    accommodation_cost_per_fte_month: float = Field(default=0.0, ge=0.0)
    annual_salary_escalation: float = Field(default=0.03, ge=-0.50, le=1.0)
    allocation_driver: LabourAllocationDriver = "Product revenue share"
    productivity_driver: ProductivityDriver = "None"
    bioethanol_share: float = Field(default=0.0, ge=0.0, le=1.0)
    sugar_share: float = Field(default=0.0, ge=0.0, le=1.0)
    electricity_share: float = Field(default=0.0, ge=0.0, le=1.0)
    bagasse_share: float = Field(default=0.0, ge=0.0, le=1.0)
    animal_feed_share: float = Field(default=0.0, ge=0.0, le=1.0)


def _default_labour_items() -> list[LabourPlanItem]:
    return [
        LabourPlanItem(
            role_id="FARM-MGR", cost_centre="Farming", department="Farm management",
            position="Farm manager", component="Farming", start_month="2025-01",
            headcount_fte=1.0, monthly_wage_per_fte=1750.0, benefits_rate=0.15,
            statutory_contribution_rate=0.08, training_cost_per_fte_year=600.0,
            ppe_cost_per_fte_year=300.0, transport_cost_per_fte_month=100.0,
            annual_salary_escalation=0.04, allocation_driver="Direct",
            productivity_driver="Cultivated hectares",
        ),
        LabourPlanItem(
            role_id="FARM-AGRON", cost_centre="Farming", department="Agronomy",
            position="Agronomist", component="Farming", start_month="2025-01",
            headcount_fte=2.0, monthly_wage_per_fte=1250.0, benefits_rate=0.15,
            statutory_contribution_rate=0.08, training_cost_per_fte_year=450.0,
            ppe_cost_per_fte_year=250.0, transport_cost_per_fte_month=87.5,
            annual_salary_escalation=0.04, allocation_driver="Direct",
            productivity_driver="Harvested hectares",
        ),
        LabourPlanItem(
            role_id="FARM-IRR", cost_centre="Farming", department="Irrigation",
            position="Irrigation technician", component="Farming", start_month="2025-01",
            headcount_fte=4.0, number_of_shifts=2, monthly_wage_per_fte=700.0,
            overtime_rate=0.05, benefits_rate=0.12, statutory_contribution_rate=0.08,
            training_cost_per_fte_year=300.0, ppe_cost_per_fte_year=250.0,
            transport_cost_per_fte_month=75.0, annual_salary_escalation=0.04,
            allocation_driver="Direct", productivity_driver="Cultivated hectares",
        ),
        LabourPlanItem(
            role_id="FARM-FIELD", cost_centre="Farming", department="Field operations",
            position="Field worker", component="Farming", start_month="2025-01",
            headcount_fte=30.0, monthly_wage_per_fte=300.0, overtime_rate=0.08,
            statutory_contribution_rate=0.08, training_cost_per_fte_year=100.0,
            ppe_cost_per_fte_year=175.0, transport_cost_per_fte_month=50.0,
            annual_salary_escalation=0.04, allocation_driver="Direct",
            productivity_driver="Harvested hectares",
        ),
        LabourPlanItem(
            role_id="CANE-LOG", cost_centre="Sourcing & Logistics",
            department="Cane procurement", position="Procurement and logistics officer",
            component="Shared Plant", start_month="2026-01", headcount_fte=4.0,
            monthly_wage_per_fte=750.0, benefits_rate=0.12,
            statutory_contribution_rate=0.08, training_cost_per_fte_year=250.0,
            ppe_cost_per_fte_year=150.0, transport_cost_per_fte_month=75.0,
            annual_salary_escalation=0.04, allocation_driver="Product revenue share",
            productivity_driver="Cane processed tonnes",
        ),
        LabourPlanItem(
            role_id="PLANT-MGR", cost_centre="Processing Plant",
            department="Plant management", position="Plant manager",
            component="Shared Plant", start_month="2026-01", headcount_fte=1.0,
            monthly_wage_per_fte=2500.0, benefits_rate=0.18,
            statutory_contribution_rate=0.08, training_cost_per_fte_year=750.0,
            ppe_cost_per_fte_year=300.0, transport_cost_per_fte_month=125.0,
            annual_salary_escalation=0.04, allocation_driver="Product revenue share",
            productivity_driver="Cane processed tonnes",
        ),
        LabourPlanItem(
            role_id="PLANT-OPS", cost_centre="Processing Plant", department="Operations",
            position="Process operator", component="Shared Plant", start_month="2026-01",
            headcount_fte=18.0, number_of_shifts=3, monthly_wage_per_fte=700.0,
            overtime_rate=0.08, benefits_rate=0.12, statutory_contribution_rate=0.08,
            training_cost_per_fte_year=250.0, ppe_cost_per_fte_year=225.0,
            transport_cost_per_fte_month=62.5, annual_salary_escalation=0.04,
            allocation_driver="Product revenue share",
            productivity_driver="Cane processed tonnes",
        ),
        LabourPlanItem(
            role_id="PLANT-MAINT", cost_centre="Maintenance & Engineering",
            department="Maintenance", position="Maintenance engineer or technician",
            component="Shared Plant", start_month="2026-01", headcount_fte=8.0,
            number_of_shifts=2, monthly_wage_per_fte=900.0, overtime_rate=0.08,
            benefits_rate=0.12, statutory_contribution_rate=0.08,
            training_cost_per_fte_year=400.0, ppe_cost_per_fte_year=275.0,
            transport_cost_per_fte_month=75.0, annual_salary_escalation=0.04,
            allocation_driver="Product revenue share",
            productivity_driver="Cane processed tonnes",
        ),
        LabourPlanItem(
            role_id="PLANT-LAB", cost_centre="Laboratory & Quality",
            department="Quality control", position="Laboratory or quality technician",
            component="Shared Plant", start_month="2026-01", headcount_fte=4.0,
            number_of_shifts=2, monthly_wage_per_fte=800.0, benefits_rate=0.12,
            statutory_contribution_rate=0.08, training_cost_per_fte_year=350.0,
            ppe_cost_per_fte_year=175.0, transport_cost_per_fte_month=62.5,
            annual_salary_escalation=0.04, allocation_driver="Product revenue share",
            productivity_driver="Cane processed tonnes",
        ),
        LabourPlanItem(
            role_id="PLANT-HSE", cost_centre="HSE",
            department="Health, safety and environment", position="HSE officer",
            component="Shared Plant", start_month="2026-01", headcount_fte=2.0,
            monthly_wage_per_fte=900.0, benefits_rate=0.12,
            statutory_contribution_rate=0.08, training_cost_per_fte_year=450.0,
            ppe_cost_per_fte_year=225.0, transport_cost_per_fte_month=75.0,
            annual_salary_escalation=0.04, allocation_driver="Product revenue share",
            productivity_driver="Cane processed tonnes",
        ),
        LabourPlanItem(
            role_id="PLANT-SEC", cost_centre="Security", department="Security",
            position="Security officer", worker_type="Contract", component="Shared Plant",
            start_month="2026-01", headcount_fte=9.0, number_of_shifts=3,
            monthly_wage_per_fte=350.0, overtime_rate=0.05,
            statutory_contribution_rate=0.05, ppe_cost_per_fte_year=125.0,
            transport_cost_per_fte_month=37.5, annual_salary_escalation=0.04,
            allocation_driver="Equal product share", productivity_driver="None",
        ),
        LabourPlanItem(
            role_id="ADMIN", cost_centre="Commercial & Administration",
            department="Finance, HR and administration",
            position="Administrative professional", component="Shared Plant",
            start_month="2026-01", headcount_fte=6.0, monthly_wage_per_fte=800.0,
            benefits_rate=0.15, statutory_contribution_rate=0.08,
            training_cost_per_fte_year=300.0, transport_cost_per_fte_month=62.5,
            annual_salary_escalation=0.04, allocation_driver="Product revenue share",
            productivity_driver="None",
        ),
    ]


class LabourAssumptions(BaseModel):
    """Consolidated role plan; non-labour OPEX remains in Farming and Costs."""

    items: list[LabourPlanItem] = Field(default_factory=_default_labour_items)


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
    debt_sizing_mode: DebtSizingMode = "fixed_ratio"
    covenant_period: CovenantPeriod = "annual"
    minimum_llcr_target: float = Field(default=1.35, ge=0.0, le=10.0)
    minimum_plcr_target: float = Field(default=1.50, ge=0.0, le=10.0)
    debt_tail_months: int = Field(default=12, ge=0, le=120)
    cash_sweep_percent: float = Field(default=0.0, ge=0.0, le=1.0)
    dividend_lockup_dscr: float = Field(default=1.30, ge=0.0, le=10.0)
    additional_debt_facilities: list[DebtFacilityAssumptions] = Field(
        default_factory=list
    )


class LiquidityAssumptions(BaseModel):
    """Minimum liquidity, reserve accounts, and committed funding support."""

    minimum_cash_balance: float = Field(default=500_000.0, ge=0.0)
    dsra_months: int = Field(default=6, ge=0, le=24)
    maintenance_reserve_rate: float = Field(default=0.01, ge=0.0, le=0.25)
    working_capital_facility_limit: float = Field(default=2_000_000.0, ge=0.0)
    working_capital_facility_interest_rate: float = Field(
        default=0.12, ge=0.0, le=1.0
    )
    sponsor_support_limit: float = Field(default=0.0, ge=0.0)


class SugarcaneBioethanolInputs(BaseModel):
    """Input landing page whose sections mirror the Cassava architecture."""

    global_assumptions: GlobalAssumptions = Field(default_factory=GlobalAssumptions)
    other_assumptions: OtherAssumptions = Field(default_factory=OtherAssumptions)
    construction: ConstructionAssumptions = Field(default_factory=ConstructionAssumptions)
    capex: CapexAssumptions = Field(default_factory=CapexAssumptions)
    cycle_planning: CyclePlanningAssumptions = Field(default_factory=CyclePlanningAssumptions)
    farm_planning: FarmPlanningAssumptions = Field(default_factory=FarmPlanningAssumptions)
    farming: FarmingAssumptions = Field(default_factory=FarmingAssumptions)
    labour: LabourAssumptions = Field(default_factory=LabourAssumptions)
    sourcing: SourcingAssumptions = Field(default_factory=SourcingAssumptions)
    processing_routing: ProcessingRoutingAssumptions = Field(default_factory=ProcessingRoutingAssumptions)
    commercialization: CommercializationAssumptions = Field(default_factory=CommercializationAssumptions)
    costs: CostAssumptions = Field(default_factory=CostAssumptions)
    working_capital: WorkingCapitalAssumptions = Field(default_factory=WorkingCapitalAssumptions)
    financing: FinancingAssumptions = Field(default_factory=FinancingAssumptions)
    liquidity: LiquidityAssumptions = Field(default_factory=LiquidityAssumptions)
    yearly_schedules: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    def grouped_sections(self) -> "OrderedDict[str, BaseModel]":
        return OrderedDict(
            [
                ("Global assumptions", self.global_assumptions),
                ("Other assumptions", self.other_assumptions),
                ("Construction & COD", self.construction),
                ("Capex", self.capex),
                ("Cycle planning", self.cycle_planning),
                ("Farm planning", self.farm_planning),
                ("Farming", self.farming),
                ("Labour planning", self.labour),
                ("Sourcing", self.sourcing),
                ("Processing & production routing", self.processing_routing),
                ("Commercialization", self.commercialization),
                ("Costs", self.costs),
                ("Working capital", self.working_capital),
                ("Financing", self.financing),
                ("Liquidity & reserves", self.liquidity),
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
