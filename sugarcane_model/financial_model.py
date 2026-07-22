"""Cassava-style orchestration layer for the Sugar Cane Bioethanol model."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from .driver_schedules import validate_driver_schedules
from .inputs import (
    SCENARIOS,
    SugarcaneBioethanolInputs,
    default_input_page,
    input_values,
    validate_farm_planning_values,
)
from .schedules import (
    CapexOutput,
    DebtOutput,
    FinancialOutput,
    ScheduleOutput,
    compute_capex_schedule,
    compute_commercialization,
    compute_cost_schedule,
    compute_cycle_plan,
    compute_debt_schedule,
    compute_farm_planning_schedule,
    compute_farming_schedule,
    compute_financials,
    compute_processing_routing,
    compute_sourcing_schedule,
    compute_working_capital,
)


@dataclass
class SugarcaneBioethanolModel:
    """Build scenario-specific schedules from one grouped input landing page."""

    input_page: SugarcaneBioethanolInputs = field(default_factory=default_input_page)
    scenario: str = "HYBRID"
    _scenario_cache: dict[str, tuple[str, dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )

    SCENARIOS = SCENARIOS

    @classmethod
    def default(cls) -> "SugarcaneBioethanolModel":
        return cls()

    def _validate(self, scenario: str) -> None:
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unsupported scenario '{scenario}'. Expected one of {self.SCENARIOS}."
            )
        inputs = self.input_page
        global_inputs = inputs.global_assumptions
        if global_inputs.end_year < global_inputs.start_year + 2:
            raise ValueError("The projection must cover at least three calendar years.")
        planning_start = pd.Period(global_inputs.planning_start, freq="M")
        projection_start = pd.Period(f"{global_inputs.start_year}-01", freq="M")
        projection_end = pd.Period(f"{global_inputs.end_year}-12", freq="M")
        if planning_start < projection_start or planning_start > projection_end:
            raise ValueError("Planning start must fall within the projection horizon.")
        if global_inputs.terminal_growth_rate >= global_inputs.discount_rate:
            raise ValueError("Terminal growth must be lower than the discount rate.")
        financing = inputs.financing
        if financing.tenor_years <= financing.grace_years:
            raise ValueError("Debt tenor must be longer than the principal grace period.")
        facility_names = {"senior debt"}
        for facility in financing.additional_debt_facilities:
            name = facility.name.strip()
            if not name:
                raise ValueError("Each additional debt facility must have a name.")
            normalized_name = name.casefold()
            if normalized_name in facility_names:
                raise ValueError(
                    f"Debt facility names must be unique; duplicate '{name}'."
                )
            facility_names.add(normalized_name)
            if facility.tenor_years <= facility.grace_years:
                raise ValueError(
                    f"Debt tenor must exceed grace period for facility '{name}'."
                )
        cycle = inputs.cycle_planning
        if cycle.establishment_months + cycle.harvest_window_months > cycle.crop_cycle_months:
            raise ValueError(
                "Establishment and harvest windows cannot exceed the crop cycle."
            )
        farm_plan_errors = validate_farm_planning_values(input_values(inputs.farm_planning))
        if farm_plan_errors:
            raise ValueError(" ".join(farm_plan_errors))

        routing = inputs.processing_routing
        bagasse_share = (
            routing.bagasse_to_electricity_share
            + routing.bagasse_to_animal_feed_share
            + routing.bagasse_to_sale_share
        )
        if abs(bagasse_share - 1.0) > 1e-9:
            raise ValueError("Bagasse routing shares must sum to 100%.")
        if not inputs.capex.items:
            raise ValueError("At least one CAPEX item is required.")
        schedule_errors = validate_driver_schedules(inputs)
        if schedule_errors:
            raise ValueError(" ".join(schedule_errors))


    def clear_cache(self) -> None:
        self._scenario_cache.clear()

    def build(self, scenario: str | None = None) -> dict[str, Any]:
        scenario_name = (scenario or self.scenario or "HYBRID").upper()
        self._validate(scenario_name)
        self.scenario = scenario_name
        signature = self.input_page.signature()
        cached = self._scenario_cache.get(scenario_name)
        if cached and cached[0] == signature:
            return copy.deepcopy(cached[1])

        cycle_plan: ScheduleOutput = compute_cycle_plan(self.input_page)
        farm_plan: ScheduleOutput = compute_farm_planning_schedule(
            self.input_page, cycle_plan
        )
        farming: ScheduleOutput = compute_farming_schedule(
            self.input_page, scenario_name, cycle_plan, farm_plan
        )
        sourcing: ScheduleOutput = compute_sourcing_schedule(
            self.input_page, scenario_name, farming, cycle_plan
        )
        processing: ScheduleOutput = compute_processing_routing(
            self.input_page, sourcing
        )
        commercialization: ScheduleOutput = compute_commercialization(
            self.input_page, processing
        )
        costs: ScheduleOutput = compute_cost_schedule(
            self.input_page, farming, sourcing, processing
        )
        capex: CapexOutput = compute_capex_schedule(
            self.input_page, scenario_name, sourcing
        )
        debt: DebtOutput = compute_debt_schedule(self.input_page, capex)
        working_capital: ScheduleOutput = compute_working_capital(
            self.input_page, commercialization, costs
        )
        financials: FinancialOutput = compute_financials(
            self.input_page,
            scenario_name,
            farming,
            farm_plan,
            sourcing,
            processing,
            commercialization,
            costs,
            capex,
            debt,
            working_capital,
        )

        results: dict[str, Any] = {
            "scenario": scenario_name,
            "input_page_snapshot": copy.deepcopy(self.input_page),
            "cycle_plan": cycle_plan,
            "farm_plan": farm_plan,
            "farming": farming,
            "sourcing": sourcing,
            "processing": processing,
            "commercialization": commercialization,
            "costs": costs,
            "capex": capex,
            "debt": debt,
            "working_capital": working_capital,
            "financials": financials,
            "metrics": financials.metrics,
        }
        self._scenario_cache[scenario_name] = (signature, copy.deepcopy(results))
        return results

    def build_all_scenarios(
        self, scenarios: Iterable[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        return {
            name: self.build(name)
            for name in (scenarios or self.SCENARIOS)
        }

    def scenario_comparison(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for scenario, result in self.build_all_scenarios().items():
            metrics = result["metrics"]
            rows.append(
                {
                    "Scenario": scenario,
                    "FarmShareTarget": metrics.get("farm_share_target"),
                    "ActualFarmShare": metrics.get("farm_share"),
                    "TotalCapex": metrics.get("total_capex"),
                    "ProjectNPV": metrics.get("project_npv"),
                    "ProjectIRR": metrics.get("project_irr"),
                    "EquityIRR": metrics.get("equity_irr"),
                    "MinimumDSCR": metrics.get("min_dscr"),
                    "TotalDebt": metrics.get("total_debt_draw"),
                    "EquityFunding": metrics.get("total_equity_contribution"),
                    "PaybackYears": metrics.get("payback_years"),
                    "ModelStatus": metrics.get("model_status"),
                }
            )
        return pd.DataFrame(rows)
