"""Refactored Sugar Cane plugin backed by the Cassava-style model package."""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, PrivateAttr

from app.plugin.contract import (
    Format,
    ModelPlugin,
    ModelResults,
    NotSupportedError,
    ReportOptions,
    Scenario,
    SubscriptionTier,
    User,
)

from sugarcane_model import (
    SCENARIOS,
    SugarcaneBioethanolInputs,
    SugarcaneBioethanolModel,
    default_input_page,
    input_from_payload,
    input_values,
)
from sugarcane_model.exporter import build_excel_report, build_pdf_report
from sugarcane_model.schedules import COMPONENTS


def _money(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"USD {float(value):,.0f}"


def _percent(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.1%}"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    if isinstance(output.index, pd.DatetimeIndex):
        output = output.reset_index(names="Date")
    elif output.index.name is not None:
        output = output.reset_index()
    return output.where(pd.notna(output), None).to_dict(orient="records")


class SugarcaneBioethanolResults(ModelResults):
    metrics: dict[str, Any]
    scenario_comparison: list[dict[str, Any]]
    production_routing: list[dict[str, Any]]
    component_financials: list[dict[str, Any]]
    consolidated_pnl: list[dict[str, Any]]
    consolidated_cashflow: list[dict[str, Any]]
    consolidated_balance_sheet: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    _tables: dict[str, pd.DataFrame] = PrivateAttr(default_factory=dict)


def _table_bundle(build: dict[str, Any], comparison: pd.DataFrame) -> dict[str, pd.DataFrame]:
    financials = build["financials"]
    capex = build["capex"]
    return {
        "Scenario Comparison": comparison,
        "Cycle Planning Monthly": build["cycle_plan"].monthly,
        "Cycle Planning Annual": build["cycle_plan"].annual,
        "Farming Monthly": build["farming"].monthly,
        "Farming Annual": build["farming"].annual,
        "Sourcing Monthly": build["sourcing"].monthly,
        "Sourcing Annual": build["sourcing"].annual,
        "Processing Routing Monthly": build["processing"].monthly,
        "Processing Routing Annual": build["processing"].annual,
        "Commercialization Monthly": build["commercialization"].monthly,
        "Commercialization Annual": build["commercialization"].annual,
        "Costs Monthly": build["costs"].monthly,
        "Costs Annual": build["costs"].annual,
        "Working Capital Monthly": build["working_capital"].monthly,
        "Working Capital Annual": build["working_capital"].annual,
        "CAPEX Items": capex.item_schedule,
        "CAPEX Monthly": capex.monthly,
        "CAPEX Annual": capex.annual,
        "Debt Monthly": build["debt"].monthly,
        "Debt Annual": build["debt"].annual,
        "Component Financials": financials.component_annual,
        "Component Reconciliation": financials.reconciliation,
        "Consolidated P&L": financials.income_annual,
        "Consolidated Cash Flow": financials.cashflow_annual,
        "Consolidated Balance Sheet": financials.balance_annual,
        "Checks": financials.checks,
    }


def compute_model(inputs: SugarcaneBioethanolInputs) -> SugarcaneBioethanolResults:
    scenario = inputs.global_assumptions.scenario
    model = SugarcaneBioethanolModel(input_page=inputs, scenario=scenario)
    build = model.build(scenario)
    comparison = model.scenario_comparison()
    tables = _table_bundle(build, comparison)
    financials = build["financials"]
    results = SugarcaneBioethanolResults(
        metrics=financials.metrics,
        scenario_comparison=_records(comparison),
        production_routing=_records(build["processing"].annual),
        component_financials=_records(financials.component_annual),
        consolidated_pnl=_records(financials.income_annual),
        consolidated_cashflow=_records(financials.cashflow_annual),
        consolidated_balance_sheet=_records(financials.balance_annual),
        checks=_records(financials.checks),
    )
    results._tables = tables
    return results


_NUMBER_SPECS: dict[str, list[tuple[str, str, float, str, Callable[[Any], Any]]]] = {
    "global_assumptions": [
        ("start_year", "Start year", 1, "%d", int),
        ("end_year", "End year", 1, "%d", int),
        ("hybrid_farm_share", "Hybrid farm share", 0.05, "%.2f", float),
        ("corporate_tax_rate", "Corporate tax rate", 0.01, "%.2f", float),
        ("discount_rate", "Discount rate", 0.01, "%.2f", float),
        ("terminal_growth_rate", "Terminal growth rate", 0.01, "%.2f", float),
        ("investor_share", "Investor share", 0.05, "%.2f", float),
    ],
    "other_assumptions": [
        ("plant_availability", "Plant availability", 0.01, "%.2f", float),
        ("process_loss", "Process loss", 0.01, "%.2f", float),
        ("startup_ramp_year_1", "Year 1 ramp", 0.05, "%.2f", float),
        ("startup_ramp_year_2", "Year 2 ramp", 0.05, "%.2f", float),
        ("price_escalation_rate", "Price escalation", 0.01, "%.2f", float),
        ("cost_inflation_rate", "Cost inflation", 0.01, "%.2f", float),
        ("minimum_dscr_target", "Minimum DSCR target", 0.05, "%.2f", float),
    ],
    "cycle_planning": [
        ("crop_cycle_months", "Crop cycle (months)", 1, "%d", int),
        ("establishment_months", "Establishment (months)", 1, "%d", int),
        ("harvest_window_months", "Harvest window (months)", 1, "%d", int),
        ("ratoon_cycles", "Ratoon cycles", 1, "%d", int),
        ("replant_share_per_cycle", "Replant share per cycle", 0.05, "%.2f", float),
    ],
    "farming": [
        ("sugarcane_yield_tonnes_per_hectare", "Cane yield (t/ha)", 1.0, "%.1f", float),
        ("harvest_recovery", "Harvest recovery", 0.01, "%.2f", float),
        ("farm_opex_per_tonne", "Farm OPEX (USD/t)", 1.0, "%.2f", float),
        ("farm_overhead_per_year", "Farm overhead (USD/year)", 25_000.0, "%.0f", float),
        ("internal_transfer_price_per_tonne", "Internal cane transfer price (USD/t)", 1.0, "%.2f", float),
    ],
    "sourcing": [
        ("cane_purchase_price_per_tonne", "Purchase price (USD/t)", 1.0, "%.2f", float),
        ("contracted_purchase_share", "Contracted purchase share", 0.05, "%.2f", float),
        ("contract_discount", "Contract discount", 0.01, "%.2f", float),
        ("logistics_cost_per_tonne", "Inbound logistics (USD/t)", 0.5, "%.2f", float),
        ("supplier_loss_rate", "Supplier loss rate", 0.01, "%.2f", float),
    ],
    "processing_routing": [
        ("annual_cane_capacity_tonnes", "Annual cane capacity (t)", 5_000.0, "%.0f", float),
        ("ethanol_litres_per_tonne", "Bioethanol yield (L/t cane)", 5.0, "%.1f", float),
        ("sugar_tonnes_per_tonne", "Sugar yield (t/t cane)", 0.01, "%.3f", float),
        ("raw_bagasse_tonnes_per_tonne", "Raw bagasse yield (t/t cane)", 0.01, "%.3f", float),
        ("bagasse_to_electricity_share", "Bagasse to electricity", 0.05, "%.2f", float),
        ("bagasse_to_animal_feed_share", "Bagasse to animal feed", 0.05, "%.2f", float),
        ("bagasse_to_sale_share", "Bagasse sold", 0.05, "%.2f", float),
        ("electricity_mwh_per_tonne_bagasse", "Electricity yield (MWh/t bagasse)", 0.05, "%.2f", float),
        ("internal_electricity_mwh_per_tonne_cane", "Internal power use (MWh/t cane)", 0.01, "%.3f", float),
        ("animal_feed_conversion_rate", "Animal feed conversion", 0.05, "%.2f", float),
    ],
    "commercialization": [
        ("ethanol_price_per_litre", "Bioethanol price (USD/L)", 0.05, "%.2f", float),
        ("sugar_price_per_tonne", "Sugar price (USD/t)", 10.0, "%.2f", float),
        ("electricity_tariff_per_mwh", "Electricity tariff (USD/MWh)", 5.0, "%.2f", float),
        ("bagasse_price_per_tonne", "Bagasse price (USD/t)", 2.0, "%.2f", float),
        ("animal_feed_price_per_tonne", "Animal feed price (USD/t)", 5.0, "%.2f", float),
        ("ethanol_sales_capture", "Bioethanol sales capture", 0.05, "%.2f", float),
        ("sugar_sales_capture", "Sugar sales capture", 0.05, "%.2f", float),
        ("power_sales_capture", "Power sales capture", 0.05, "%.2f", float),
        ("coproduct_sales_capture", "Coproduct sales capture", 0.05, "%.2f", float),
    ],
    "costs": [
        ("processing_variable_cost_per_tonne_cane", "Processing variable cost (USD/t cane)", 1.0, "%.2f", float),
        ("ethanol_variable_cost_per_litre", "Bioethanol variable cost (USD/L)", 0.01, "%.3f", float),
        ("sugar_variable_cost_per_tonne", "Sugar variable cost (USD/t)", 2.0, "%.2f", float),
        ("electricity_variable_cost_per_mwh", "Electricity variable cost (USD/MWh)", 1.0, "%.2f", float),
        ("bagasse_handling_cost_per_tonne", "Bagasse handling cost (USD/t)", 0.5, "%.2f", float),
        ("animal_feed_variable_cost_per_tonne", "Animal feed variable cost (USD/t)", 2.0, "%.2f", float),
        ("fixed_processing_opex_per_year", "Fixed processing OPEX (USD/year)", 100_000.0, "%.0f", float),
        ("commercial_and_admin_cost_per_year", "Commercial & admin (USD/year)", 50_000.0, "%.0f", float),
    ],
    "working_capital": [
        ("receivable_days", "Receivable days", 1.0, "%.1f", float),
        ("inventory_days", "Inventory days", 1.0, "%.1f", float),
        ("payable_days", "Payable days", 1.0, "%.1f", float),
    ],
    "financing": [
        ("debt_ratio", "Debt ratio", 0.05, "%.2f", float),
        ("interest_rate", "Interest rate", 0.01, "%.2f", float),
        ("tenor_years", "Debt tenor (years)", 1, "%d", int),
        ("grace_years", "Principal grace (years)", 1, "%d", int),
    ],
}


def _number_grid(st, section: str, values: dict[str, Any]) -> dict[str, Any]:
    updated = dict(values)
    columns = st.columns(2)
    for index, (key, label, step, fmt, caster) in enumerate(_NUMBER_SPECS[section]):
        value = values[key]
        updated[key] = caster(
            columns[index % 2].number_input(
                label,
                value=caster(value),
                step=caster(step),
                format=fmt,
                key=f"sugarcane_{section}_{key}",
            )
        )
    return updated


class SugarcaneBioethanolPlugin:
    slug = "sugar-cane-bioethanol"
    name = "Sugar Cane Bioethanol Financial Model"
    version = "2.0.0"
    description = (
        "Integrated farm-to-market Sugar Cane model with Cassava-style modular "
        "assumptions, operating schedules, component economics, and consolidated statements."
    )
    icon = "🌾"
    category = "Bioenergy project finance"
    badge_gradient = "linear-gradient(135deg, #4d7c0f, #ca8a04)"
    features = [
        "FARM_ONLY, BUY_ONLY, and HYBRID sourcing",
        "Five-product processing and production routing",
        "Six component financial views",
        "Consolidated three-statement model",
    ]
    minimum_tier = SubscriptionTier.PRO
    bundle = None
    supported_formats = {Format.XLSX, Format.PDF}
    input_schema: type[BaseModel] = SugarcaneBioethanolInputs
    results_schema: type[ModelResults] = SugarcaneBioethanolResults

    def default_inputs(self) -> SugarcaneBioethanolInputs:
        return default_input_page()

    def compute(self, inputs: BaseModel) -> SugarcaneBioethanolResults:
        assert isinstance(inputs, SugarcaneBioethanolInputs)
        return compute_model(inputs)

    def render(self, *, user: User, scenario: Scenario | None = None) -> None:
        import streamlit as st

        defaults = self.default_inputs()
        saved_inputs = st.session_state.get("_sugarcane_refactored_inputs")
        if isinstance(saved_inputs, SugarcaneBioethanolInputs):
            defaults = saved_inputs
        if scenario is not None:
            try:
                defaults = input_from_payload(scenario.inputs_json)
            except Exception:
                pass
        payload = input_values(defaults)

        st.subheader(self.name)
        st.caption(
            "Cassava-style architecture: grouped assumptions → operating schedules → "
            "component financials → consolidated statements."
        )
        with st.form("sugarcane_refactored_inputs"):
            with st.expander("Global assumptions", expanded=True):
                global_values = _number_grid(st, "global_assumptions", payload["global_assumptions"])
                global_values["scenario"] = st.selectbox(
                    "Feedstock scenario",
                    SCENARIOS,
                    index=SCENARIOS.index(payload["global_assumptions"]["scenario"]),
                )
                global_values["planning_start"] = st.text_input(
                    "Planning start (YYYY-MM)", payload["global_assumptions"]["planning_start"]
                )
            with st.expander("Other assumptions"):
                other_values = _number_grid(st, "other_assumptions", payload["other_assumptions"])
            with st.expander("Capex"):
                capex_editor = st.data_editor(
                    pd.DataFrame(payload["capex"]["items"]),
                    use_container_width=True,
                    num_rows="dynamic",
                    key="sugarcane_capex_editor",
                )
            with st.expander("Cycle planning"):
                cycle_values = _number_grid(st, "cycle_planning", payload["cycle_planning"])
            with st.expander("Farming"):
                farming_values = _number_grid(st, "farming", payload["farming"])
            with st.expander("Sourcing"):
                sourcing_values = _number_grid(st, "sourcing", payload["sourcing"])
            with st.expander("Processing & production routing"):
                routing_values = _number_grid(st, "processing_routing", payload["processing_routing"])
                st.caption("Bagasse routing shares must total 1.00.")
            with st.expander("Commercialization"):
                commercial_values = _number_grid(st, "commercialization", payload["commercialization"])
            with st.expander("Costs"):
                cost_values = _number_grid(st, "costs", payload["costs"])
            with st.expander("Working capital"):
                wc_values = _number_grid(st, "working_capital", payload["working_capital"])
            with st.expander("Financing"):
                financing_values = _number_grid(st, "financing", payload["financing"])
                financing_values["amortization_type"] = st.selectbox(
                    "Amortization",
                    ("straight", "annuity"),
                    index=("straight", "annuity").index(payload["financing"]["amortization_type"]),
                )
                financing_values["capitalize_idc"] = st.checkbox(
                    "Capitalize construction interest",
                    value=bool(payload["financing"]["capitalize_idc"]),
                )
            submitted = st.form_submit_button(
                "Run Model", type="primary", use_container_width=True
            )

        result_key = "_sugarcane_refactored_result"
        inputs_key = "_sugarcane_refactored_inputs"
        if submitted:
            payload.update(
                {
                    "global_assumptions": global_values,
                    "other_assumptions": other_values,
                    "capex": {
                        "items": capex_editor.where(pd.notna(capex_editor), None).to_dict(orient="records")
                    },
                    "cycle_planning": cycle_values,
                    "farming": farming_values,
                    "sourcing": sourcing_values,
                    "processing_routing": routing_values,
                    "commercialization": commercial_values,
                    "costs": cost_values,
                    "working_capital": wc_values,
                    "financing": financing_values,
                }
            )
            try:
                model_inputs = input_from_payload(payload)
                st.session_state[result_key] = self.compute(model_inputs)
                st.session_state[inputs_key] = model_inputs
            except Exception as exc:
                st.error(f"Model inputs could not be calculated: {exc}")

        result = st.session_state.get(result_key)
        model_inputs = st.session_state.get(inputs_key)
        if result is None or model_inputs is None:
            st.info("Review the grouped assumptions and select **Run Model** to calculate results.")
            return

        metrics = result.metrics
        columns = st.columns(5)
        columns[0].metric("Project NPV", _money(metrics.get("project_npv")))
        columns[1].metric("Project IRR", _percent(metrics.get("project_irr")))
        columns[2].metric("Equity IRR", _percent(metrics.get("equity_irr")))
        columns[3].metric(
            "Minimum DSCR",
            "n/a" if metrics.get("min_dscr") is None else f"{metrics['min_dscr']:.2f}x",
        )
        columns[4].metric("Model status", str(metrics.get("model_status", "CHECK")))

        overview, operations, components, consolidated, debt_tab, checks_tab = st.tabs(
            [
                "Overview",
                "Operations & routing",
                "Component financials",
                "Consolidated statements",
                "Debt & coverage",
                "Model checks",
            ]
        )
        with overview:
            comparison = pd.DataFrame(result.scenario_comparison)
            st.subheader("Scenario comparison")
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.subheader("Annual revenue and EBITDA")
            pnl = pd.DataFrame(result.consolidated_pnl).set_index("Year")
            st.line_chart(pnl[["Revenue", "EBITDA"]])
        with operations:
            st.subheader("Processing and production routing")
            st.dataframe(
                pd.DataFrame(result.production_routing),
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Annual farming and sourcing")
            st.dataframe(result._tables["Farming Annual"].reset_index(), use_container_width=True, hide_index=True)
            st.dataframe(result._tables["Sourcing Annual"].reset_index(), use_container_width=True, hide_index=True)
        with components:
            selected = st.selectbox("Component", COMPONENTS)
            component_frame = pd.DataFrame(result.component_financials)
            selected_frame = component_frame.loc[component_frame["Component"] == selected]
            st.dataframe(selected_frame, use_container_width=True, hide_index=True)
            st.line_chart(selected_frame.set_index("Year")[["Revenue", "EBITDA", "NetIncome"]])
            with st.expander("All component financials and consolidation reconciliation"):
                st.dataframe(component_frame, use_container_width=True, hide_index=True)
                st.dataframe(
                    result._tables["Component Reconciliation"],
                    use_container_width=True,
                    hide_index=True,
                )
        with consolidated:
            st.subheader("Consolidated profit and loss")
            st.dataframe(pd.DataFrame(result.consolidated_pnl), use_container_width=True, hide_index=True)
            st.subheader("Consolidated cash flow")
            st.dataframe(pd.DataFrame(result.consolidated_cashflow), use_container_width=True, hide_index=True)
            st.subheader("Consolidated balance sheet")
            st.dataframe(
                pd.DataFrame(result.consolidated_balance_sheet),
                use_container_width=True,
                hide_index=True,
            )
        with debt_tab:
            debt = result._tables["Debt Annual"].reset_index()
            st.line_chart(debt.set_index("Year")[["ClosingBalance", "DebtService"]])
            st.dataframe(debt, use_container_width=True, hide_index=True)
        with checks_tab:
            st.dataframe(pd.DataFrame(result.checks), use_container_width=True, hide_index=True)

        from app.reports.ui import render_report_downloads

        render_report_downloads(self, model_inputs, result, user)

    def generate_report(
        self,
        inputs: BaseModel,
        results: ModelResults,
        formats: set[Format],
        options: ReportOptions,
        user: User,
    ) -> dict[Format, bytes]:
        unsupported = formats - self.supported_formats
        if unsupported:
            raise NotSupportedError(
                f"{self.slug} does not support: {sorted(fmt.value for fmt in unsupported)}"
            )
        assert isinstance(inputs, SugarcaneBioethanolInputs)
        assert isinstance(results, SugarcaneBioethanolResults)
        if not results._tables:
            results = self.compute(inputs)
        output: dict[Format, bytes] = {}
        if Format.XLSX in formats:
            output[Format.XLSX] = build_excel_report(inputs, results.metrics, results._tables)
        if Format.PDF in formats:
            output[Format.PDF] = build_pdf_report(
                results.metrics,
                results._tables,
                user.email,
                options.watermark,
            )
        return output


MODEL: ModelPlugin = SugarcaneBioethanolPlugin()
