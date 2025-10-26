"""Streamlit front-end for the Sugar Cane Bioethanol multi-product finance model.

This app wraps the `model.py` engine and exposes interactive controls for
adjusting key drivers and reviewing the resulting financial outputs, dashboards,
sensitivities, and scenarios.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import math
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException


def _streamlit_runtime_exists() -> bool:
    """Return True when executed inside an active Streamlit runtime."""
    try:  # Streamlit >= 1.35 exposes runtime.exists()
        from streamlit.runtime import exists as runtime_exists  # type: ignore import

        return bool(runtime_exists())
    except Exception:  # pragma: no cover - fallback for older versions
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

            return get_script_run_ctx() is not None
        except Exception:
            return False


MODEL_IMPORT_ERROR: ModuleNotFoundError | None = None
try:  # noqa: SIM105 - streamlit feedback when dependencies missing
    from model import (
        DEFAULTS,
        FEEDSTOCK_SCENARIOS,
        INPUT_SCHEMAS,
        PRODUCTS,
        InputTables,
        build_config,
        monte_carlo,
        run_full_model,
        run_scenarios,
        sensitivity_tornado,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - executed only when deps missing
    MODEL_IMPORT_ERROR = exc


def _format_metric(value: object, kind: str = "number") -> str:
    """Format metric values for display in metric cards."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "N/A"
        if kind == "currency":
            return f"${value:,.2f}"
        if kind == "percent":
            return f"{value:.2%}"
        if kind == "ratio":
            return f"{value:,.2f}"
        if kind == "year":
            return str(int(round(value)))
        return f"{value:,.2f}"
    return str(value)


def _render_dataframe(df: pd.DataFrame, title: str, key: str) -> None:
    """Render a dataframe with a download button."""
    st.subheader(title)
    st.dataframe(df, use_container_width=True)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {title}",
        csv_data,
        file_name=f"{title.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=key,
    )


def _seed_tables_with_defaults(tables: InputTables) -> None:
    """Populate session tables with sensible defaults for every schema."""
    horizon_defaults = dict(DEFAULTS["horizon"])
    tables.set_table("projection_horizon", pd.DataFrame([horizon_defaults]))
    tables.set_table("production_horizon", pd.DataFrame([DEFAULTS["production_horizon"]]))
    tables.set_table("global_inputs", pd.DataFrame([DEFAULTS["global"]]))
    tables.set_table("working_capital_days", pd.DataFrame([DEFAULTS["working_capital"]]))

    start_year = int(horizon_defaults["start_year"])
    start_month = int(horizon_defaults.get("start_month", 1))

    capex_rows = [
        {
            "item_name": "Land Acquisition",
            "category": "land",
            "amount": 5_000_000.0,
            "currency": "USD",
            "fx_curve": "",
            "start_date": f"{start_year}-01",
            "end_date": f"{start_year}-12",
            "life_years": 40,
            "depr_method": "straight",
            "depr_rate_override": np.nan,
            "vat_rate": 0.0,
            "vat_recovery_lag_months": 0,
            "capitalized": True,
            "is_farm_capex": True,
        },
        {
            "item_name": "Civil Works",
            "category": "civil",
            "amount": 8_500_000.0,
            "currency": "USD",
            "fx_curve": "",
            "start_date": f"{start_year}-01",
            "end_date": f"{start_year}-12",
            "life_years": 20,
            "depr_method": "straight",
            "depr_rate_override": np.nan,
            "vat_rate": 0.0,
            "vat_recovery_lag_months": 0,
            "capitalized": True,
            "is_farm_capex": False,
        },
        {
            "item_name": "Process Equipment",
            "category": "equipment",
            "amount": 12_500_000.0,
            "currency": "USD",
            "fx_curve": "",
            "start_date": f"{start_year}-01",
            "end_date": f"{start_year}-12",
            "life_years": 12,
            "depr_method": "straight",
            "depr_rate_override": np.nan,
            "vat_rate": 0.0,
            "vat_recovery_lag_months": 0,
            "capitalized": True,
            "is_farm_capex": False,
        },
        {
            "item_name": "Farm Machinery",
            "category": "farm_machinery",
            "amount": 4_000_000.0,
            "currency": "USD",
            "fx_curve": "",
            "start_date": f"{start_year}-01",
            "end_date": f"{start_year}-12",
            "life_years": 10,
            "depr_method": "straight",
            "depr_rate_override": np.nan,
            "vat_rate": 0.0,
            "vat_recovery_lag_months": 0,
            "capitalized": True,
            "is_farm_capex": True,
        },
    ]
    tables.set_table("capex_lines", pd.DataFrame(capex_rows))

    price_rows = []
    for product, params in DEFAULTS["prices"].items():
        price_rows.append(
            {
                "product": product,
                "base_price": params.get("base_price", 0.0),
                "price_escalation_pa": params.get("price_escalation_pa", 0.0),
                "price_indexation": params.get("price_indexation", "cpi"),
                "uom": params.get("uom", ""),
                "tariff_structure": "",
                "revenue_share": 1.0,
            }
        )
    tables.set_table("revenue_params", pd.DataFrame(price_rows))

    prod_defaults = DEFAULTS["production"]
    feedstock = prod_defaults["annual_feedstock_ton"]
    availability = prod_defaults["plant_availability"]
    loss = prod_defaults["loss_factor"]
    ramp = "0.7;0.9;1.0"

    production_rows = [
        {
            "product": "ethanol",
            "annual_volume": feedstock * prod_defaults["ethanol_litre_per_ton"] * availability * (1 - loss),
            "availability": availability,
            "loss_factor": loss,
            "startup_ramp": ramp,
            "boe_conversion": np.nan,
            "sugarcane_yield_ton_per_ha": prod_defaults["sugarcane_yield_ton_per_ha"],
            "farm_area_ha": feedstock / prod_defaults["sugarcane_yield_ton_per_ha"],
        },
        {
            "product": "sugar",
            "annual_volume": feedstock * prod_defaults["sugar_ton_per_ton_cane"] * availability * (1 - loss),
            "availability": availability,
            "loss_factor": loss,
            "startup_ramp": ramp,
            "boe_conversion": np.nan,
            "sugarcane_yield_ton_per_ha": prod_defaults["sugarcane_yield_ton_per_ha"],
            "farm_area_ha": feedstock / prod_defaults["sugarcane_yield_ton_per_ha"],
        },
        {
            "product": "electricity",
            "annual_volume": feedstock * prod_defaults["electricity_mwh_per_ton_cane"] * availability * (1 - loss),
            "availability": availability,
            "loss_factor": loss,
            "startup_ramp": ramp,
            "boe_conversion": np.nan,
            "sugarcane_yield_ton_per_ha": prod_defaults["sugarcane_yield_ton_per_ha"],
            "farm_area_ha": feedstock / prod_defaults["sugarcane_yield_ton_per_ha"],
        },
        {
            "product": "animal_feed",
            "annual_volume": feedstock * prod_defaults["animal_feed_ton_per_ton_cane"] * availability * (1 - loss),
            "availability": availability,
            "loss_factor": loss,
            "startup_ramp": ramp,
            "boe_conversion": np.nan,
            "sugarcane_yield_ton_per_ha": prod_defaults["sugarcane_yield_ton_per_ha"],
            "farm_area_ha": feedstock / prod_defaults["sugarcane_yield_ton_per_ha"],
        },
    ]
    tables.set_table("production_annual", pd.DataFrame(production_rows))

    first_year_months = pd.date_range(f"{start_year}-{start_month:02d}-01", periods=12, freq="MS")
    monthly_rows = []
    for row in production_rows:
        monthly_volume = float(row["annual_volume"]) / 12.0
        for date in first_year_months:
            monthly_rows.append(
                {
                    "date": date.strftime("%Y-%m"),
                    "product": row["product"],
                    "volume": monthly_volume,
                    "availability_override": np.nan,
                    "maintenance_downtime": np.nan,
                    "loss_override": np.nan,
                }
            )
    tables.set_table("production_monthly", pd.DataFrame(monthly_rows))

    direct_rows = [
        {
            "date": f"{start_year}-01",
            "cost_type": "feedstock purchase",
            "product_link": "ethanol",
            "amount": 500_000.0,
            "currency": "USD",
        }
    ]
    tables.set_table("direct_costs_monthly", pd.DataFrame(direct_rows))

    staff_rows = [
        {
            "date": f"{start_year}-01",
            "dept": "Operations",
            "headcount": 50,
            "gross_pay": 150_000.0,
            "benefits": 25_000.0,
            "training": 5_000.0,
            "other": 10_000.0,
            "currency": "USD",
        }
    ]
    tables.set_table("staff_costs_monthly", pd.DataFrame(staff_rows))

    other_rows = [
        {"date": f"{start_year}-01", "category": "insurance", "amount": 20_000.0, "currency": "USD"},
        {"date": f"{start_year}-01", "category": "service_contract", "amount": 35_000.0, "currency": "USD"},
        {"date": f"{start_year}-01", "category": "general_admin", "amount": 50_000.0, "currency": "USD"},
        {"date": f"{start_year}-01", "category": "energy_cost", "amount": 15_000.0, "currency": "USD"},
    ]
    tables.set_table("other_opex_monthly", pd.DataFrame(other_rows))

    ar_rows = [
        {
            "date": f"{start_year}-01",
            "receivables": 0.0,
            "prepaid_expenses": 0.0,
            "other_current_assets": 0.0,
            "dso_days": DEFAULTS["working_capital"]["dso_days"],
        }
    ]
    tables.set_table("ar_other_assets", pd.DataFrame(ar_rows))

    inventory_rows = [
        {
            "date": f"{start_year}-01",
            "inventory_raw": 0.0,
            "inventory_wip": 0.0,
            "inventory_fg": 0.0,
            "accounts_payable": 0.0,
            "dio_days": DEFAULTS["working_capital"]["dio_days"],
            "dpo_days": DEFAULTS["working_capital"]["dpo_days"],
        }
    ]
    tables.set_table("inventory_ap", pd.DataFrame(inventory_rows))

    debt_rows = []
    for tranche in DEFAULTS["debt"]["tranches"]:
        debt_rows.append(tranche)
    tables.set_table("debt_tranches", pd.DataFrame(debt_rows))

    tax_rows = [DEFAULTS["tax"]]
    tables.set_table("tax_schedule", pd.DataFrame(tax_rows))

    tables.set_table("inflation_index", DEFAULTS["inflation_index"])
    tables.set_table("risk_params", DEFAULTS["risk_params"])


def _get_tables() -> InputTables:
    if "input_tables" not in st.session_state:
        tables = InputTables()
        _seed_tables_with_defaults(tables)
        st.session_state["input_tables"] = tables
    return st.session_state["input_tables"]


def _sync_tables_from_state(tables: InputTables) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    for table_name in INPUT_SCHEMAS.keys():
        state_key = f"editor_{table_name}"
        value = st.session_state.get(state_key)
        if isinstance(value, pd.DataFrame):
            try:
                tables.set_table(table_name, value)
            except Exception as exc:  # pragma: no cover - validation feedback for UI edits
                errors[table_name] = str(exc)
    return errors


LANDING_TABLES: List[Tuple[str, str, Optional[str]]] = [
    ("Projection Horizon", "projection_horizon", "Define the calendar start and end of the modeling period."),
    ("Production Horizon", "production_horizon", "Limit operating volumes to the active production window."),
    ("Global Inputs", "global_inputs", "Corporate tax, discount rate, and ownership split."),
    (
        "Working Capital Assumptions",
        "working_capital_days",
        "DSO, DIO, and DPO assumptions controlling receivables, inventory, and payables timing.",
    ),
    (
        "Initial Investment (CAPEX)",
        "capex_lines",
        "Detailed plant and farm investment lines with depreciation lives and VAT timing.",
    ),
    ("Product Pricing Inputs", "revenue_params", "Product pricing, escalation, and indexation."),
    (
        "Production Volumes (Annual)",
        "production_annual",
        "Annual production volumes, availability, and ramp-up assumptions by product.",
    ),
    (
        "Production Volumes (Monthly)",
        "production_monthly",
        "Monthly production schedule by product within the operating window.",
    ),
    ("Operating Costs - Direct", "direct_costs_monthly", "Feedstock and variable operating expenses."),
    ("Operating Costs - Staff", "staff_costs_monthly", "Headcount and payroll assumptions."),
    (
        "Operating Costs - Other Opex",
        "other_opex_monthly",
        "Insurance, services, energy, and overhead costs.",
    ),
    (
        "Working Capital Balances - Receivables",
        "ar_other_assets",
        "Accounts receivable, prepaid expenses, and other current assets.",
    ),
    (
        "Working Capital Balances - Inventory & Payables",
        "inventory_ap",
        "Inventory positions and supplier payables by month.",
    ),
    ("Debt Schedule", "debt_tranches", "Debt facilities with rates, tenors, amortisation, and IDC."),
    ("Tax Schedule", "tax_schedule", "Tax rate, incentives, timing adjustments, and loss carryforwards."),
    ("Inflation Schedule", "inflation_index", "Inflation and FX indexation curves."),
    ("Risk Schedule", "risk_params", "Drivers for sensitivities, Monte Carlo, and scenario analysis."),
]


def _render_table_editor(
    tables: InputTables,
    table_name: str,
    label: str,
    error_message: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    st.markdown(f"#### {label}")
    if description:
        st.caption(description)
    df = tables.ensure_table(table_name).copy()
    controls = st.columns(2)
    if controls[0].button(f"Add row", key=f"add_{table_name}"):
        try:
            tables.add_row(table_name, {})
        except Exception as exc:  # pragma: no cover - validation feedback
            st.error(f"Unable to add row: {exc}")
        st.experimental_rerun()
    if not df.empty:
        remove_idx = controls[1].selectbox(
            "Row to remove",
            options=list(df.index),
            format_func=lambda idx: f"Row {idx + 1}",
            key=f"remove_select_{table_name}",
        )
        if controls[1].button("Remove selected", key=f"remove_{table_name}"):
            try:
                tables.remove_row(table_name, int(remove_idx))
            except Exception as exc:  # pragma: no cover - defensive feedback
                st.error(f"Unable to remove row: {exc}")
            st.experimental_rerun()
    editor = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{table_name}",
    )
    if error_message:
        st.error(f"Validation error: {error_message}")
    st.divider()


def main() -> None:
    try:
        if _streamlit_runtime_exists():
            st.set_page_config(title="Sugarcane Bioethanol Finance Model", layout="wide")
    except (StreamlitAPIException, RuntimeError, Exception):  # pragma: no cover - defensive guard
        # Some Streamlit versions raise a generic Exception when page config is invoked
        # outside a live runtime; swallow and continue so local execution still works.
        pass

    st.markdown(
        """
        <style>
        html, body, .stApp {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
        }
        [data-testid="stAppViewContainer"] {
            padding: 0 1.5rem !important;
            margin: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding: 0 !important;
            margin: 0 auto !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        [data-testid="stAppViewContainer"] .main .block-container,
        .block-container {
            padding: 0 1.5rem !important;
            margin: 0 auto !important;
            width: 100% !important;
            max-width: calc(100vw - 3rem) !important;
        }
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if MODEL_IMPORT_ERROR is not None:
        st.error(
            "Required dependency missing when importing the finance engine: "
            f"{MODEL_IMPORT_ERROR}. Install the project requirements (numpy, pandas, etc.) "
            "and restart the app."
        )
        st.stop()

    st.title("Sugarcane Bioethanol Project Finance Model")
    st.markdown(
        "Use this Streamlit interface to explore the integrated bioethanol, sugar, "
        "electricity, and animal feed project finance model. Adjust critical drivers "
        "in the control tabs above and review the resulting statements, dashboards, "
        "sensitivities, and scenarios."
    )

    page_tabs_container = st.container()

    st.subheader("Model Controls")

    control_tabs = st.tabs(
        [
            "Projection",
            "Financial",
            "Production",
            "Pricing",
            "Risk & Scenarios",
        ]
    )

    tables = _get_tables()
    sync_errors = _sync_tables_from_state(tables)
    assumptions: Dict[str, object] = {}
    cfg = build_config(assumptions, tables)

    horizon = cfg["projection_horizon"]
    production_horizon = cfg.get("production_horizon", {"start_year": horizon["start_year"], "end_year": horizon["end_year"]})
    with control_tabs[0]:
        st.markdown("### Projection horizon")
        start_year = st.number_input("Start year", value=int(horizon["start_year"]), step=1)
        end_year = st.number_input("End year", value=int(horizon["end_year"]), min_value=int(start_year), step=1)
        start_month = st.number_input("Start month", min_value=1, max_value=12, value=int(horizon.get("start_month", 1)))
        horizon.update({"start_year": int(start_year), "end_year": int(end_year), "start_month": int(start_month)})
        st.markdown("### Production horizon")
        prod_start_year = st.number_input(
            "Production start year",
            value=int(production_horizon.get("start_year", start_year)),
            min_value=int(start_year),
            step=1,
        )
        prod_end_year = st.number_input(
            "Production end year",
            value=int(production_horizon.get("end_year", end_year)),
            min_value=int(prod_start_year),
            max_value=int(end_year),
            step=1,
        )
        production_horizon.update({"start_year": int(prod_start_year), "end_year": int(prod_end_year)})
        cfg["production_horizon"] = production_horizon
        st.caption("Production volumes are set to zero outside the defined production horizon.")
        try:
            tables.set_table("projection_horizon", pd.DataFrame([horizon]))
            tables.set_table("production_horizon", pd.DataFrame([production_horizon]))
        except Exception as exc:
            st.warning(f"Projection inputs not saved due to validation error: {exc}")

    global_inputs = cfg["global_inputs"]
    with control_tabs[1]:
        st.markdown("### Financial assumptions")
        discount_rate = st.number_input(
            "Discount rate (WACC)",
            min_value=0.0,
            max_value=1.0,
            value=float(global_inputs.get("discount_rate", 0.12)),
            step=0.005,
            format="%.4f",
        )
        corp_tax = st.number_input(
            "Corporate tax rate",
            min_value=0.0,
            max_value=1.0,
            value=float(global_inputs.get("corp_tax_rate", 0.28)),
            step=0.01,
            format="%.4f",
        )
        investor_share = st.slider(
            "Investor equity share",
            min_value=0.0,
            max_value=1.0,
            value=float(global_inputs.get("investor_share", 0.6)),
            step=0.05,
        )
        global_inputs.update(
            {
                "discount_rate": float(discount_rate),
                "corp_tax_rate": float(corp_tax),
                "investor_share": float(investor_share),
                "owner_share": float(1.0 - investor_share),
            }
        )
        st.caption(f"Owner equity share automatically set to {1.0 - investor_share:.2f}")
        try:
            tables.set_table("global_inputs", pd.DataFrame([global_inputs]))
            tables.set_table("working_capital_days", pd.DataFrame([cfg["working_capital"]]))
        except Exception as exc:
            st.warning(f"Global inputs not saved due to validation error: {exc}")

    production_cfg = cfg.setdefault("production", {})
    with control_tabs[2]:
        st.markdown("### Production assumptions")
        feedstock = st.number_input(
            "Annual feedstock (t)",
            min_value=0.0,
            value=float(production_cfg.get("annual_feedstock_ton", 100_000.0)),
            step=10_000.0,
            format="%.0f",
        )
        availability = st.slider(
            "Plant availability",
            min_value=0.5,
            max_value=1.0,
            value=float(production_cfg.get("plant_availability", 0.9)),
            step=0.01,
        )
        loss_factor = st.slider(
            "Process loss factor",
            min_value=0.0,
            max_value=0.2,
            value=float(production_cfg.get("loss_factor", 0.02)),
            step=0.005,
        )
        production_cfg.update(
            {
                "annual_feedstock_ton": float(feedstock),
                "plant_availability": float(availability),
                "loss_factor": float(loss_factor),
            }
        )
        scenario = st.selectbox(
            "Feedstock sourcing scenario",
            FEEDSTOCK_SCENARIOS,
            index=FEEDSTOCK_SCENARIOS.index(production_cfg.get("feedstock_scenario", "HYBRID")),
        )
        production_cfg["feedstock_scenario"] = scenario
        if scenario == "HYBRID":
            farm_share = st.slider(
                "Hybrid farm share",
                min_value=0.0,
                max_value=1.0,
                value=float(production_cfg.get("farm_share", 0.5)),
                step=0.05,
            )
            production_cfg["farm_share"] = float(farm_share)

    pricing_cfg = cfg.setdefault("prices", {})
    with control_tabs[3]:
        st.markdown("### Product pricing")
        for product in PRODUCTS:
            params = pricing_cfg.setdefault(product, {})
            base_price = st.number_input(
                f"{product.replace('_', ' ').title()} base price",
                min_value=0.0,
                value=float(params.get("base_price", 0.0 if product != "ethanol" else 0.7)),
                step=1.0 if product != "ethanol" else 0.05,
                format="%.4f",
            )
            escalation = st.number_input(
                f"{product.replace('_', ' ').title()} price escalation (pa)",
                min_value=0.0,
                max_value=0.3,
                value=float(params.get("price_escalation_pa", 0.02)),
                step=0.005,
                format="%.4f",
            )
            params.update({"base_price": float(base_price), "price_escalation_pa": float(escalation)})
        revenue_table = tables.ensure_table("revenue_params").copy()
        if revenue_table.empty:
            revenue_table = pd.DataFrame(columns=list(INPUT_SCHEMAS["revenue_params"].columns.keys()))
        for product in PRODUCTS:
            params = pricing_cfg.get(product, {})
            mask = (
                revenue_table["product"].astype(str).str.lower() == product
                if "product" in revenue_table
                else pd.Series(dtype=bool)
            )
            updated = {
                "product": product,
                "base_price": params.get("base_price", np.nan),
                "price_escalation_pa": params.get("price_escalation_pa", np.nan),
                "price_indexation": params.get("price_indexation", "cpi"),
                "uom": params.get("uom", ""),
                "tariff_structure": params.get("tariff_structure", ""),
                "revenue_share": params.get("revenue_share", 1.0),
            }
            if mask.any():
                for key, value in updated.items():
                    revenue_table.loc[mask, key] = value
            else:
                revenue_table = pd.concat([revenue_table, pd.DataFrame([updated])], ignore_index=True)
        try:
            tables.set_table("revenue_params", revenue_table)
        except Exception as exc:
            st.warning(f"Pricing table not saved due to validation error: {exc}")
    with control_tabs[4]:
        st.markdown("### Risk and scenario options")
        run_tornado = st.checkbox("Compute sensitivity tornado", value=False)
        run_monte_carlo = st.checkbox("Run Monte Carlo", value=False)
        run_scenario_analysis = st.checkbox("Run scenario comparison", value=True)

        monte_iterations = 1000
        monte_seed = 42
        if run_monte_carlo:
            monte_iterations = st.slider("Monte Carlo iterations", min_value=200, max_value=5000, value=1000, step=100)
            monte_seed = st.number_input("Monte Carlo random seed", value=42, step=1)

    with st.spinner("Running base model..."):
        try:
            results = run_full_model(cfg)
        except Exception as exc:  # pragma: no cover - runtime feedback for the UI
            st.error(f"Model execution failed: {exc}")
            st.stop()

    metrics = results["metrics"]
    dashboard = results["dashboard"]

    metric_items = [
        ("Project_NPV", "Project NPV", "currency"),
        ("Project_IRR", "Project IRR", "percent"),
        ("Equity_IRR", "Equity IRR", "percent"),
        ("Payback_Year", "Payback Year", "year"),
        ("DSCR_min", "Min DSCR", "ratio"),
        ("DSCR_avg", "Avg DSCR", "ratio"),
    ]

    with page_tabs_container:
        landing_tab, summary_tab, financial_tab, production_tab, sensitivity_tab, scenario_tab = st.tabs(
            [
                "Input & Assumptions",
                "Summary",
                "Financial Statements",
                "Production & Pricing",
                "Sensitivities",
                "Scenarios",
            ]
        )

        with landing_tab:
            st.subheader("Input & assumptions tables")
            st.markdown(
                "Review, add, or remove records from each canonical input table. Updates apply across the model "
                "on the next run."
            )
            if sync_errors:
                for table_name, message in sync_errors.items():
                    st.error(f"{table_name}: {message}")
            for label, table_name, description in LANDING_TABLES:
                _render_table_editor(tables, table_name, label, sync_errors.get(table_name), description)

        with summary_tab:
            st.subheader("Headline metrics")
            for idx in range(0, len(metric_items), 3):
                cols = st.columns(3)
                for col, (key, label, kind) in zip(cols, metric_items[idx: idx + 3]):
                    col.metric(label, _format_metric(metrics.get(key), kind))
            st.subheader("Assumptions snapshot")
            st.dataframe(dashboard["assumptions_snapshot"], use_container_width=True)
            st.subheader("Global block")
            st.dataframe(dashboard["global_block"], use_container_width=True)
            latest = dashboard.get("latest_drivers", {})
            if latest:
                st.subheader("Latest drivers")
                latest_df = pd.DataFrame([latest])
                st.dataframe(latest_df, use_container_width=True)
            annual_prod = dashboard.get("annual_production")
            if isinstance(annual_prod, pd.DataFrame) and not annual_prod.empty:
                st.subheader("Annual production by product")
                prod_chart = annual_prod.set_index("year")
                st.bar_chart(prod_chart)
            annual_cashflow = dashboard.get("annual_cashflow")
            if isinstance(annual_cashflow, pd.DataFrame) and not annual_cashflow.empty:
                st.subheader("Annual cash flows")
                cash_chart = annual_cashflow.set_index("year")[["CFO", "CFI", "CFF", "NetCashFlow"]]
                st.bar_chart(cash_chart)

        statement_configs = [
            ("Income Statement (P&L)", "pnl"),
            ("Statement of Cash Flows", "cashflow"),
            ("Statement of Financial Position", "balancesheet"),
        ]
        with financial_tab:
            fs_tabs = st.tabs([label for label, _ in statement_configs])
            for tab, (label, key) in zip(fs_tabs, statement_configs):
                with tab:
                    monthly_df = results["statements_monthly"].get(key, pd.DataFrame())
                    annual_df = results["statements_annual"].get(key, pd.DataFrame())
                    safe_key = re.sub(r"[^a-z0-9]+", "_", label.lower())
                    _render_dataframe(monthly_df, f"Monthly {label}", key=f"monthly_{safe_key}")
                    _render_dataframe(annual_df, f"Annual {label}", key=f"annual_{safe_key}")

        with production_tab:
            prod_monthly = results["production_monthly"].copy()
            prod_monthly["date"] = pd.to_datetime(prod_monthly["date"])
            _render_dataframe(prod_monthly, "Monthly production", key="production_monthly")
            prod_annual_src = results.get("production_annual")
            if (
                isinstance(prod_annual_src, pd.DataFrame)
                and {"year", "product", "volume"}.issubset(prod_annual_src.columns)
                and not prod_annual_src.empty
            ):
                prod_annual = (
                    prod_annual_src.pivot_table(index="year", columns="product", values="volume", aggfunc="sum")
                    .reset_index()
                    .fillna(0.0)
                )
            else:
                prod_annual = pd.DataFrame(columns=["year", *PRODUCTS])
            _render_dataframe(prod_annual, "Annual production", key="production_annual")
            price_curves = results["price_curves"].copy()
            price_curves["date"] = pd.to_datetime(price_curves["date"])
            _render_dataframe(price_curves, "Price curves", key="price_curves")
            revenue_df = results["revenue"].copy()
            revenue_df["date"] = pd.to_datetime(revenue_df["date"])
            _render_dataframe(revenue_df, "Revenue stack", key="revenue")

        with sensitivity_tab:
            if run_tornado:
                with st.spinner("Calculating sensitivity tornado..."):
                    tornado_df = sensitivity_tornado(cfg, {"metrics": metrics}, lambda c: run_full_model(c), None)
                _render_dataframe(tornado_df, "Tornado sensitivity", key="tornado")
            else:
                st.info("Enable 'Compute sensitivity tornado' in the controls tabs to evaluate sensitivities.")

            if run_monte_carlo:
                with st.spinner("Running Monte Carlo simulation..."):
                    monte_results = monte_carlo(
                        cfg,
                        lambda c: run_full_model(c),
                        iterations=int(monte_iterations),
                        random_seed=int(monte_seed),
                    )
                _render_dataframe(
                    monte_results["percentiles"].reset_index().rename(columns={"index": "Percentile"}),
                    "Monte Carlo percentiles",
                    key="monte_percentiles",
                )
                _render_dataframe(monte_results["samples"], "Monte Carlo samples", key="monte_samples")
            else:
                st.info("Enable 'Run Monte Carlo' in the controls tabs to sample risk drivers.")

        with scenario_tab:
            if run_scenario_analysis:
                scenarios = {
                    "FARM_ONLY": {"production": {"feedstock_scenario": "FARM_ONLY"}},
                    "BUY_ONLY": {"production": {"feedstock_scenario": "BUY_ONLY"}},
                    "HYBRID": {"production": {"feedstock_scenario": "HYBRID"}},
                }
                with st.spinner("Evaluating scenarios..."):
                    scenario_df = run_scenarios(cfg, lambda c: run_full_model(c), scenarios)
                base_metrics = pd.DataFrame([metrics]).assign(scenario="Base")
                scenario_df = pd.concat([base_metrics, scenario_df], ignore_index=True)
                _render_dataframe(scenario_df, "Scenario comparison", key="scenarios")
            else:
                st.info("Enable 'Run scenario comparison' in the controls tabs to compare FARM/BUY/HYBRID structures.")

    st.success("Model run complete.")


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    if _streamlit_runtime_exists():
        main()
    else:
        try:
            from streamlit.web import bootstrap  # type: ignore[attr-defined]

            bootstrap.run(__file__, "", [])
        except ModuleNotFoundError as exc:
            print(
                "Streamlit is not installed in this environment. Install it with "
                "'pip install streamlit' and re-run the app.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        except Exception as exc:  # pragma: no cover - defensive feedback
            print(
                "Unable to start the Streamlit runtime automatically. "
                "If the Streamlit CLI is unavailable, try installing Streamlit or "
                "launching via 'python -m streamlit run streamlit_app.py'.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
