"""Streamlit front-end for the Sugar Cane Bioethanol multi-product finance model.

This app wraps the `model.py` engine and exposes interactive controls for loading
assumption workbooks, adjusting key drivers, and reviewing the resulting
financial outputs, dashboards, sensitivities, and scenarios.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

# Streamlit requires page config to be set before other st.* calls; do so at import time.
try:  # pragma: no cover - harmless when running via `python streamlit_app.py`
    st.set_page_config(title="Sugarcane Bioethanol Finance Model", layout="wide")
except (StreamlitAPIException, RuntimeError):
    # If not running inside `streamlit run`, the API raises; tolerate so the module
    # remains importable for linting/py_compile or other tooling contexts.
    pass

try:  # noqa: SIM105 - streamlit feedback when dependencies missing
    from model import (
        FEEDSTOCK_SCENARIOS,
        PRODUCTS,
        InputTables,
        build_config,
        load_inputs_from_excel,
        monte_carlo,
        run_full_model,
        run_scenarios,
        sensitivity_tornado,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - executed only when deps missing
    st.error(
        "Required dependency missing when importing the finance engine: "
        f"{exc}. Install the project requirements (numpy, pandas, etc.) and "
        "restart the app."
    )
    st.stop()


def _load_tables_from_upload(uploaded_file: io.BytesIO | None) -> tuple[InputTables, Dict[str, object]]:
    """Persist an uploaded Excel file temporarily and load it via the model helpers."""
    if uploaded_file is None:
        return InputTables(), {}

    suffix = Path(uploaded_file.name).suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        tables, assumptions = load_inputs_from_excel(tmp_path, preview=0)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:  # pragma: no cover - best effort cleanup
            pass
    return tables, assumptions


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


def main() -> None:
    st.title("Sugarcane Bioethanol Project Finance Model")
    st.markdown(
        "Use this Streamlit interface to explore the integrated bioethanol, sugar, "
        "electricity, and animal feed project finance model. Upload an Excel "
        "assumptions workbook or rely on defaults, tweak critical drivers in the "
        "sidebar, and review the resulting statements, dashboards, sensitivities, "
        "and scenarios."
    )

    st.sidebar.header("Model Controls")
    uploaded_file = st.sidebar.file_uploader("Upload assumptions workbook", type=["xlsx", "xlsm", "xls"], key="workbook")

    tables, assumptions = _load_tables_from_upload(uploaded_file)
    cfg = build_config(assumptions, tables)

    horizon = cfg["projection_horizon"]
    with st.sidebar.expander("Projection horizon", expanded=False):
        start_year = st.number_input("Start year", value=int(horizon["start_year"]), step=1)
        end_year = st.number_input("End year", value=int(horizon["end_year"]), min_value=int(start_year), step=1)
        start_month = st.number_input("Start month", min_value=1, max_value=12, value=int(horizon.get("start_month", 1)))
        horizon.update({"start_year": int(start_year), "end_year": int(end_year), "start_month": int(start_month)})

    global_inputs = cfg["global_inputs"]
    with st.sidebar.expander("Financial assumptions", expanded=False):
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

    production_cfg = cfg.setdefault("production", {})
    with st.sidebar.expander("Production assumptions", expanded=False):
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

    pricing_cfg = cfg.setdefault("prices", {})
    with st.sidebar.expander("Product pricing", expanded=False):
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

    scenario = st.sidebar.selectbox(
        "Feedstock sourcing scenario",
        FEEDSTOCK_SCENARIOS,
        index=FEEDSTOCK_SCENARIOS.index(production_cfg.get("feedstock_scenario", "HYBRID")),
    )
    production_cfg["feedstock_scenario"] = scenario
    if scenario == "HYBRID":
        farm_share = st.sidebar.slider(
            "Hybrid farm share",
            min_value=0.0,
            max_value=1.0,
            value=float(production_cfg.get("farm_share", 0.5)),
            step=0.05,
        )
        production_cfg["farm_share"] = float(farm_share)

    run_tornado = st.sidebar.checkbox("Compute sensitivity tornado", value=False)
    run_monte_carlo = st.sidebar.checkbox("Run Monte Carlo", value=False)
    monte_iterations = 1000
    monte_seed = 42
    if run_monte_carlo:
        monte_iterations = st.sidebar.slider("Monte Carlo iterations", min_value=200, max_value=5000, value=1000, step=100)
        monte_seed = st.sidebar.number_input("Monte Carlo random seed", value=42, step=1)
    run_scenario_analysis = st.sidebar.checkbox("Run scenario comparison", value=True)

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

    tabs = st.tabs(["Summary", "Financial Statements", "Production & Pricing", "Sensitivities", "Scenarios"])

    with tabs[0]:
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

    statement_map = {"P&L": "pnl", "Cash Flow": "cashflow", "Balance Sheet": "balancesheet"}
    with tabs[1]:
        statement_choice = st.selectbox("Statement", list(statement_map.keys()), index=0)
        key = statement_map[statement_choice]
        monthly_df = results["statements_monthly"][key]
        annual_df = results["statements_annual"][key]
        _render_dataframe(monthly_df, f"Monthly {statement_choice}", key=f"monthly_{key}")
        _render_dataframe(annual_df, f"Annual {statement_choice}", key=f"annual_{key}")

    with tabs[2]:
        prod_monthly = results["production_monthly"].copy()
        prod_monthly["date"] = pd.to_datetime(prod_monthly["date"])
        _render_dataframe(prod_monthly, "Monthly production", key="production_monthly")
        prod_annual = results["production_annual"].pivot_table(index="year", columns="product", values="volume", aggfunc="sum")
        prod_annual = prod_annual.reset_index().fillna(0.0)
        _render_dataframe(prod_annual, "Annual production", key="production_annual")
        price_curves = results["price_curves"].copy()
        price_curves["date"] = pd.to_datetime(price_curves["date"])
        _render_dataframe(price_curves, "Price curves", key="price_curves")
        revenue_df = results["revenue"].copy()
        revenue_df["date"] = pd.to_datetime(revenue_df["date"])
        _render_dataframe(revenue_df, "Revenue stack", key="revenue")

    with tabs[3]:
        if run_tornado:
            with st.spinner("Calculating sensitivity tornado..."):
                tornado_df = sensitivity_tornado(cfg, {"metrics": metrics}, lambda c: run_full_model(c), None)
            _render_dataframe(tornado_df, "Tornado sensitivity", key="tornado")
        else:
            st.info("Enable 'Compute sensitivity tornado' in the sidebar to evaluate sensitivities.")

        if run_monte_carlo:
            with st.spinner("Running Monte Carlo simulation..."):
                monte_results = monte_carlo(cfg, lambda c: run_full_model(c), iterations=int(monte_iterations), random_seed=int(monte_seed))
            _render_dataframe(monte_results["percentiles"].reset_index().rename(columns={"index": "Percentile"}), "Monte Carlo percentiles", key="monte_percentiles")
            _render_dataframe(monte_results["samples"], "Monte Carlo samples", key="monte_samples")
        else:
            st.info("Enable 'Run Monte Carlo' in the sidebar to sample risk drivers.")

    with tabs[4]:
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
            st.info("Enable 'Run scenario comparison' in the sidebar to compare FARM/BUY/HYBRID structures.")

    st.success("Model run complete.")


if __name__ == "__main__":
    main()
