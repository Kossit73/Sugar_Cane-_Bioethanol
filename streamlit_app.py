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
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
from pandas.testing import assert_frame_equal


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


def _safe_rerun() -> None:
    """Attempt to trigger a Streamlit rerun only when a runtime exists."""

    if not _streamlit_runtime_exists():
        return
    try:
        st.experimental_rerun()
    except StreamlitAPIException:  # pragma: no cover - defensive catch
        pass
    except Exception:
        # Older Streamlit builds or embedded executions may surface bespoke
        # rerun exceptions; swallow them so non-Streamlit contexts keep running.
        pass


def _editor_state_key(table_name: str) -> str:
    """Return the canonical widget state key for a given table editor."""

    return f"table_editor__{table_name}"


def _update_editor_state(table_name: str, tables: "InputTables") -> None:
    """Synchronise the Streamlit data editor state with the backing table."""

    _ = tables  # retained for signature compatibility
    state_key = _editor_state_key(table_name)
    # Streamlit forbids direct writes to widget-managed keys once the widget is
    # instantiated. Clearing the state entry ensures the next render picks up
    # the refreshed DataFrame without violating session-state policies. Also
    # purge the legacy key from earlier builds to avoid conflicts.
    st.session_state.pop(state_key, None)
    st.session_state.pop(f"editor_{table_name}", None)


@contextmanager
def _modal_container(title: str):
    """Yield a modal-like container, falling back when `st.modal` is unavailable."""

    if hasattr(st, "modal"):
        with st.modal(title):
            yield True
        return

    # Fallback for older Streamlit releases – use an expander to host the form.
    with st.expander(title, expanded=True):
        st.info(
            "Modal dialogs are not supported in this Streamlit version; "
            "editing is displayed inline instead.",
            icon="ℹ️",
        )
        yield False


def _format_default_for_entry(value: object, dtype: str) -> object:
    """Return a widget-friendly default for modal data-entry fields."""

    if dtype == "bool":
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        return bool(value)
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _parse_modal_entry(value: object, dtype: str) -> object:
    """Convert modal widget values back into schema-compatible data."""

    if dtype == "bool":
        return bool(value)
    text = str(value).strip() if value is not None else ""
    if text == "":
        return np.nan
    try:
        if dtype == "int":
            return int(float(text))
        if dtype == "float":
            return float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a valid {dtype} value")
    return text


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
        normalize_key,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - executed only when deps missing
    MODEL_IMPORT_ERROR = exc

if MODEL_IMPORT_ERROR is not None:
    _KEY_NORMALIZER = re.compile(r"[^0-9a-zA-Z]+")

    def normalize_key(value: str) -> str:
        key = _KEY_NORMALIZER.sub("_", str(value).strip().lower())
        key = re.sub(r"_+", "_", key).strip("_")
        return key


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


DEFAULT_TABLE_STORE_KEY = "table_defaults_store"
DEFAULT_EDIT_STATE_KEY = "default_edit_state"
_FACTORY_DEFAULT_FRAMES_CACHE: Optional[Dict[str, pd.DataFrame]] = None


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


def _display_default_value(value: object) -> str:
    """Pretty-print values for the default-row manager."""
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "—"
        magnitude = abs(float(value))
        if magnitude >= 1_000:
            return f"{float(value):,.2f}"
        if magnitude >= 1:
            return f"{float(value):,.2f}"
        return (f"{float(value):.6f}").rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _factory_default_frames() -> Dict[str, pd.DataFrame]:
    """Return deep copies of the model's factory default tables."""

    global _FACTORY_DEFAULT_FRAMES_CACHE
    if _FACTORY_DEFAULT_FRAMES_CACHE is None:
        frames: Dict[str, pd.DataFrame] = {}

        horizon_defaults = dict(DEFAULTS["horizon"])
        frames["projection_horizon"] = pd.DataFrame([horizon_defaults])
        frames["production_horizon"] = pd.DataFrame([DEFAULTS["production_horizon"]])
        frames["global_inputs"] = pd.DataFrame([DEFAULTS["global"]])
        frames["working_capital_days"] = pd.DataFrame([DEFAULTS["working_capital"]])

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
        frames["capex_lines"] = pd.DataFrame(capex_rows)

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
        frames["revenue_params"] = pd.DataFrame(price_rows)

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
        frames["production_annual"] = pd.DataFrame(production_rows)

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
        frames["production_monthly"] = pd.DataFrame(monthly_rows)

        frames["direct_costs_monthly"] = pd.DataFrame(
            [
                {
                    "date": f"{start_year}-01",
                    "cost_type": "feedstock purchase",
                    "product_link": "ethanol",
                    "amount": 500_000.0,
                    "currency": "USD",
                }
            ]
        )

        frames["staff_costs_monthly"] = pd.DataFrame(
            [
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
        )

        frames["other_opex_monthly"] = pd.DataFrame(
            [
                {"date": f"{start_year}-01", "category": "insurance", "amount": 20_000.0, "currency": "USD"},
                {"date": f"{start_year}-01", "category": "service_contract", "amount": 35_000.0, "currency": "USD"},
                {"date": f"{start_year}-01", "category": "general_admin", "amount": 50_000.0, "currency": "USD"},
                {"date": f"{start_year}-01", "category": "energy_cost", "amount": 15_000.0, "currency": "USD"},
            ]
        )

        frames["ar_other_assets"] = pd.DataFrame(
            [
                {
                    "date": f"{start_year}-01",
                    "receivables": 0.0,
                    "prepaid_expenses": 0.0,
                    "other_current_assets": 0.0,
                    "dso_days": DEFAULTS["working_capital"]["dso_days"],
                }
            ]
        )

        frames["inventory_ap"] = pd.DataFrame(
            [
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
        )

        frames["debt_tranches"] = pd.DataFrame(DEFAULTS["debt"]["tranches"])
        frames["tax_schedule"] = pd.DataFrame([DEFAULTS["tax"]])
        frames["inflation_index"] = DEFAULTS["inflation_index"].copy()
        frames["risk_params"] = DEFAULTS["risk_params"].copy()

        for table_name, schema in INPUT_SCHEMAS.items():
            frames.setdefault(table_name, pd.DataFrame(columns=list(schema.columns.keys())))

        _FACTORY_DEFAULT_FRAMES_CACHE = {name: df.copy(deep=True) for name, df in frames.items()}

    return {name: df.copy(deep=True) for name, df in _FACTORY_DEFAULT_FRAMES_CACHE.items()}


def _ensure_default_table_store() -> Dict[str, pd.DataFrame]:
    """Ensure session state carries a mutable, schema-aligned default table store."""

    if DEFAULT_TABLE_STORE_KEY not in st.session_state:
        factory_frames = _factory_default_frames()
        seeded_tables = InputTables()
        for table_name, df in factory_frames.items():
            seeded_tables.set_table(table_name, df)
        st.session_state[DEFAULT_TABLE_STORE_KEY] = {
            name: seeded_tables.ensure_table(name).copy()
            for name in INPUT_SCHEMAS.keys()
        }

    store = st.session_state[DEFAULT_TABLE_STORE_KEY]
    for table_name, schema in INPUT_SCHEMAS.items():
        if table_name not in store:
            store[table_name] = pd.DataFrame(columns=list(schema.columns.keys()))
    return store


def _get_default_table(table_name: str) -> pd.DataFrame:
    store = _ensure_default_table_store()
    df = store.get(table_name)
    if df is None:
        schema = INPUT_SCHEMAS[table_name]
        return pd.DataFrame(columns=list(schema.columns.keys()))
    return df.copy(deep=True)


def _save_default_table(table_name: str, df: pd.DataFrame) -> None:
    temp = InputTables()
    temp.set_table(table_name, df)
    store = _ensure_default_table_store()
    store[table_name] = temp.ensure_table(table_name).copy()


def _restore_factory_default_table(table_name: str) -> pd.DataFrame:
    factory_frames = _factory_default_frames()
    df = factory_frames.get(
        table_name,
        pd.DataFrame(columns=list(INPUT_SCHEMAS[table_name].columns.keys())),
    )
    _save_default_table(table_name, df)
    return _get_default_table(table_name)


def _render_default_row_controls(table_name: str, label: str, schema, default_df: pd.DataFrame) -> None:
    if default_df.empty:
        st.info("No default rows defined yet – save the current table to create a baseline.")
        return

    st.markdown("##### Default baseline rows")
    display_df = default_df.reset_index(drop=True)
    for idx, row in display_df.iterrows():
        col_widths = [1] * len(schema.columns) + [0.6]
        row_cols = st.columns(col_widths)
        for position, column_name in enumerate(schema.columns.keys()):
            value = row.get(column_name, np.nan)
            display_value = _display_default_value(value)
            row_cols[position].markdown(
                f"<span style='font-size:0.7rem;color:#6b6b6b'>{column_name.replace('_', ' ').title()}</span><br>"
                f"<span style='font-weight:600'>{display_value}</span>",
                unsafe_allow_html=True,
            )
        if row_cols[-1].button("Edit", key=f"default_edit_{table_name}_{idx}"):
            st.session_state[DEFAULT_EDIT_STATE_KEY] = {"table": table_name, "row": int(idx)}
            _safe_rerun()


def _render_default_edit_modal(table_name: str, label: str, schema) -> None:
    edit_state = st.session_state.get(DEFAULT_EDIT_STATE_KEY)
    if not edit_state or edit_state.get("table") != table_name:
        return

    row_index = int(edit_state.get("row", -1))
    store = _ensure_default_table_store()
    df = store.get(table_name)
    if df is None or not (0 <= row_index < len(df)):
        st.session_state.pop(DEFAULT_EDIT_STATE_KEY, None)
        return

    row = df.iloc[row_index]
    with _modal_container(f"Edit default row {row_index + 1} – {label}") as modal_supported:
        form = st.form(key=f"default_edit_form_{table_name}_{row_index}")
        updated_values: Dict[str, object] = {}
        for column_name, dtype in schema.columns.items():
            field_label = column_name.replace("_", " ").title()
            current_value = row.get(column_name, np.nan)
            if dtype == "int":
                default_val = schema.defaults.get(column_name, 0)
                base_value = int(default_val) if pd.isna(current_value) else int(current_value)
                input_value = form.number_input(
                    field_label,
                    value=float(base_value),
                    step=1.0,
                    format="%d",
                    key=f"default_edit_field_{table_name}_{row_index}_{column_name}",
                )
                updated_values[column_name] = int(input_value)
            elif dtype == "float":
                default_val = schema.defaults.get(column_name, 0.0)
                base_value = float(default_val) if pd.isna(current_value) else float(current_value)
                input_value = form.number_input(
                    field_label,
                    value=base_value,
                    step=0.01,
                    format="%.6f",
                    key=f"default_edit_field_{table_name}_{row_index}_{column_name}",
                )
                updated_values[column_name] = float(input_value)
            elif dtype == "bool":
                default_val = bool(schema.defaults.get(column_name, False))
                base_value = default_val if pd.isna(current_value) else bool(current_value)
                input_value = form.checkbox(
                    field_label,
                    value=base_value,
                    key=f"default_edit_field_{table_name}_{row_index}_{column_name}",
                )
                updated_values[column_name] = bool(input_value)
            else:
                if current_value is None or (isinstance(current_value, float) and math.isnan(current_value)):
                    text_value = ""
                else:
                    text_value = str(current_value)
                input_value = form.text_input(
                    field_label,
                    value=text_value,
                    key=f"default_edit_field_{table_name}_{row_index}_{column_name}",
                )
                updated_values[column_name] = input_value if input_value != "" else None

        submit_label = "Save changes" if modal_supported else "Save inline changes"
        if form.form_submit_button(submit_label, type="primary"):
            updated_df = df.copy()
            for column_name, value in updated_values.items():
                updated_df.at[row_index, column_name] = value
            _save_default_table(table_name, updated_df)
            st.session_state.pop(DEFAULT_EDIT_STATE_KEY, None)
            st.session_state[f"default_feedback_{table_name}"] = "Default row updated."
            _safe_rerun()

        if st.button("Cancel", key=f"default_edit_cancel_{table_name}_{row_index}"):
            st.session_state.pop(DEFAULT_EDIT_STATE_KEY, None)
            _safe_rerun()


def _seed_tables_with_defaults(tables: InputTables) -> None:
    """Populate session tables with the active default frames."""

    default_store = _ensure_default_table_store()
    for table_name, schema in INPUT_SCHEMAS.items():
        default_df = default_store.get(table_name)
        if default_df is None:
            empty_df = pd.DataFrame(columns=list(schema.columns.keys()))
            tables.set_table(table_name, empty_df)
        else:
            tables.set_table(table_name, default_df.copy())


def _get_tables() -> InputTables:
    if "input_tables" not in st.session_state:
        tables = InputTables()
        _seed_tables_with_defaults(tables)
        st.session_state["input_tables"] = tables
    return st.session_state["input_tables"]


def _sync_tables_from_state(tables: InputTables) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    for table_name in INPUT_SCHEMAS.keys():
        state_key = _editor_state_key(table_name)
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
    schema = INPUT_SCHEMAS[table_name]
    df = tables.ensure_table(table_name).copy()

    feedback_key = f"default_feedback_{table_name}"
    feedback_message = st.session_state.pop(feedback_key, None)
    if feedback_message:
        st.success(feedback_message)
    controls = st.columns(2)
    add_modal_key = f"show_add_modal_{table_name}"
    if controls[0].button(f"Add row", key=f"add_{table_name}"):
        st.session_state[add_modal_key] = True

    if st.session_state.get(add_modal_key):
        modal_title = f"Add {label} row"
        with _modal_container(modal_title):
            schema_columns = list(schema.columns.items())
            with st.form(f"add_row_form_{table_name}"):
                form_values: Dict[str, object] = {}
                for col_name, dtype in schema_columns:
                    pretty_label = col_name.replace("_", " ").title()
                    default_value = schema.defaults.get(col_name, np.nan)
                    if dtype == "bool":
                        form_values[col_name] = st.checkbox(
                            pretty_label,
                            value=_format_default_for_entry(default_value, dtype),
                            key=f"add_row_{table_name}_{col_name}",
                        )
                    else:
                        form_values[col_name] = st.text_input(
                            pretty_label,
                            value=_format_default_for_entry(default_value, dtype),
                            key=f"add_row_{table_name}_{col_name}",
                        )

                submit_col, cancel_col = st.columns(2)
                submitted = submit_col.form_submit_button("Save row", use_container_width=True)
                cancelled = cancel_col.form_submit_button("Cancel", use_container_width=True, type="secondary")

            if cancelled:
                st.session_state.pop(add_modal_key, None)
                for col_name, _ in schema_columns:
                    st.session_state.pop(f"add_row_{table_name}_{col_name}", None)
                _safe_rerun()
            elif submitted:
                try:
                    row_payload: Dict[str, object] = {}
                    for col_name, dtype in schema_columns:
                        widget_value = form_values[col_name]
                        row_payload[col_name] = _parse_modal_entry(widget_value, dtype)
                    tables.add_row(table_name, row_payload)
                except Exception as exc:
                    st.error(f"Unable to add row: {exc}")
                else:
                    st.session_state[feedback_key] = (
                        "Row added successfully. You can fine-tune the values directly "
                        "in the table below."
                    )
                    st.session_state.pop(add_modal_key, None)
                    for col_name, _ in schema_columns:
                        st.session_state.pop(f"add_row_{table_name}_{col_name}", None)
                    _update_editor_state(table_name, tables)
                    _safe_rerun()

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
            df = tables.ensure_table(table_name).copy()
            _update_editor_state(table_name, tables)
            _safe_rerun()
    state_key = _editor_state_key(table_name)
    # Clear any legacy widget state that may have been set by previous builds
    # using the old key to avoid Streamlit policy violations on render.
    st.session_state.pop(f"editor_{table_name}", None)

    editor = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=state_key,
    )
    if isinstance(editor, pd.DataFrame):
        editor_clean = editor.copy()
        if "_index" in editor_clean.columns:
            editor_clean = editor_clean.drop(columns=["_index"])

        drop_columns: List[str] = []
        rename_map: Dict[str, str] = {}
        for col in list(editor_clean.columns):
            if col in schema.columns:
                continue
            norm = normalize_key(col)
            if norm in schema.columns:
                rename_map[col] = norm
            else:
                drop_columns.append(col)
        if drop_columns:
            editor_clean = editor_clean.drop(columns=drop_columns)
        if rename_map:
            editor_clean = editor_clean.rename(columns=rename_map)

        canonical_cols = list(schema.columns.keys())
        editor_clean = editor_clean.reindex(columns=canonical_cols, fill_value=np.nan)
        editor_clean = editor_clean.replace({None: np.nan})
        editor_clean = editor_clean.dropna(how="all").reset_index(drop=True)

        try:
            assert_frame_equal(
                df.reset_index(drop=True),
                editor_clean.reset_index(drop=True),
                check_dtype=False,
                check_like=True,
            )
            frames_equal = True
        except AssertionError:
            frames_equal = False

        if not frames_equal:
            try:
                tables.set_table(table_name, editor_clean)
            except Exception as exc:
                st.error(f"Unable to update table: {exc}")
            else:
                _update_editor_state(table_name, tables)
                _safe_rerun()

    current_df = tables.ensure_table(table_name).copy()

    with st.expander("Manage defaults & clean start", expanded=False):
        st.caption(
            "Reset this table to the stored defaults or update the baseline values used when creating a clean workbook."
        )
        action_cols = st.columns(3)
        if action_cols[0].button("Reset table to defaults", key=f"reset_defaults_{table_name}"):
            try:
                tables.set_table(table_name, _get_default_table(table_name))
            except Exception as exc:
                st.error(f"Unable to reset table: {exc}")
            else:
                st.session_state[feedback_key] = "Table reset to stored defaults."
                _update_editor_state(table_name, tables)
                _safe_rerun()
        if action_cols[1].button("Save current as defaults", key=f"save_defaults_{table_name}"):
            try:
                _save_default_table(table_name, current_df)
            except Exception as exc:
                st.error(f"Unable to save defaults: {exc}")
            else:
                st.session_state[feedback_key] = "Stored defaults updated from current table."
                _safe_rerun()
        if action_cols[2].button("Restore factory defaults", key=f"factory_defaults_{table_name}"):
            try:
                restored_df = _restore_factory_default_table(table_name)
                tables.set_table(table_name, restored_df)
            except Exception as exc:
                st.error(f"Unable to restore factory defaults: {exc}")
            else:
                st.session_state[feedback_key] = "Factory defaults restored and applied."
                _update_editor_state(table_name, tables)
                _safe_rerun()

        defaults_df = _get_default_table(table_name)
        _render_default_row_controls(table_name, label, schema, defaults_df)

    _render_default_edit_modal(table_name, label, schema)

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

    # Ensure the hero title has sufficient breathing room while keeping the layout
    # nearly full-width on large monitors.
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
            margin: 0 !important;
            padding: 3.5rem 0 2.5rem !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            margin: 0 auto !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        [data-testid="stAppViewContainer"] .main .block-container,
        .block-container {
            margin: 0 auto !important;
            width: 100% !important;
            max-width: none !important;
            padding-left: clamp(20px, 3vw, 48px) !important;
            padding-right: clamp(20px, 3vw, 48px) !important;
        }
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] {
            margin-left: 0 !important;
            margin-right: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        h1, h1 span, .stMarkdown h1 {
            font-size: clamp(2.2rem, 3vw, 2.8rem) !important;
            line-height: 1.2 !important;
            margin-top: 0 !important;
            margin-bottom: 1.2rem !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
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
    (
        model_controls_tab,
        landing_tab,
        summary_tab,
        financial_tab,
        production_tab,
        sensitivity_tab,
        scenario_tab,
    ) = page_tabs_container.tabs(
        [
            "Model Controls",
            "Input & Assumptions",
            "Summary",
            "Financial Statements",
            "Production & Pricing",
            "Sensitivities",
            "Scenarios",
        ]
    )

    tables = _get_tables()
    sync_errors = _sync_tables_from_state(tables)
    assumptions: Dict[str, object] = {}
    cfg = build_config(assumptions, tables)

    horizon = cfg["projection_horizon"]
    production_horizon = cfg.get(
        "production_horizon",
        {"start_year": horizon["start_year"], "end_year": horizon["end_year"]},
    )

    with model_controls_tab:
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

        with control_tabs[0]:
            st.markdown("### Projection horizon")
            start_year = st.number_input("Start year", value=int(horizon["start_year"]), step=1)
            end_year = st.number_input(
                "End year",
                value=int(horizon["end_year"]),
                min_value=int(start_year),
                step=1,
            )
            start_month = st.number_input(
                "Start month",
                min_value=1,
                max_value=12,
                value=int(horizon.get("start_month", 1)),
            )
            horizon.update(
                {
                    "start_year": int(start_year),
                    "end_year": int(end_year),
                    "start_month": int(start_month),
                }
            )
            st.markdown("### Production horizon")
            prod_start_default = int(production_horizon.get("start_year", start_year))
            prod_start_default = max(prod_start_default, int(start_year))
            prod_start_default = min(prod_start_default, int(end_year))
            prod_start_year = st.number_input(
                "Production start year",
                value=prod_start_default,
                min_value=int(start_year),
                max_value=int(end_year),
                step=1,
            )
            prod_end_default = int(production_horizon.get("end_year", end_year))
            prod_end_default = max(prod_end_default, int(prod_start_year))
            prod_end_default = min(prod_end_default, int(end_year))
            prod_end_year = st.number_input(
                "Production end year",
                value=prod_end_default,
                min_value=int(prod_start_year),
                max_value=int(end_year),
                step=1,
            )
            if prod_end_year < prod_start_year:
                st.warning("Production end year adjusted to be no earlier than the start year.")
                prod_end_year = prod_start_year
            production_horizon.update({"start_year": int(prod_start_year), "end_year": int(prod_end_year)})
            cfg["production_horizon"] = production_horizon
            st.caption("Production volumes are set to zero outside the defined production horizon.")
            try:
                tables.set_table("projection_horizon", pd.DataFrame([horizon]))
                _update_editor_state("projection_horizon", tables)
                tables.set_table("production_horizon", pd.DataFrame([production_horizon]))
                _update_editor_state("production_horizon", tables)
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
                _update_editor_state("global_inputs", tables)
                tables.set_table("working_capital_days", pd.DataFrame([cfg["working_capital"]]))
                _update_editor_state("working_capital_days", tables)
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
                revenue_table = pd.DataFrame(
                    columns=list(INPUT_SCHEMAS["revenue_params"].columns.keys())
                )
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
                    revenue_table = pd.concat(
                        [revenue_table, pd.DataFrame([updated])],
                        ignore_index=True,
                    )
            try:
                tables.set_table("revenue_params", revenue_table)
            except Exception as exc:
                st.warning(f"Pricing table not saved due to validation error: {exc}")
            else:
                _update_editor_state("revenue_params", tables)

        with control_tabs[4]:
            st.markdown("### Risk and scenario options")
            run_tornado = st.checkbox("Compute sensitivity tornado", value=False)
            run_monte_carlo = st.checkbox("Run Monte Carlo", value=False)
            run_scenario_analysis = st.checkbox("Run scenario comparison", value=True)

            monte_iterations = 1000
            monte_seed = 42
            if run_monte_carlo:
                monte_iterations = st.slider(
                    "Monte Carlo iterations",
                    min_value=200,
                    max_value=5000,
                    value=1000,
                    step=100,
                )
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
