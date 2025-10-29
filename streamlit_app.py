"""Streamlit front-end for the Sugar Cane Bioethanol multi-product finance model.

This app wraps the `model.py` engine and exposes interactive controls for
adjusting key drivers and reviewing the resulting financial outputs, dashboards,
sensitivities, and scenarios.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import copy
import math
import re
import sys
from contextlib import contextmanager
from io import BytesIO
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def _option_index(options: Sequence[str], value: str, default: int = 0) -> int:
    """Return the index of ``value`` inside ``options`` with a safe fallback."""

    try:
        return options.index(value)
    except ValueError:
        return default if 0 <= default < len(options) else 0


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
        align_with_projection_horizon,
        build_config,
        MONTE_CARLO_DISTRIBUTIONS,
        MONTE_CARLO_VARIABLES,
        MONTE_CARLO_VARIABLE_LABELS,
        compute_risk_profile,
        decision_tree_analysis,
        metaheuristic_optimize,
        monte_carlo,
        neural_forecast_production,
        parse_ramp,
        run_full_model,
        run_scenarios,
        sensitivity_tornado,
        statistical_forecast,
        normalize_key,
        Timeline,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - executed only when deps missing
    MODEL_IMPORT_ERROR = exc

if MODEL_IMPORT_ERROR is not None:
    _KEY_NORMALIZER = re.compile(r"[^0-9a-zA-Z]+")

    def normalize_key(value: str) -> str:
        key = _KEY_NORMALIZER.sub("_", str(value).strip().lower())
        key = re.sub(r"_+", "_", key).strip("_")
        return key

    PRODUCTS = ("ethanol", "sugar", "electricity", "animal_feed")

    MONTE_CARLO_DISTRIBUTIONS = ("normal", "lognormal", "triangular", "uniform")
    MONTE_CARLO_VARIABLE_ITEMS = (
        ("opex", "Operating expenditure (all)"),
        ("interest_rate", "Interest rate"),
        ("capex", "Total CAPEX"),
        ("initial_investment", "Initial Investment (CAPEX)"),
        ("debt_schedule", "Debt schedule"),
        ("production", "Production volumes (all products)"),
        ("production_ethanol", "Production volumes (annual) – Ethanol"),
        ("production_sugar", "Production volumes (annual) – Sugar"),
        ("production_electricity", "Production volumes (annual) – Electricity"),
        ("production_animal_feed", "Production volumes (annual) – Animal feed"),
        ("pricing", "Pricing (all products)"),
        ("pricing_ethanol", "Pricing – Ethanol"),
        ("pricing_sugar", "Pricing – Sugar"),
        ("pricing_electricity", "Pricing – Electricity"),
        ("pricing_animal_feed", "Pricing – Animal feed"),
        ("revenue", "Revenue"),
        ("sugarcane_yield", "Sugarcane yield"),
        ("operating_cost_direct", "Operating Costs - Direct"),
        ("operating_cost_staff", "Operating Costs - Staff"),
        ("operating_cost_other", "Operating Costs - Other Opex"),
        ("labour", "Labour costs"),
        ("availability", "Plant availability"),
        ("other", "Other"),
    )
    MONTE_CARLO_VARIABLES = tuple(key for key, _ in MONTE_CARLO_VARIABLE_ITEMS)
    MONTE_CARLO_VARIABLE_LABELS = {key: label for key, label in MONTE_CARLO_VARIABLE_ITEMS}

    def compute_risk_profile(risk_params):  # pragma: no cover - fallback stub
        return {}

    def metaheuristic_optimize(*args, **kwargs):  # pragma: no cover - fallback stub
        return pd.DataFrame()

    def neural_forecast_production(*args, **kwargs):  # pragma: no cover - fallback stub
        return pd.DataFrame(columns=["date", "product", "forecast_volume"])

    def statistical_forecast(*args, **kwargs):  # pragma: no cover - fallback stub
        return {"historical": pd.DataFrame(), "forecast": pd.DataFrame(), "residual_std": np.nan, "equipment_failure_risk": np.nan}

    def decision_tree_analysis(*args, **kwargs):  # pragma: no cover - fallback stub
        return {"paths": pd.DataFrame(), "expected_metric": np.nan, "objective": "Project_NPV", "total_probability": 0.0}


if MODEL_IMPORT_ERROR is None:
    RISK_DISTRIBUTION_OPTIONS: Tuple[str, ...] = tuple(dict.fromkeys(MONTE_CARLO_DISTRIBUTIONS))
    _default_targets = set(str(x).lower() for x in DEFAULTS["risk_params"].get("target", []) if pd.notna(x))
    _default_targets.update(["risk", "price", "availability", "opex", "capex"])
    RISK_TARGET_OPTIONS: Tuple[str, ...] = tuple(sorted(_default_targets))
    _default_applies = {"global", "market"}
    for col in ("applies_to",):
        if col in DEFAULTS["risk_params"]:
            _default_applies.update(
                str(x).lower() for x in DEFAULTS["risk_params"][col].dropna().unique()
            )
    _default_applies.update(PRODUCTS)
    RISK_APPLIES_OPTIONS: Tuple[str, ...] = tuple(sorted(_default_applies))
else:  # pragma: no cover - fallback values when engine unavailable
    RISK_DISTRIBUTION_OPTIONS = MONTE_CARLO_DISTRIBUTIONS
    RISK_TARGET_OPTIONS = ("risk", "price", "availability", "opex", "capex")
    RISK_APPLIES_OPTIONS = tuple(sorted({"global", "market", *PRODUCTS}))

MONTE_CARLO_VARIABLE_LABEL_TO_KEY: Dict[str, str] = {
    label: key for key, label in MONTE_CARLO_VARIABLE_LABELS.items()
}

OPTIMIZER_VARIABLE_OPTIONS: Dict[str, str] = {
    "ethanol_price": "Ethanol price multiplier",
    "sugar_price": "Sugar price multiplier",
    "electricity_price": "Electricity tariff multiplier",
    "animal_feed_price": "Animal feed price multiplier",
    "availability": "Plant availability multiplier",
    "capex": "Total CAPEX multiplier",
    "opex": "Operating cost multiplier",
    "debt_rate": "Debt rate shift",
}

STATISTICAL_SERIES_OPTIONS: Dict[str, str] = {
    "revenue": "Revenue",
    "cogs": "Cost of goods sold",
    "opex": "Operating expenditure",
    "ebitda": "EBITDA",
    "staff_costs": "Staff costs",
    "production_ethanol": "Production – Ethanol",
    "production_sugar": "Production – Sugar",
    "production_electricity": "Production – Electricity",
    "production_animal_feed": "Production – Animal feed",
}


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


def _metric_kind(metric_name: str) -> str:
    key = str(metric_name or "").lower()
    if "irr" in key:
        return "percent"
    if "npv" in key or "value" in key:
        return "currency"
    if "dscr" in key or "ratio" in key:
        return "ratio"
    return "number"


DEFAULT_TABLE_STORE_KEY = "table_defaults_store"
DEFAULT_EDIT_STATE_KEY = "default_edit_state"
_FACTORY_DEFAULT_FRAMES_CACHE: Optional[Dict[str, pd.DataFrame]] = None


def _render_dataframe(df: pd.DataFrame, title: str, key: str) -> None:
    """Render a dataframe without exposing download controls."""
    st.subheader(title)
    st.dataframe(df, use_container_width=True, key=f"df_{key}")


def _scenario_overrides_from_table(scenario_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Convert the scenario comparison table into override dictionaries."""

    if not isinstance(scenario_df, pd.DataFrame) or scenario_df.empty:
        return {}

    working = scenario_df.copy()
    working = working.replace({"": np.nan})

    enabled = working.get("enabled", True)
    if not isinstance(enabled, pd.Series):
        enabled = pd.Series(True, index=working.index)
    working = working[enabled.fillna(True)]
    working = working.dropna(subset=["scenario_name"], how="any")

    overrides: Dict[str, Dict[str, object]] = {}
    for _, row in working.iterrows():
        name = str(row.get("scenario_name", "")).strip()
        if not name:
            continue
        override: Dict[str, object] = {}
        production_override: Dict[str, object] = {}
        feedstock = str(row.get("feedstock_scenario", "")).strip()
        if feedstock:
            production_override["feedstock_scenario"] = feedstock.upper()
        farm_share_val = row.get("farm_share")
        if pd.notna(farm_share_val):
            try:
                production_override["farm_share"] = float(farm_share_val)
            except (TypeError, ValueError):
                pass
        if production_override:
            override["production"] = production_override
        if override:
            overrides[name] = override

    return overrides


def _apply_scenario_override(base_cfg: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    """Return a deep-copied configuration with scenario overrides applied."""

    scenario_cfg = copy.deepcopy(base_cfg)
    for section, values in override.items():
        if isinstance(values, Mapping) and isinstance(scenario_cfg.get(section), Mapping):
            target = copy.deepcopy(scenario_cfg.get(section))
            for key, value in values.items():
                target[key] = value
            scenario_cfg[section] = target
        else:
            scenario_cfg[section] = values
    return scenario_cfg


BASE_SCENARIO_LABEL = "Base case"


def _ensure_scenario_payload(
    selected: str,
    base_cfg: Dict[str, object],
    base_results: Mapping[str, object],
    overrides: Mapping[str, Dict[str, object]],
) -> Tuple[Dict[str, object], Mapping[str, object]]:
    """Return (cfg, results) for the selected scenario, caching evaluations."""

    cache: Dict[str, Tuple[Dict[str, object], Mapping[str, object]]] = st.session_state.setdefault(
        "scenario_payload_cache", {}
    )

    if selected == BASE_SCENARIO_LABEL or selected not in overrides:
        cache[BASE_SCENARIO_LABEL] = (copy.deepcopy(base_cfg), base_results)
        st.session_state.scenario_payload_cache = cache
        return base_cfg, base_results

    if selected in cache:
        cfg_cached, results_cached = cache[selected]
        return cfg_cached, results_cached

    override = overrides[selected]
    scenario_cfg = _apply_scenario_override(base_cfg, override)
    scenario_results = run_full_model(scenario_cfg)
    cache[selected] = (scenario_cfg, scenario_results)
    st.session_state.scenario_payload_cache = cache
    return scenario_cfg, scenario_results


def _generate_excel_bytes(
    cfg: Mapping[str, object],
    results: Mapping[str, object],
    scenario_name: str,
) -> bytes:
    """Create an Excel workbook for the provided results and return its bytes."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        dashboard = results.get("dashboard") if isinstance(results, Mapping) else None
        if isinstance(dashboard, Mapping):
            snapshot = dashboard.get("assumptions_snapshot")
            if isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
                snapshot.to_excel(writer, sheet_name="Summary", index=False)
            overview = dashboard.get("overview_metrics")
            if isinstance(overview, pd.DataFrame) and not overview.empty:
                overview.to_excel(writer, sheet_name="Metrics", index=False)
            annual_prod = dashboard.get("annual_production")
            if isinstance(annual_prod, pd.DataFrame) and not annual_prod.empty:
                annual_prod.to_excel(writer, sheet_name="Production", index=False)

        statements_annual = results.get("statements_annual") if isinstance(results, Mapping) else None
        if isinstance(statements_annual, Mapping):
            for key in ("pnl", "cashflow", "balancesheet"):
                df = statements_annual.get(key)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    sheet = f"Annual_{key}"
                    df.to_excel(writer, sheet_name=sheet, index=False)

        capex_df = results.get("capex") if isinstance(results, Mapping) else None
        if isinstance(capex_df, pd.DataFrame) and not capex_df.empty:
            capex_df.to_excel(writer, sheet_name="CAPEX", index=False)

        debt_df = results.get("debt_schedule") if isinstance(results, Mapping) else None
        if isinstance(debt_df, pd.DataFrame) and not debt_df.empty:
            debt_df.to_excel(writer, sheet_name="Debt", index=False)

        wc_df = results.get("working_capital") if isinstance(results, Mapping) else None
        if isinstance(wc_df, pd.DataFrame) and not wc_df.empty:
            wc_df.to_excel(writer, sheet_name="WorkingCapital", index=False)

        meta_rows: List[Dict[str, object]] = []
        global_inputs = cfg.get("global_inputs") if isinstance(cfg, Mapping) else {}
        if isinstance(global_inputs, Mapping):
            meta_rows.append(
                {
                    "Scenario": scenario_name,
                    "Discount rate": global_inputs.get("discount_rate"),
                    "Corporate tax": global_inputs.get("corp_tax_rate"),
                    "Investor share": global_inputs.get("investor_share"),
                    "Owner share": global_inputs.get("owner_share"),
                }
            )
        if meta_rows:
            pd.DataFrame(meta_rows).to_excel(writer, sheet_name="Scenario", index=False)

    buffer.seek(0)
    return buffer.read()

def _get_risk_select_options(tables: "InputTables") -> Tuple[List[str], List[str]]:
    """Return dropdown option sets for risk targets and scope fields."""

    target_set = set(RISK_TARGET_OPTIONS)
    applies_set = set(RISK_APPLIES_OPTIONS)

    default_risk = DEFAULTS.get("risk_params")
    sources: List[pd.DataFrame] = []
    if isinstance(default_risk, pd.DataFrame) and not default_risk.empty:
        sources.append(default_risk)
    try:
        active_risk = tables.ensure_table("risk_params")
    except Exception:
        active_risk = pd.DataFrame()
    if isinstance(active_risk, pd.DataFrame) and not active_risk.empty:
        sources.append(active_risk)

    for source in sources:
        if "target" in source:
            target_set.update(str(val).lower() for val in source["target"].dropna().unique())
        if "applies_to" in source:
            applies_set.update(str(val).lower() for val in source["applies_to"].dropna().unique())

    target_options = sorted(option for option in target_set if option)
    applies_options = sorted(option for option in applies_set if option)
    return target_options, applies_options


def _render_risk_schedule_preview(tables: "InputTables") -> None:
    """Display the normalised risk schedule and consolidated multipliers."""

    risk_df = tables.ensure_table("risk_params").copy()
    risk_df = risk_df.replace({"": np.nan}).dropna(how="all")
    if risk_df.empty:
        st.info("Add at least one risk driver above to populate the active schedule.")
        return

    risk_df = risk_df.reset_index(drop=True)
    _render_dataframe(risk_df, "Active risk schedule", key="risk_schedule_preview")

    profile = compute_risk_profile(risk_df)
    if not isinstance(profile, dict) or not profile:
        return

    summary_rows: List[Dict[str, object]] = []
    profile_labels = [
        ("production", "Production multiplier"),
        ("labour", "Labour multiplier"),
        ("price", "Price multiplier"),
        ("revenue", "Revenue multiplier"),
        ("yield", "Yield multiplier"),
    ]
    for key_name, label in profile_labels:
        value = profile.get(key_name)
        if value is None:
            continue
        try:
            summary_rows.append({"Metric": label, "Multiplier": float(value)})
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            continue
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        _render_dataframe(summary_df, "Consolidated risk multipliers", key="risk_profile_summary")

    price_map = profile.get("price_by_product")
    if isinstance(price_map, dict) and price_map:
        price_rows: List[Dict[str, object]] = []
        for product, value in price_map.items():
            try:
                multiplier = float(value)
            except (TypeError, ValueError):
                continue
            product_label = str(product).replace("_", " ").title()
            price_rows.append({"Product": product_label, "Multiplier": multiplier})
        if price_rows:
            price_df = pd.DataFrame(price_rows)
            _render_dataframe(price_df, "Price multipliers by product", key="risk_profile_prices")


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
                    "unit_price": 65.0,
                    "quantity": 8_000.0,
                    "amount": 520_000.0,
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
                    "gross_pay_per_head": 3_000.0,
                    "benefits_per_head": 500.0,
                    "training_per_head": 100.0,
                    "other_per_head": 200.0,
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
        frames["tornado_drivers"] = DEFAULTS["tornado_drivers"].copy()
        frames["monte_carlo_settings"] = DEFAULTS["monte_carlo_settings"].copy()
        frames["scenario_comparison"] = DEFAULTS["scenario_comparison"].copy()
        frames["optimizer_settings"] = DEFAULTS["optimizer_settings"].copy()
        frames["neural_forecast_settings"] = DEFAULTS["neural_forecast_settings"].copy()
        frames["statistical_forecast_settings"] = DEFAULTS["statistical_forecast_settings"].copy()
        frames["decision_tree_paths"] = DEFAULTS["decision_tree_paths"].copy()

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


def _render_default_row_controls(
    table_name: str,
    label: str,
    schema,
    default_df: pd.DataFrame,
) -> None:
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


def _render_default_edit_modal(table_name: str, label: str, schema, tables: "InputTables") -> None:
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
    risk_select_cache: Optional[Tuple[List[str], List[str]]] = None

    def _field_options(column_name: str) -> Optional[List[str]]:
        nonlocal risk_select_cache
        col = column_name.lower()
        if table_name == "risk_params":
            if col == "distribution":
                return list(RISK_DISTRIBUTION_OPTIONS)
            if col in {"target", "applies_to"}:
                if risk_select_cache is None:
                    risk_select_cache = _get_risk_select_options(tables)
                targets, applies = risk_select_cache
                return list(targets if col == "target" else applies)
        elif table_name == "monte_carlo_settings":
            if col == "distribution":
                return list(MONTE_CARLO_DISTRIBUTIONS)
            if col == "variable":
                return list(MONTE_CARLO_VARIABLE_LABELS.values())
            if col == "applies_to":
                return ["global", *PRODUCTS]
        return None

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
                    value=base_value,
                    step=1,
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
                options = _field_options(column_name)
                if options:
                    default_option = schema.defaults.get(column_name)
                    if table_name == "monte_carlo_settings" and column_name.lower() == "variable":
                        default_label = MONTE_CARLO_VARIABLE_LABELS.get(str(default_option), options[0])
                        current_option = MONTE_CARLO_VARIABLE_LABELS.get(
                            str(current_value), default_label
                        )
                    else:
                        if default_option is None or (
                            isinstance(default_option, float) and math.isnan(default_option)
                        ):
                            default_option = options[0] if options else ""
                        if current_value is None or (
                            isinstance(current_value, float) and math.isnan(current_value)
                        ):
                            current_option = str(default_option)
                        else:
                            current_option = str(current_value)
                    index = _option_index(options, current_option, default=0)
                    input_value = form.selectbox(
                        field_label,
                        options,
                        index=index,
                        key=f"default_edit_field_{table_name}_{row_index}_{column_name}",
                    )
                    if table_name == "monte_carlo_settings" and column_name.lower() == "variable":
                        updated_values[column_name] = MONTE_CARLO_VARIABLE_LABEL_TO_KEY.get(
                            str(input_value), str(input_value)
                        )
                    else:
                        updated_values[column_name] = str(input_value)
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
            # Apply the refreshed defaults to the live table so downstream
            # schedules pick up the edited baseline immediately.
            try:
                refreshed_defaults = _get_default_table(table_name)
                tables.set_table(table_name, refreshed_defaults)
            except Exception as exc:
                st.error(f"Default saved but unable to update table: {exc}")
            else:
                _update_editor_state(table_name, tables)
                st.session_state.pop(DEFAULT_EDIT_STATE_KEY, None)
                st.session_state[f"default_feedback_{table_name}"] = "Default row updated and applied."
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
]


HORIZON_SYNC_TABLES: Tuple[str, ...] = (
    "capex_lines",
    "production_monthly",
    "direct_costs_monthly",
    "staff_costs_monthly",
    "other_opex_monthly",
    "ar_other_assets",
    "inventory_ap",
    "inflation_index",
)


YEARLY_INCREMENT_CONFIG = {
    "production_annual": {
        "kind": "production",
        "columns": {
            "annual_volume": "Annual volume annual change (%)",
        },
    },
    "direct_costs_monthly": {
        "kind": "monthly",
        "columns": {
            "unit_price": "Unit price annual change (%)",
            "quantity": "Quantity annual change (%)",
        },
        "reset_amount": True,
    },
    "staff_costs_monthly": {
        "kind": "monthly",
        "columns": {
            "headcount": "Headcount annual change (%)",
            "gross_pay": "Gross pay annual change (%)",
            "benefits": "Benefits annual change (%)",
            "training": "Training annual change (%)",
            "other": "Other staff cost annual change (%)",
        },
        "per_head_map": {
            "gross_pay": "gross_pay_per_head",
            "benefits": "benefits_per_head",
            "training": "training_per_head",
            "other": "other_per_head",
        },
    },
    "other_opex_monthly": {
        "kind": "monthly",
        "columns": {
            "amount": "Other opex annual change (%)",
        },
    },
    "inflation_index": {
        "kind": "monthly",
        "columns": {
            "cpi": "CPI annual change (%)",
            "fx_index": "FX index annual change (%)",
        },
    },
}


def _sync_tables_to_horizon(tables: InputTables, cfg: Dict[str, object]) -> None:
    """Update time-indexed tables in the UI after horizon edits."""

    errors: List[str] = []
    for table in HORIZON_SYNC_TABLES:
        df = cfg.get(table)
        if not isinstance(df, pd.DataFrame):
            continue
        try:
            tables.set_table(table, df.copy())
        except Exception as exc:  # pragma: no cover - UI validation feedback
            errors.append(f"{table}: {exc}")
        else:
            _update_editor_state(table, tables)
            cfg[table] = tables.ensure_table(table).copy()
    for message in errors:
        st.warning(f"Unable to align {message}")


def _auto_step(value: float) -> float:
    magnitude = abs(float(value))
    if magnitude == 0:
        return 1.0
    step = 10 ** math.floor(math.log10(magnitude))
    return max(step * 0.1, 0.01)


def _infer_date_format(template: object) -> str:
    if isinstance(template, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", template):
            return "%Y-%m-%d"
        if re.fullmatch(r"\d{4}-\d{2}", template):
            return "%Y-%m"
    return "%Y-%m-%d"


def _format_date_like(date_value: pd.Timestamp, template: object) -> str:
    fmt = _infer_date_format(template)
    return date_value.strftime(fmt)


def _growth_factor(percent: float, offset: int, mode: str) -> float:
    if offset <= 0:
        return 1.0
    pct = float(percent or 0.0)
    if mode == "increase":
        pct = abs(pct)
    elif mode == "decrease":
        pct = -abs(pct)
    elif mode == "copy":
        pct = 0.0
    base = 1.0 + pct / 100.0
    if base <= 0:
        return 0.0
    return base ** offset


def _apply_yearly_increment_monthly(
    df: pd.DataFrame,
    value_columns: Iterable[str],
    percent_map: Dict[str, float],
    timeline: Timeline,
    base_year: int,
    mode: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Add at least one row for the selected table before applying yearly increments.")
    if "date" not in df.columns:
        raise ValueError("The selected table does not include a 'date' column.")

    working = df.copy()
    working["__date_ts"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["__date_ts"])  # remove rows without valid dates
    if working.empty:
        raise ValueError("No valid dates detected in the current table.")

    base_rows = working[working["__date_ts"].dt.year == int(base_year)].copy()
    if base_rows.empty:
        raise ValueError("No rows found for the selected base year. Update the table and try again.")

    for column in value_columns:
        if column in base_rows.columns:
            base_rows[column] = pd.to_numeric(base_rows[column], errors="coerce").fillna(0.0)

    template_sample = base_rows.iloc[0]["date"]
    years = [year for year in timeline.annual_index() if year >= int(base_year)]
    if not years:
        raise ValueError("Projection horizon does not extend beyond the selected base year.")

    generated_rows: List[Dict[str, object]] = []
    for year in years:
        offset = year - int(base_year)
        for _, row in base_rows.iterrows():
            new_row = row.copy()
            new_date = row["__date_ts"].replace(year=int(year))
            new_row["date"] = _format_date_like(new_date, template_sample)
            for column in value_columns:
                if column not in new_row:
                    continue
                base_value = float(row.get(column, 0.0) or 0.0)
                factor = _growth_factor(percent_map.get(column, 0.0), offset, mode)
                new_row[column] = base_value * factor
            generated_rows.append(new_row)

    result = pd.DataFrame(generated_rows)
    if result.empty:
        raise ValueError("No rows generated from the yearly increment helper.")

    result = result.drop(columns=["__date_ts"], errors="ignore")
    result["__sort"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values(["__sort"] + [col for col in base_rows.columns if col not in {"__date_ts", "date"}])
    result = result.drop(columns=["__sort"], errors="ignore").reset_index(drop=True)

    # Reorder columns to match the original DataFrame structure
    result = result.reindex(columns=list(df.columns), fill_value=np.nan)
    return result


def _apply_yearly_increment_production(
    df: pd.DataFrame,
    base_values: Dict[str, float],
    percent_map: Dict[str, float],
    timeline: Timeline,
    production_horizon: Mapping[str, object],
    mode: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Add at least one product row before applying yearly increments.")

    years = timeline.annual_index()
    if not years:
        raise ValueError("Projection horizon is not defined.")

    prod_start = max(int(production_horizon.get("start_year", years[0])), years[0])
    prod_end = min(int(production_horizon.get("end_year", years[-1])), years[-1])
    if prod_end < prod_start:
        raise ValueError("Production horizon is not aligned with the projection horizon.")

    updated = df.copy()
    for idx, row in updated.iterrows():
        product = str(row.get("product", "")).strip()
        if not product:
            continue
        base_volume = float(base_values.get(product, row.get("annual_volume", 0.0)) or 0.0)
        base_volume = max(base_volume, 0.0)
        percent = float(percent_map.get(product, 0.0) or 0.0)

        volumes_by_year: List[float] = []
        for year in years:
            if year < prod_start or year > prod_end:
                volumes_by_year.append(0.0)
                continue
            offset = year - prod_start
            factor = _growth_factor(percent, offset, mode)
            volumes_by_year.append(base_volume * factor)

        if base_volume <= 0:
            ramp_values = [0.0 for _ in years]
        else:
            ramp_values = []
            for year, volume in zip(years, volumes_by_year):
                if year < prod_start or year > prod_end:
                    ramp_values.append(0.0)
                else:
                    ramp_values.append(volume / base_volume if base_volume else 0.0)

        ramp_str = ";".join(f"{value:.6f}" for value in ramp_values)
        updated.at[idx, "annual_volume"] = base_volume
        updated.at[idx, "startup_ramp"] = ramp_str

    return updated


def _render_yearly_increment_helper(
    tables: InputTables,
    table_name: str,
    df: pd.DataFrame,
    timeline: Timeline,
    production_horizon: Mapping[str, object],
) -> None:
    config = YEARLY_INCREMENT_CONFIG.get(table_name)
    if not config:
        return

    helper_key = f"{table_name}_yearly_helper"
    with st.expander("Yearly increment helper", expanded=False):
        st.caption(
            "Fill the first year's values and specify the annual percentage change to populate future years automatically."
        )

        if config["kind"] == "production":
            if df.empty:
                st.info("Add production rows before applying yearly changes.")
                return

            years = timeline.annual_index()
            if not years:
                st.warning("Projection horizon years are not available.")
                return

            prod_start = max(int(production_horizon.get("start_year", years[0])), years[0])
            prod_end = min(int(production_horizon.get("end_year", years[-1])), years[-1])
            st.write(
                f"First production year: **{prod_start}**. Values outside {prod_start}–{prod_end} are set to zero."
            )

            base_values: Dict[str, float] = {}
            percent_map: Dict[str, float] = {}
            controls = []
            for _, row in df.iterrows():
                product = str(row.get("product", "")).strip()
                if not product:
                    continue
                ramp = parse_ramp(row.get("startup_ramp"), len(years))
                base_index = years.index(prod_start) if prod_start in years else 0
                base_actual = float(row.get("annual_volume", 0.0) or 0.0)
                if ramp:
                    base_actual *= float(ramp[base_index])
                base_actual = max(base_actual, 0.0)

                base_key = f"{helper_key}_{product}_base"
                change_key = f"{helper_key}_{product}_pct"

                base_input = st.number_input(
                    f"{product.replace('_', ' ').title()} first production year volume",
                    min_value=0.0,
                    value=base_actual,
                    step=_auto_step(base_actual),
                    format="%.4f",
                    key=base_key,
                )
                pct_input = st.number_input(
                    f"{product.replace('_', ' ').title()} annual change (%)",
                    value=0.0,
                    step=0.1,
                    format="%.4f",
                    key=change_key,
                )
                base_values[product] = float(base_input)
                percent_map[product] = float(pct_input)
                controls.append(product)

            if not controls:
                st.info("No product rows detected. Add products to the Production Volumes table first.")
                return

            action_cols = st.columns(3)
            copy_clicked = action_cols[0].button("Copy forward", key=f"{helper_key}_copy")
            inc_clicked = action_cols[1].button("Apply increases", key=f"{helper_key}_inc")
            dec_clicked = action_cols[2].button("Apply decreases", key=f"{helper_key}_dec")

            mode = None
            if copy_clicked:
                mode = "copy"
            elif inc_clicked:
                mode = "increase"
            elif dec_clicked:
                mode = "decrease"

            if mode:
                try:
                    updated_df = _apply_yearly_increment_production(
                        df,
                        base_values,
                        percent_map,
                        timeline,
                        production_horizon,
                        mode,
                    )
                except ValueError as exc:
                    st.error(f"Unable to apply yearly increments: {exc}")
                else:
                    try:
                        tables.set_table(table_name, updated_df)
                    except Exception as exc:
                        st.error(f"Unable to update production table: {exc}")
                    else:
                        st.success("Production ramp updated from yearly changes.")
                        _update_editor_state(table_name, tables)
                        _safe_rerun()

        else:  # monthly-style tables
            if df.empty:
                st.info("Add at least one row before applying yearly changes.")
                return

            years = timeline.annual_index()
            if not years:
                st.warning("Projection horizon years are not available.")
                return

            base_year = st.number_input(
                "Base year",
                min_value=int(years[0]),
                max_value=int(years[-1]),
                value=int(years[0]),
                step=1,
                key=f"{helper_key}_base_year",
            )

            percent_map = {
                column: float(
                    st.number_input(
                        label,
                        value=0.0,
                        step=0.1,
                        format="%.4f",
                        key=f"{helper_key}_{column}_pct",
                    )
                )
                for column, label in config["columns"].items()
            }

            action_cols = st.columns(3)
            copy_clicked = action_cols[0].button("Copy forward", key=f"{helper_key}_copy")
            inc_clicked = action_cols[1].button("Apply increases", key=f"{helper_key}_inc")
            dec_clicked = action_cols[2].button("Apply decreases", key=f"{helper_key}_dec")

            mode = None
            if copy_clicked:
                mode = "copy"
            elif inc_clicked:
                mode = "increase"
            elif dec_clicked:
                mode = "decrease"

            if mode:
                try:
                    updated_df = _apply_yearly_increment_monthly(
                        df,
                        config["columns"].keys(),
                        percent_map,
                        timeline,
                        int(base_year),
                        mode,
                    )
                except ValueError as exc:
                    st.error(f"Unable to apply yearly increments: {exc}")
                else:
                    if config.get("reset_amount") and "amount" in updated_df.columns:
                        updated_df["amount"] = np.nan

                    if table_name == "staff_costs_monthly":
                        headcount = pd.to_numeric(updated_df.get("headcount"), errors="coerce").fillna(0.0)
                        updated_df["headcount"] = headcount
                        per_head_map = config.get("per_head_map", {})
                        for total_col, per_col in per_head_map.items():
                            if total_col in updated_df.columns:
                                updated_df[total_col] = pd.to_numeric(
                                    updated_df[total_col], errors="coerce"
                                ).fillna(0.0)
                                if per_col in updated_df.columns:
                                    updated_df[per_col] = np.where(
                                        headcount > 0,
                                        updated_df[total_col] / headcount,
                                        0.0,
                                    )

                    try:
                        tables.set_table(table_name, updated_df)
                    except Exception as exc:
                        st.error(f"Unable to update table: {exc}")
                    else:
                        st.success("Yearly changes applied to the table.")
                        _update_editor_state(table_name, tables)
                        _safe_rerun()
def _render_table_editor(
    tables: InputTables,
    table_name: str,
    label: str,
    error_message: Optional[str] = None,
    description: Optional[str] = None,
    helper: Optional[Callable[[InputTables, str, pd.DataFrame], None]] = None,
    column_config: Optional[Dict[str, object]] = None,
) -> None:
    st.markdown(f"#### {label}")
    if description:
        st.caption(description)
    schema = INPUT_SCHEMAS[table_name]
    df = tables.ensure_table(table_name).copy()

    if helper is not None:
        helper(tables, table_name, df)
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

    display_df = df.copy()
    if table_name == "monte_carlo_settings" and "variable" in display_df.columns:
        display_df["variable"] = display_df["variable"].map(MONTE_CARLO_VARIABLE_LABELS).fillna(
            display_df["variable"].astype(str)
        )

    editor = st.data_editor(
        display_df,
        num_rows="dynamic",
        use_container_width=True,
        key=state_key,
        column_config=column_config,
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

        if table_name == "monte_carlo_settings" and "variable" in editor_clean.columns:
            editor_clean["variable"] = editor_clean["variable"].map(MONTE_CARLO_VARIABLE_LABEL_TO_KEY).fillna(
                editor_clean["variable"]
            )

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

    _render_default_edit_modal(table_name, label, schema, tables)

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
    st.session_state.pop("scenario_payload_cache", None)

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
            else:
                cfg = align_with_projection_horizon(cfg)
                _sync_tables_to_horizon(tables, cfg)

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
            risk_targets, risk_applies = _get_risk_select_options(tables)
            risk_column_config = {
                "distribution": st.column_config.SelectboxColumn(
                    "Distribution",
                    options=list(RISK_DISTRIBUTION_OPTIONS),
                    help="Probability distribution used when sampling this risk driver.",
                ),
                "target": st.column_config.SelectboxColumn(
                    "Target",
                    options=risk_targets,
                    help="Choose whether the draw scales general risk multipliers, prices, availability, opex, or capex.",
                ),
                "applies_to": st.column_config.SelectboxColumn(
                    "Applies to",
                    options=risk_applies,
                    help="Restrict the driver to a specific product/segment or leave as global.",
                ),
                "p1": st.column_config.NumberColumn("P1", help="Distribution parameter 1 (e.g., mean or minimum)."),
                "p2": st.column_config.NumberColumn("P2", help="Distribution parameter 2 (e.g., stdev or mode)."),
                "p3": st.column_config.NumberColumn("P3", help="Distribution parameter 3 (e.g., triangular maximum)."),
                "production_multiplier": st.column_config.NumberColumn(
                    "Production multiplier",
                    min_value=0.0,
                    help="Baseline production multiplier applied before any sampled draw.",
                ),
                "labour_multiplier": st.column_config.NumberColumn(
                    "Labour multiplier",
                    min_value=0.0,
                    help="Baseline labour cost multiplier applied before any sampled draw.",
                ),
                "price_multiplier": st.column_config.NumberColumn(
                    "Price multiplier",
                    min_value=0.0,
                    help="Baseline price multiplier applied before any sampled draw.",
                ),
                "revenue_multiplier": st.column_config.NumberColumn(
                    "Revenue multiplier",
                    min_value=0.0,
                    help="Baseline revenue multiplier applied before any sampled draw.",
                ),
                "yield_multiplier": st.column_config.NumberColumn(
                    "Yield multiplier",
                    min_value=0.0,
                    help="Baseline yield multiplier applied before any sampled draw.",
                ),
            }
            _render_table_editor(
                tables,
                "risk_params",
                "Risk Schedule",
                sync_errors.get("risk_params"),
                "Political, environmental, and market risk multipliers for production, pricing, and labour assumptions.",
                column_config=risk_column_config,
            )
            _render_risk_schedule_preview(tables)

    timeline = Timeline(
        int(horizon["start_year"]),
        int(horizon["end_year"]),
        int(horizon.get("start_month", 1)),
    )

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

    scenario_table_for_download = tables.ensure_table("scenario_comparison").copy()
    scenario_overrides = _scenario_overrides_from_table(scenario_table_for_download)
    scenario_options: List[str] = [BASE_SCENARIO_LABEL, *list(scenario_overrides.keys())]


    with landing_tab:
        top_left, top_right = st.columns([3, 2])
        with top_left:
            st.subheader("Input & assumptions tables")
            st.markdown(
                "Review, add, or remove records from each canonical input table. Updates apply across the model "
                "on the next run."
            )
        with top_right:
            st.markdown("#### Excel model download")
            if not scenario_options:
                scenario_options = [BASE_SCENARIO_LABEL]
            default_option = st.session_state.get("excel_download_selected", scenario_options[0])
            if default_option not in scenario_options:
                default_option = scenario_options[0]
            selected_scenario = st.selectbox(
                "Scenario",
                scenario_options,
                index=scenario_options.index(default_option),
                key="excel_download_scenario",
            )
            st.session_state["excel_download_selected"] = selected_scenario

            download_container = st.container()
            excel_map: Dict[str, bytes] = st.session_state.setdefault("excel_bytes_map", {})
            stale_keys = [key for key in excel_map if key not in scenario_options]
            for key in stale_keys:
                excel_map.pop(key, None)
            st.session_state.excel_bytes_map = excel_map

            scenario_cfg_payload, scenario_results_payload = _ensure_scenario_payload(
                selected_scenario,
                cfg,
                results,
                scenario_overrides,
            )
            cfg_for_excel = copy.deepcopy(scenario_cfg_payload)
            metadata = cfg_for_excel.setdefault("metadata", {}) if isinstance(cfg_for_excel, dict) else {}
            if isinstance(metadata, dict):
                metadata["scenario"] = selected_scenario
            st.session_state.model_results = (cfg_for_excel, scenario_results_payload)

            excel_bytes = excel_map.get(selected_scenario)

            with download_container:
                if not excel_bytes:
                    if st.button(
                        "Prepare Excel Model",
                        key=f"prepare_excel_{normalize_key(selected_scenario) or 'base'}",
                    ):
                        with st.spinner("Preparing Excel workbook..."):
                            excel_bytes = _generate_excel_bytes(
                                cfg_for_excel,
                                scenario_results_payload,
                                selected_scenario,
                            )
                        excel_map[selected_scenario] = excel_bytes
                        st.session_state.excel_bytes_map = excel_map
                if excel_bytes:
                    file_scenario = normalize_key(selected_scenario) or "base"
                    st.download_button(
                        "Download Excel Model",
                        data=excel_bytes,
                        file_name=f"Sugarcane_Financial_Model_{file_scenario}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_excel_{file_scenario}",
                    )
                    if st.button(
                        "Clear Prepared Excel",
                        key=f"clear_excel_{normalize_key(selected_scenario) or 'base'}",
                    ):
                        excel_map.pop(selected_scenario, None)
                        st.session_state.excel_bytes_map = excel_map
                        excel_bytes = None
                if not excel_bytes:
                    st.info("Click 'Prepare Excel Model' to generate the workbook for download.")

        if sync_errors:
            for table_name, message in sync_errors.items():
                st.error(f"{table_name}: {message}")
        for label, table_name, description in LANDING_TABLES:
            helper = None
            if table_name in YEARLY_INCREMENT_CONFIG:
                helper = lambda tbls, _name, df, tn=table_name: _render_yearly_increment_helper(
                    tbls,
                    tn,
                    df,
                    timeline,
                    production_horizon,
                )
            _render_table_editor(
                tables,
                table_name,
                label,
                sync_errors.get(table_name),
                description,
                helper=helper,
            )

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

        staff_monthly = results.get("staff_costs_detail")
        if isinstance(staff_monthly, pd.DataFrame) and not staff_monthly.empty:
            staff_monthly_view = staff_monthly.copy()
            staff_monthly_view["date"] = pd.to_datetime(staff_monthly_view["date"])
            staff_monthly_view = staff_monthly_view.sort_values(["date", "dept"]).reset_index(drop=True)
            _render_dataframe(
                staff_monthly_view,
                "Staff operating cost breakdown (monthly)",
                key="staff_costs_detail_monthly",
            )

        staff_annual = results.get("staff_costs_detail_annual")
        if isinstance(staff_annual, pd.DataFrame) and not staff_annual.empty:
            sort_cols = ["year"]
            if "dept" in staff_annual.columns:
                sort_cols.append("dept")
            if "currency" in staff_annual.columns and "currency" not in sort_cols:
                sort_cols.append("currency")
            staff_annual_view = staff_annual.sort_values(sort_cols).reset_index(drop=True)
            _render_dataframe(
                staff_annual_view,
                "Staff operating cost breakdown (annual)",
                key="staff_costs_detail_annual",
            )

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
        st.subheader("Advanced sensitivity analytics")
        sensitivity_sections = st.tabs(
            [
                "Metaheuristic optimiser",
                "Neural forecasts",
                "Statistical forecasts",
                "Decision tree",
            ]
        )

        with sensitivity_sections[0]:
            optimizer_column_config = {
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "variable": st.column_config.SelectboxColumn(
                    "Variable",
                    options=list(OPTIMIZER_VARIABLE_OPTIONS.keys()),
                    format_func=lambda key: OPTIMIZER_VARIABLE_OPTIONS.get(key, key.replace("_", " ").title()),
                ),
                "lower_bound": st.column_config.NumberColumn("Lower bound"),
                "upper_bound": st.column_config.NumberColumn("Upper bound"),
                "notes": st.column_config.TextColumn("Notes"),
            }
            _render_table_editor(
                tables,
                "optimizer_settings",
                "Optimizer variables",
                sync_errors.get("optimizer_settings"),
                "Define variable bounds for the metaheuristic optimiser.",
                column_config=optimizer_column_config,
            )

            optimizer_table = tables.ensure_table("optimizer_settings").copy()
            enabled_mask = optimizer_table.get("enabled", True)
            if not isinstance(enabled_mask, pd.Series):
                enabled_mask = pd.Series(True, index=optimizer_table.index)
            optimizer_active = optimizer_table[enabled_mask.fillna(True)]
            if optimizer_active.empty:
                st.info("Add at least one enabled variable to run the optimiser.")
            else:
                objective_options = list(metrics.keys())
                default_objective = "Project_NPV" if "Project_NPV" in objective_options else objective_options[0]
                selected_objective = st.selectbox(
                    "Objective metric",
                    objective_options,
                    index=objective_options.index(default_objective),
                )
                iterations = int(
                    st.number_input("Iterations", min_value=1, value=10, step=1, key="optimizer_iterations")
                )
                population = int(
                    st.number_input("Population size", min_value=2, value=6, step=1, key="optimizer_population")
                )
                seed_value = st.number_input("Random seed (optional)", value=42, step=1, key="optimizer_seed")
                if st.button("Run metaheuristic optimisation", key="run_optimizer"):
                    with st.spinner("Running optimisation across variable bounds..."):
                        optimiser_results = metaheuristic_optimize(
                            cfg,
                            metrics,
                            lambda c: run_full_model(c),
                            optimizer_active,
                            selected_objective,
                            iterations=iterations,
                            population=population,
                            seed=int(seed_value),
                        )
                    if optimiser_results.empty:
                        st.info("Optimiser did not produce any results with the current configuration.")
                    else:
                        _render_dataframe(
                            optimiser_results,
                            f"Optimiser results – {selected_objective}",
                            key="optimizer_results",
                        )
                        best_row = optimiser_results.sort_values("objective", ascending=False).iloc[0]
                        metric_kind = _metric_kind(selected_objective)
                        st.metric(
                            f"Best {selected_objective}",
                            _format_metric(best_row.get("objective"), metric_kind),
                            delta=_format_metric(best_row.get("delta_vs_base"), metric_kind),
                        )

        with sensitivity_sections[1]:
            neural_column_config = {
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "product": st.column_config.SelectboxColumn(
                    "Product",
                    options=list(PRODUCTS),
                    format_func=lambda key: key.replace("_", " ").title(),
                ),
                "lookback_months": st.column_config.NumberColumn("Lookback (months)", min_value=3, step=1),
                "forecast_months": st.column_config.NumberColumn("Forecast horizon (months)", min_value=1, step=1),
                "hidden_units": st.column_config.NumberColumn("Hidden units", min_value=1, step=1),
                "learning_rate": st.column_config.NumberColumn("Learning rate", min_value=0.0001, step=0.0001, format="%.4f"),
                "epochs": st.column_config.NumberColumn("Epochs", min_value=50, step=50),
            }
            _render_table_editor(
                tables,
                "neural_forecast_settings",
                "Neural forecasting settings",
                sync_errors.get("neural_forecast_settings"),
                "Configure neural network parameters to forecast production volumes.",
                column_config=neural_column_config,
            )

            neural_cfg = tables.ensure_table("neural_forecast_settings").copy()
            enabled_mask = neural_cfg.get("enabled", True)
            if not isinstance(enabled_mask, pd.Series):
                enabled_mask = pd.Series(True, index=neural_cfg.index)
            neural_active = neural_cfg[enabled_mask.fillna(True)]
            if neural_active.empty:
                st.info("Enable at least one neural forecast configuration to generate projections.")
            elif st.button("Run neural forecasts", key="run_neural"):
                for idx, row in neural_active.iterrows():
                    product_key = str(row.get("product", "ethanol") or "ethanol")
                    lookback_val = int(row.get("lookback_months", 12) or 12)
                    horizon_val = int(row.get("forecast_months", 12) or 12)
                    hidden_units = int(row.get("hidden_units", 8) or 8)
                    learning_rate = float(row.get("learning_rate", 0.01) or 0.01)
                    epochs_val = int(row.get("epochs", 300) or 300)
                    with st.spinner(f"Forecasting {product_key.title()} volumes..."):
                        forecast_df = neural_forecast_production(
                            results,
                            product_key,
                            lookback=lookback_val,
                            horizon=horizon_val,
                            hidden_units=hidden_units,
                            learning_rate=learning_rate,
                            epochs=epochs_val,
                            seed=int(idx + 1),
                        )
                    if forecast_df.empty:
                        st.warning(f"No forecast generated for {product_key.title()} (insufficient data).")
                    else:
                        _render_dataframe(
                            forecast_df,
                            f"Neural forecast – {product_key.title()}",
                            key=f"neural_{idx}",
                        )

        with sensitivity_sections[2]:
            stat_column_config = {
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "series": st.column_config.SelectboxColumn(
                    "Series",
                    options=list(STATISTICAL_SERIES_OPTIONS.keys()),
                    format_func=lambda key: STATISTICAL_SERIES_OPTIONS.get(key, key.replace("_", " ").title()),
                ),
                "alpha": st.column_config.NumberColumn("Alpha", min_value=0.01, max_value=1.0, format="%.2f"),
                "forecast_months": st.column_config.NumberColumn("Forecast horizon (months)", min_value=1, step=1),
            }
            _render_table_editor(
                tables,
                "statistical_forecast_settings",
                "Statistical forecast settings",
                sync_errors.get("statistical_forecast_settings"),
                "Set exponential-smoothing parameters for demand and cost forecasting.",
                column_config=stat_column_config,
            )

            stat_cfg = tables.ensure_table("statistical_forecast_settings").copy()
            enabled_mask = stat_cfg.get("enabled", True)
            if not isinstance(enabled_mask, pd.Series):
                enabled_mask = pd.Series(True, index=stat_cfg.index)
            stat_active = stat_cfg[enabled_mask.fillna(True)]
            if stat_active.empty:
                st.info("Enable at least one statistical series to forecast.")
            elif st.button("Run statistical forecasts", key="run_statistical"):
                for idx, row in stat_active.iterrows():
                    series_key = str(row.get("series", "revenue") or "revenue")
                    alpha_val = float(row.get("alpha", 0.3) or 0.3)
                    horizon_val = int(row.get("forecast_months", 12) or 12)
                    with st.spinner(f"Forecasting {STATISTICAL_SERIES_OPTIONS.get(series_key, series_key)}..."):
                        forecast_result = statistical_forecast(
                            results,
                            series_key,
                            alpha=alpha_val,
                            horizon=horizon_val,
                        )
                    historical_df = forecast_result.get("historical", pd.DataFrame())
                    forecast_df = forecast_result.get("forecast", pd.DataFrame())
                    if historical_df.empty and forecast_df.empty:
                        st.warning(f"No data available for {STATISTICAL_SERIES_OPTIONS.get(series_key, series_key)}.")
                        continue
                    if not historical_df.empty:
                        _render_dataframe(
                            historical_df,
                            f"Historical series – {STATISTICAL_SERIES_OPTIONS.get(series_key, series_key)}",
                            key=f"stat_hist_{idx}",
                        )
                    if not forecast_df.empty:
                        _render_dataframe(
                            forecast_df,
                            f"Forecast – {STATISTICAL_SERIES_OPTIONS.get(series_key, series_key)}",
                            key=f"stat_forecast_{idx}",
                        )
                    st.caption(
                        f"Residual standard deviation: {forecast_result.get('residual_std', 0.0):.2f} · "
                        f"Equipment failure risk proxy: {forecast_result.get('equipment_failure_risk', 0.0):.2%}"
                    )

        with sensitivity_sections[3]:
            decision_column_config = {
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "path_name": st.column_config.TextColumn("Path name"),
                "probability": st.column_config.NumberColumn("Probability", min_value=0.0, max_value=1.0, format="%.2f"),
                "ethanol_price_multiplier": st.column_config.NumberColumn("Ethanol price multiplier", format="%.3f"),
                "sugar_price_multiplier": st.column_config.NumberColumn("Sugar price multiplier", format="%.3f"),
                "electricity_price_multiplier": st.column_config.NumberColumn("Electricity price multiplier", format="%.3f"),
                "animal_feed_price_multiplier": st.column_config.NumberColumn("Animal feed price multiplier", format="%.3f"),
                "capex_multiplier": st.column_config.NumberColumn("CAPEX multiplier", format="%.3f"),
                "opex_multiplier": st.column_config.NumberColumn("Opex multiplier", format="%.3f"),
                "debt_rate_shift": st.column_config.NumberColumn("Debt rate shift", format="%.4f"),
                "notes": st.column_config.TextColumn("Notes"),
            }
            _render_table_editor(
                tables,
                "decision_tree_paths",
                "Decision tree paths",
                sync_errors.get("decision_tree_paths"),
                "Define scenario branches with probabilities and multipliers for pricing, CAPEX, and OPEX.",
                column_config=decision_column_config,
            )

            decision_table = tables.ensure_table("decision_tree_paths").copy()
            if decision_table.empty:
                st.info("Add decision paths to evaluate expected outcomes.")
            else:
                objective_options = list(metrics.keys())
                default_objective = "Project_NPV" if "Project_NPV" in objective_options else objective_options[0]
                selected_objective = st.selectbox(
                    "Objective metric",
                    objective_options,
                    index=objective_options.index(default_objective),
                    key="decision_objective",
                )
                if st.button("Evaluate decision tree", key="run_decision_tree"):
                    with st.spinner("Evaluating decision tree paths..."):
                        decision_result = decision_tree_analysis(
                            cfg,
                            lambda c: run_full_model(c),
                            decision_table,
                            objective=selected_objective,
                        )
                    paths_df = decision_result.get("paths", pd.DataFrame())
                    if paths_df.empty:
                        st.info("No active decision paths with valid probabilities were found.")
                    else:
                        _render_dataframe(paths_df, "Decision tree evaluation", key="decision_tree_results")
                        metric_kind = _metric_kind(selected_objective)
                        st.metric(
                            f"Expected {selected_objective}",
                            _format_metric(decision_result.get("expected_metric"), metric_kind),
                        )
                        st.caption(
                            f"Total probability weight: {decision_result.get('total_probability', 0.0):.2f}"
                        )


    with scenario_tab:
        st.markdown("### Scenario analytics workspace")
        scenario_sections = st.tabs([
            "Sensitivity tornado",
            "Monte Carlo simulation",
            "Scenario comparison",
        ])

        with scenario_sections[0]:
            _render_table_editor(
                tables,
                "tornado_drivers",
                "Tornado drivers",
                sync_errors.get("tornado_drivers"),
                "Enable drivers and adjust percentage shocks to analyse NPV sensitivity.",
            )
            tornado_df_cfg = tables.ensure_table("tornado_drivers").copy()
            if tornado_df_cfg.empty:
                st.info("Add at least one driver row to evaluate the tornado chart.")
            else:
                enabled_mask = tornado_df_cfg.get("enabled", True)
                if not isinstance(enabled_mask, pd.Series):
                    enabled_mask = pd.Series(True, index=tornado_df_cfg.index)
                enabled_rows = tornado_df_cfg[enabled_mask.fillna(True)]
                enabled_rows = enabled_rows.replace({"": np.nan})
                enabled_rows = enabled_rows.dropna(subset=["driver", "pct_change"], how="any")
                drivers: List[Tuple[str, float]] = []
                for _, driver_row in enabled_rows.iterrows():
                    driver_key = str(driver_row.get("driver", "")).strip()
                    if not driver_key:
                        continue
                    driver_key = normalize_key(driver_key)
                    try:
                        pct_change = float(driver_row.get("pct_change", 0.0))
                    except (TypeError, ValueError):
                        pct_change = 0.0
                    drivers.append((driver_key, pct_change))
                if not drivers:
                    st.info("Enable at least one driver with a valid percentage change to run the tornado analysis.")
                else:
                    with st.spinner("Calculating sensitivity tornado..."):
                        tornado_results = sensitivity_tornado(
                            cfg,
                            {"metrics": metrics},
                            lambda c: run_full_model(c),
                            drivers,
                        )
                    _render_dataframe(tornado_results, "Tornado sensitivity", key="tornado")

        with scenario_sections[1]:
            distribution_options = list(MONTE_CARLO_DISTRIBUTIONS)
            variable_label_map = dict(MONTE_CARLO_VARIABLE_LABELS)
            variable_options = list(variable_label_map.values())
            applies_options = ["global", *PRODUCTS]
            monte_column_config = {
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "distribution": st.column_config.SelectboxColumn(
                    "Probability distribution",
                    options=distribution_options,
                    help="Select the probability distribution used to draw this Monte Carlo driver.",
                ),
                "variable": st.column_config.SelectboxColumn(
                    "Variable",
                    options=variable_options,
                    help="Choose which driver the sampled draw should adjust during the simulation.",
                ),
                "applies_to": st.column_config.SelectboxColumn(
                    "Applies to",
                    options=applies_options,
                    help="Restrict the driver to a specific product or leave as global for portfolio-wide adjustments.",
                ),
                "p1": st.column_config.NumberColumn("P1", help="Distribution parameter 1 (mean/min)."),
                "p2": st.column_config.NumberColumn("P2", help="Distribution parameter 2 (std/mode/max)."),
                "p3": st.column_config.NumberColumn("P3", help="Distribution parameter 3 (triangular max)."),
            }
            _render_table_editor(
                tables,
                "monte_carlo_settings",
                "Monte Carlo settings",
                sync_errors.get("monte_carlo_settings"),
                "Set iterations, random seed, and enable the simulation to sample risk drivers.",
                column_config=monte_column_config,
            )
            monte_cfg = tables.ensure_table("monte_carlo_settings").copy()
            if not monte_cfg.empty:
                row_indices = list(monte_cfg.index)

                def _format_driver(idx: int) -> str:
                    row = monte_cfg.loc[idx]
                    variable_key = str(row.get("variable", "driver") or "driver")
                    variable = variable_label_map.get(variable_key, variable_key)
                    distribution = str(row.get("distribution", "normal") or "normal")
                    scope = str(row.get("applies_to", "global") or "global")
                    return f"Row {idx + 1}: {variable} · {distribution} ({scope})"

                selected_idx = st.selectbox(
                    "Select Monte Carlo driver to edit",
                    row_indices,
                    format_func=_format_driver,
                    key="monte_driver_select",
                )

                selected_row = monte_cfg.loc[selected_idx]
                with st.form(f"monte_driver_form_{selected_idx}"):
                    enabled_value = st.checkbox(
                        "Enable driver",
                        value=bool(selected_row.get("enabled", False)),
                        key=f"monte_enabled_{selected_idx}",
                    )
                    iterations_value = st.number_input(
                        "Iterations",
                        min_value=1,
                        value=int(selected_row.get("iterations", 1000)
                                  if pd.notna(selected_row.get("iterations"))
                                  else 1000),
                        step=1,
                        key=f"monte_iterations_{selected_idx}",
                    )
                    random_seed_value = st.number_input(
                        "Random seed",
                        value=int(selected_row.get("random_seed", 42)
                                  if pd.notna(selected_row.get("random_seed"))
                                  else 42),
                        step=1,
                        key=f"monte_seed_{selected_idx}",
                    )
                    distribution_value = st.selectbox(
                        "Probability distribution type",
                        distribution_options,
                        index=_option_index(
                            distribution_options,
                            str(selected_row.get("distribution", "normal") or "normal"),
                        ),
                        key=f"monte_distribution_{selected_idx}",
                    )
                    selected_key = str(selected_row.get("variable", "") or "")
                    selected_label = variable_label_map.get(selected_key, variable_options[0])
                    variable_value = st.selectbox(
                        "Variable",
                        variable_options,
                        index=_option_index(variable_options, selected_label),
                        key=f"monte_variable_{selected_idx}",
                    )
                    applies_value = st.selectbox(
                        "Applies to",
                        applies_options,
                        index=_option_index(
                            applies_options,
                            str(selected_row.get("applies_to", "global") or "global"),
                        ),
                        key=f"monte_applies_{selected_idx}",
                    )
                    p1_value = st.number_input(
                        "P1",
                        value=float(selected_row.get("p1", 0.0)
                                    if pd.notna(selected_row.get("p1"))
                                    else 0.0),
                        key=f"monte_p1_{selected_idx}",
                    )
                    p2_value = st.number_input(
                        "P2",
                        value=float(selected_row.get("p2", 0.05)
                                    if pd.notna(selected_row.get("p2"))
                                    else 0.05),
                        key=f"monte_p2_{selected_idx}",
                    )
                    p3_value = st.number_input(
                        "P3",
                        value=float(selected_row.get("p3", 0.0)
                                    if pd.notna(selected_row.get("p3"))
                                    else 0.0),
                        key=f"monte_p3_{selected_idx}",
                    )
                    submitted = st.form_submit_button("Save Monte Carlo driver")

                if submitted:
                    updated_cfg = monte_cfg.copy()
                    updated_cfg.at[selected_idx, "enabled"] = bool(enabled_value)
                    updated_cfg.at[selected_idx, "iterations"] = int(iterations_value)
                    updated_cfg.at[selected_idx, "random_seed"] = int(random_seed_value)
                    updated_cfg.at[selected_idx, "distribution"] = str(distribution_value)
                    updated_cfg.at[selected_idx, "variable"] = MONTE_CARLO_VARIABLE_LABEL_TO_KEY.get(
                        str(variable_value), str(variable_value)
                    )
                    updated_cfg.at[selected_idx, "applies_to"] = str(applies_value)
                    updated_cfg.at[selected_idx, "p1"] = float(p1_value)
                    updated_cfg.at[selected_idx, "p2"] = float(p2_value)
                    updated_cfg.at[selected_idx, "p3"] = float(p3_value)
                    tables.set_table("monte_carlo_settings", updated_cfg)
                    _update_editor_state("monte_carlo_settings", tables)
                    st.success("Monte Carlo driver updated.")
                    _safe_rerun()
            enabled_mc = pd.Series(dtype=bool)
            if not monte_cfg.empty:
                enabled_mc = monte_cfg.get("enabled", False)
                if not isinstance(enabled_mc, pd.Series):
                    enabled_mc = pd.Series(False, index=monte_cfg.index)
                enabled_mc = enabled_mc.fillna(False)
            if monte_cfg.empty or not enabled_mc.any():
                st.info("Enable a Monte Carlo row to execute the simulation.")
            else:
                active_row = monte_cfg.loc[enabled_mc].iloc[0]
                try:
                    iterations = int(float(active_row.get("iterations", 1000)))
                except (TypeError, ValueError):
                    iterations = 1000
                iterations = max(iterations, 1)
                try:
                    random_seed = int(float(active_row.get("random_seed", 42)))
                except (TypeError, ValueError):
                    random_seed = 42
                with st.spinner("Running Monte Carlo simulation..."):
                    monte_results = monte_carlo(
                        cfg,
                        lambda c: run_full_model(c),
                        iterations=iterations,
                        random_seed=random_seed,
                    )
                percentiles = (
                    monte_results["percentiles"].reset_index().rename(columns={"index": "Percentile"})
                )
                _render_dataframe(percentiles, "Monte Carlo percentiles", key="monte_percentiles")
                _render_dataframe(monte_results["samples"], "Monte Carlo samples", key="monte_samples")

        with scenario_sections[2]:
            _render_table_editor(
                tables,
                "scenario_comparison",
                "Scenario definitions",
                sync_errors.get("scenario_comparison"),
                "Toggle and edit scenario overrides (feedstock sourcing and farm share) for comparison against the base case.",
            )
            scenario_cfg = tables.ensure_table("scenario_comparison").copy()
            scenario_overrides_active = _scenario_overrides_from_table(scenario_cfg)
            if not scenario_overrides_active:
                st.info("Add scenario rows with overrides to compare against the base configuration.")
            else:
                with st.spinner("Evaluating scenarios..."):
                    scenario_df = run_scenarios(
                        cfg,
                        lambda c: run_full_model(c),
                        scenario_overrides_active,
                    )
                base_metrics = pd.DataFrame([metrics]).assign(scenario="Base")
                scenario_df = pd.concat([base_metrics, scenario_df], ignore_index=True)
                _render_dataframe(scenario_df, "Scenario comparison", key="scenarios")

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
