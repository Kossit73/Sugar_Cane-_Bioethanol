"""Sugar Cane Bioethanol Multi-Product Project Finance Model
=============================================================

Quickstart
----------
python model.py --excel "/path/to/workbook.xlsx" --export ./out_csv --excel-pack ./Finance_Pack.xlsx --preview 5

This script loads an Excel workbook, auto-detects assumptions and structured input tables, and builds a
comprehensive monthly project finance model for an integrated sugarcane complex producing bioethanol,
sugar, electricity, and animal feed. Outputs include monthly and annual financial statements, dashboard
metrics, sensitivity and scenario analytics, and optional CSV/Excel exports.

The implementation relies on :mod:`pandas` and :mod:`numpy`, with optional chart
generation via :mod:`matplotlib` when available. No external services are
required.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import importlib
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
import zipfile
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)
from xml.sax.saxutils import escape as xml_escape

from dependencies import ensure_package, get_package_error

MATPLOTLIB_IMPORT_ERROR: Optional[str]
if ensure_package("matplotlib"):
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib import failed unexpectedly
        plt = None
        MATPLOTLIB_IMPORT_ERROR = str(exc)
else:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = get_package_error("matplotlib")
import numpy as np
import pandas as pd

###############################################################################
# Section 0: Global constants and utility helpers
###############################################################################

PRODUCTS: Tuple[str, ...] = ("ethanol", "sugar", "electricity", "animal_feed")
FEEDSTOCK_SCENARIOS: Tuple[str, ...] = ("FARM_ONLY", "BUY_ONLY", "HYBRID")
MONTE_CARLO_DISTRIBUTIONS: Tuple[str, ...] = ("normal", "lognormal", "triangular", "uniform")
MONTE_CARLO_VARIABLE_ITEMS: Tuple[Tuple[str, str], ...] = (
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
MONTE_CARLO_VARIABLES: Tuple[str, ...] = tuple(key for key, _ in MONTE_CARLO_VARIABLE_ITEMS)
MONTE_CARLO_VARIABLE_LABELS: Dict[str, str] = {key: label for key, label in MONTE_CARLO_VARIABLE_ITEMS}
MONTHS_IN_YEAR = 12
RISK_MULTIPLIER_COLUMNS: Dict[str, str] = {
    "production_multiplier": "production",
    "labour_multiplier": "labour",
    "price_multiplier": "price",
    "revenue_multiplier": "revenue",
    "yield_multiplier": "yield",
}

SIMPLE_XLSX_ENGINE = "__simple_xlsx__"

DEFAULTS = {
    "horizon": {"start_year": 2025, "end_year": 2035, "start_month": 1, "frequency": "monthly"},
    "production_horizon": {"start_year": 2025, "end_year": 2035},
    "global": {
        "corp_tax_rate": 0.28,
        "investor_share": 0.6,
        "owner_share": 0.4,
        "capital_gains_tax_rate": 0.0,
        "terminal_growth": 0.02,
        "discount_rate": 0.12,
        "inflation_rate": 0.02,
        "base_currency": "USD",
        "fx_index_name": None,
    },
    "prices": {
        "ethanol": {"base_price": 0.70, "price_escalation_pa": 0.02, "price_indexation": "cpi", "uom": "USD/L"},
        "sugar": {"base_price": 450.0, "price_escalation_pa": 0.02, "price_indexation": "cpi", "uom": "USD/t"},
        "electricity": {"base_price": 80.0, "price_escalation_pa": 0.02, "price_indexation": "cpi", "uom": "USD/MWh"},
        "animal_feed": {"base_price": 180.0, "price_escalation_pa": 0.02, "price_indexation": "cpi", "uom": "USD/t"},
    },
    "production": {
        "sugarcane_yield_ton_per_ha": 70.0,
        "plant_availability": 0.9,
        "loss_factor": 0.02,
        "ramp": [0.7, 0.9, 1.0],
        "ethanol_litre_per_ton": 160.0,
        "sugar_ton_per_ton_cane": 0.1,
        "electricity_mwh_per_ton_cane": 0.12,
        "animal_feed_ton_per_ton_cane": 0.05,
        "annual_feedstock_ton": 100_000.0,
    },
    "opex": {
        "farm_opex_per_ton": 18.0,
        "purchase_price_per_ton": 65.0,
        "other_variable_cost_per_unit": {"ethanol": 0.08, "sugar": 50.0, "electricity": 10.0, "animal_feed": 25.0},
        "fixed_opex_per_month": 2_500_000.0 / MONTHS_IN_YEAR,
    },
    "working_capital": {"dso_days": 30.0, "dio_days": 20.0, "dpo_days": 25.0},
    "debt": {
        "tranches": [
            {
                "name": "Senior Loan",
                "draw_curve": None,
                "interest_rate": 0.1,
                "base_rate": 0.0,
                "margin": 0.0,
                "type": "term",
                "tenor_years": 8,
                "grace_years": 1,
                "amortization": "straight",
                "fees_upfront": 0.0,
                "fees_annual": 0.0,
                "capitalize_idc": True,
                "currency": "USD",
                "fx_curve": None,
                "share": 0.6,
                "start_year": 2025,
            }
        ]
    },
    "tax": {
        "base_tax_rate": 0.28,
        "timing_adjustment_rules": None,
        "loss_carryforward_years": None,
        "min_tax": 0.0,
        "capex_incentives": {},
    },
    "inflation_index": pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=240, freq="MS"),
            "cpi": 1.0,
            "fx_pair": np.nan,
            "fx_index": np.nan,
        }
    ),
    "risk_params": pd.DataFrame(
        [
            {
                "driver_name": "political_risk",
                "distribution": "triangular",
                "p1": -0.015,
                "p2": 0.0,
                "p3": 0.015,
                "target": "risk",
                "applies_to": "global",
                "production_multiplier": 0.99,
                "labour_multiplier": 1.02,
                "price_multiplier": 1.005,
                "revenue_multiplier": 0.995,
                "yield_multiplier": 0.995,
            },
            {
                "driver_name": "environmental_risk",
                "distribution": "normal",
                "p1": -0.025,
                "p2": 0.012,
                "p3": np.nan,
                "target": "risk",
                "applies_to": "global",
                "production_multiplier": 0.97,
                "labour_multiplier": 1.01,
                "price_multiplier": 1.0,
                "revenue_multiplier": 0.97,
                "yield_multiplier": 0.95,
            },
            {
                "driver_name": "market_risk",
                "distribution": "normal",
                "p1": -0.03,
                "p2": 0.018,
                "p3": np.nan,
                "target": "risk",
                "applies_to": "market",
                "production_multiplier": 0.99,
                "labour_multiplier": 1.0,
                "price_multiplier": 0.96,
                "revenue_multiplier": 0.96,
                "yield_multiplier": 1.0,
            },
            {
                "driver_name": "ethanol_price",
                "distribution": "normal",
                "p1": 0.0,
                "p2": 0.05,
                "p3": np.nan,
                "target": "price",
                "applies_to": "ethanol",
                "production_multiplier": 1.0,
                "labour_multiplier": 1.0,
                "price_multiplier": 1.0,
                "revenue_multiplier": 1.0,
                "yield_multiplier": 1.0,
            },
            {
                "driver_name": "electricity_price",
                "distribution": "normal",
                "p1": 0.0,
                "p2": 0.06,
                "p3": np.nan,
                "target": "price",
                "applies_to": "electricity",
                "production_multiplier": 1.0,
                "labour_multiplier": 1.0,
                "price_multiplier": 1.0,
                "revenue_multiplier": 1.0,
                "yield_multiplier": 1.0,
            },
            {
                "driver_name": "sugar_price",
                "distribution": "normal",
                "p1": 0.0,
                "p2": 0.07,
                "p3": np.nan,
                "target": "price",
                "applies_to": "sugar",
                "production_multiplier": 1.0,
                "labour_multiplier": 1.0,
                "price_multiplier": 1.0,
                "revenue_multiplier": 1.0,
                "yield_multiplier": 1.0,
            },
            {
                "driver_name": "animal_feed_price",
                "distribution": "normal",
                "p1": 0.0,
                "p2": 0.08,
                "p3": np.nan,
                "target": "price",
                "applies_to": "animal_feed",
                "production_multiplier": 1.0,
                "labour_multiplier": 1.0,
                "price_multiplier": 1.0,
                "revenue_multiplier": 1.0,
                "yield_multiplier": 1.0,
            },
            {
                "driver_name": "availability",
                "distribution": "normal",
                "p1": 0.0,
                "p2": 0.02,
                "p3": np.nan,
                "target": "availability",
                "applies_to": "global",
                "production_multiplier": 1.0,
                "labour_multiplier": 1.0,
                "price_multiplier": 1.0,
                "revenue_multiplier": 1.0,
                "yield_multiplier": 1.0,
            },
        ]
    ),
    "tornado_drivers": pd.DataFrame(
        [
            {"enabled": True, "driver": "ethanol_price", "pct_change": 0.20},
            {"enabled": True, "driver": "sugar_price", "pct_change": 0.20},
            {"enabled": True, "driver": "availability", "pct_change": 0.05},
            {"enabled": True, "driver": "capex", "pct_change": 0.20},
            {"enabled": True, "driver": "debt_rate", "pct_change": 0.02},
        ]
    ),
    "monte_carlo_settings": pd.DataFrame(
        [
            {
                "enabled": False,
                "iterations": 1000,
                "random_seed": 42,
                "distribution": "normal",
                "variable": "opex",
                "applies_to": "global",
                "p1": 0.0,
                "p2": 0.05,
                "p3": np.nan,
            },
        ]
    ),
    "scenario_comparison": pd.DataFrame(
        [
            {
                "enabled": True,
                "scenario_name": "FARM_ONLY",
                "feedstock_scenario": "FARM_ONLY",
                "farm_share": 1.0,
                "increment_profile": "base",
                "notes": "All feedstock grown internally",
            },
            {
                "enabled": True,
                "scenario_name": "BUY_ONLY",
                "feedstock_scenario": "BUY_ONLY",
                "farm_share": 0.0,
                "increment_profile": "base",
                "notes": "All feedstock purchased from market",
            },
            {
                "enabled": True,
                "scenario_name": "HYBRID",
                "feedstock_scenario": "HYBRID",
                "farm_share": 0.5,
                "increment_profile": "base",
                "notes": "Blend of internal farming and market purchases",
            },
        ]
    ),
    "optimizer_settings": pd.DataFrame(
        [
            {"enabled": True, "variable": "ethanol_price", "lower_bound": 0.85, "upper_bound": 1.15, "notes": "Scale ethanol tariff"},
            {"enabled": True, "variable": "capex", "lower_bound": 0.85, "upper_bound": 1.15, "notes": "Adjust total project CAPEX"},
            {"enabled": False, "variable": "debt_rate", "lower_bound": -0.02, "upper_bound": 0.02, "notes": "Shift debt interest rate"},
        ]
    ),
    "neural_forecast_settings": pd.DataFrame(
        [
            {
                "enabled": True,
                "product": "ethanol",
                "lookback_months": 12,
                "forecast_months": 12,
                "hidden_units": 8,
                "learning_rate": 0.01,
                "epochs": 300,
            },
            {
                "enabled": False,
                "product": "sugar",
                "lookback_months": 12,
                "forecast_months": 12,
                "hidden_units": 8,
                "learning_rate": 0.01,
                "epochs": 300,
            },
        ]
    ),
    "statistical_forecast_settings": pd.DataFrame(
        [
            {"enabled": True, "series": "revenue", "alpha": 0.3, "forecast_months": 12},
            {"enabled": True, "series": "cogs", "alpha": 0.3, "forecast_months": 12},
        ]
    ),
    "decision_tree_paths": pd.DataFrame(
        [
            {
                "enabled": True,
                "path_name": "Upside demand",
                "probability": 0.35,
                "ethanol_price_multiplier": 1.1,
                "sugar_price_multiplier": 1.05,
                "electricity_price_multiplier": 1.02,
                "animal_feed_price_multiplier": 1.03,
                "capex_multiplier": 1.0,
                "opex_multiplier": 1.0,
                "debt_rate_shift": -0.005,
                "notes": "Higher pricing environment",
            },
            {
                "enabled": True,
                "path_name": "Base case",
                "probability": 0.40,
                "ethanol_price_multiplier": 1.0,
                "sugar_price_multiplier": 1.0,
                "electricity_price_multiplier": 1.0,
                "animal_feed_price_multiplier": 1.0,
                "capex_multiplier": 1.0,
                "opex_multiplier": 1.0,
                "debt_rate_shift": 0.0,
                "notes": "Central outlook",
            },
            {
                "enabled": True,
                "path_name": "Downside",
                "probability": 0.25,
                "ethanol_price_multiplier": 0.9,
                "sugar_price_multiplier": 0.92,
                "electricity_price_multiplier": 0.95,
                "animal_feed_price_multiplier": 0.9,
                "capex_multiplier": 1.05,
                "opex_multiplier": 1.08,
                "debt_rate_shift": 0.01,
                "notes": "Pricing pressure and cost inflation",
            },
        ]
    ),
    "offtake_terms": pd.DataFrame(
        [
            {
                "product": "ethanol",
                "contracted_share": 0.7,
                "merchant_share": 0.3,
                "floor_price": 0.65,
                "collar_price": 0.85,
                "indexation_rule": "cpi",
                "counterparty_haircut": 0.02,
            },
            {
                "product": "electricity",
                "contracted_share": 0.8,
                "merchant_share": 0.2,
                "floor_price": 70.0,
                "collar_price": 95.0,
                "indexation_rule": "cpi",
                "counterparty_haircut": 0.01,
            },
        ]
    ),
    "dispatch_constraints": pd.DataFrame(
        [
            {
                "product": "electricity",
                "max_dispatch_share": 0.95,
                "curtailment_penalty_per_unit": 2.0,
                "availability_floor": 0.88,
            }
        ]
    ),
    "construction_schedule": pd.DataFrame(
        [
            {
                "epc_package": "Plant EPC",
                "start_date": "2025-01",
                "end_date": "2025-12",
                "amount": 30_000_000.0,
                "draw_curve": "",
                "contingency_rank": 1,
                "contingency_amount": 2_000_000.0,
                "delay_months": 0,
            }
        ]
    ),
    "completion_tests": pd.DataFrame(
        [
            {"enabled": True, "metric": "plant_availability", "threshold": 0.85, "test_date": "2026-01-01"},
            {"enabled": True, "metric": "dscr_min", "threshold": 1.1, "test_date": "2026-12-01"},
        ]
    ),
    "reserve_accounts": pd.DataFrame(
        [
            {
                "dsra_months": 6.0,
                "mmra_monthly": 75_000.0,
                "wc_facility_limit": 2_000_000.0,
                "wc_facility_rate": 0.08,
                "restricted_cash_min": 250_000.0,
                "sweep_pct": 0.5,
                "sweep_dscr_trigger": 1.25,
            }
        ]
    ),
    "covenant_thresholds": pd.DataFrame(
        [
            {
                "start_date": "2025-01-01",
                "end_date": "2035-12-01",
                "case": "base",
                "adsc_min": 1.2,
                "dscr_lockup": 1.1,
                "dscr_default": 1.0,
                "llcr_min": 1.25,
            },
            {
                "start_date": "2025-01-01",
                "end_date": "2035-12-01",
                "case": "downside",
                "adsc_min": 1.1,
                "dscr_lockup": 1.05,
                "dscr_default": 0.95,
                "llcr_min": 1.15,
            },
        ]
    ),
    "lender_cases": pd.DataFrame(
        [
            {
                "enabled": True,
                "case_name": "Base case (P50)",
                "increment_profile": "base",
                "production_multiplier": 1.0,
                "price_multiplier": 1.0,
                "opex_multiplier": 1.0,
                "capex_multiplier": 1.0,
                "delay_months": 0,
                "debt_rate_shift": 0.0,
            },
            {
                "enabled": True,
                "case_name": "Downside case (P90-like)",
                "increment_profile": "downside",
                "production_multiplier": 0.9,
                "price_multiplier": 0.9,
                "opex_multiplier": 1.1,
                "capex_multiplier": 1.08,
                "delay_months": 6,
                "debt_rate_shift": 0.01,
            },
            {
                "enabled": True,
                "case_name": "Break case",
                "increment_profile": "break",
                "production_multiplier": 0.8,
                "price_multiplier": 0.82,
                "opex_multiplier": 1.2,
                "capex_multiplier": 1.15,
                "delay_months": 12,
                "debt_rate_shift": 0.02,
            },
        ]
    ),
    "yearly_increments": pd.DataFrame(
        [
            {"profile": "base", "year": 2025, "global_increment_pct": 0.0, "price_increment_pct": 0.0, "opex_increment_pct": 0.0, "debt_increment_pct": 0.0, "capex_increment_pct": 0.0, "wc_increment_pct": 0.0, "tax_increment_pct": 0.0},
            {"profile": "base", "year": 2026, "global_increment_pct": 0.0, "price_increment_pct": 0.0, "opex_increment_pct": 0.0, "debt_increment_pct": 0.0, "capex_increment_pct": 0.0, "wc_increment_pct": 0.0, "tax_increment_pct": 0.0},
            {"profile": "base", "year": 2027, "global_increment_pct": 0.0, "price_increment_pct": 0.0, "opex_increment_pct": 0.0, "debt_increment_pct": 0.0, "capex_increment_pct": 0.0, "wc_increment_pct": 0.0, "tax_increment_pct": 0.0},
            {"profile": "downside", "year": 2025, "global_increment_pct": 0.0, "price_increment_pct": -0.01, "opex_increment_pct": 0.01, "debt_increment_pct": 0.005, "capex_increment_pct": 0.01, "wc_increment_pct": 0.0, "tax_increment_pct": 0.0},
            {"profile": "break", "year": 2025, "global_increment_pct": 0.0, "price_increment_pct": -0.02, "opex_increment_pct": 0.02, "debt_increment_pct": 0.01, "capex_increment_pct": 0.02, "wc_increment_pct": 0.01, "tax_increment_pct": 0.0},
        ]
    ),
    "debt_sizing_params": pd.DataFrame(
        [
            {
                "enabled": True,
                "min_dscr": 1.25,
                "llcr_floor": 1.35,
                "plcr_floor": 1.5,
                "max_tenor_years": 12,
                "reserve_buffer_pct": 0.05,
                "target_debt_amount": np.nan,
            }
        ]
    ),
    "covenant_cure_rules": pd.DataFrame(
        [
            {
                "enabled": True,
                "case": "base",
                "lockup_grace_months": 3,
                "default_grace_months": 1,
                "cash_sweep_stepup_pct": 0.15,
                "waiver_fee_pct": 0.005,
                "waiver_probability": 0.5,
            }
        ]
    ),
    "construction_risk_mc": pd.DataFrame(
        [
            {
                "enabled": True,
                "iterations": 200,
                "delay_mean_months": 1.0,
                "delay_std_months": 2.0,
                "ld_per_day": 20_000.0,
                "dsu_monthly_payout": 500_000.0,
                "idc_capitalize": True,
            }
        ]
    ),
    "hedging_program": pd.DataFrame(
        [
            {
                "product": "ethanol",
                "hedge_ratio": 0.5,
                "tenor_months": 12,
                "fx_basis_pct": 0.01,
                "correlation_shock": 0.15,
                "hedge_cost_pct": 0.01,
                "collateral_pct": 0.03,
                "effectiveness": 0.85,
            }
        ]
    ),
    "sustainability_compliance": pd.DataFrame(
        [
            {
                "standard": "Bonsucro",
                "status": "compliant",
                "price_premium_pct": 0.01,
                "low_carbon_credit_per_unit": 0.0,
                "capex_gap_closure_pct": 0.0,
                "opex_gap_closure_pct": 0.0,
                "delay_penalty_pct": 0.0,
            }
        ]
    ),
    "benchmark_library": pd.DataFrame(
        [
            {"metric": "capex_per_litre", "region": "LATAM", "year": 2025, "value": 0.6, "risk_tier": "base"},
            {"metric": "opex_per_litre", "region": "LATAM", "year": 2025, "value": 0.35, "risk_tier": "base"},
            {"metric": "dscr_min_threshold", "region": "GLOBAL", "year": 2025, "value": 1.2, "risk_tier": "senior"},
            {"metric": "irr_threshold", "region": "GLOBAL", "year": 2025, "value": 0.14, "risk_tier": "equity"},
        ]
    ),
}


def _build_default_break_even_inputs() -> pd.DataFrame:
    """Construct default break-even inputs per product using base assumptions."""

    production_defaults = DEFAULTS["production"]
    price_defaults = DEFAULTS["prices"]
    variable_defaults = DEFAULTS["opex"].get("other_variable_cost_per_unit", {})
    feedstock = float(production_defaults.get("annual_feedstock_ton", 0.0) or 0.0)
    availability = float(production_defaults.get("plant_availability", 1.0) or 1.0)
    loss_factor = float(production_defaults.get("loss_factor", 0.0) or 0.0)
    effective_factor = availability * (1.0 - loss_factor)
    fixed_total = float(DEFAULTS["opex"].get("fixed_opex_per_month", 0.0) or 0.0) * MONTHS_IN_YEAR
    fixed_per_product = fixed_total / max(len(PRODUCTS), 1)

    conversion_map = {
        "ethanol": production_defaults.get("ethanol_litre_per_ton", 0.0),
        "sugar": production_defaults.get("sugar_ton_per_ton_cane", 0.0),
        "electricity": production_defaults.get("electricity_mwh_per_ton_cane", 0.0),
        "animal_feed": production_defaults.get("animal_feed_ton_per_ton_cane", 0.0),
    }

    records: List[Dict[str, object]] = []
    for product in PRODUCTS:
        conversion = float(conversion_map.get(product, 0.0) or 0.0)
        reference_volume = feedstock * conversion * effective_factor
        price_info = price_defaults.get(product, {})
        unit_price = float(price_info.get("base_price", 0.0) or 0.0)
        variable_cost = float(variable_defaults.get(product, 0.0) or 0.0)
        records.append(
            {
                "product": product,
                "unit_price": unit_price,
                "variable_cost_per_unit": variable_cost,
                "fixed_cost": fixed_per_product,
                "reference_volume": reference_volume,
            }
        )

    return pd.DataFrame(records)


DEFAULTS["break_even_inputs"] = _build_default_break_even_inputs()

###############################################################################
# Section 1: Excel Loader
###############################################################################

_SANITIZE_REGEX = re.compile(r"[^0-9a-zA-Z]+")


def sanitize_sheet_name(name: str) -> str:
    clean = _SANITIZE_REGEX.sub("_", name.strip().lower()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    if clean == "":
        clean = "sheet"
    if clean[0].isdigit():
        clean = f"sheet_{clean}"
    return clean


def excel_inventory(excel_path: Path, header: Optional[int] = 0) -> Tuple[pd.ExcelFile, Dict[str, str]]:
    xls = pd.ExcelFile(excel_path)
    mapping: Dict[str, str] = {}
    for name in xls.sheet_names:
        key = sanitize_sheet_name(name)
        suffix = 1
        base = key
        while key in mapping:
            suffix += 1
            key = f"{base}_{suffix}"
        mapping[key] = name
    return xls, mapping


def load_sheet(excel_path: Path, sheet_name: str, header: Optional[int] = 0, **kwargs) -> pd.DataFrame:
    return pd.read_excel(excel_path, sheet_name=sheet_name, header=header, **kwargs)


def build_registry(excel_path: Path, header: Optional[int] = 0) -> Dict[str, Callable[..., pd.DataFrame]]:
    xls, mapping = excel_inventory(excel_path, header=header)
    registry: Dict[str, Callable[..., pd.DataFrame]] = {}
    for key, sheet_name in mapping.items():
        def _loader(sheet=sheet_name):
            return pd.read_excel(excel_path, sheet_name=sheet, header=header)
        _loader.__name__ = f"load_{key}"
        _loader.__doc__ = f"Load sheet '{sheet_name}' from {excel_path}"
        registry[key] = _loader
    return registry


def load_all(excel_path: Path, header: Optional[int] = 0) -> Dict[str, pd.DataFrame]:
    xls, mapping = excel_inventory(excel_path, header=header)
    data: Dict[str, pd.DataFrame] = {}
    for key, sheet_name in mapping.items():
        data[key] = pd.read_excel(xls, sheet_name=sheet_name, header=header)
    return data


def export_registry_to_csv(excel_path: Path, out_dir: Path, header: Optional[int] = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(excel_path, header=header)
    for key, df in data.items():
        df.to_csv(out_dir / f"{key}.csv", index=False)


###############################################################################
# Section 2: Assumption detection and normalization
###############################################################################

_KEY_NORMALIZER = re.compile(r"[^0-9a-zA-Z]+")


def normalize_key(value: str) -> str:
    key = _KEY_NORMALIZER.sub("_", str(value).strip().lower())
    key = re.sub(r"_+", "_", key).strip("_")
    return key


ALIASES: Dict[str, str] = {
    "corporate_tax": "corp_tax_rate",
    "corporate_tax_rate": "corp_tax_rate",
    "tax_rate": "corp_tax_rate",
    "investor_equity_share": "investor_share",
    "owner_equity_share": "owner_share",
    "wacc": "discount_rate",
    "discount": "discount_rate",
    "inflation": "inflation_rate",
    "ethanol_price": "ethanol_price_per_litre",
    "ethanol_price_l": "ethanol_price_per_litre",
    "ethanol_price_per_liter": "ethanol_price_per_litre",
    "ethanol_price_per_litre": "ethanol_price_per_litre",
    "sugar_price": "sugar_price_per_ton",
    "electricity_tariff": "electricity_tariff_per_mwh",
    "animal_feed_price": "animal_feed_price_per_ton",
    "dso": "dso_days",
    "dio": "dio_days",
    "dpo": "dpo_days",
    "start": "start_year",
    "end": "end_year",
    "project_start_year": "start_year",
    "project_end_year": "end_year",
    "scenario": "feedstock_scenario",
    "feedstock_scenario": "feedstock_scenario",
    "hybrid_share": "farm_share",
}


def detect_assumptions(sheets: Mapping[str, pd.DataFrame]) -> Dict[str, object]:
    assumptions: Dict[str, object] = {}

    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        try:
            df = df.dropna(how="all")
        except Exception:
            continue
        if df.empty:
            continue

        if df.shape[1] >= 2:
            sample = df.iloc[:, :3]
        else:
            sample = df

        for row in sample.itertuples(index=False):
            if len(row) >= 2 and isinstance(row[0], str):
                key = normalize_key(row[0])
                if not key:
                    continue
                value = row[1]
                if key in ALIASES:
                    key = ALIASES[key]
                assumptions[key] = value
            if len(row) >= 3 and isinstance(row[0], str) and isinstance(row[1], str):
                section = normalize_key(row[0])
                key = normalize_key(row[1])
                if section and key:
                    compound = f"{section}__{key}"
                    assumptions[compound] = row[2]

    return assumptions


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return int(default)
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return int(default)
        try:
            return int(float(value))
        except ValueError:
            return int(default)
    try:
        numeric = float(value)
        if math.isnan(numeric):
            return int(default)
        return int(numeric)
    except Exception:
        return int(default)


def _coerce_float(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _coerce_str(value: object, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    value = str(value).strip()
    if value == "":
        return default
    return value


def _projection_horizon_bounds(horizon: Mapping[str, object]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive timestamps for the active projection horizon."""

    start_year = _coerce_int(horizon.get("start_year"), DEFAULTS["horizon"]["start_year"])
    end_year = _coerce_int(horizon.get("end_year"), DEFAULTS["horizon"]["end_year"])
    start_month = _coerce_int(horizon.get("start_month"), DEFAULTS["horizon"].get("start_month", 1))

    if end_year < start_year:
        end_year = start_year

    start_ts = pd.Timestamp(year=start_year, month=start_month, day=1)
    end_ts = pd.Timestamp(year=end_year, month=12, day=1)
    if end_ts < start_ts:
        end_ts = start_ts
    return start_ts, end_ts


def _filter_dates_to_horizon(
    df: pd.DataFrame,
    column: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Clamp a dataframe to the projection horizon based on a date column."""

    if column not in df.columns:
        return df

    df_local = df.copy()
    converted = pd.to_datetime(df_local[column], errors="coerce")
    valid_mask = converted.notna()

    if not valid_mask.any():
        # No parsable dates – retain the original rows so users can continue
        # editing values (e.g. newly added rows awaiting a date entry).
        return df_local.reset_index(drop=True)

    valid = df_local[valid_mask].copy()
    valid[column] = converted[valid_mask]
    within_mask = (valid[column] >= start_ts) & (valid[column] <= end_ts)
    valid = valid[within_mask]

    invalid = df_local[~valid_mask].copy()

    if invalid.empty:
        return valid.reset_index(drop=True)

    combined = pd.concat([valid, invalid]).sort_index()
    return combined.reset_index(drop=True)


def align_with_projection_horizon(cfg: Dict[str, object]) -> Dict[str, object]:
    """Propagate projection horizon changes across time-based input tables."""

    start_ts, end_ts = _projection_horizon_bounds(cfg.get("projection_horizon", DEFAULTS["horizon"]))

    # Production horizon is fused with the projection horizon for a simpler UX.
    cfg["production_horizon"] = {"start_year": int(start_ts.year), "end_year": int(end_ts.year)}

    capex_df = cfg.get("capex_lines")
    if isinstance(capex_df, pd.DataFrame) and not capex_df.empty:
        capex_adj = capex_df.copy()
        capex_adj["start_date"] = capex_adj["start_date"].apply(lambda v: parse_date_str(v, start_ts))
        capex_adj["end_date"] = capex_adj["end_date"].apply(lambda v: parse_date_str(v, end_ts))
        capex_adj.loc[capex_adj["start_date"] < start_ts, "start_date"] = start_ts
        capex_adj.loc[capex_adj["start_date"] > end_ts, "start_date"] = end_ts
        capex_adj.loc[capex_adj["end_date"] < start_ts, "end_date"] = start_ts
        capex_adj.loc[capex_adj["end_date"] > end_ts, "end_date"] = end_ts
        capex_adj.loc[capex_adj["end_date"] < capex_adj["start_date"], "end_date"] = capex_adj["start_date"]
        capex_adj["start_date"] = capex_adj["start_date"].dt.to_period("M").dt.to_timestamp()
        capex_adj["end_date"] = capex_adj["end_date"].dt.to_period("M").dt.to_timestamp()
        cfg["capex_lines"] = capex_adj.reset_index(drop=True)

    for table in (
        "production_monthly",
        "direct_costs_monthly",
        "staff_costs_monthly",
        "other_opex_monthly",
        "ar_other_assets",
        "inventory_ap",
    ):
        df = cfg.get(table)
        if isinstance(df, pd.DataFrame) and not df.empty:
            cfg[table] = _filter_dates_to_horizon(df, "date", start_ts, end_ts)

    prod_annual = cfg.get("production_annual")
    if isinstance(prod_annual, pd.DataFrame) and not prod_annual.empty and "year" in prod_annual.columns:
        prod_annual = prod_annual.copy()
        prod_annual["year"] = pd.to_numeric(prod_annual["year"], errors="coerce")
        mask = prod_annual["year"].between(start_ts.year, end_ts.year)
        cfg["production_annual"] = prod_annual[mask].reset_index(drop=True)

    inflation_df = cfg.get("inflation_index")
    if isinstance(inflation_df, pd.DataFrame) and not inflation_df.empty and "date" in inflation_df.columns:
        filtered = _filter_dates_to_horizon(inflation_df, "date", start_ts, end_ts)
        if filtered.empty:
            monthly_index = pd.date_range(start_ts, end_ts, freq="MS")
            inflation_rate = cfg.get("global_inputs", {}).get(
                "inflation_rate", DEFAULTS["global"].get("inflation_rate", 0.0)
            )
            filtered = pd.DataFrame(
                {
                    "date": monthly_index,
                    "cpi": (1 + inflation_rate / 12) ** np.arange(len(monthly_index)),
                    "fx_pair": np.nan,
                    "fx_index": np.nan,
                }
            )
        cfg["inflation_index"] = filtered

    return cfg


def compute_risk_profile(risk_params: Optional[pd.DataFrame]) -> Dict[str, object]:
    profile: Dict[str, object] = {name: 1.0 for name in RISK_MULTIPLIER_COLUMNS.values()}
    profile["price_by_product"] = {}
    if isinstance(risk_params, pd.DataFrame) and not risk_params.empty:
        for _, row in risk_params.iterrows():
            target = str(row.get("target", "risk")).lower()
            applies_to = str(row.get("applies_to", "")).lower()
            if target == "price" and applies_to in PRODUCTS:
                multiplier = row.get("price_multiplier", 1.0)
                try:
                    multiplier = float(multiplier)
                except (TypeError, ValueError):
                    multiplier = 1.0
                multiplier = max(multiplier, 0.0)
                price_map: Dict[str, float] = profile.setdefault("price_by_product", {})  # type: ignore[assignment]
                price_map[applies_to] = price_map.get(applies_to, 1.0) * (multiplier or 1.0)
                continue

            for column, key in RISK_MULTIPLIER_COLUMNS.items():
                if column in row and pd.notna(row[column]):
                    try:
                        value = float(row[column])
                    except (TypeError, ValueError):
                        continue
                    profile[key] = float(profile.get(key, 1.0)) * max(value, 0.0)
    return profile


def _validate_projection_horizon(row: pd.Series) -> None:
    start = row.get("start_year")
    end = row.get("end_year")
    if pd.isna(start) or pd.isna(end):
        return
    start_val = _coerce_int(start, DEFAULTS["horizon"]["start_year"])
    end_val = _coerce_int(end, start_val)
    if end_val < start_val:
        raise ValueError("end_year must be >= start_year")


def _validate_production_horizon(row: pd.Series) -> None:
    start = row.get("start_year")
    end = row.get("end_year")
    if pd.isna(start) or pd.isna(end):
        return
    start_val = _coerce_int(start, DEFAULTS["production_horizon"]["start_year"])
    end_val = _coerce_int(end, start_val)
    if end_val < start_val:
        raise ValueError("production end_year must be >= start_year")


def _validate_break_even_inputs(row: pd.Series) -> None:
    product = str(row.get("product", "")).strip().lower()
    if product == "":
        raise ValueError("break-even inputs require a product identifier")
    if product not in PRODUCTS:
        raise ValueError(
            "product must be one of: " + ", ".join(PRODUCTS)
        )


def _derive_direct_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure direct cost lines carry a calculated amount from unit rates."""

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "cost_type", "product_link", "unit_price", "quantity", "amount", "currency"])

    result = df.copy()
    for column in ("unit_price", "quantity", "amount"):
        if column not in result.columns:
            result[column] = np.nan

    result["unit_price"] = pd.to_numeric(result["unit_price"], errors="coerce")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce")
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")

    mask = result["unit_price"].notna() & result["quantity"].notna()
    result.loc[mask, "amount"] = result.loc[mask, "unit_price"] * result.loc[mask, "quantity"]

    return result


def _derive_staff_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise staff cost entries and recompute totals from per-head inputs."""

    required_columns = [
        "date",
        "dept",
        "headcount",
        "gross_pay",
        "benefits",
        "training",
        "other",
        "gross_pay_per_head",
        "benefits_per_head",
        "training_per_head",
        "other_per_head",
        "currency",
    ]

    if df is None or df.empty:
        empty = pd.DataFrame(columns=required_columns)
        for column in ("headcount", "gross_pay", "benefits", "training", "other", "gross_pay_per_head", "benefits_per_head", "training_per_head", "other_per_head"):
            empty[column] = empty[column].astype(float)
        return empty

    result = df.copy()
    for column in required_columns:
        if column not in result.columns:
            result[column] = np.nan

    # Ensure textual columns carry sensible defaults while numeric fields are coerced
    result["dept"] = result["dept"].astype(str).where(result["dept"].notna(), "Unassigned")
    result["currency"] = result["currency"].astype(str).where(result["currency"].notna(), DEFAULTS["global"].get("base_currency", "USD"))

    headcount = pd.to_numeric(result["headcount"], errors="coerce")
    result["headcount"] = headcount.fillna(0.0)

    pairs = (
        ("gross_pay", "gross_pay_per_head"),
        ("benefits", "benefits_per_head"),
        ("training", "training_per_head"),
        ("other", "other_per_head"),
    )

    for total_col, per_head_col in pairs:
        total_series = pd.to_numeric(result[total_col], errors="coerce")
        per_head_series = pd.to_numeric(result[per_head_col], errors="coerce")

        # Compute totals from per-head values whenever available
        mask_per_head = per_head_series.notna() & headcount.notna()
        if mask_per_head.any():
            total_series.loc[mask_per_head] = (
                per_head_series.loc[mask_per_head].fillna(0.0)
                * headcount.loc[mask_per_head].fillna(0.0)
            ).values

        # Derive per-head figures from totals where missing and headcount is positive
        mask_total = total_series.notna() & headcount.notna() & (headcount != 0)
        if mask_total.any():
            per_head_series.loc[mask_total] = (
                total_series.loc[mask_total] / headcount.loc[mask_total]
            ).values

        result[total_col] = total_series.fillna(0.0)
        result[per_head_col] = per_head_series.fillna(0.0)

    return result[required_columns]
###############################################################################
# Section 3: Input tables and CRUD helpers
###############################################################################

@dataclass
class TableSchema:
    columns: Dict[str, str]
    defaults: Dict[str, object] = field(default_factory=dict)
    validators: List[Callable[[pd.Series], None]] = field(default_factory=list)
    derived: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


INPUT_SCHEMAS: Dict[str, TableSchema] = {
    "projection_horizon": TableSchema(
        columns={"start_year": "int", "end_year": "int", "start_month": "int", "frequency": "str"},
        defaults={"frequency": "monthly", "start_month": 1},
        validators=[_validate_projection_horizon],
    ),
    "production_horizon": TableSchema(
        columns={"start_year": "int", "end_year": "int"},
        defaults=dict(DEFAULTS["production_horizon"]),
        validators=[_validate_production_horizon],
    ),
    "global_inputs": TableSchema(
        columns={
            "corp_tax_rate": "float",
            "investor_share": "float",
            "owner_share": "float",
            "capital_gains_tax_rate": "float",
            "terminal_growth": "float",
            "discount_rate": "float",
            "inflation_rate": "float",
            "base_currency": "str",
            "fx_index_name": "str",
        },
        defaults={
            **DEFAULTS["global"],
        },
        validators=[
            lambda row: (_ for _ in ()).throw(ValueError("investor_share + owner_share must equal 1"))
            if not math.isclose(float(row.get("investor_share", 0.0)) + float(row.get("owner_share", 0.0)), 1.0, rel_tol=1e-4, abs_tol=1e-4)
            else None,
        ],
    ),
    "working_capital_days": TableSchema(
        columns={"dso_days": "float", "dio_days": "float", "dpo_days": "float"},
        defaults={**DEFAULTS["working_capital"]},
    ),
    "capex_lines": TableSchema(
        columns={
            "item_name": "str",
            "category": "str",
            "amount": "float",
            "currency": "str",
            "fx_curve": "str",
            "start_date": "str",
            "end_date": "str",
            "life_years": "int",
            "depr_method": "str",
            "depr_rate_override": "float",
            "vat_rate": "float",
            "vat_recovery_lag_months": "int",
            "capitalized": "bool",
            "is_farm_capex": "bool",
        },
        defaults={
            "category": "other",
            "currency": "USD",
            "life_years": 10,
            "depr_method": "straight",
            "vat_rate": 0.0,
            "vat_recovery_lag_months": 0,
            "capitalized": True,
            "is_farm_capex": False,
        },
    ),
    "revenue_params": TableSchema(
        columns={
            "product": "str",
            "base_price": "float",
            "price_escalation_pa": "float",
            "price_indexation": "str",
            "uom": "str",
            "tariff_structure": "str",
            "revenue_share": "float",
        },
        defaults={"price_escalation_pa": 0.0, "price_indexation": "cpi", "revenue_share": 1.0},
    ),
    "break_even_inputs": TableSchema(
        columns={
            "product": "str",
            "unit_price": "float",
            "variable_cost_per_unit": "float",
            "fixed_cost": "float",
            "reference_volume": "float",
        },
        defaults={
            "unit_price": np.nan,
            "variable_cost_per_unit": np.nan,
            "fixed_cost": 0.0,
            "reference_volume": np.nan,
        },
        validators=[_validate_break_even_inputs],
    ),
    "production_annual": TableSchema(
        columns={
            "product": "str",
            "annual_volume": "float",
            "availability": "float",
            "loss_factor": "float",
            "startup_ramp": "str",
            "boe_conversion": "float",
            "sugarcane_yield_ton_per_ha": "float",
            "farm_area_ha": "float",
        },
        defaults={"availability": DEFAULTS["production"]["plant_availability"], "loss_factor": DEFAULTS["production"]["loss_factor"]},
    ),
    "production_monthly": TableSchema(
        columns={"date": "str", "product": "str", "volume": "float", "availability_override": "float", "maintenance_downtime": "float", "loss_override": "float"},
    ),
    "direct_costs_monthly": TableSchema(
        columns={
            "date": "str",
            "cost_type": "str",
            "product_link": "str",
            "unit_price": "float",
            "quantity": "float",
            "amount": "float",
            "currency": "str",
        },
        defaults={"currency": "USD", "unit_price": 0.0, "quantity": 0.0, "amount": 0.0},
        derived=lambda df: _derive_direct_costs(df),
    ),
    "staff_costs_monthly": TableSchema(
        columns={
            "date": "str",
            "dept": "str",
            "headcount": "float",
            "gross_pay": "float",
            "benefits": "float",
            "training": "float",
            "other": "float",
            "gross_pay_per_head": "float",
            "benefits_per_head": "float",
            "training_per_head": "float",
            "other_per_head": "float",
            "currency": "str",
        },
        defaults={"currency": "USD", "gross_pay_per_head": 0.0, "benefits_per_head": 0.0, "training_per_head": 0.0, "other_per_head": 0.0},
        derived=_derive_staff_costs,
    ),
    "other_opex_monthly": TableSchema(
        columns={"date": "str", "category": "str", "amount": "float", "currency": "str"},
        defaults={"currency": "USD"},
    ),
    "ar_other_assets": TableSchema(
        columns={"date": "str", "receivables": "float", "prepaid_expenses": "float", "other_current_assets": "float", "dso_days": "float"},
    ),
    "inventory_ap": TableSchema(
        columns={"date": "str", "inventory_raw": "float", "inventory_wip": "float", "inventory_fg": "float", "accounts_payable": "float", "dio_days": "float", "dpo_days": "float"},
    ),
    "debt_tranches": TableSchema(
        columns={
            "name": "str",
            "draw_curve": "str",
            "interest_rate": "float",
            "base_rate": "float",
            "margin": "float",
            "type": "str",
            "tenor_years": "int",
            "grace_years": "int",
            "amortization": "str",
            "fees_upfront": "float",
            "fees_annual": "float",
            "capitalize_idc": "bool",
            "currency": "str",
            "fx_curve": "str",
            "share": "float",
            "start_year": "int",
            "target_dscr": "float",
            "sculpting_enabled": "bool",
        },
        defaults={
            "type": "term",
            "amortization": "straight",
            "currency": "USD",
            "share": 1.0,
            "start_year": 2025,
            "target_dscr": 1.3,
            "sculpting_enabled": False,
        },
    ),
    "tax_schedule": TableSchema(
        columns={"base_tax_rate": "float", "timing_adjustment_rules": "str", "loss_carryforward_years": "float", "min_tax": "float", "capex_incentives": "str"},
        defaults={**DEFAULTS["tax"]},
    ),
    "inflation_index": TableSchema(
        columns={"date": "str", "cpi": "float", "fx_pair": "str", "fx_index": "float"},
    ),
    "tornado_drivers": TableSchema(
        columns={"enabled": "bool", "driver": "str", "pct_change": "float"},
        defaults={"enabled": True, "pct_change": 0.1},
    ),
    "monte_carlo_settings": TableSchema(
        columns={
            "enabled": "bool",
            "iterations": "int",
            "random_seed": "int",
            "distribution": "str",
            "variable": "str",
            "applies_to": "str",
            "p1": "float",
            "p2": "float",
            "p3": "float",
        },
        defaults={
            "enabled": False,
            "iterations": 1000,
            "random_seed": 42,
            "distribution": "normal",
            "variable": "opex",
            "applies_to": "global",
            "p1": 0.0,
            "p2": 0.05,
            "p3": np.nan,
        },
    ),
    "scenario_comparison": TableSchema(
        columns={
            "enabled": "bool",
            "scenario_name": "str",
            "feedstock_scenario": "str",
            "farm_share": "float",
            "increment_profile": "str",
            "notes": "str",
        },
        defaults={
            "enabled": True,
            "feedstock_scenario": "HYBRID",
            "farm_share": 0.5,
            "increment_profile": "base",
            "notes": "",
        },
    ),
    "optimizer_settings": TableSchema(
        columns={
            "enabled": "bool",
            "variable": "str",
            "lower_bound": "float",
            "upper_bound": "float",
            "notes": "str",
        },
        defaults={"enabled": True, "lower_bound": 0.9, "upper_bound": 1.1, "notes": ""},
    ),
    "neural_forecast_settings": TableSchema(
        columns={
            "enabled": "bool",
            "product": "str",
            "lookback_months": "int",
            "forecast_months": "int",
            "hidden_units": "int",
            "learning_rate": "float",
            "epochs": "int",
        },
        defaults={
            "enabled": True,
            "lookback_months": 12,
            "forecast_months": 12,
            "hidden_units": 8,
            "learning_rate": 0.01,
            "epochs": 300,
        },
    ),
    "statistical_forecast_settings": TableSchema(
        columns={
            "enabled": "bool",
            "series": "str",
            "alpha": "float",
            "forecast_months": "int",
        },
        defaults={"enabled": True, "alpha": 0.3, "forecast_months": 12},
    ),
    "decision_tree_paths": TableSchema(
        columns={
            "enabled": "bool",
            "path_name": "str",
            "probability": "float",
            "ethanol_price_multiplier": "float",
            "sugar_price_multiplier": "float",
            "electricity_price_multiplier": "float",
            "animal_feed_price_multiplier": "float",
            "capex_multiplier": "float",
            "opex_multiplier": "float",
            "debt_rate_shift": "float",
            "notes": "str",
        },
        defaults={
            "enabled": True,
            "probability": 0.33,
            "ethanol_price_multiplier": 1.0,
            "sugar_price_multiplier": 1.0,
            "electricity_price_multiplier": 1.0,
            "animal_feed_price_multiplier": 1.0,
            "capex_multiplier": 1.0,
            "opex_multiplier": 1.0,
            "debt_rate_shift": 0.0,
            "notes": "",
        },
    ),
    "risk_params": TableSchema(
        columns={
            "driver_name": "str",
            "distribution": "str",
            "p1": "float",
            "p2": "float",
            "p3": "float",
            "target": "str",
            "applies_to": "str",
            "production_multiplier": "float",
            "labour_multiplier": "float",
            "price_multiplier": "float",
            "revenue_multiplier": "float",
            "yield_multiplier": "float",
        },
        defaults={
            "distribution": "normal",
            "p1": 0.0,
            "p2": 0.0,
            "p3": np.nan,
            "target": "risk",
            "applies_to": "global",
            "production_multiplier": 1.0,
            "labour_multiplier": 1.0,
            "price_multiplier": 1.0,
            "revenue_multiplier": 1.0,
            "yield_multiplier": 1.0,
        },
    ),
    "offtake_terms": TableSchema(
        columns={
            "product": "str",
            "contracted_share": "float",
            "merchant_share": "float",
            "floor_price": "float",
            "collar_price": "float",
            "indexation_rule": "str",
            "counterparty_haircut": "float",
        },
        defaults={"contracted_share": 0.0, "merchant_share": 1.0, "indexation_rule": "cpi", "counterparty_haircut": 0.0},
    ),
    "dispatch_constraints": TableSchema(
        columns={"product": "str", "max_dispatch_share": "float", "curtailment_penalty_per_unit": "float", "availability_floor": "float"},
        defaults={"max_dispatch_share": 1.0, "curtailment_penalty_per_unit": 0.0, "availability_floor": 0.0},
    ),
    "construction_schedule": TableSchema(
        columns={
            "epc_package": "str",
            "start_date": "str",
            "end_date": "str",
            "amount": "float",
            "draw_curve": "str",
            "contingency_rank": "int",
            "contingency_amount": "float",
            "delay_months": "int",
        },
        defaults={"contingency_rank": 1, "contingency_amount": 0.0, "delay_months": 0},
    ),
    "completion_tests": TableSchema(
        columns={"enabled": "bool", "metric": "str", "threshold": "float", "test_date": "str"},
        defaults={"enabled": True, "threshold": 0.0},
    ),
    "reserve_accounts": TableSchema(
        columns={
            "dsra_months": "float",
            "mmra_monthly": "float",
            "wc_facility_limit": "float",
            "wc_facility_rate": "float",
            "restricted_cash_min": "float",
            "sweep_pct": "float",
            "sweep_dscr_trigger": "float",
        },
        defaults={"dsra_months": 6.0, "mmra_monthly": 0.0, "wc_facility_limit": 0.0, "wc_facility_rate": 0.0, "restricted_cash_min": 0.0, "sweep_pct": 0.0, "sweep_dscr_trigger": 1.25},
    ),
    "covenant_thresholds": TableSchema(
        columns={"start_date": "str", "end_date": "str", "case": "str", "adsc_min": "float", "dscr_lockup": "float", "dscr_default": "float", "llcr_min": "float"},
        defaults={"case": "base", "adsc_min": 1.2, "dscr_lockup": 1.1, "dscr_default": 1.0, "llcr_min": 1.25},
    ),
    "lender_cases": TableSchema(
        columns={
            "enabled": "bool",
            "case_name": "str",
            "increment_profile": "str",
            "production_multiplier": "float",
            "price_multiplier": "float",
            "opex_multiplier": "float",
            "capex_multiplier": "float",
            "delay_months": "int",
            "debt_rate_shift": "float",
        },
        defaults={"enabled": True, "increment_profile": "base", "production_multiplier": 1.0, "price_multiplier": 1.0, "opex_multiplier": 1.0, "capex_multiplier": 1.0, "delay_months": 0, "debt_rate_shift": 0.0},
    ),
    "yearly_increments": TableSchema(
        columns={
            "profile": "str",
            "year": "int",
            "global_increment_pct": "float",
            "price_increment_pct": "float",
            "opex_increment_pct": "float",
            "debt_increment_pct": "float",
            "capex_increment_pct": "float",
            "wc_increment_pct": "float",
            "tax_increment_pct": "float",
        },
        defaults={
            "profile": "base",
            "global_increment_pct": 0.0,
            "price_increment_pct": 0.0,
            "opex_increment_pct": 0.0,
            "debt_increment_pct": 0.0,
            "capex_increment_pct": 0.0,
            "wc_increment_pct": 0.0,
            "tax_increment_pct": 0.0,
        },
    ),
    "debt_sizing_params": TableSchema(
        columns={
            "enabled": "bool",
            "min_dscr": "float",
            "llcr_floor": "float",
            "plcr_floor": "float",
            "max_tenor_years": "int",
            "reserve_buffer_pct": "float",
            "target_debt_amount": "float",
        },
        defaults={"enabled": True, "min_dscr": 1.25, "llcr_floor": 1.35, "plcr_floor": 1.5, "max_tenor_years": 12, "reserve_buffer_pct": 0.05},
    ),
    "covenant_cure_rules": TableSchema(
        columns={
            "enabled": "bool",
            "case": "str",
            "lockup_grace_months": "int",
            "default_grace_months": "int",
            "cash_sweep_stepup_pct": "float",
            "waiver_fee_pct": "float",
            "waiver_probability": "float",
        },
        defaults={"enabled": True, "case": "base", "lockup_grace_months": 3, "default_grace_months": 1, "cash_sweep_stepup_pct": 0.15, "waiver_fee_pct": 0.005, "waiver_probability": 0.5},
    ),
    "construction_risk_mc": TableSchema(
        columns={"enabled": "bool", "iterations": "int", "delay_mean_months": "float", "delay_std_months": "float", "ld_per_day": "float", "dsu_monthly_payout": "float", "idc_capitalize": "bool"},
        defaults={"enabled": True, "iterations": 200, "delay_mean_months": 1.0, "delay_std_months": 2.0, "ld_per_day": 20_000.0, "dsu_monthly_payout": 500_000.0, "idc_capitalize": True},
    ),
    "hedging_program": TableSchema(
        columns={"product": "str", "hedge_ratio": "float", "tenor_months": "int", "fx_basis_pct": "float", "correlation_shock": "float", "hedge_cost_pct": "float", "collateral_pct": "float", "effectiveness": "float"},
        defaults={"hedge_ratio": 0.0, "tenor_months": 12, "fx_basis_pct": 0.0, "correlation_shock": 0.0, "hedge_cost_pct": 0.0, "collateral_pct": 0.0, "effectiveness": 1.0},
    ),
    "sustainability_compliance": TableSchema(
        columns={"standard": "str", "status": "str", "price_premium_pct": "float", "low_carbon_credit_per_unit": "float", "capex_gap_closure_pct": "float", "opex_gap_closure_pct": "float", "delay_penalty_pct": "float"},
        defaults={"status": "compliant", "price_premium_pct": 0.0, "low_carbon_credit_per_unit": 0.0, "capex_gap_closure_pct": 0.0, "opex_gap_closure_pct": 0.0, "delay_penalty_pct": 0.0},
    ),
    "benchmark_library": TableSchema(
        columns={"metric": "str", "region": "str", "year": "int", "value": "float", "risk_tier": "str"},
        defaults={"region": "GLOBAL", "risk_tier": "base"},
    ),
}


class InputTables:
    def __init__(self):
        self.tables: Dict[str, pd.DataFrame] = {}

    def ensure_table(self, table_name: str) -> pd.DataFrame:
        if table_name not in INPUT_SCHEMAS:
            raise KeyError(f"Unknown table '{table_name}'")
        schema = INPUT_SCHEMAS[table_name]
        if table_name not in self.tables:
            df = pd.DataFrame(columns=list(schema.columns.keys()))
            for col, typ in schema.columns.items():
                if typ in {"float", "int"}:
                    df[col] = df[col].astype(float)
            self.tables[table_name] = df
        return self.tables[table_name]

    def add_row(self, table_name: str, row: Mapping[str, object]) -> None:
        schema = INPUT_SCHEMAS[table_name]
        df = self.ensure_table(table_name)
        data = dict(schema.defaults)
        data.update(row)
        for col in schema.columns:
            if col not in data:
                data[col] = np.nan
        row_series = pd.Series(data)
        for validator in schema.validators:
            validator(row_series)
        self.tables[table_name] = pd.concat([df, pd.DataFrame([row_series])], ignore_index=True)
        if schema.derived is not None:
            self.tables[table_name] = schema.derived(self.tables[table_name])

    def set_table(self, table_name: str, df: pd.DataFrame) -> None:
        if table_name not in INPUT_SCHEMAS:
            raise KeyError(f"Unknown table '{table_name}'")
        schema = INPUT_SCHEMAS[table_name]
        df_copy = pd.DataFrame(df).copy()
        if df_copy.empty:
            self.tables[table_name] = pd.DataFrame(columns=list(schema.columns.keys()))
            return

        # Align incoming column labels to the schema using normalized keys so that
        # edits made via UI data editors (which may alter capitalisation or add
        # whitespace) still map back to the canonical column names. This guards
        # against downstream KeyError issues such as missing the required
        # ``product`` column in production schedules.
        rename_map: Dict[str, str] = {}
        for col in list(df_copy.columns):
            if not isinstance(col, str):
                continue
            normalised = normalize_key(col)
            if normalised in schema.columns and col != normalised:
                rename_map[col] = normalised
        if rename_map:
            df_copy = df_copy.rename(columns=rename_map)

        for col in schema.columns:
            if col not in df_copy.columns:
                default_value = schema.defaults.get(col, np.nan)
                df_copy[col] = default_value

        df_copy = df_copy[list(schema.columns.keys())]
        df_copy = df_copy.replace({"": np.nan})

        for col, dtype in schema.columns.items():
            if dtype == "float":
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            elif dtype == "int":
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            elif dtype == "bool":
                df_copy[col] = df_copy[col].fillna(schema.defaults.get(col, False)).astype(bool)
            elif dtype == "str":
                df_copy[col] = df_copy[col].astype(str).where(df_copy[col].notna(), None)

        df_copy = df_copy.dropna(how="all").reset_index(drop=True)

        for _, row in df_copy.iterrows():
            for validator in schema.validators:
                validator(row)

        if schema.derived is not None and not df_copy.empty:
            df_copy = schema.derived(df_copy)

        self.tables[table_name] = df_copy.reset_index(drop=True)

    def remove_row(self, table_name: str, row_id: int) -> None:
        df = self.ensure_table(table_name)
        if not 0 <= row_id < len(df):
            raise IndexError("row_id out of bounds")
        self.tables[table_name] = df.drop(df.index[row_id]).reset_index(drop=True)

    def load_from_workbook(self, sheets: Mapping[str, pd.DataFrame]) -> None:
        for table_name in INPUT_SCHEMAS:
            schema = INPUT_SCHEMAS[table_name]
            best_match = None
            for key, df in sheets.items():
                cols = {normalize_key(c): c for c in df.columns if isinstance(c, str)}
                required = set(schema.columns.keys()) & set(cols.keys())
                if required:
                    best_match = key
                    break
            if best_match:
                df_raw = sheets[best_match]
                df_norm = pd.DataFrame()
                for col in schema.columns.keys():
                    matches = [c for c in df_raw.columns if normalize_key(c) == col]
                    if matches:
                        df_norm[col] = df_raw[matches[0]]
                try:
                    self.set_table(table_name, df_norm)
                except Exception:
                    # fall back to raw dropna if validation fails; retain best effort load
                    df_basic = df_norm.copy()
                    for col in schema.columns.keys():
                        if col not in df_basic.columns:
                            df_basic[col] = schema.defaults.get(col, np.nan)
                    self.tables[table_name] = df_basic.dropna(how="all").reset_index(drop=True)
                    if schema.derived is not None and not self.tables[table_name].empty:
                        self.tables[table_name] = schema.derived(self.tables[table_name])
###############################################################################
# Section 4: Configuration builder
###############################################################################


def build_config(assumptions: Mapping[str, object], tables: InputTables) -> Dict[str, object]:
    cfg = {
        "projection_horizon": dict(DEFAULTS["horizon"]),
        "production_horizon": dict(DEFAULTS["production_horizon"]),
        "increment_profile": "base",
        "global_inputs": dict(DEFAULTS["global"]),
        "prices": {k: dict(v) for k, v in DEFAULTS["prices"].items()},
        "production": dict(DEFAULTS["production"]),
        "opex": dict(DEFAULTS["opex"]),
        "working_capital": dict(DEFAULTS["working_capital"]),
        "debt": {"tranches": [dict(t) for t in DEFAULTS["debt"]["tranches"]]},
        "tax": dict(DEFAULTS["tax"]),
        "risk_params": DEFAULTS["risk_params"].copy(),
        "inflation_index": DEFAULTS["inflation_index"].copy(),
        "break_even_inputs": DEFAULTS["break_even_inputs"].copy(),
        "offtake_terms": DEFAULTS["offtake_terms"].copy(),
        "dispatch_constraints": DEFAULTS["dispatch_constraints"].copy(),
        "construction_schedule": DEFAULTS["construction_schedule"].copy(),
        "completion_tests": DEFAULTS["completion_tests"].copy(),
        "reserve_accounts": DEFAULTS["reserve_accounts"].copy(),
        "covenant_thresholds": DEFAULTS["covenant_thresholds"].copy(),
        "lender_cases": DEFAULTS["lender_cases"].copy(),
        "yearly_increments": DEFAULTS["yearly_increments"].copy(),
        "debt_sizing_params": DEFAULTS["debt_sizing_params"].copy(),
        "covenant_cure_rules": DEFAULTS["covenant_cure_rules"].copy(),
        "construction_risk_mc": DEFAULTS["construction_risk_mc"].copy(),
        "hedging_program": DEFAULTS["hedging_program"].copy(),
        "sustainability_compliance": DEFAULTS["sustainability_compliance"].copy(),
        "benchmark_library": DEFAULTS["benchmark_library"].copy(),
    }

    for key, value in assumptions.items():
        if key in {"start_year", "end_year", "start_month"}:
            cfg["projection_horizon"][key] = _coerce_int(value, cfg["projection_horizon"][key])
        elif key == "frequency":
            cfg["projection_horizon"]["frequency"] = str(value).lower()
        elif key in cfg["global_inputs"]:
            if isinstance(cfg["global_inputs"][key], str):
                cfg["global_inputs"][key] = _coerce_str(value, cfg["global_inputs"][key])
            else:
                cfg["global_inputs"][key] = _coerce_float(value, cfg["global_inputs"][key])
        elif key.endswith("price_per_litre") or key.endswith("price_per_ton") or key.endswith("tariff_per_mwh"):
            for product in PRODUCTS:
                if product in key:
                    cfg["prices"].setdefault(product, dict(DEFAULTS["prices"][product]))
                    cfg["prices"][product]["base_price"] = _coerce_float(value, cfg["prices"][product]["base_price"])
        elif key.endswith("price_escalation") or key.endswith("price_escalation_pa"):
            for product in PRODUCTS:
                if product in key:
                    cfg["prices"].setdefault(product, dict(DEFAULTS["prices"][product]))
                    cfg["prices"][product]["price_escalation_pa"] = _coerce_float(value, cfg["prices"][product]["price_escalation_pa"])
        elif key in {"plant_availability", "loss_factor"}:
            cfg["production"][key] = _coerce_float(value, cfg["production"][key])
        elif key in {"investor_share", "owner_share"}:
            cfg["global_inputs"][key] = _coerce_float(value, cfg["global_inputs"][key])
        elif key in {"feedstock_scenario", "scenario"}:
            cfg["production"]["feedstock_scenario"] = _coerce_str(value, "HYBRID").upper()
        elif key in {"farm_share", "hybrid_farm_share"}:
            cfg["production"]["farm_share"] = _coerce_float(value, cfg["production"].get("farm_share", 0.5))
        elif key in {"dso_days", "dio_days", "dpo_days"}:
            cfg["working_capital"][key] = _coerce_float(value, cfg["working_capital"][key])
        elif key in {"debt_ratio", "loan_to_value"}:
            cfg["debt"]["target_ratio"] = _coerce_float(value, 0.6)
        elif key in {"debt_rate", "interest_rate"}:
            cfg["debt"].setdefault("global_rate", _coerce_float(value, 0.1))
        elif key in {"increment_profile", "yearly_increment_profile"}:
            cfg["increment_profile"] = str(value)

    tables.ensure_table("projection_horizon")
    if not tables.tables["projection_horizon"].empty:
        horizon_row = tables.tables["projection_horizon"].iloc[0]
        cfg["projection_horizon"].update({
            "start_year": _coerce_int(horizon_row.get("start_year"), cfg["projection_horizon"]["start_year"]),
            "end_year": _coerce_int(horizon_row.get("end_year"), cfg["projection_horizon"]["end_year"]),
            "start_month": _coerce_int(horizon_row.get("start_month"), cfg["projection_horizon"]["start_month"]),
            "frequency": horizon_row.get("frequency", cfg["projection_horizon"]["frequency"]),
        })

    cfg["production_horizon"] = {
        "start_year": int(cfg["projection_horizon"]["start_year"]),
        "end_year": int(cfg["projection_horizon"]["end_year"]),
    }

    if not tables.ensure_table("global_inputs").empty:
        global_row = tables.tables["global_inputs"].iloc[0]
        for key in cfg["global_inputs"].keys():
            val = global_row.get(key)
            if pd.notna(val):
                if isinstance(cfg["global_inputs"][key], str):
                    cfg["global_inputs"][key] = str(val)
                else:
                    cfg["global_inputs"][key] = float(val)

    tables.ensure_table("working_capital_days")
    if not tables.tables["working_capital_days"].empty:
        wc_row = tables.tables["working_capital_days"].iloc[0]
        for key in cfg["working_capital"].keys():
            val = wc_row.get(key)
            if pd.notna(val):
                cfg["working_capital"][key] = float(val)

    for table_name in (
        "revenue_params",
        "break_even_inputs",
        "production_annual",
        "production_monthly",
        "direct_costs_monthly",
        "staff_costs_monthly",
        "other_opex_monthly",
        "ar_other_assets",
        "inventory_ap",
        "debt_tranches",
        "tax_schedule",
        "inflation_index",
        "risk_params",
        "capex_lines",
        "tornado_drivers",
        "monte_carlo_settings",
        "scenario_comparison",
        "offtake_terms",
        "dispatch_constraints",
        "construction_schedule",
        "completion_tests",
        "reserve_accounts",
        "covenant_thresholds",
        "lender_cases",
        "yearly_increments",
        "debt_sizing_params",
        "covenant_cure_rules",
        "construction_risk_mc",
        "hedging_program",
        "sustainability_compliance",
        "benchmark_library",
    ):
        df = tables.ensure_table(table_name)
        if not df.empty:
            cfg[table_name] = df.copy().reset_index(drop=True)

    revenue_table = tables.ensure_table("revenue_params")
    if not revenue_table.empty:
        for _, row in revenue_table.iterrows():
            product = str(row.get("product", "")).strip().lower()
            if not product:
                continue
            if product not in PRODUCTS:
                continue
            params = cfg["prices"].setdefault(product, dict(DEFAULTS["prices"][product]))
            if pd.notna(row.get("base_price")):
                params["base_price"] = float(row["base_price"])
            if pd.notna(row.get("price_escalation_pa")):
                params["price_escalation_pa"] = float(row["price_escalation_pa"])
            if isinstance(row.get("price_indexation"), str) and row.get("price_indexation"):
                params["price_indexation"] = str(row["price_indexation"])
            if isinstance(row.get("uom"), str) and row.get("uom"):
                params["uom"] = str(row["uom"])
            if pd.notna(row.get("revenue_share")):
                params["revenue_share"] = float(row["revenue_share"])
            if isinstance(row.get("tariff_structure"), str) and row.get("tariff_structure"):
                params["tariff_structure"] = str(row["tariff_structure"])

    total_share = cfg["global_inputs"]["investor_share"] + cfg["global_inputs"]["owner_share"]
    if not math.isclose(total_share, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        cfg["global_inputs"]["owner_share"] = 1.0 - cfg["global_inputs"]["investor_share"]

    cfg.setdefault("production", {})
    cfg["production"].setdefault("feedstock_scenario", "HYBRID")
    cfg["production"].setdefault("farm_share", 0.5)

    if not isinstance(cfg.get("tornado_drivers"), pd.DataFrame) or cfg["tornado_drivers"].empty:
        cfg["tornado_drivers"] = DEFAULTS["tornado_drivers"].copy()
    if not isinstance(cfg.get("monte_carlo_settings"), pd.DataFrame) or cfg["monte_carlo_settings"].empty:
        cfg["monte_carlo_settings"] = DEFAULTS["monte_carlo_settings"].copy()
    if not isinstance(cfg.get("scenario_comparison"), pd.DataFrame) or cfg["scenario_comparison"].empty:
        cfg["scenario_comparison"] = DEFAULTS["scenario_comparison"].copy()
    for table_name in (
        "offtake_terms",
        "dispatch_constraints",
        "construction_schedule",
        "completion_tests",
        "reserve_accounts",
        "covenant_thresholds",
        "lender_cases",
        "yearly_increments",
    ):
        if not isinstance(cfg.get(table_name), pd.DataFrame) or cfg[table_name].empty:
            cfg[table_name] = DEFAULTS[table_name].copy()

    return align_with_projection_horizon(cfg)
###############################################################################
# Section 5: Timeline utilities
###############################################################################


@dataclass
class Timeline:
    start_year: int
    end_year: int
    start_month: int = 1

    def monthly_index(self) -> pd.DatetimeIndex:
        start = f"{self.start_year:04d}-{self.start_month:02d}-01"
        total_years = self.end_year - self.start_year + 1
        periods = total_years * 12 - (self.start_month - 1)
        return pd.date_range(start=start, periods=periods, freq="MS")

    def annual_index(self) -> List[int]:
        return list(range(self.start_year, self.end_year + 1))


def build_yearly_increment_factors(cfg: Mapping[str, object], timeline: Timeline) -> pd.DataFrame:
    """Return monthly cumulative factors for global/price/opex/debt/capex/wc/tax."""

    monthly_index = timeline.monthly_index()
    columns = [
        "global_factor",
        "price_factor",
        "opex_factor",
        "debt_factor",
        "capex_factor",
        "wc_factor",
        "tax_factor",
    ]
    increments_df = cfg.get("yearly_increments")
    if not isinstance(increments_df, pd.DataFrame) or increments_df.empty:
        return pd.DataFrame({"date": monthly_index, **{col: 1.0 for col in columns}})

    inc = increments_df.copy()
    profile_name = str(cfg.get("increment_profile", "base")).strip().lower()
    if "profile" in inc.columns:
        inc["profile"] = inc["profile"].astype(str).str.strip().str.lower()
        selected = inc[inc["profile"] == profile_name]
        if selected.empty:
            selected = inc[inc["profile"] == "base"]
        inc = selected if not selected.empty else inc
    if "year" not in inc.columns:
        return pd.DataFrame({"date": monthly_index, **{col: 1.0 for col in columns}})
    inc["year"] = pd.to_numeric(inc["year"], errors="coerce").fillna(0).astype(int)
    inc = inc.sort_values("year")
    year_rows = pd.DataFrame({"year": sorted(set(monthly_index.year))})
    inc = year_rows.merge(inc, on="year", how="left").fillna(0.0)

    pct_map = {
        "global_factor": "global_increment_pct",
        "price_factor": "price_increment_pct",
        "opex_factor": "opex_increment_pct",
        "debt_factor": "debt_increment_pct",
        "capex_factor": "capex_increment_pct",
        "wc_factor": "wc_increment_pct",
        "tax_factor": "tax_increment_pct",
    }
    for factor_col, pct_col in pct_map.items():
        pct = pd.to_numeric(inc.get(pct_col, 0.0), errors="coerce").fillna(0.0)
        global_pct = pd.to_numeric(inc.get("global_increment_pct", 0.0), errors="coerce").fillna(0.0)
        combined = (1.0 + global_pct) * (1.0 + pct) - 1.0
        inc[factor_col] = (1.0 + combined).cumprod()

    monthly = pd.DataFrame({"date": monthly_index})
    monthly["year"] = monthly["date"].dt.year
    monthly = monthly.merge(inc[["year", *columns]], on="year", how="left").drop(columns=["year"])
    for col in columns:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce").ffill().fillna(1.0)
    return monthly


def get_increment_series(cfg: Mapping[str, object], timeline: Timeline, factor_col: str) -> pd.Series:
    factors = cfg.get("_increment_factors")
    monthly_index = timeline.monthly_index()
    if isinstance(factors, pd.DataFrame) and not factors.empty and {"date", factor_col}.issubset(factors.columns):
        work = factors.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).set_index("date")
        return pd.to_numeric(work[factor_col], errors="coerce").reindex(monthly_index, fill_value=1.0).fillna(1.0)
    return pd.Series(1.0, index=monthly_index)
###############################################################################
# Section 6: Production modeling
###############################################################################


def parse_ramp(ramp_value: object, years: int) -> List[float]:
    if isinstance(ramp_value, (list, tuple, np.ndarray)):
        values = [float(x) for x in ramp_value]
    elif isinstance(ramp_value, str):
        parts = [p.strip() for p in re.split(r"[;,]", ramp_value) if p.strip()]
        values = [float(p) for p in parts] if parts else []
    else:
        values = []
    if not values:
        values = DEFAULTS["production"]["ramp"]
    if len(values) < years:
        values.extend([values[-1]] * (years - len(values)))
    return values[:years]


def build_production_tables(cfg: Mapping[str, object], timeline: Timeline) -> Tuple[pd.DataFrame, pd.DataFrame]:
    monthly_index = timeline.monthly_index()
    annual_years = timeline.annual_index()
    prod_horizon = cfg.get("production_horizon", DEFAULTS["production_horizon"])
    prod_start_year = _coerce_int(prod_horizon.get("start_year"), timeline.start_year)
    prod_end_year = _coerce_int(prod_horizon.get("end_year"), timeline.end_year)
    risk_profile = cfg.get("risk_profile", {})
    production_factor = max(float(risk_profile.get("production", 1.0)), 0.0)
    yield_factor = max(float(risk_profile.get("yield", 1.0)), 0.0)
    scaling_factor = production_factor * yield_factor
    risk_curve = get_increment_series(cfg, timeline, "global_factor")
    yearly_risk_curve = risk_curve.groupby(risk_curve.index.year).mean()

    def _default_production_table() -> pd.DataFrame:
        base = DEFAULTS["production"]
        feedstock = base["annual_feedstock_ton"]
        adjusted_yield = base["sugarcane_yield_ton_per_ha"] * (yield_factor if yield_factor > 0 else 1.0)
        adjusted_yield = adjusted_yield if adjusted_yield > 0 else base["sugarcane_yield_ton_per_ha"]
        return pd.DataFrame(
            [
                {
                    "product": "ethanol",
                    "annual_volume": feedstock
                    * base["ethanol_litre_per_ton"]
                    * base["plant_availability"]
                    * (1 - base["loss_factor"]),
                    "availability": base["plant_availability"],
                    "loss_factor": base["loss_factor"],
                    "startup_ramp": "0.7;0.9;1.0",
                    "boe_conversion": np.nan,
                    "sugarcane_yield_ton_per_ha": adjusted_yield,
                    "farm_area_ha": feedstock / adjusted_yield if adjusted_yield else np.nan,
                },
                {
                    "product": "sugar",
                    "annual_volume": feedstock
                    * base["sugar_ton_per_ton_cane"]
                    * base["plant_availability"]
                    * (1 - base["loss_factor"]),
                    "availability": base["plant_availability"],
                    "loss_factor": base["loss_factor"],
                    "startup_ramp": "0.7;0.9;1.0",
                    "boe_conversion": np.nan,
                    "sugarcane_yield_ton_per_ha": adjusted_yield,
                    "farm_area_ha": feedstock / adjusted_yield if adjusted_yield else np.nan,
                },
                {
                    "product": "electricity",
                    "annual_volume": feedstock
                    * base["electricity_mwh_per_ton_cane"]
                    * base["plant_availability"]
                    * (1 - base["loss_factor"]),
                    "availability": base["plant_availability"],
                    "loss_factor": base["loss_factor"],
                    "startup_ramp": "0.7;0.9;1.0",
                    "boe_conversion": np.nan,
                    "sugarcane_yield_ton_per_ha": adjusted_yield,
                    "farm_area_ha": feedstock / adjusted_yield if adjusted_yield else np.nan,
                },
                {
                    "product": "animal_feed",
                    "annual_volume": feedstock
                    * base["animal_feed_ton_per_ton_cane"]
                    * base["plant_availability"]
                    * (1 - base["loss_factor"]),
                    "availability": base["plant_availability"],
                    "loss_factor": base["loss_factor"],
                    "startup_ramp": "0.7;0.9;1.0",
                    "boe_conversion": np.nan,
                    "sugarcane_yield_ton_per_ha": adjusted_yield,
                    "farm_area_ha": feedstock / adjusted_yield if adjusted_yield else np.nan,
                },
            ]
        )

    def _prepare_annual_table(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            prod_annual = raw.copy()
        else:
            prod_annual = _default_production_table()

        required_columns = {"product", "annual_volume"}
        if not required_columns.issubset(prod_annual.columns):
            prod_annual = _default_production_table()

        for col, default_val in {
            "availability": DEFAULTS["production"]["plant_availability"],
            "loss_factor": DEFAULTS["production"]["loss_factor"],
            "startup_ramp": "0.7;0.9;1.0",
        }.items():
            if col not in prod_annual.columns:
                prod_annual[col] = default_val

        prod_annual["product"] = prod_annual["product"].astype(str).str.strip()
        prod_annual = prod_annual[prod_annual["product"].str.lower() != "nan"]
        prod_annual = prod_annual[prod_annual["product"] != ""]

        if prod_annual.empty:
            prod_annual = _default_production_table()

        prod_annual["annual_volume"] = pd.to_numeric(prod_annual["annual_volume"], errors="coerce").fillna(0.0)
        prod_annual["annual_volume"] *= production_factor if production_factor else 0.0
        prod_annual["annual_volume"] *= yield_factor if yield_factor else 0.0
        if "sugarcane_yield_ton_per_ha" in prod_annual.columns:
            prod_annual["sugarcane_yield_ton_per_ha"] = (
                pd.to_numeric(prod_annual["sugarcane_yield_ton_per_ha"], errors="coerce").fillna(0.0)
                * (yield_factor if yield_factor else 1.0)
            )
        if "farm_area_ha" in prod_annual.columns and yield_factor not in {0.0, 0}:
            prod_annual["farm_area_ha"] = (
                pd.to_numeric(prod_annual["farm_area_ha"], errors="coerce").fillna(0.0)
                / yield_factor
            )
        return prod_annual

    def _monthly_from_annual(source: pd.DataFrame) -> pd.DataFrame:
        monthly_rows: List[Dict[str, object]] = []
        seasonality = np.ones(MONTHS_IN_YEAR) / MONTHS_IN_YEAR
        for _, row in source.iterrows():
            year_value = _coerce_int(row.get("year"), timeline.start_year)
            volume_value = float(row.get("volume", 0.0) or 0.0)
            if math.isnan(volume_value):
                volume_value = 0.0
            for month in range(1, MONTHS_IN_YEAR + 1):
                date = pd.Timestamp(year=year_value, month=month, day=1)
                if date not in monthly_index:
                    continue
                if date.year < prod_start_year or date.year > prod_end_year:
                    volume = 0.0
                else:
                    volume = volume_value * seasonality[month - 1]
                monthly_rows.append(
                    {
                        "date": date,
                        "product": row["product"],
                        "volume": volume,
                        "availability_override": np.nan,
                        "maintenance_downtime": 0.0,
                        "loss_override": np.nan,
                    }
                )
        return pd.DataFrame(monthly_rows)

    try:
        prod_annual = _prepare_annual_table(cfg.get("production_annual"))

        annual_rows: List[Dict[str, object]] = []
        for _, row in prod_annual.iterrows():
            ramp = parse_ramp(row.get("startup_ramp"), len(annual_years))
            for idx, year in enumerate(annual_years):
                annual_rows.append(
                    {
                        "year": year,
                        "product": row["product"],
                        "volume": float(row["annual_volume"]) * float(ramp[idx]) * float(yearly_risk_curve.get(year, 1.0)),
                        "availability": row.get("availability", DEFAULTS["production"]["plant_availability"]),
                        "loss_factor": row.get("loss_factor", DEFAULTS["production"]["loss_factor"]),
                    }
                )
        annual_df = pd.DataFrame(annual_rows)
        if not annual_df.empty:
            annual_df.loc[(annual_df["year"] < prod_start_year) | (annual_df["year"] > prod_end_year), "volume"] = 0.0

        raw_monthly = cfg.get("production_monthly") if isinstance(cfg.get("production_monthly"), pd.DataFrame) else None
        monthly_from_manual = False
        if raw_monthly is not None and not raw_monthly.empty:
            monthly_df = raw_monthly.copy()
            monthly_df.columns = [normalize_key(col) if isinstance(col, str) else col for col in monthly_df.columns]
            required_cols = {"date", "volume", "product"}
            if not required_cols.issubset(set(monthly_df.columns)):
                monthly_df = _monthly_from_annual(annual_df)
            else:
                monthly_df = monthly_df.rename(columns={col: col for col in required_cols})
                monthly_df["date"] = pd.to_datetime(monthly_df["date"], errors="coerce")
                monthly_df["product"] = monthly_df["product"].astype(str).str.strip()
                monthly_df = monthly_df[monthly_df["product"] != ""]
                monthly_df = monthly_df[monthly_df["product"].str.lower() != "nan"]
                monthly_df = monthly_df.dropna(subset=["product"])
                if monthly_df.empty:
                    monthly_df = _monthly_from_annual(annual_df)
                else:
                    monthly_df["volume"] = pd.to_numeric(monthly_df["volume"], errors="coerce").fillna(0.0)
                    mask = monthly_df["date"].dt.year.between(prod_start_year, prod_end_year)
                    monthly_df.loc[~mask, "volume"] = 0.0
                    monthly_from_manual = True
        else:
            monthly_df = _monthly_from_annual(annual_df)
            monthly_from_manual = False
    except KeyError:
        # Any unexpected column issues fall back to a fully defaulted schedule so the
        # broader model can continue executing without raising ``KeyError: 'product'``.
        prod_annual = _prepare_annual_table(None)
        annual_rows = []
        for _, row in prod_annual.iterrows():
            ramp = parse_ramp(row.get("startup_ramp"), len(annual_years))
            for idx, year in enumerate(annual_years):
                annual_rows.append(
                    {
                        "year": year,
                        "product": row["product"],
                        "volume": float(row["annual_volume"]) * float(ramp[idx]) * float(yearly_risk_curve.get(year, 1.0)),
                        "availability": row.get("availability", DEFAULTS["production"]["plant_availability"]),
                        "loss_factor": row.get("loss_factor", DEFAULTS["production"]["loss_factor"]),
                    }
                )
        annual_df = pd.DataFrame(annual_rows)
        monthly_df = _monthly_from_annual(annual_df)
        monthly_from_manual = False

    required_order = ["date", "product", "volume", "availability_override", "maintenance_downtime", "loss_override"]
    for col in required_order:
        if col not in monthly_df.columns:
            monthly_df[col] = np.nan if col != "date" else pd.NaT
    monthly_df = monthly_df[required_order]

    monthly_df["date"] = pd.to_datetime(monthly_df["date"], errors="coerce")
    monthly_df = monthly_df[monthly_df["date"].isin(monthly_index)]
    monthly_df = monthly_df.sort_values(["date", "product"]).reset_index(drop=True)

    if "year" not in annual_df.columns:
        # When the annual fallback above regenerates the schedule, the helper already
        # populates the ``year`` column. This guard simply guarantees consistency if
        # upstream inputs were malformed but not severe enough to trigger the
        # ``KeyError`` branch.
        annual_df["year"] = [year for year in annual_years for _ in range(len(prod_annual))][: len(annual_df)]

    monthly_df["volume"] = pd.to_numeric(monthly_df["volume"], errors="coerce").fillna(0.0)
    annual_df["volume"] = pd.to_numeric(annual_df["volume"], errors="coerce").fillna(0.0)
    if monthly_from_manual:
        monthly_df["volume"] *= scaling_factor

    return monthly_df, annual_df
###############################################################################
# Section 7: Pricing and revenue
###############################################################################


def build_price_curves(cfg: Mapping[str, object], timeline: Timeline) -> pd.DataFrame:
    monthly_index = timeline.monthly_index()
    price_increment = get_increment_series(cfg, timeline, "price_factor")
    inflation_rate = cfg["global_inputs"].get("inflation_rate", DEFAULTS["global"]["inflation_rate"])
    inflation_index = cfg.get("inflation_index")
    risk_profile = cfg.get("risk_profile", {})
    price_multiplier = float(risk_profile.get("price", 1.0))
    price_by_product = {}
    if isinstance(risk_profile, dict):
        price_by_product = risk_profile.get("price_by_product", {}) or {}
    if isinstance(inflation_index, pd.DataFrame) and not inflation_index.empty and "date" in inflation_index.columns:
        idx = inflation_index.copy()
        idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
        idx = idx.set_index("date").reindex(monthly_index).ffill().bfill()
    else:
        idx = pd.DataFrame(index=monthly_index, data={"cpi": (1 + inflation_rate / 12) ** np.arange(len(monthly_index))})

    records: List[Dict[str, object]] = []
    for product in PRODUCTS:
        params = cfg["prices"].get(product, DEFAULTS["prices"][product])
        base_price = params.get("base_price", DEFAULTS["prices"][product]["base_price"])
        escalation = params.get("price_escalation_pa", DEFAULTS["prices"][product]["price_escalation_pa"])
        monthly_escalation = (1 + escalation) ** (1 / 12) - 1
        product_multiplier = 1.0
        if isinstance(price_by_product, dict):
            try:
                product_multiplier = float(price_by_product.get(product, 1.0))
            except (TypeError, ValueError):
                product_multiplier = 1.0
        for i, date in enumerate(monthly_index):
            price = base_price * ((1 + monthly_escalation) ** i)
            if params.get("price_indexation", "cpi").lower() == "cpi" and "cpi" in idx:
                price *= idx.loc[date, "cpi"]
            price *= float(price_increment.loc[date]) if date in price_increment.index else 1.0
            price *= price_multiplier * product_multiplier
            records.append({"date": date, "product": product, "price": price, "uom": params.get("uom", "")})
    return pd.DataFrame(records)


def build_revenue_stack(cfg: Mapping[str, object], production_monthly: pd.DataFrame, price_curves: pd.DataFrame) -> pd.DataFrame:
    timeline = Timeline(
        start_year=int(cfg["projection_horizon"]["start_year"]),
        end_year=int(cfg["projection_horizon"]["end_year"]),
        start_month=int(cfg["projection_horizon"].get("start_month", 1)),
    )
    price_factor = get_increment_series(cfg, timeline, "price_factor")
    opex_factor = get_increment_series(cfg, timeline, "opex_factor")
    global_factor = get_increment_series(cfg, timeline, "global_factor")
    df = production_monthly.merge(price_curves, on=["date", "product"], how="left")
    df["price"].fillna(0.0, inplace=True)
    df["contracted_share"] = 0.0
    df["merchant_share"] = 1.0
    df["floor_price"] = np.nan
    df["collar_price"] = np.nan
    df["counterparty_haircut"] = 0.0
    offtake_df = cfg.get("offtake_terms")
    if isinstance(offtake_df, pd.DataFrame) and not offtake_df.empty:
        terms = offtake_df.copy()
        terms["product"] = terms["product"].astype(str).str.lower()
        df = df.merge(
            terms[
                ["product", "contracted_share", "merchant_share", "floor_price", "collar_price", "counterparty_haircut"]
            ],
            on="product",
            how="left",
            suffixes=("", "_term"),
        )
        for col in ("contracted_share", "merchant_share", "floor_price", "collar_price", "counterparty_haircut"):
            term_col = f"{col}_term"
            if term_col in df.columns:
                df[col] = df[term_col].where(df[term_col].notna(), df[col])
                df.drop(columns=[term_col], inplace=True)
    contracted_price = df["price"].clip(lower=df["floor_price"].fillna(-np.inf), upper=df["collar_price"].fillna(np.inf))
    date_values = pd.to_datetime(df["date"], errors="coerce")
    price_scale = date_values.map(lambda d: float(price_factor.get(d, 1.0)) if pd.notna(d) else 1.0)
    opex_scale = date_values.map(lambda d: float(opex_factor.get(d, 1.0)) if pd.notna(d) else 1.0)
    global_scale = date_values.map(lambda d: float(global_factor.get(d, 1.0)) if pd.notna(d) else 1.0)
    contracted_price = contracted_price * price_scale.values
    effective_price = (
        contracted_price * df["contracted_share"].fillna(0.0)
        + df["price"] * df["merchant_share"].fillna(1.0)
    )
    df["effective_price"] = effective_price
    df["contracted_volume"] = df["volume"] * df["contracted_share"].fillna(0.0)
    df["merchant_volume"] = df["volume"] * df["merchant_share"].fillna(1.0)
    haircut = (df["counterparty_haircut"].fillna(0.0) * global_scale.values).clip(lower=0.0, upper=0.95)
    df["revenue"] = df["volume"] * df["effective_price"] * (1.0 - haircut)
    dispatch_df = cfg.get("dispatch_constraints")
    if isinstance(dispatch_df, pd.DataFrame) and not dispatch_df.empty:
        dispatch = dispatch_df.copy()
        dispatch["product"] = dispatch["product"].astype(str).str.lower()
        df = df.merge(dispatch[["product", "max_dispatch_share", "curtailment_penalty_per_unit"]], on="product", how="left")
        constrained = df["max_dispatch_share"].notna()
        allowed_volume = df["volume"] * df["max_dispatch_share"].fillna(1.0)
        curtailed = np.where(constrained, np.maximum(df["volume"] - allowed_volume, 0.0), 0.0)
        df["curtailed_volume"] = curtailed
        penalty_rate = df["curtailment_penalty_per_unit"].fillna(0.0) * opex_scale.values
        df["curtailment_penalty"] = df["curtailed_volume"] * penalty_rate
        df["revenue"] = np.where(constrained, allowed_volume * df["effective_price"] - df["curtailment_penalty"], df["revenue"])
    df["currency"] = cfg["global_inputs"].get("base_currency", "USD")
    risk_profile = cfg.get("risk_profile", {})
    revenue_multiplier = float(risk_profile.get("revenue", 1.0))
    if not math.isclose(revenue_multiplier, 1.0):
        df["revenue"] *= revenue_multiplier
    return df
###############################################################################
# Section 8: CAPEX and depreciation
###############################################################################


def parse_date_str(value: object, default: pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value)
    if isinstance(value, str) and value:
        if len(value) == 7 and value.count("-") == 1:
            value = value + "-01"
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return default
        return pd.Timestamp(parsed)
    if isinstance(value, (int, float)) and not math.isnan(value):
        year = int(value)
        return pd.Timestamp(year=year, month=1, day=1)
    return default


def build_capex_depr_monthly(cfg: Mapping[str, object], timeline: Timeline) -> Dict[str, pd.DataFrame]:
    monthly_index = timeline.monthly_index()
    capex_increment = get_increment_series(cfg, timeline, "capex_factor")
    capex_lines = cfg.get("capex_lines")
    if not isinstance(capex_lines, pd.DataFrame) or capex_lines.empty:
        capex_lines = pd.DataFrame(
            [
                {
                    "item_name": "Plant",
                    "category": "plant",
                    "amount": 35_000_000.0,
                    "currency": "USD",
                    "start_date": f"{timeline.start_year}-01",
                    "end_date": f"{timeline.start_year}-12",
                    "life_years": 15,
                    "depr_method": "straight",
                    "depr_rate_override": np.nan,
                    "vat_rate": 0.0,
                    "vat_recovery_lag_months": 0,
                    "capitalized": True,
                    "is_farm_capex": False,
                }
            ]
        )

    capex_records: List[Dict[str, object]] = []
    depr_records: List[Dict[str, object]] = []

    for _, row in capex_lines.iterrows():
        amount = float(row.get("amount", 0.0))
        start = parse_date_str(row.get("start_date"), monthly_index[0])
        end = parse_date_str(row.get("end_date"), monthly_index[0])
        if end < start:
            end = start
        months = max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)
        monthly_amount = amount / months
        dates = pd.date_range(start=start, periods=months, freq="MS")
        for date in dates:
            if date not in monthly_index:
                continue
            escalated_amount = monthly_amount * float(capex_increment.loc[date]) if date in capex_increment.index else monthly_amount
            capex_records.append(
                {
                    "date": date,
                    "item_name": row.get("item_name"),
                    "category": row.get("category"),
                    "amount": escalated_amount,
                    "currency": row.get("currency", "USD"),
                    "vat_outflow": escalated_amount * float(row.get("vat_rate", 0.0)),
                    "is_farm_capex": bool(row.get("is_farm_capex", False)),
                }
            )
        life_years = max(1, _coerce_int(row.get("life_years"), 10))
        depr_rate_override = row.get("depr_rate_override")
        if pd.notna(depr_rate_override) and float(depr_rate_override) > 0:
            life_months = int(round(12 / float(depr_rate_override)))
        else:
            life_months = life_years * 12
        start_idx = np.searchsorted(monthly_index, start)
        for m in range(life_months):
            idx = start_idx + m
            if idx >= len(monthly_index):
                break
            depr_records.append(
                {
                    "date": monthly_index[idx],
                    "item_name": row.get("item_name"),
                    "depr": (amount / life_months) * float(capex_increment.loc[monthly_index[idx]]),
                }
            )

    capex_df = pd.DataFrame(capex_records)
    if capex_df.empty:
        capex_df = pd.DataFrame({"date": monthly_index, "amount": 0.0})
    capex_df = capex_df.groupby("date").sum(numeric_only=True).reindex(monthly_index, fill_value=0.0).reset_index()
    capex_df.rename(columns={"index": "date"}, inplace=True)
    capex_df["cumulative_capex"] = capex_df["amount"].cumsum()

    depr_df = pd.DataFrame(depr_records)
    if depr_df.empty:
        depr_df = pd.DataFrame({"date": monthly_index, "depr": 0.0})
    depr_df = depr_df.groupby("date").sum(numeric_only=True).reindex(monthly_index, fill_value=0.0).reset_index()
    depr_df.rename(columns={"index": "date"}, inplace=True)
    depr_df["accum_depr"] = depr_df["depr"].cumsum()

    return {"capex": capex_df, "depreciation": depr_df}


def apply_construction_risk(cfg: Mapping[str, object], timeline: Timeline, capex_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply construction draw S-curves, delays, and contingency hierarchy."""
    monthly_index = timeline.monthly_index()
    construction = cfg.get("construction_schedule")
    if not isinstance(construction, pd.DataFrame) or construction.empty:
        empty = pd.DataFrame(columns=["epc_package", "base_amount", "adjusted_amount", "delay_months", "contingency_amount", "rank"])
        return capex_df, empty
    records: List[Dict[str, object]] = []
    capex_series = pd.Series(0.0, index=monthly_index)
    capex_increment = get_increment_series(cfg, timeline, "capex_factor")
    for _, row in construction.iterrows():
        amount = float(row.get("amount", 0.0) or 0.0)
        contingency = float(row.get("contingency_amount", 0.0) or 0.0)
        delay_months = max(0, _coerce_int(row.get("delay_months"), 0))
        start = parse_date_str(row.get("start_date"), monthly_index[0]) + pd.DateOffset(months=delay_months)
        end = parse_date_str(row.get("end_date"), monthly_index[0]) + pd.DateOffset(months=delay_months)
        local_index = pd.date_range(start=start, end=end, freq="MS")
        if local_index.empty:
            local_index = pd.DatetimeIndex([start])
        curve = parse_draw_curve(row.get("draw_curve"), local_index)
        start_factor = float(capex_increment.loc[start]) if start in capex_increment.index else 1.0
        package_total = (amount + contingency) * start_factor
        for dt, val in curve.items():
            if dt in capex_series.index:
                capex_series.loc[dt] += float(val) * package_total
        records.append(
            {
                "epc_package": row.get("epc_package", "EPC"),
                "base_amount": amount,
                "adjusted_amount": package_total,
                "delay_months": delay_months,
                "contingency_amount": contingency,
                "rank": _coerce_int(row.get("contingency_rank"), 1),
            }
        )
    adjusted = capex_df.copy()
    if "date" not in adjusted.columns:
        adjusted["date"] = monthly_index[: len(adjusted)] if len(adjusted) else pd.NaT
    adjusted["date"] = pd.to_datetime(adjusted["date"], errors="coerce")
    adjusted = adjusted.dropna(subset=["date"])
    adjusted = adjusted.set_index("date").reindex(monthly_index, fill_value=0.0).reset_index()
    adjusted["amount"] = capex_series.values
    adjusted["cumulative_capex"] = adjusted["amount"].cumsum()
    return adjusted, pd.DataFrame(records)
###############################################################################
# Section 9: Debt modeling
###############################################################################


def parse_draw_curve(draw_curve: object, monthly_index: pd.DatetimeIndex) -> pd.Series:
    if draw_curve is None or (isinstance(draw_curve, float) and math.isnan(draw_curve)):
        series = pd.Series(0.0, index=monthly_index)
        series.iloc[0] = 1.0
        return series
    if isinstance(draw_curve, str):
        try:
            data = json.loads(draw_curve)
            if isinstance(data, list):
                arr = np.array(data, dtype=float)
                if arr.sum() != 0:
                    arr = arr / arr.sum()
                series = pd.Series(0.0, index=monthly_index)
                series.iloc[: len(arr)] = arr
                return series
            if isinstance(data, dict):
                series = pd.Series(0.0, index=monthly_index)
                for key, value in data.items():
                    date = parse_date_str(key, monthly_index[0])
                    if date in series.index:
                        series.loc[date] = float(value)
                total = series.sum()
                if total != 0:
                    series = series / total
                return series
        except Exception:
            pass
    if isinstance(draw_curve, (list, tuple, np.ndarray)):
        arr = np.array(draw_curve, dtype=float)
        if arr.sum() != 0:
            arr = arr / arr.sum()
        series = pd.Series(0.0, index=monthly_index)
        series.iloc[: len(arr)] = arr
        return series
    series = pd.Series(0.0, index=monthly_index)
    series.iloc[0] = 1.0
    return series


def build_debt_schedule(
    cfg: Mapping[str, object],
    timeline: Timeline,
    capex_df: pd.DataFrame,
    cfads_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    monthly_index = timeline.monthly_index()
    debt_increment = get_increment_series(cfg, timeline, "debt_factor")
    tranches = cfg.get("debt_tranches") if "debt_tranches" in cfg else cfg["debt"].get("tranches", [])
    if isinstance(tranches, pd.DataFrame):
        tranches = tranches.to_dict("records")
    schedule_frames: List[pd.DataFrame] = []
    total_capex = capex_df["amount"].sum()

    for tranche in tranches:
        name = tranche.get("name", "Tranche")
        share = float(tranche.get("share", 1.0))
        principal_total = total_capex * share
        start_year_value = _coerce_int(tranche.get("start_year"), monthly_index[0].year)
        start_year_value = max(monthly_index[0].year, min(monthly_index[-1].year, start_year_value))
        start_mask = monthly_index.year >= start_year_value
        if start_mask.any():
            loan_start_idx = int(np.argmax(start_mask))
        else:  # pragma: no cover - defensive fallback
            loan_start_idx = 0
        draw_curve = parse_draw_curve(tranche.get("draw_curve"), monthly_index)
        if loan_start_idx > 0:
            draw_curve = draw_curve.copy()
            draw_curve.iloc[:loan_start_idx] = 0.0
            total = draw_curve.sum()
            if total <= 0:
                draw_curve.iloc[loan_start_idx] = 1.0
            else:
                draw_curve = draw_curve / total
        draws = draw_curve * principal_total
        base_rate_total = float(tranche.get("interest_rate", cfg["debt"].get("global_rate", 0.1))) + float(tranche.get("base_rate", 0.0)) + float(tranche.get("margin", 0.0))
        tenor_years = max(0, _coerce_int(tranche.get("tenor_years"), 8))
        grace_years = max(0, _coerce_int(tranche.get("grace_years"), 1))
        capitalize_idc = bool(tranche.get("capitalize_idc", True))
        amortization = str(tranche.get("amortization", "straight")).lower()
        target_dscr = float(tranche.get("target_dscr", 1.3) or 1.3)
        sculpting_enabled = bool(tranche.get("sculpting_enabled", False)) or amortization == "sculpted"

        balances = []
        interests = []
        principals = []
        fees = []
        services = []
        balance = 0.0
        annuity_payment = None
        first_factor = float(debt_increment.iloc[loan_start_idx]) if len(debt_increment) > loan_start_idx else 1.0
        monthly_rate = (base_rate_total * first_factor) / 12
        for i, date in enumerate(monthly_index):
            effective_rate = base_rate_total * float(debt_increment.loc[date]) if date in debt_increment.index else base_rate_total
            monthly_rate_i = effective_rate / 12.0
            draw = draws.iloc[i]
            balance += draw
            interest = balance * monthly_rate_i
            principal_payment = 0.0
            fee_payment = 0.0
            months_since_start = max(0, i - loan_start_idx)
            years_since_start = months_since_start // 12
            if years_since_start >= grace_years and tenor_years > 0:
                if amortization == "annuity":
                    if annuity_payment is None:
                        n = tenor_years * 12
                        if monthly_rate_i == 0:
                            annuity_payment = principal_total / n
                        else:
                            factor = (1 + monthly_rate_i) ** n
                            annuity_payment = principal_total * (monthly_rate_i * factor) / max(1e-9, factor - 1)
                        principal_payment = max(0.0, annuity_payment - interest)
                elif sculpting_enabled and cfads_series is not None:
                    cfads_value = float(cfads_series.reindex(monthly_index, fill_value=0.0).iloc[i])
                    allowed_debt_service = max(0.0, cfads_value / max(target_dscr, 1e-6))
                    principal_payment = max(0.0, allowed_debt_service - interest)
                else:
                    principal_payment = principal_total / max(1, tenor_years * 12)
                principal_payment = min(principal_payment, balance)
                balance -= principal_payment
            else:
                if capitalize_idc:
                    balance += interest
                    interest = 0.0
                else:
                    balance += 0.0
            if i == loan_start_idx:
                upfront_raw = float(tranche.get("fees_upfront", 0.0) or 0.0)
                fee_payment += principal_total * upfront_raw if abs(upfront_raw) <= 1.0 else upfront_raw
            annual_fee_rate = float(tranche.get("fees_annual", 0.0) or 0.0)
            if annual_fee_rate > 0:
                fee_payment += balance * annual_fee_rate / 12.0
            balances.append(balance)
            interests.append(interest)
            principals.append(principal_payment)
            fees.append(fee_payment)
            services.append(interest + principal_payment + fee_payment)
        schedule_frames.append(
            pd.DataFrame(
                {
                    "date": monthly_index,
                    "tranche": name,
                    "draw": draws.values,
                    "interest": interests,
                    "principal": principals,
                    "fees": fees,
                    "debt_service": services,
                    "balance": balances,
                }
            )
        )
    if schedule_frames:
        schedule = pd.concat(schedule_frames, ignore_index=True)
    else:
        schedule = pd.DataFrame({"date": monthly_index, "tranche": [], "draw": [], "interest": [], "principal": [], "fees": [], "debt_service": [], "balance": []})
    return schedule
###############################################################################
# Section 10: Working capital and tax
###############################################################################


def working_capital_block(revenue_df: pd.DataFrame, cost_df: pd.DataFrame, cfg: Mapping[str, object], timeline: Timeline) -> pd.DataFrame:
    monthly_index = timeline.monthly_index()
    cost_df = _derive_direct_costs(cost_df)
    revenue = revenue_df.groupby("date")["revenue"].sum().reindex(monthly_index, fill_value=0.0)
    cost = cost_df.groupby("date")["amount"].sum().reindex(monthly_index, fill_value=0.0)
    wc_days = cfg.get("working_capital", DEFAULTS["working_capital"])
    dso = wc_days.get("dso_days", DEFAULTS["working_capital"]["dso_days"])
    dio = wc_days.get("dio_days", DEFAULTS["working_capital"]["dio_days"])
    dpo = wc_days.get("dpo_days", DEFAULTS["working_capital"]["dpo_days"])
    wc_factor = get_increment_series(cfg, timeline, "wc_factor")
    dso_series = float(dso) * wc_factor
    dio_series = float(dio) * wc_factor
    dpo_series = float(dpo) * wc_factor

    ar = revenue * dso_series / 365.0
    inv = cost * dio_series / 365.0
    ap = cost * dpo_series / 365.0
    net_wc = ar + inv - ap
    delta_wc = net_wc.diff().fillna(net_wc)

    return pd.DataFrame(
        {
            "date": monthly_index,
            "accounts_receivable": ar.values,
            "inventory": inv.values,
            "accounts_payable": ap.values,
            "net_working_capital": net_wc.values,
            "delta_working_capital": delta_wc.values,
        }
    )


def tax_block(pnl_df: pd.DataFrame, cfg: Mapping[str, object]) -> pd.DataFrame:
    base_rate = cfg.get("tax", {}).get("base_tax_rate", DEFAULTS["tax"]["base_tax_rate"])
    min_tax = cfg.get("tax", {}).get("min_tax", 0.0)
    nol_years = cfg.get("tax", {}).get("loss_carryforward_years")

    nol_queue: List[Tuple[int, float]] = []
    rows: List[Dict[str, object]] = []
    tax_factor_by_year: Dict[int, float] = {}
    increments_df = cfg.get("yearly_increments")
    if isinstance(increments_df, pd.DataFrame) and not increments_df.empty and "year" in increments_df.columns:
        inc = increments_df.copy()
        inc["year"] = pd.to_numeric(inc["year"], errors="coerce").fillna(0).astype(int)
        global_pct = pd.to_numeric(inc.get("global_increment_pct", 0.0), errors="coerce").fillna(0.0)
        tax_pct = pd.to_numeric(inc.get("tax_increment_pct", 0.0), errors="coerce").fillna(0.0)
        combined = (1.0 + global_pct) * (1.0 + tax_pct) - 1.0
        inc["tax_factor"] = (1.0 + combined).cumprod()
        tax_factor_by_year = dict(zip(inc["year"], inc["tax_factor"]))
    for idx, row in pnl_df.iterrows():
        date = row["date"]
        year = pd.Timestamp(date).year if not pd.isna(date) else 0
        tax_factor = float(tax_factor_by_year.get(year, 1.0))
        effective_base_rate = max(float(base_rate) * tax_factor, 0.0)
        effective_min_tax = max(float(min_tax) * tax_factor, 0.0)
        taxable_income = row["EBIT"]
        available_loss = sum(val for _, val in nol_queue)
        taxable_after_loss = taxable_income - available_loss
        tax = 0.0
        if taxable_after_loss > 0:
            tax = max(taxable_after_loss * effective_base_rate, taxable_after_loss * effective_min_tax)
            nol_queue.clear()
        else:
            nol_queue.append((idx, -taxable_after_loss))
            if nol_years is not None:
                nol_horizon = max(0, _coerce_int(nol_years, 0))
                if nol_horizon > 0:
                    nol_queue = [(age, val) for age, val in nol_queue if idx - age < nol_horizon * 12]
        rows.append({"date": date, "taxable_income": taxable_after_loss, "tax": tax, "loss_carryforward": sum(val for _, val in nol_queue)})
    return pd.DataFrame(rows)
###############################################################################
# Section 11: Financial statements
###############################################################################


def statements_monthly(cfg: Mapping[str, object], timeline: Timeline, revenue_df: pd.DataFrame, production_df: pd.DataFrame, capex_info: Dict[str, pd.DataFrame], debt_schedule: pd.DataFrame, wc_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    monthly_index = timeline.monthly_index()
    opex_factor = get_increment_series(cfg, timeline, "opex_factor")
    depreciation_df = capex_info.get("depreciation", pd.DataFrame()).copy()
    if "date" not in depreciation_df.columns:
        depreciation_df = pd.DataFrame({"date": monthly_index, "depr": 0.0})
    depreciation_df["date"] = pd.to_datetime(depreciation_df["date"], errors="coerce")
    depreciation_df = depreciation_df.dropna(subset=["date"])
    if "depr" not in depreciation_df.columns:
        depreciation_df["depr"] = 0.0
    depreciation = depreciation_df.set_index("date").reindex(monthly_index, fill_value=0.0)["depr"].values

    capex_df = capex_info.get("capex", pd.DataFrame()).copy()
    if "date" not in capex_df.columns:
        capex_df = pd.DataFrame({"date": monthly_index, "amount": 0.0})
    capex_df["date"] = pd.to_datetime(capex_df["date"], errors="coerce")
    capex_df = capex_df.dropna(subset=["date"])
    if "amount" not in capex_df.columns:
        capex_df["amount"] = 0.0
    capex = capex_df.set_index("date").reindex(monthly_index, fill_value=0.0)["amount"].values

    direct_costs = cfg.get("direct_costs_monthly") if "direct_costs_monthly" in cfg else pd.DataFrame()
    if isinstance(direct_costs, pd.DataFrame) and not direct_costs.empty:
        direct_costs = direct_costs.copy()
        direct_costs["date"] = pd.to_datetime(direct_costs["date"], errors="coerce")
    else:
        direct_costs = pd.DataFrame({
            "date": monthly_index,
            "unit_price": 0.0,
            "quantity": 0.0,
            "amount": 0.0,
        })
    direct_costs = _derive_direct_costs(direct_costs)
    direct_costs_total = direct_costs.groupby("date")["amount"].sum().reindex(monthly_index, fill_value=0.0)
    direct_costs_total = direct_costs_total * opex_factor

    staff_costs = cfg.get("staff_costs_monthly") if "staff_costs_monthly" in cfg else pd.DataFrame()
    default_currency = cfg.get("global_inputs", {}).get(
        "base_currency", DEFAULTS["global"].get("base_currency", "USD")
    )
    if isinstance(staff_costs, pd.DataFrame) and not staff_costs.empty:
        staff_costs = staff_costs.copy()
        staff_costs["date"] = pd.to_datetime(staff_costs["date"], errors="coerce")
    else:
        staff_costs = pd.DataFrame(
            {
                "date": monthly_index,
                "dept": "Operations",
                "headcount": 0.0,
                "gross_pay": 0.0,
                "benefits": 0.0,
                "training": 0.0,
                "other": 0.0,
                "currency": default_currency,
            }
        )
    for col in ("dept", "currency"):
        if col not in staff_costs.columns:
            default_val = "" if col == "dept" else default_currency
            staff_costs[col] = default_val
    staff_costs["dept"] = (
        staff_costs["dept"].fillna("Unassigned").astype(str).replace({"": "Unassigned"})
    )
    staff_costs["currency"] = (
        staff_costs["currency"].fillna(default_currency).astype(str).replace({"": default_currency})
    )
    if "headcount" not in staff_costs.columns:
        staff_costs["headcount"] = 0.0

    staff_costs = _derive_staff_costs(staff_costs)

    headcount_series = staff_costs["headcount"].fillna(0.0)
    cost_components = ("gross_pay", "benefits", "training", "other")

    staff_costs_detail = (
        staff_costs.groupby(["date", "dept", "currency"], dropna=False)[
            ["headcount", "gross_pay", "benefits", "training", "other"]
        ]
        .sum()
        .reset_index()
    )
    staff_costs_detail = staff_costs_detail.sort_values(["date", "dept"]).reset_index(drop=True)
    labour_multiplier = float(cfg.get("risk_profile", {}).get("labour", 1.0))
    if not math.isclose(labour_multiplier, 1.0):
        for col in ("gross_pay", "benefits", "training", "other"):
            staff_costs_detail[col] *= labour_multiplier
    staff_costs_detail = staff_costs_detail.merge(
        opex_factor.rename("opex_factor").rename_axis("date").reset_index(),
        on="date",
        how="left",
    )
    for col in ("gross_pay", "benefits", "training", "other", "total_cost"):
        if col in staff_costs_detail.columns:
            staff_costs_detail[col] = pd.to_numeric(staff_costs_detail[col], errors="coerce").fillna(0.0) * staff_costs_detail["opex_factor"].fillna(1.0)
    for col in ("gross_pay", "benefits", "training", "other"):
        staff_costs_detail[f"{col}_per_head"] = np.where(
            staff_costs_detail["headcount"] > 0,
            staff_costs_detail[col] / staff_costs_detail["headcount"],
            0.0,
        )
    staff_costs_detail["total_cost"] = staff_costs_detail[
        ["gross_pay", "benefits", "training", "other"]
    ].sum(axis=1)
    staff_costs_detail["total_cost_per_head"] = np.where(
        staff_costs_detail["headcount"] > 0,
        staff_costs_detail["total_cost"] / staff_costs_detail["headcount"],
        0.0,
    )

    staff_costs_total = (
        staff_costs_detail.groupby("date")[["gross_pay", "benefits", "training", "other"]]
        .sum()
        .reindex(monthly_index, fill_value=0.0)
        .sum(axis=1)
    )

    other_opex = cfg.get("other_opex_monthly") if "other_opex_monthly" in cfg else pd.DataFrame()
    if isinstance(other_opex, pd.DataFrame) and not other_opex.empty:
        other_opex = other_opex.copy()
        other_opex["date"] = pd.to_datetime(other_opex["date"], errors="coerce")
    else:
        other_opex = pd.DataFrame({"date": monthly_index, "amount": DEFAULTS["opex"]["fixed_opex_per_month"]})
    other_opex_total = other_opex.groupby("date")["amount"].sum().reindex(monthly_index, fill_value=DEFAULTS["opex"]["fixed_opex_per_month"])
    other_opex_total = other_opex_total * opex_factor

    revenue_total = revenue_df.groupby("date")["revenue"].sum().reindex(monthly_index, fill_value=0.0)
    cogs_total = direct_costs_total + other_opex_total
    gross_profit = revenue_total - cogs_total
    opex_total = staff_costs_total
    ebitda = gross_profit - opex_total

    interest = debt_schedule.groupby("date")["interest"].sum().reindex(monthly_index, fill_value=0.0)
    pnl_df = pd.DataFrame(
        {
            "date": monthly_index,
            "Revenue": revenue_total.values,
            "COGS": cogs_total.values,
            "DirectCosts": direct_costs_total.reindex(monthly_index, fill_value=0.0).values,
            "OtherOpexCosts": other_opex_total.reindex(monthly_index, fill_value=0.0).values,
            "StaffCosts": staff_costs_total.reindex(monthly_index, fill_value=0.0).values,
            "GrossProfit": gross_profit.values,
            "Opex": opex_total.values,
            "EBITDA": ebitda.values,
            "Depreciation": depreciation,
            "EBIT": (ebitda - depreciation - interest.values),
            "Interest": interest.values,
        }
    )
    tax_df = tax_block(pnl_df, cfg)
    pnl_df = pnl_df.merge(tax_df[["date", "tax"]], on="date", how="left")
    pnl_df["NetIncome"] = pnl_df["EBIT"] - pnl_df["tax"]

    wc_work = wc_df.copy() if isinstance(wc_df, pd.DataFrame) else pd.DataFrame()
    if "date" not in wc_work.columns:
        wc_work = pd.DataFrame({"date": monthly_index, "delta_working_capital": 0.0})
    wc_work["date"] = pd.to_datetime(wc_work["date"], errors="coerce")
    wc_work = wc_work.dropna(subset=["date"])
    if "delta_working_capital" not in wc_work.columns:
        wc_work["delta_working_capital"] = 0.0
    delta_wc = wc_work.set_index("date")["delta_working_capital"].reindex(monthly_index, fill_value=0.0)
    cash_from_ops = pnl_df["EBITDA"] - pnl_df["tax"] - delta_wc.values
    capex_outflow = capex
    debt_service = debt_schedule.groupby("date")["debt_service"].sum().reindex(monthly_index, fill_value=0.0)
    draws = debt_schedule.groupby("date")["draw"].sum().reindex(monthly_index, fill_value=0.0)
    cash_flow_df = pd.DataFrame(
        {
            "date": monthly_index,
            "CFO": cash_from_ops.values,
            "CFI": -capex_outflow,
            "CFF": draws.values - debt_service.values,
        }
    )
    cash_flow_df["NetCashFlow"] = cash_flow_df[["CFO", "CFI", "CFF"]].sum(axis=1)
    cash_flow_df["CashBalance"] = cash_flow_df["NetCashFlow"].cumsum()

    balance_sheet = pd.DataFrame(
        {
            "date": monthly_index,
            "Cash": cash_flow_df["CashBalance"].values,
            "AccountsReceivable": wc_df["accounts_receivable"].values,
            "Inventory": wc_df["inventory"].values,
            "PPE_Gross": capex_info["capex"]["cumulative_capex"].values,
            "PPE_Accumulated": capex_info["depreciation"]["accum_depr"].values,
            "Debt": debt_schedule.groupby("date")["balance"].sum().reindex(monthly_index, fill_value=0.0).values,
            "AccountsPayable": wc_df["accounts_payable"].values,
        }
    )
    balance_sheet["PPE_Net"] = balance_sheet["PPE_Gross"] - balance_sheet["PPE_Accumulated"]
    balance_sheet["TotalAssets"] = balance_sheet[["Cash", "AccountsReceivable", "Inventory", "PPE_Net"]].sum(axis=1)
    balance_sheet["TotalLiabilities"] = balance_sheet[["Debt", "AccountsPayable"]].sum(axis=1)
    balance_sheet["Equity"] = balance_sheet["TotalAssets"] - balance_sheet["TotalLiabilities"]

    diff = balance_sheet["TotalAssets"] - (balance_sheet["TotalLiabilities"] + balance_sheet["Equity"])
    assert (diff.abs() < 1e-3).all(), "Balance sheet does not balance"

    return {
        "pnl": pnl_df,
        "cashflow": cash_flow_df,
        "balancesheet": balance_sheet,
        "staff_costs_detail": staff_costs_detail,
    }


def aggregate_annual(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame()

    df = monthly_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    numeric_cols = df.select_dtypes(include=[float, int, np.number]).columns.tolist()
    per_head_cols = [col for col in numeric_cols if col.endswith("_per_head")]
    sum_cols = [col for col in numeric_cols if col not in per_head_cols and col != "year"]

    group_cols: List[str] = ["year"]
    for col in df.columns:
        if col in {"date", "year"}:
            continue
        if col in sum_cols or col in per_head_cols:
            continue
        group_cols.append(col)

    grouped = df.groupby(group_cols, dropna=False)
    if "headcount" in sum_cols:
        sum_targets = [col for col in sum_cols if col != "headcount"]
        if sum_targets:
            agg_df = grouped[sum_targets].sum().reset_index()
        else:
            agg_df = grouped.size().reset_index(name="_tmp")
            agg_df = agg_df.drop(columns=["_tmp"])
        headcount_series = grouped["headcount"].mean().reset_index(name="headcount")
        agg_df = agg_df.merge(headcount_series, on=group_cols, how="left")
    else:
        agg_df = grouped[sum_cols].sum().reset_index()

    if "headcount" in agg_df.columns:
        headcount = agg_df["headcount"].replace({0: np.nan})
    else:
        headcount = None

    for col in per_head_cols:
        base_col = col[: -len("_per_head")]
        if headcount is not None and base_col in agg_df.columns:
            agg_df[col] = np.where(headcount.fillna(0.0) > 0, agg_df[base_col] / headcount, 0.0)
        else:
            agg_df[col] = 0.0

    numeric_result_cols = agg_df.select_dtypes(include=[float, int, np.number]).columns
    agg_df[numeric_result_cols] = agg_df[numeric_result_cols].fillna(0.0)
    return agg_df
###############################################################################
# Section 12: Valuation metrics
###############################################################################


def npv(rate: float, cashflows: Sequence[float]) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def build_cfads_bridge(
    statements: Dict[str, pd.DataFrame],
    timeline: Timeline,
    debt_schedule: Optional[pd.DataFrame] = None,
    reserve_effects: Optional[pd.DataFrame] = None,
    cfg: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Build lender-style CFADS bridge used consistently across debt metrics."""

    monthly_index = timeline.monthly_index()
    pnl = statements.get("pnl", pd.DataFrame()).copy()
    cashflow = statements.get("cashflow", pd.DataFrame()).copy()
    for frame in (pnl, cashflow):
        if isinstance(frame, pd.DataFrame) and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    pnl = pnl.set_index("date").reindex(monthly_index, fill_value=0.0) if isinstance(pnl, pd.DataFrame) and not pnl.empty else pd.DataFrame(index=monthly_index)
    cashflow = cashflow.set_index("date").reindex(monthly_index, fill_value=0.0) if isinstance(cashflow, pd.DataFrame) and not cashflow.empty else pd.DataFrame(index=monthly_index)

    ebitda = pd.to_numeric(pnl.get("EBITDA", pd.Series(0.0, index=monthly_index)), errors="coerce").fillna(0.0)
    cash_tax = pd.to_numeric(pnl.get("tax", pd.Series(0.0, index=monthly_index)), errors="coerce").fillna(0.0)
    maintenance_capex_pct = float(cfg.get("global_inputs", {}).get("maintenance_capex_pct", 0.05)) if isinstance(cfg, Mapping) else 0.05
    maintenance_capex = np.maximum(ebitda.values, 0.0) * maintenance_capex_pct
    delta_wc = pd.to_numeric(cashflow.get("CFO", pd.Series(0.0, index=monthly_index)), errors="coerce").fillna(0.0) - (ebitda - cash_tax)
    reserve_moves = pd.Series(0.0, index=monthly_index)
    if isinstance(reserve_effects, pd.DataFrame) and not reserve_effects.empty and "date" in reserve_effects.columns:
        tmp = reserve_effects.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna(subset=["date"]).set_index("date")
        reserve_moves = pd.to_numeric(tmp.get("reserve_net_movement", 0.0), errors="coerce").reindex(monthly_index, fill_value=0.0)
    lender_adjustments = pd.Series(0.0, index=monthly_index)
    if isinstance(debt_schedule, pd.DataFrame) and not debt_schedule.empty and {"date", "fees"}.issubset(debt_schedule.columns):
        fee_series = debt_schedule.groupby("date")["fees"].sum()
        lender_adjustments = -pd.to_numeric(fee_series, errors="coerce").reindex(monthly_index, fill_value=0.0)

    cfads = ebitda - cash_tax - maintenance_capex - delta_wc - reserve_moves + lender_adjustments
    bridge = pd.DataFrame(
        {
            "date": monthly_index,
            "EBITDA": ebitda.values,
            "cash_taxes": cash_tax.values,
            "maintenance_capex": maintenance_capex,
            "working_capital_movement": delta_wc.values,
            "reserve_movement": reserve_moves.values,
            "lender_adjustments": lender_adjustments.values,
            "CFADS": cfads.values,
        }
    )
    return bridge


def size_debt_from_cfads(
    cfg: Mapping[str, object],
    timeline: Timeline,
    cfads_bridge: pd.DataFrame,
    discount_rate: float,
) -> Dict[str, object]:
    """Compute lender-style debt sizing envelope using DSCR/LLCR/PLCR constraints."""

    sizing = cfg.get("debt_sizing_params")
    if not isinstance(sizing, pd.DataFrame) or sizing.empty or not bool(sizing.iloc[0].get("enabled", True)):
        return {"enabled": False, "sized_debt": np.nan, "debt_scale_factor": 1.0}
    row = sizing.iloc[0]
    min_dscr = float(row.get("min_dscr", 1.25) or 1.25)
    llcr_floor = float(row.get("llcr_floor", 1.35) or 1.35)
    plcr_floor = float(row.get("plcr_floor", 1.5) or 1.5)
    reserve_buffer = float(row.get("reserve_buffer_pct", 0.05) or 0.0)

    cfads = pd.to_numeric(cfads_bridge.get("CFADS", pd.Series(dtype=float)), errors="coerce").fillna(0.0).values
    if cfads.size == 0:
        return {"enabled": True, "sized_debt": 0.0, "debt_scale_factor": 0.0}
    monthly_discount = (1 + max(discount_rate, 0.0)) ** (1 / 12) - 1
    rem = np.arange(len(cfads))
    pv_cfads = float(np.sum(np.maximum(cfads, 0.0) / ((1 + monthly_discount) ** rem)))
    dscr_capacity = float(np.sum(np.maximum(cfads, 0.0) / max(min_dscr, 1e-6)))
    llcr_capacity = pv_cfads / max(llcr_floor, 1e-6)
    plcr_capacity = pv_cfads / max(plcr_floor, 1e-6)
    sized_debt = max(0.0, min(dscr_capacity, llcr_capacity, plcr_capacity) * max(0.0, 1.0 - reserve_buffer))
    capex_df = cfg.get("capex_lines")
    total_capex = float(pd.to_numeric(capex_df.get("amount"), errors="coerce").fillna(0.0).sum()) if isinstance(capex_df, pd.DataFrame) and "amount" in capex_df.columns else 0.0
    scale_factor = sized_debt / total_capex if total_capex > 1e-9 else 1.0
    return {
        "enabled": True,
        "sized_debt": sized_debt,
        "capacity_dscr": dscr_capacity,
        "capacity_llcr": llcr_capacity,
        "capacity_plcr": plcr_capacity,
        "debt_scale_factor": max(0.0, min(scale_factor, 1.5)),
    }


def irr_bisection(cashflows: Sequence[float], lo: float = -0.9, hi: float = 1.5, tol: float = 1e-6, max_iter: int = 200) -> float:
    def f(r):
        return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows))

    try:
        f_lo, f_hi = f(lo), f(hi)
    except ZeroDivisionError:
        return float("nan")
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid


def project_cashflows(statements: Dict[str, pd.DataFrame], cfg: Mapping[str, object], timeline: Timeline) -> Dict[str, object]:
    cashflow = statements["cashflow"].copy()
    pnl = statements["pnl"].copy()
    monthly_index = timeline.monthly_index()
    base_discount_rate = float(cfg["global_inputs"].get("discount_rate", DEFAULTS["global"]["discount_rate"]))
    wacc_factor = get_increment_series(cfg, timeline, "global_factor")
    annual_discount_curve = np.maximum(base_discount_rate * wacc_factor.reindex(monthly_index, fill_value=1.0).values, 0.0)
    monthly_rate_curve = np.power(1 + annual_discount_curve, 1 / 12) - 1
    discount_factors = np.cumprod(1 + monthly_rate_curve)
    fcf = cashflow["NetCashFlow"].values
    project_npv = np.sum(fcf / discount_factors)

    equity_cf = cashflow["CFO"] + cashflow["CFI"] + cashflow["CFF"]
    project_irr = irr_bisection(fcf)
    equity_irr = irr_bisection(equity_cf.values)
    cumulative_fcf = np.cumsum(fcf)
    cumulative_equity = np.cumsum(equity_cf.values)
    payback_month = next((i for i, val in enumerate(cumulative_fcf) if val >= 0), None)
    payback_year = monthly_index[payback_month].year if payback_month is not None else None

    debt_service = -cashflow["CFF"].values
    cfads = cashflow["CFO"].values
    dscr_series = [cf / ds if ds != 0 else np.nan for cf, ds in zip(cfads, debt_service)]
    metrics = {
        "Project_NPV": project_npv,
        "Project_IRR": project_irr,
        "Equity_IRR": equity_irr,
        "Payback_Year": payback_year,
        "Cumulative_FCF": cumulative_fcf[-1] if len(cumulative_fcf) else 0.0,
        "Cumulative_Equity_CF": cumulative_equity[-1] if len(cumulative_equity) else 0.0,
        "DSCR_min": float(np.nanmin(dscr_series)) if dscr_series else np.nan,
        "DSCR_avg": float(np.nanmean(dscr_series)) if dscr_series else np.nan,
    }
    return {"metrics": metrics, "cashflows": {"project": fcf, "equity": equity_cf.values, "discount_factors": discount_factors}}


def _annual_credit_distributions(dscr_df: pd.DataFrame) -> pd.DataFrame:
    if dscr_df.empty:
        return pd.DataFrame(columns=["year", "ADSCR_p10", "ADSCR_p50", "ADSCR_p90", "ADSCR_avg"])
    rows: List[Dict[str, float]] = []
    work = dscr_df.copy()
    work["year"] = work["date"].dt.year
    for year, group in work.groupby("year"):
        adsc = pd.to_numeric(group["DSCR"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "year": int(year),
                "ADSCR_p10": float(np.nanpercentile(adsc, 10)) if not adsc.empty else np.nan,
                "ADSCR_p50": float(np.nanpercentile(adsc, 50)) if not adsc.empty else np.nan,
                "ADSCR_p90": float(np.nanpercentile(adsc, 90)) if not adsc.empty else np.nan,
                "ADSCR_avg": float(np.nanmean(adsc)) if not adsc.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_credit_and_waterfall(
    cfg: Mapping[str, object],
    timeline: Timeline,
    statements: Dict[str, pd.DataFrame],
    debt_schedule: pd.DataFrame,
    case_name: str = "base",
) -> Dict[str, pd.DataFrame]:
    monthly_index = timeline.monthly_index()
    global_factor = get_increment_series(cfg, timeline, "global_factor")
    wc_factor = get_increment_series(cfg, timeline, "wc_factor")
    debt_factor = get_increment_series(cfg, timeline, "debt_factor")
    cashflow = statements.get("cashflow", pd.DataFrame()).copy()
    if cashflow.empty or "date" not in cashflow.columns:
        empty = pd.DataFrame()
        return {"credit_metrics_yearly": empty, "covenant_monitor": empty, "cash_waterfall": empty, "completion_tests": empty}
    cashflow["date"] = pd.to_datetime(cashflow["date"], errors="coerce")
    cashflow = cashflow.dropna(subset=["date"])
    if cashflow.empty:
        empty = pd.DataFrame()
        return {"credit_metrics_yearly": empty, "covenant_monitor": empty, "cash_waterfall": empty, "completion_tests": empty}
    debt_monthly = debt_schedule.groupby("date")[["debt_service", "balance"]].sum().reindex(monthly_index, fill_value=0.0)
    if "CFO" not in cashflow.columns:
        cashflow["CFO"] = 0.0
    if "NetCashFlow" not in cashflow.columns:
        cashflow["NetCashFlow"] = 0.0
    cfads = cashflow.set_index("date")["CFO"].reindex(monthly_index, fill_value=0.0)
    dscr = np.where(debt_monthly["debt_service"].values > 1e-9, cfads.values / debt_monthly["debt_service"].values, np.nan)
    dscr_df = pd.DataFrame({"date": monthly_index, "CFADS": cfads.values, "debt_service": debt_monthly["debt_service"].values, "DSCR": dscr, "balance": debt_monthly["balance"].values})
    annual_dist = _annual_credit_distributions(dscr_df)

    discount_rate = float(cfg.get("global_inputs", {}).get("discount_rate", DEFAULTS["global"]["discount_rate"]))
    monthly_discount = (1 + discount_rate) ** (1 / 12) - 1
    project_cf = cashflow.set_index("date")["NetCashFlow"].reindex(monthly_index, fill_value=0.0)
    llcr_vals: List[float] = []
    plcr_vals: List[float] = []
    for i, bal in enumerate(dscr_df["balance"].values):
        rem = np.arange(len(monthly_index) - i)
        pv_cfads = float(np.sum(cfads.values[i:] / ((1 + monthly_discount) ** rem)))
        pv_project = float(np.sum(project_cf.values[i:] / ((1 + monthly_discount) ** rem)))
        if abs(bal) <= 1e-9:
            llcr_vals.append(np.nan)
            plcr_vals.append(np.nan)
        else:
            llcr_vals.append(pv_cfads / bal)
            plcr_vals.append(pv_project / bal)
    dscr_df["LLCR"] = llcr_vals
    dscr_df["PLCR"] = plcr_vals
    ratio_yearly = dscr_df.assign(year=dscr_df["date"].dt.year).groupby("year")[["LLCR", "PLCR"]].agg(["min", "median", "max"]).reset_index()
    ratio_yearly.columns = ["year", "LLCR_min", "LLCR_p50", "LLCR_max", "PLCR_min", "PLCR_p50", "PLCR_max"]
    credit_metrics = annual_dist.merge(ratio_yearly, on="year", how="outer").sort_values("year")

    cov_table = cfg.get("covenant_thresholds")
    threshold_defaults = {"adsc_min": 1.2, "dscr_lockup": 1.1, "dscr_default": 1.0, "llcr_min": 1.25}
    threshold_frame = pd.DataFrame({"date": monthly_index, **threshold_defaults})
    if isinstance(cov_table, pd.DataFrame) and not cov_table.empty:
        cov = cov_table.copy()
        cov["case"] = cov["case"].astype(str).str.lower()
        case_rows = cov[cov["case"] == str(case_name).lower()]
        if case_rows.empty:
            case_rows = cov[cov["case"] == "base"]
        for _, row in case_rows.iterrows():
            start = parse_date_str(row.get("start_date"), monthly_index[0])
            end = parse_date_str(row.get("end_date"), monthly_index[-1])
            mask = (threshold_frame["date"] >= start) & (threshold_frame["date"] <= end)
            for key in threshold_defaults:
                if pd.notna(row.get(key)):
                    threshold_frame.loc[mask, key] = float(row.get(key))
    dscr_df = dscr_df.merge(threshold_frame, on="date", how="left")
    for col in ("adsc_min", "dscr_lockup", "dscr_default", "llcr_min"):
        dscr_df[col] = pd.to_numeric(dscr_df[col], errors="coerce").fillna(0.0) * global_factor.reindex(monthly_index, fill_value=1.0).values
    dscr_df["lockup_triggered"] = dscr_df["DSCR"] < dscr_df["dscr_lockup"]
    dscr_df["default_triggered"] = dscr_df["DSCR"] < dscr_df["dscr_default"]
    dscr_df["llcr_breach"] = dscr_df["LLCR"] < dscr_df["llcr_min"]

    reserve_cfg = cfg.get("reserve_accounts")
    reserve_row = reserve_cfg.iloc[0] if isinstance(reserve_cfg, pd.DataFrame) and not reserve_cfg.empty else {}
    sweep_pct = float(reserve_row.get("sweep_pct", 0.0) or 0.0)
    sweep_trigger = float(reserve_row.get("sweep_dscr_trigger", 1.25) or 1.25)
    dsra_months_base = float(reserve_row.get("dsra_months", 6.0) or 0.0)
    mmra_monthly_base = float(reserve_row.get("mmra_monthly", 0.0) or 0.0)
    restricted_cash_min_base = float(reserve_row.get("restricted_cash_min", 0.0) or 0.0)
    wc_limit_base = float(reserve_row.get("wc_facility_limit", 0.0) or 0.0)
    wc_rate_base = float(reserve_row.get("wc_facility_rate", 0.0) or 0.0)
    dsra_months_series = dsra_months_base * wc_factor
    mmra_monthly_series = mmra_monthly_base * wc_factor
    restricted_cash_min_series = restricted_cash_min_base * wc_factor
    wc_limit_series = wc_limit_base * wc_factor
    wc_rate_series = wc_rate_base * debt_factor
    sweep_trigger_series = sweep_trigger * global_factor
    rolling_ds = debt_monthly["debt_service"].rolling(12, min_periods=1).mean()
    dsra_target = rolling_ds * dsra_months_series.values
    sweep_amount = np.where(dscr_df["DSCR"].fillna(0.0) >= sweep_trigger_series.values, np.maximum(cfads.values - debt_monthly["debt_service"].values, 0.0) * sweep_pct, 0.0)
    mmra_balance = np.cumsum(mmra_monthly_series.values)
    dsra_funding = np.maximum(dsra_target.values - np.concatenate([[0.0], dsra_target.values[:-1]]), 0.0)
    restricted_cash = np.maximum(dsra_target.values + mmra_balance, restricted_cash_min_series.values)
    wc_draw = np.minimum(np.maximum(-project_cf.values, 0.0), wc_limit_series.values)
    wc_interest = wc_draw * wc_rate_series.values / 12.0
    distributable = np.maximum(cfads.values - debt_monthly["debt_service"].values - dsra_funding - mmra_monthly_series.values - sweep_amount - wc_interest, 0.0)
    distributable = np.where(dscr_df["lockup_triggered"], 0.0, distributable)
    cash_waterfall = pd.DataFrame({"date": monthly_index, "CFADS": cfads.values, "debt_service": debt_monthly["debt_service"].values, "dsra_target": dsra_target.values, "dsra_funding": dsra_funding, "mmra_contribution": mmra_monthly_series.values, "cash_sweep": sweep_amount, "wc_facility_draw": wc_draw, "wc_facility_interest": wc_interest, "restricted_cash": restricted_cash, "distributable_cash": distributable})

    completion_table = cfg.get("completion_tests")
    completion_rows: List[Dict[str, object]] = []
    if isinstance(completion_table, pd.DataFrame) and not completion_table.empty:
        for _, row in completion_table.iterrows():
            if not bool(row.get("enabled", True)):
                continue
            metric_name = str(row.get("metric", "")).strip().lower()
            threshold_value = float(row.get("threshold", 0.0) or 0.0)
            if metric_name == "plant_availability":
                measure = float(pd.to_numeric(statements.get("production", pd.DataFrame()).get("availability", pd.Series([0.0])), errors="coerce").mean())
            elif metric_name == "dscr_min":
                measure = float(np.nanmin(dscr_df["DSCR"].values)) if not dscr_df["DSCR"].dropna().empty else np.nan
            else:
                measure = np.nan
            completion_rows.append({"metric": metric_name, "threshold": threshold_value, "measured": measure, "passed": bool(pd.notna(measure) and measure >= threshold_value), "test_date": row.get("test_date")})
    completion_df = pd.DataFrame(completion_rows)
    return {"credit_metrics_yearly": credit_metrics, "covenant_monitor": dscr_df, "cash_waterfall": cash_waterfall, "completion_tests": completion_df}
###############################################################################
# Section 13: Dashboard and charts
###############################################################################


def build_dashboard(
    cfg: Mapping[str, object],
    statements: Dict[str, pd.DataFrame],
    production_monthly: pd.DataFrame,
    revenue_df: pd.DataFrame,
    valuation: Dict[str, object],
    working_capital: Optional[pd.DataFrame] = None,
    debt_schedule: Optional[pd.DataFrame] = None,
    staff_detail: Optional[pd.DataFrame] = None,
    break_even: Optional[Mapping[str, object]] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, object]:
    horizon = cfg["projection_horizon"]
    production_horizon = {
        "start_year": horizon["start_year"],
        "end_year": horizon["end_year"],
    }
    global_inputs = cfg["global_inputs"]
    metrics = valuation["metrics"]

    monthly_pnl = statements.get("pnl", pd.DataFrame()).copy()
    cashflow = statements.get("cashflow", pd.DataFrame()).copy()

    for frame in (monthly_pnl, cashflow, production_monthly, revenue_df):
        if isinstance(frame, pd.DataFrame) and not frame.empty and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    dashboard: Dict[str, object] = {}
    dashboard["assumptions_snapshot"] = pd.DataFrame(
        {
            "Metric": [
                "Projection Start",
                "Projection End",
                "Production Start",
                "Production End",
                "Tax Rate",
                "Investor Share",
                "Owner Share",
                "Terminal Growth",
                "Discount Rate",
                "Inflation",
            ],
            "Value": [
                f"{horizon['start_year']}-{horizon['start_month']:02d}",
                horizon["end_year"],
                production_horizon.get("start_year", horizon["start_year"]),
                production_horizon.get("end_year", horizon["end_year"]),
                global_inputs["corp_tax_rate"],
                global_inputs["investor_share"],
                global_inputs["owner_share"],
                global_inputs.get("terminal_growth", 0.02),
                global_inputs["discount_rate"],
                global_inputs.get("inflation_rate", 0.02),
            ],
        }
    )

    dashboard["global_block"] = pd.DataFrame(
        {
            "Metric": ["Corporate Tax", "Investor Share", "Owner Share", "Terminal Growth", "Capital Gains Tax", "Payback Year"],
            "Value": [
                global_inputs["corp_tax_rate"],
                global_inputs["investor_share"],
                global_inputs["owner_share"],
                global_inputs.get("terminal_growth", 0.02),
                global_inputs.get("capital_gains_tax_rate", 0.0),
                metrics.get("Payback_Year"),
            ],
        }
    )

    final_row = monthly_pnl.iloc[-1] if not monthly_pnl.empty else pd.Series(dtype=float)
    final_cashflow = cashflow.iloc[-1] if not cashflow.empty else pd.Series(dtype=float)
    dashboard["latest_drivers"] = {
        "final_month_revenue": float(final_row.get("Revenue", np.nan)),
        "final_month_ebitda": float(final_row.get("EBITDA", np.nan)),
        "final_month_equity_cf": float(final_cashflow.get("NetCashFlow", np.nan)),
        "cumulative_fcf_to_date": metrics.get("Cumulative_FCF"),
        "cumulative_equity_cf": metrics.get("Cumulative_Equity_CF"),
    }

    dashboard["overview_metrics"] = pd.DataFrame(
        {
            "Metric": ["Project NPV", "Project IRR", "Equity IRR", "Payback Year", "DSCR (min)", "DSCR (avg)"],
            "Value": [
                metrics.get("Project_NPV"),
                metrics.get("Project_IRR"),
                metrics.get("Equity_IRR"),
                metrics.get("Payback_Year"),
                metrics.get("DSCR_min"),
                metrics.get("DSCR_avg"),
            ],
        }
    )

    prod_for_annual = production_monthly.copy() if isinstance(production_monthly, pd.DataFrame) else pd.DataFrame()
    if not prod_for_annual.empty and {"date", "product", "volume"}.issubset(prod_for_annual.columns):
        prod_for_annual = prod_for_annual.dropna(subset=["date"])
        prod_for_annual["product"] = prod_for_annual["product"].astype(str).str.lower()
        prod_for_annual["year"] = prod_for_annual["date"].dt.year
        volume_annual = prod_for_annual.groupby(["year", "product"], as_index=False)["volume"].sum()
    else:
        volume_annual = pd.DataFrame(columns=["year", "product", "volume"])

    revenue_by_product = revenue_df.copy() if isinstance(revenue_df, pd.DataFrame) else pd.DataFrame()
    if not revenue_by_product.empty and {"date", "product", "revenue"}.issubset(revenue_by_product.columns):
        revenue_by_product = revenue_by_product.dropna(subset=["date"])
        revenue_by_product["product"] = revenue_by_product["product"].astype(str).str.lower()
        revenue_by_product["year"] = revenue_by_product["date"].dt.year
        revenue_annual = revenue_by_product.groupby(["year", "product"], as_index=False)["revenue"].sum()
    else:
        revenue_annual = pd.DataFrame(columns=["year", "product", "revenue"])

    annual_production = volume_annual.merge(revenue_annual, on=["year", "product"], how="left")
    if "revenue" in annual_production:
        annual_production["revenue"].fillna(0.0, inplace=True)
    annual_production_chart = (
        annual_production.pivot(index="year", columns="product", values="volume").fillna(0.0)
        if not annual_production.empty
        else pd.DataFrame(columns=PRODUCTS)
    )

    # Risk-adjusted DSCR and debt summaries
    dscr_df = pd.DataFrame()
    debt_summary = pd.DataFrame()
    if isinstance(debt_schedule, pd.DataFrame) and not debt_schedule.empty:
        debt_local = debt_schedule.copy()
        debt_local["date"] = pd.to_datetime(debt_local["date"], errors="coerce")
        debt_local = debt_local.dropna(subset=["date"])
        debt_summary = (
            debt_local.groupby("date")[[col for col in ["interest", "principal", "fees", "debt_service", "draw"] if col in debt_local.columns]]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        if not cashflow.empty:
            dscr_df = cashflow.merge(debt_summary, on="date", how="left")
            fill_cols = [col for col in ["interest", "principal", "fees", "debt_service"] if col in dscr_df.columns]
            dscr_df[fill_cols] = dscr_df[fill_cols].fillna(0.0)
            dscr_df["CFADS"] = dscr_df["CFO"] + dscr_df["interest"]
            dscr_df["DSCR"] = np.where(
                dscr_df["debt_service"].abs() > 1e-9,
                dscr_df["CFADS"] / dscr_df["debt_service"],
                np.nan,
            )
            dscr_df = dscr_df[["date", "CFADS", "debt_service", "interest", "principal", "DSCR"]]
    dashboard["dscr_trend"] = dscr_df

    debt_service_annual = pd.DataFrame()
    if not debt_summary.empty:
        debt_summary["year"] = debt_summary["date"].dt.year
        annual_cols = [col for col in ["interest", "principal", "fees", "debt_service", "draw"] if col in debt_summary.columns]
        debt_service_annual = debt_summary.groupby("year")[annual_cols].sum().reset_index()
    dashboard["debt_service_summary"] = debt_service_annual

    wc_trend = working_capital.copy() if isinstance(working_capital, pd.DataFrame) else pd.DataFrame()
    if not wc_trend.empty and "date" in wc_trend.columns:
        wc_trend = wc_trend.copy()
        wc_trend["date"] = pd.to_datetime(wc_trend["date"], errors="coerce")
        wc_trend = wc_trend.dropna(subset=["date"])
    dashboard["working_capital_trend"] = wc_trend

    cost_structure_monthly = pd.DataFrame()
    if not monthly_pnl.empty:
        cost_cols = [col for col in ("DirectCosts", "StaffCosts", "OtherOpexCosts") if col in monthly_pnl.columns]
        if cost_cols:
            cost_structure_monthly = monthly_pnl[["date", *cost_cols]].copy()
            cost_structure_monthly[cost_cols] = cost_structure_monthly[cost_cols].fillna(0.0)
    dashboard["cost_structure_monthly"] = cost_structure_monthly
    cost_structure_annual = aggregate_annual(cost_structure_monthly) if not cost_structure_monthly.empty else pd.DataFrame()
    dashboard["cost_structure_annual"] = cost_structure_annual

    labour_summary = pd.DataFrame()
    if isinstance(staff_detail, pd.DataFrame) and not staff_detail.empty:
        labour_summary = staff_detail.copy()
        labour_summary["date"] = pd.to_datetime(labour_summary["date"], errors="coerce")
        labour_summary = labour_summary.dropna(subset=["date"])
    dashboard["labour_summary"] = labour_summary

    break_even_products = pd.DataFrame()
    if isinstance(break_even, Mapping):
        per_product = break_even.get("per_product")
        if isinstance(per_product, pd.DataFrame):
            break_even_products = per_product.copy()
    dashboard["break_even_per_product"] = break_even_products

    valuation_cashflows = valuation.get("cashflows", {}) if isinstance(valuation, Mapping) else {}
    cumulative_cf = pd.DataFrame()
    if isinstance(valuation_cashflows, Mapping):
        project_cf = valuation_cashflows.get("project")
        equity_cf = valuation_cashflows.get("equity")
        if (
            project_cf is not None
            and equity_cf is not None
            and not cashflow.empty
            and len(project_cf)
            and len(equity_cf)
        ):
            min_len = min(len(project_cf), len(equity_cf), len(cashflow))
            cf_df = pd.DataFrame(
                {
                    "date": cashflow["date"].values[:min_len],
                    "project_cf": np.asarray(project_cf, dtype=float)[:min_len],
                    "equity_cf": np.asarray(equity_cf, dtype=float)[:min_len],
                }
            )
            cf_df["project_cumulative"] = cf_df["project_cf"].cumsum()
            cf_df["equity_cumulative"] = cf_df["equity_cf"].cumsum()
            cumulative_cf = cf_df
    dashboard["cumulative_cashflows"] = cumulative_cf

    revenue_vs_production = annual_production.copy()
    if not revenue_vs_production.empty and "volume" in revenue_vs_production:
        revenue_vs_production["average_price"] = np.where(
            revenue_vs_production["volume"].abs() > 1e-9,
            revenue_vs_production.get("revenue", 0.0) / revenue_vs_production["volume"],
            np.nan,
        )
    dashboard["revenue_vs_production"] = revenue_vs_production

    charts: Dict[str, Optional[Path]] = {}
    if plt is not None:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        def _save_chart(fig, name: str) -> Optional[Path]:
            if out_dir is None:
                plt.close(fig)
                return None
            path = out_dir / f"{name}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path

        if not annual_production_chart.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            annual_production_chart.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Annual Production by Product")
            ax.set_ylabel("Volume")
            charts["annual_production"] = _save_chart(fig, "annual_production")

        annual_cashflow = aggregate_annual(cashflow) if not cashflow.empty else pd.DataFrame()
        if not annual_cashflow.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(annual_cashflow["year"], annual_cashflow["NetCashFlow"], label="Net CF", color="tab:blue", alpha=0.6)
            ax.bar(annual_cashflow["year"], annual_cashflow["CFO"], label="CFO", color="tab:green", alpha=0.4)
            ax.set_title("Annual Cash Flow")
            ax.legend()
            charts["cash_flow"] = _save_chart(fig, "cash_flow")

        if not revenue_by_product.empty:
            revenue_mix = revenue_by_product.groupby(["year", "product"])["revenue"].sum().unstack(fill_value=0.0)
            fig, ax = plt.subplots(figsize=(8, 4))
            revenue_mix.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Revenue Mix")
            charts["revenue_mix"] = _save_chart(fig, "revenue_mix")

        if not dscr_df.empty:
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(dscr_df["date"], dscr_df["DSCR"], color="tab:blue", label="DSCR")
            ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax1.set_ylabel("DSCR")
            ax2 = ax1.twinx()
            ax2.plot(dscr_df["date"], dscr_df["CFADS"], color="tab:green", label="CFADS")
            ax2.plot(dscr_df["date"], dscr_df["debt_service"], color="tab:red", label="Debt service")
            ax1.set_title("DSCR and Debt Service Trend")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")
            charts["dscr_trend"] = _save_chart(fig, "dscr_trend")

        if not debt_service_annual.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            bottom = np.zeros(len(debt_service_annual))
            for col, color in (("interest", "tab:red"), ("principal", "tab:purple")):
                ax.bar(debt_service_annual["year"], debt_service_annual[col], bottom=bottom, label=col.title(), color=color, alpha=0.7)
                bottom += debt_service_annual[col].values
            ax.plot(debt_service_annual["year"], debt_service_annual["draw"], color="tab:orange", marker="o", label="Draws")
            ax.set_title("Debt Service Waterfall")
            ax.legend()
            charts["debt_service"] = _save_chart(fig, "debt_service")

        if not wc_trend.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.stackplot(
                wc_trend["date"],
                wc_trend["accounts_receivable"],
                wc_trend["inventory"],
                wc_trend["accounts_payable"],
                labels=["Accounts Receivable", "Inventory", "Accounts Payable"],
            )
            ax.set_title("Working Capital Components")
            ax.legend(loc="upper left")
            charts["working_capital"] = _save_chart(fig, "working_capital")

        if not cost_structure_annual.empty:
            df = cost_structure_annual.set_index("year")[[col for col in cost_structure_annual.columns if col not in {"year"}]]
            fig, ax = plt.subplots(figsize=(8, 4))
            df.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Annual Cost Structure")
            charts["cost_structure"] = _save_chart(fig, "cost_structure")

        if not labour_summary.empty:
            labour_plot = labour_summary.groupby(["date", "dept"])["total_cost"].sum().unstack(fill_value=0.0)
            if not labour_plot.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                labour_plot.plot(kind="area", stacked=True, ax=ax)
                ax.set_title("Labour Cost by Department")
                charts["labour_cost"] = _save_chart(fig, "labour_cost")

        if not break_even_products.empty:
            be_plot = break_even_products[["product", "actual_volume", "break_even_units"]].dropna()
            if not be_plot.empty:
                be_plot = be_plot.set_index("product")
                fig, ax = plt.subplots(figsize=(8, 4))
                be_plot.plot(kind="bar", ax=ax)
                ax.set_title("Actual vs Break-even Volume")
                ax.set_ylabel("Units")
                charts["break_even"] = _save_chart(fig, "break_even")

        if not cumulative_cf.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cumulative_cf["date"], cumulative_cf["project_cumulative"], label="Project cumulative")
            ax.plot(cumulative_cf["date"], cumulative_cf["equity_cumulative"], label="Equity cumulative")
            ax.set_title("Cumulative Cash Flows")
            ax.legend()
            charts["cumulative_cashflows"] = _save_chart(fig, "cumulative_cashflows")

        if not revenue_vs_production.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            for product, grp in revenue_vs_production.groupby("product"):
                ax.scatter(grp["volume"], grp.get("revenue", 0.0), label=product)
            ax.set_xlabel("Volume")
            ax.set_ylabel("Revenue")
            ax.set_title("Revenue vs Production")
            ax.legend()
            charts["revenue_vs_production"] = _save_chart(fig, "revenue_vs_production")

    else:
        charts = {
            "annual_production": None,
            "cash_flow": None,
            "revenue_mix": None,
            "dscr_trend": None,
            "debt_service": None,
            "working_capital": None,
            "cost_structure": None,
            "labour_cost": None,
            "break_even": None,
            "cumulative_cashflows": None,
            "revenue_vs_production": None,
        }

    dashboard["charts"] = charts
    dashboard["annual_production"] = annual_production_chart.reset_index() if not annual_production_chart.empty else annual_production_chart
    dashboard["annual_production_detail"] = annual_production
    dashboard["annual_cashflow"] = aggregate_annual(cashflow) if not cashflow.empty else pd.DataFrame()

    return dashboard
###############################################################################
# Section 14: Sensitivity, Monte Carlo, Goal Seek, Scenarios
###############################################################################


OPTIMIZATION_VARIABLE_LIBRARY: Dict[str, Dict[str, object]] = {
    "ethanol_price": {"label": "Ethanol price multiplier", "mode": "scale", "bounds": (0.8, 1.2)},
    "sugar_price": {"label": "Sugar price multiplier", "mode": "scale", "bounds": (0.8, 1.2)},
    "electricity_price": {"label": "Electricity tariff multiplier", "mode": "scale", "bounds": (0.8, 1.25)},
    "animal_feed_price": {"label": "Animal feed price multiplier", "mode": "scale", "bounds": (0.8, 1.25)},
    "availability": {"label": "Plant availability multiplier", "mode": "scale", "bounds": (0.8, 1.05)},
    "capex": {"label": "Total CAPEX multiplier", "mode": "scale", "bounds": (0.8, 1.2)},
    "opex": {"label": "Operating cost multiplier", "mode": "scale", "bounds": (0.85, 1.2)},
    "debt_rate": {"label": "Debt interest rate shift", "mode": "shift", "bounds": (-0.03, 0.03)},
}

DECISION_TREE_COLUMN_MAP: Dict[str, str] = {
    "ethanol_price_multiplier": "ethanol_price",
    "sugar_price_multiplier": "sugar_price",
    "electricity_price_multiplier": "electricity_price",
    "animal_feed_price_multiplier": "animal_feed_price",
    "capex_multiplier": "capex",
    "opex_multiplier": "opex",
    "debt_rate_shift": "debt_rate",
}


@dataclass
class OptimizationVariableSpec:
    name: str
    label: str
    bounds: Tuple[float, float]
    mode: Literal["scale", "shift"]
    base: object

    def clip(self, value: float) -> float:
        low, high = self.bounds
        if low > high:
            low, high = high, low
        return float(min(max(value, low), high))

    def apply(self, cfg_copy: Dict[str, object], value: float) -> None:
        value = self.clip(value)
        if self.name in {"ethanol_price", "sugar_price", "electricity_price", "animal_feed_price"}:
            product = self.name.replace("_price", "")
            base_price = float(self.base) if self.base is not None else 0.0
            cfg_copy.setdefault("prices", {}).setdefault(product, {})
            cfg_copy["prices"][product]["base_price"] = base_price * value
            return
        if self.name == "availability":
            base_availability = float(self.base) if self.base is not None else DEFAULTS["production"]["plant_availability"]
            cfg_copy.setdefault("production", {})
            cfg_copy["production"]["plant_availability"] = min(max(base_availability * value, 0.0), 1.0)
            return
        if self.name == "capex":
            if isinstance(self.base, pd.DataFrame):
                df = self.base.copy()
                if "amount" in df.columns:
                    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0) * value
                cfg_copy["capex_lines"] = df
            elif isinstance(self.base, list):
                new_lines: List[Dict[str, object]] = []
                for line in self.base:
                    new_line = copy.deepcopy(line)
                    if isinstance(new_line, dict) and "amount" in new_line and new_line["amount"] is not None:
                        try:
                            new_line["amount"] = float(new_line["amount"]) * value
                        except (TypeError, ValueError):
                            pass
                    new_lines.append(new_line)
                cfg_copy["capex_lines"] = new_lines
            return
        if self.name == "opex":
            base_mapping = self.base if isinstance(self.base, dict) else {}
            fixed = float(base_mapping.get("fixed_opex_per_month", DEFAULTS["opex"]["fixed_opex_per_month"]))
            var_map = copy.deepcopy(base_mapping.get("other_variable_cost_per_unit", DEFAULTS["opex"]["other_variable_cost_per_unit"]))
            cfg_copy.setdefault("opex", {})
            cfg_copy["opex"]["fixed_opex_per_month"] = fixed * value
            cfg_copy["opex"]["other_variable_cost_per_unit"] = {k: float(v) * value for k, v in var_map.items()}
            return
        if self.name == "debt_rate":
            delta = value if self.mode == "shift" else value
            debt_df = cfg_copy.get("debt_tranches")
            if isinstance(debt_df, pd.DataFrame) and not debt_df.empty:
                df = debt_df.copy()
                if "interest_rate" in df.columns:
                    df["interest_rate"] = pd.to_numeric(df["interest_rate"], errors="coerce").fillna(0.0) + delta
                cfg_copy["debt_tranches"] = df
            else:
                tranches = cfg_copy.get("debt", {}).get("tranches", [])
                if isinstance(tranches, list):
                    for idx, tranche in enumerate(tranches):
                        base_rate = 0.0
                        if isinstance(self.base, list) and idx < len(self.base):
                            base_rate = float(self.base[idx])
                        elif isinstance(tranche, dict) and "interest_rate" in tranche:
                            try:
                                base_rate = float(tranche["interest_rate"])
                            except (TypeError, ValueError):
                                base_rate = 0.0
                        if isinstance(tranche, dict):
                            tranche["interest_rate"] = base_rate + delta


def _extract_optimizer_base(cfg: Mapping[str, object], key: str) -> object:
    if key in {"ethanol_price", "sugar_price", "electricity_price", "animal_feed_price"}:
        product = key.replace("_price", "")
        price_cfg = cfg.get("prices", {})
        base_price = None
        if isinstance(price_cfg, Mapping):
            product_cfg = price_cfg.get(product, {})
            if isinstance(product_cfg, Mapping):
                base_price = product_cfg.get("base_price")
        if base_price is None:
            base_price = DEFAULTS["prices"][product]["base_price"]
        return float(base_price)
    if key == "availability":
        production_cfg = cfg.get("production", {})
        if isinstance(production_cfg, Mapping):
            base_availability = production_cfg.get("plant_availability")
            if base_availability is not None:
                try:
                    return float(base_availability)
                except (TypeError, ValueError):
                    pass
        return float(DEFAULTS["production"]["plant_availability"])
    if key == "capex":
        capex_lines = cfg.get("capex_lines")
        if isinstance(capex_lines, pd.DataFrame):
            return capex_lines.copy()
        if isinstance(capex_lines, list):
            return copy.deepcopy(capex_lines)
        return pd.DataFrame(columns=["item_name", "amount"])
    if key == "opex":
        opex_cfg = cfg.get("opex", {}) if isinstance(cfg, Mapping) else {}
        if not isinstance(opex_cfg, Mapping):
            opex_cfg = {}
        base_map = {
            "fixed_opex_per_month": float(
                opex_cfg.get("fixed_opex_per_month", DEFAULTS["opex"]["fixed_opex_per_month"])
            ),
            "other_variable_cost_per_unit": copy.deepcopy(
                opex_cfg.get("other_variable_cost_per_unit", DEFAULTS["opex"]["other_variable_cost_per_unit"])
            ),
        }
        return base_map
    if key == "debt_rate":
        debt_df = cfg.get("debt_tranches")
        rates: List[float] = []
        if isinstance(debt_df, pd.DataFrame) and not debt_df.empty and "interest_rate" in debt_df.columns:
            rates = pd.to_numeric(debt_df["interest_rate"], errors="coerce").fillna(0.0).tolist()
        else:
            tranches = cfg.get("debt", {}).get("tranches", []) if isinstance(cfg, Mapping) else []
            if isinstance(tranches, list):
                for tranche in tranches:
                    if isinstance(tranche, Mapping):
                        try:
                            rates.append(float(tranche.get("interest_rate", 0.0)))
                        except (TypeError, ValueError):
                            rates.append(0.0)
        if not rates:
            rates = [DEFAULTS["debt"]["tranches"][0].get("interest_rate", 0.1)]
        return rates
    return None


def _make_optimizer_spec(cfg: Mapping[str, object], key: str, lower: float, upper: float) -> OptimizationVariableSpec:
    definition = OPTIMIZATION_VARIABLE_LIBRARY[key]
    bounds = (float(lower), float(upper))
    base_value = _extract_optimizer_base(cfg, key)
    return OptimizationVariableSpec(
        name=key,
        label=definition["label"],
        bounds=bounds,
        mode=definition["mode"],
        base=base_value,
    )


def build_optimizer_specs(
    cfg: Mapping[str, object],
    variable_table: Optional[pd.DataFrame] = None,
) -> List[OptimizationVariableSpec]:
    specs: List[OptimizationVariableSpec] = []
    base_cfg = copy.deepcopy(cfg)

    def _coerce_float(value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            val = float(value)
            if np.isnan(val):
                return None
            return float(val)
        except (TypeError, ValueError):
            return None

    if isinstance(variable_table, pd.DataFrame) and not variable_table.empty:
        for _, row in variable_table.iterrows():
            if bool(row.get("enabled", True)) is False:
                continue
            variable_key = str(row.get("variable", "")).strip().lower()
            if not variable_key:
                continue
            if variable_key not in OPTIMIZATION_VARIABLE_LIBRARY:
                continue
            default_bounds = OPTIMIZATION_VARIABLE_LIBRARY[variable_key]["bounds"]
            lower = _coerce_float(row.get("lower_bound"))
            upper = _coerce_float(row.get("upper_bound"))
            lo, hi = default_bounds
            if lower is not None:
                lo = lower
            if upper is not None:
                hi = upper
            if lo == hi:
                hi = lo + (abs(lo) * 0.05 if lo != 0 else 0.05)
            if lo > hi:
                lo, hi = hi, lo
            spec = _make_optimizer_spec(base_cfg, variable_key, lo, hi)
            specs.append(spec)
    else:
        for key, definition in OPTIMIZATION_VARIABLE_LIBRARY.items():
            lo, hi = definition["bounds"]
            specs.append(_make_optimizer_spec(base_cfg, key, lo, hi))

    return specs


def metaheuristic_optimize(
    cfg: Mapping[str, object],
    base_metrics: Mapping[str, object],
    run_model_fn: Callable[[Mapping[str, object]], Mapping[str, object]],
    variable_table: Optional[pd.DataFrame] = None,
    objective: str = "Project_NPV",
    iterations: int = 10,
    population: int = 6,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    specs = build_optimizer_specs(cfg, variable_table)
    if not specs:
        return pd.DataFrame()

    try:
        base_metric = float(base_metrics.get(objective, np.nan))
    except Exception:
        base_metric = float("nan")

    rng = np.random.default_rng(seed)
    population_size = max(population, len(specs))

    def _random_candidate() -> Dict[str, float]:
        return {spec.name: rng.uniform(spec.bounds[0], spec.bounds[1]) for spec in specs}

    population_vectors: List[Dict[str, float]] = [_random_candidate() for _ in range(population_size)]
    records: List[Dict[str, object]] = []
    for iteration in range(max(1, iterations)):
        evaluated: List[Tuple[Dict[str, float], float]] = []
        for candidate in population_vectors:
            cfg_candidate = copy.deepcopy(cfg)
            for spec in specs:
                spec.apply(cfg_candidate, candidate[spec.name])
            result = run_model_fn(cfg_candidate)
            metric_value = result.get("metrics", {}).get(objective, np.nan)
            evaluated.append((candidate, metric_value))
            row: Dict[str, object] = {
                "iteration": iteration,
                "objective": metric_value,
                "delta_vs_base": metric_value - base_metric if np.isfinite(metric_value) and np.isfinite(base_metric) else np.nan,
            }
            for spec in specs:
                suffix = "delta" if spec.mode == "shift" else "factor"
                row[f"{spec.name}_{suffix}"] = candidate[spec.name]
            records.append(row)

        evaluated.sort(key=lambda item: (float("-inf") if not np.isfinite(item[1]) else item[1]), reverse=True)
        elite_count = max(1, population_size // 2)
        elites = [candidate for candidate, _ in evaluated[:elite_count]] or [population_vectors[0]]
        new_population: List[Dict[str, float]] = elites.copy()
        while len(new_population) < population_size:
            parent = elites[rng.integers(0, len(elites))]
            child = parent.copy()
            for spec in specs:
                span = spec.bounds[1] - spec.bounds[0]
                perturb = rng.normal(0.0, span * 0.1)
                child[spec.name] = spec.clip(child[spec.name] + perturb)
            new_population.append(child)
        population_vectors = new_population

    results_df = pd.DataFrame(records)
    if not results_df.empty:
        results_df = results_df.sort_values(["iteration", "objective"], ascending=[True, False]).reset_index(drop=True)
    return results_df


class SimpleMLP:
    def __init__(
        self,
        input_size: int,
        hidden_units: int = 8,
        learning_rate: float = 0.01,
        epochs: int = 300,
        seed: Optional[int] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_units = max(1, hidden_units)
        self.learning_rate = max(1e-5, float(learning_rate))
        self.epochs = max(10, int(epochs))
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.1, size=(self.input_size, self.hidden_units))
        self.b1 = np.zeros(self.hidden_units)
        self.W2 = rng.normal(scale=0.1, size=(self.hidden_units, 1))
        self.b2 = np.zeros(1)
        self.x_mean = np.zeros(self.input_size)
        self.x_std = np.ones(self.input_size)
        self.y_mean = 0.0
        self.y_std = 1.0

    @staticmethod
    def _activation(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _activation_derivative(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            return
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1.0
        X_norm = (X - self.x_mean) / self.x_std
        self.y_mean = y.mean()
        self.y_std = y.std() if y.std() > 0 else 1.0
        y_norm = (y - self.y_mean) / self.y_std

        for _ in range(self.epochs):
            z1 = X_norm @ self.W1 + self.b1
            a1 = self._activation(z1)
            z2 = a1 @ self.W2 + self.b2
            y_pred = z2.reshape(-1)
            error = y_pred - y_norm
            grad_W2 = a1.T @ error[:, None] / len(X_norm)
            grad_b2 = error.mean()
            delta1 = (error[:, None] @ self.W2.T) * self._activation_derivative(z1)
            grad_W1 = X_norm.T @ delta1 / len(X_norm)
            grad_b1 = delta1.mean(axis=0)

            self.W2 -= self.learning_rate * grad_W2
            self.b2 -= self.learning_rate * grad_b2
            self.W1 -= self.learning_rate * grad_W1
            self.b1 -= self.learning_rate * grad_b1

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self.x_mean) / self.x_std
        z1 = X_norm @ self.W1 + self.b1
        a1 = self._activation(z1)
        z2 = a1 @ self.W2 + self.b2
        y_pred = z2.reshape(-1)
        return y_pred * self.y_std + self.y_mean


def neural_forecast_production(
    results: Mapping[str, object],
    product: str,
    lookback: int = 12,
    horizon: int = 12,
    hidden_units: int = 8,
    learning_rate: float = 0.01,
    epochs: int = 300,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    production_df = results.get("production_monthly")
    if not isinstance(production_df, pd.DataFrame) or production_df.empty:
        return pd.DataFrame(columns=["date", "product", "forecast_volume"])

    df = production_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["product"] = df["product"].astype(str).str.lower()
    product_key = product.lower()
    df = df[df["product"] == product_key].sort_values("date")
    if df.empty:
        return pd.DataFrame(columns=["date", "product", "forecast_volume"])

    lookback = max(3, int(lookback))
    horizon = max(1, int(horizon))
    values = df["volume"].astype(float).values
    if len(values) < lookback + 2:
        avg = float(values.mean()) if len(values) else 0.0
        future_dates = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return pd.DataFrame({"date": future_dates, "product": product_key, "forecast_volume": np.full(horizon, avg)})

    X = []
    y = []
    for idx in range(lookback, len(values)):
        X.append(values[idx - lookback : idx])
        y.append(values[idx])
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    mlp = SimpleMLP(
        input_size=lookback,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )
    mlp.fit(X_arr, y_arr)

    history = list(values)
    future_dates = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    forecasts: List[float] = []
    for _ in range(horizon):
        input_vec = np.asarray(history[-lookback:])
        predicted = float(mlp.predict(input_vec)[0])
        predicted = max(predicted, 0.0)
        forecasts.append(predicted)
        history.append(predicted)

    return pd.DataFrame({"date": future_dates, "product": product_key, "forecast_volume": forecasts})


def _series_from_statements(results: Mapping[str, object], column: str) -> pd.Series:
    statements = results.get("statements_monthly")
    if isinstance(statements, Mapping):
        pnl = statements.get("pnl")
        if isinstance(pnl, pd.DataFrame) and column in pnl.columns:
            df = pnl.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            if df.empty:
                return pd.Series(dtype=float)
            return df.set_index("date")[column].astype(float)
    return pd.Series(dtype=float)


def _series_staff_costs(results: Mapping[str, object]) -> pd.Series:
    staff_detail = results.get("staff_costs_detail")
    if isinstance(staff_detail, pd.DataFrame) and not staff_detail.empty:
        df = staff_detail.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return pd.Series(dtype=float)
        grouped = df.groupby("date")["total_cost"].sum().sort_index()
        return grouped.astype(float)
    return pd.Series(dtype=float)


def _series_production(results: Mapping[str, object], product: str) -> pd.Series:
    prod = results.get("production_monthly")
    if isinstance(prod, pd.DataFrame) and not prod.empty:
        df = prod.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["product"] = df["product"].astype(str).str.lower()
        df = df[df["product"] == product]
        if df.empty:
            return pd.Series(dtype=float)
        return df.set_index("date")["volume"].astype(float)
    return pd.Series(dtype=float)


STATISTICAL_SERIES_LIBRARY: Dict[str, Callable[[Mapping[str, object]], pd.Series]] = {
    "revenue": lambda res: _series_from_statements(res, "Revenue"),
    "cogs": lambda res: _series_from_statements(res, "COGS"),
    "opex": lambda res: _series_from_statements(res, "Opex"),
    "ebitda": lambda res: _series_from_statements(res, "EBITDA"),
    "staff_costs": _series_staff_costs,
    "production_ethanol": lambda res: _series_production(res, "ethanol"),
    "production_sugar": lambda res: _series_production(res, "sugar"),
    "production_electricity": lambda res: _series_production(res, "electricity"),
    "production_animal_feed": lambda res: _series_production(res, "animal_feed"),
}


def statistical_forecast(
    results: Mapping[str, object],
    series_key: str,
    alpha: float = 0.3,
    horizon: int = 12,
) -> Dict[str, object]:
    extractor = STATISTICAL_SERIES_LIBRARY.get(series_key)
    if extractor is None:
        return {"historical": pd.DataFrame(), "forecast": pd.DataFrame(), "residual_std": np.nan, "equipment_failure_risk": np.nan}

    series = extractor(results)
    if series is None or series.empty:
        return {"historical": pd.DataFrame(), "forecast": pd.DataFrame(), "residual_std": np.nan, "equipment_failure_risk": np.nan}

    series = series.sort_index()
    history_df = series.reset_index()
    history_df.columns = ["date", "value"]

    values = series.values.astype(float)
    alpha = min(max(alpha, 0.01), 1.0)
    level = values[0]
    residuals: List[float] = []
    for actual in values:
        forecast_val = level
        residuals.append(actual - forecast_val)
        level = alpha * actual + (1 - alpha) * level

    residuals_arr = np.asarray(residuals[1:])
    residual_std = float(residuals_arr.std(ddof=1)) if residuals_arr.size > 1 else 0.0
    conf_int = 1.96 * residual_std

    start_date = series.index[-1] + pd.offsets.MonthBegin(1)
    future_dates = pd.date_range(start_date, periods=max(1, int(horizon)), freq="MS")
    forecasts = np.full(len(future_dates), level)
    lower = np.maximum(forecasts - conf_int, 0.0)
    upper = forecasts + conf_int
    forecast_df = pd.DataFrame({"date": future_dates, "forecast": forecasts, "lower": lower, "upper": upper})

    risk_profile = results.get("risk_profile", {})
    production_multiplier = float(risk_profile.get("production", 1.0)) if isinstance(risk_profile, Mapping) else 1.0
    equipment_failure_risk = max(0.0, 1.0 - production_multiplier)

    return {
        "historical": history_df,
        "forecast": forecast_df,
        "residual_std": residual_std,
        "equipment_failure_risk": equipment_failure_risk,
    }


def decision_tree_analysis(
    cfg: Mapping[str, object],
    run_model_fn: Callable[[Mapping[str, object]], Mapping[str, object]],
    paths_table: Optional[pd.DataFrame],
    objective: str = "Project_NPV",
) -> Dict[str, object]:
    if not isinstance(paths_table, pd.DataFrame) or paths_table.empty:
        return {"paths": pd.DataFrame(), "expected_metric": np.nan, "objective": objective, "total_probability": 0.0}

    specs = {spec.name: spec for spec in build_optimizer_specs(cfg)}
    rows: List[Dict[str, object]] = []
    total_probability = 0.0
    expected_metric = 0.0

    for _, path_row in paths_table.iterrows():
        if bool(path_row.get("enabled", True)) is False:
            continue
        try:
            probability = float(path_row.get("probability", np.nan))
        except (TypeError, ValueError):
            probability = np.nan
        if not np.isfinite(probability) or probability <= 0:
            continue

        cfg_candidate = copy.deepcopy(cfg)
        applied: Dict[str, object] = {}
        for column, spec_name in DECISION_TREE_COLUMN_MAP.items():
            if column not in path_row or spec_name not in specs:
                continue
            value = path_row.get(column)
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            specs[spec_name].apply(cfg_candidate, value_float)
            suffix = "delta" if specs[spec_name].mode == "shift" else "factor"
            applied[f"{spec_name}_{suffix}"] = value_float

        result = run_model_fn(cfg_candidate)
        metric_value = result.get("metrics", {}).get(objective, np.nan)
        row_record = {
            "path_name": path_row.get("path_name", "Path"),
            "probability": probability,
            "metric": metric_value,
            "notes": path_row.get("notes", ""),
        }
        row_record.update(applied)
        rows.append(row_record)

        if np.isfinite(metric_value):
            total_probability += probability
            expected_metric += probability * metric_value

    if total_probability > 0:
        expected_metric /= total_probability
    else:
        expected_metric = float("nan")

    paths_df = pd.DataFrame(rows)
    return {
        "paths": paths_df,
        "expected_metric": expected_metric,
        "objective": objective,
        "total_probability": total_probability,
    }


def sensitivity_tornado(cfg: Mapping[str, object], base_results: Dict[str, object], run_model_fn: Callable[[Mapping[str, object]], Dict[str, object]], drivers: Optional[List[Tuple[str, float]]] = None) -> pd.DataFrame:
    if drivers is None:
        drivers = [
            ("ethanol_price", 0.2),
            ("sugar_price", 0.2),
            ("availability", 0.05),
            ("capex", 0.2),
            ("debt_rate", 0.02),
        ]
    base_npv = base_results["metrics"].get("Project_NPV", 0.0)
    rows = []
    for key, pct in drivers:
        cfg_up = copy.deepcopy(cfg)
        cfg_down = copy.deepcopy(cfg)
        if "price" in key:
            prod = key.split("_")[0]
            for variant, factor in ((cfg_up, 1 + pct), (cfg_down, 1 - pct)):
                variant["prices"][prod]["base_price"] *= factor
        elif key == "availability":
            for variant, factor in ((cfg_up, 1 + pct), (cfg_down, 1 - pct)):
                variant["production"]["plant_availability"] *= factor
        elif key == "capex":
            for variant, factor in ((cfg_up, 1 + pct), (cfg_down, 1 - pct)):
                capex_lines = variant.get("capex_lines")
                if isinstance(capex_lines, pd.DataFrame):
                    variant["capex_lines"] = capex_lines.copy()
                    if "amount" in variant["capex_lines"]:
                        variant["capex_lines"]["amount"] = variant["capex_lines"]["amount"].astype(float) * factor
                elif isinstance(capex_lines, list):
                    for line in capex_lines:
                        line["amount"] *= factor
        elif key == "debt_rate":
            for variant, factor in ((cfg_up, pct), (cfg_down, -pct)):
                for tranche in variant["debt"]["tranches"]:
                    tranche["interest_rate"] += factor
        up_results = run_model_fn(cfg_up)
        down_results = run_model_fn(cfg_down)
        rows.append({"driver": key, "scenario": "High", "npv": up_results["metrics"].get("Project_NPV", np.nan), "delta": up_results["metrics"].get("Project_NPV", np.nan) - base_npv})
        rows.append({"driver": key, "scenario": "Low", "npv": down_results["metrics"].get("Project_NPV", np.nan), "delta": down_results["metrics"].get("Project_NPV", np.nan) - base_npv})
    tornado = pd.DataFrame(rows)
    tornado["abs_delta"] = tornado["delta"].abs()
    tornado.sort_values("abs_delta", ascending=False, inplace=True)
    return tornado


def _sample_distribution_value(
    rng: np.random.Generator,
    distribution: str,
    p1: float,
    p2: float,
    p3: float,
) -> float:
    dist = (distribution or "normal").lower()
    if dist not in MONTE_CARLO_DISTRIBUTIONS:
        dist = "normal"
    if dist == "normal":
        sigma = max(p2, 0.0)
        return rng.normal(p1, sigma)
    if dist == "lognormal":
        sigma = max(p2, 0.0)
        return rng.lognormal(p1, sigma)
    if dist == "triangular":
        left = p1
        mode = p2 if not math.isnan(p2) else left
        right = p3 if not math.isnan(p3) else mode
        if math.isnan(left):
            left = 0.0
        if math.isnan(mode):
            mode = left
        if math.isnan(right):
            right = mode
        if right <= left:
            # ensure a valid range for triangular distribution
            adjustment = max(abs(mode - left), 1e-6)
            right = left + adjustment
        mode = min(max(mode, left), right)
        return rng.triangular(left, mode, right)
    if dist == "uniform":
        low = p1
        high = p2
        if math.isnan(low):
            low = 0.0
        if math.isnan(high):
            high = low
        if low == high:
            return low
        if high < low:
            low, high = high, low
        return rng.uniform(low, high)
    return p1


def _apply_monte_carlo_variable(sample_cfg: Dict[str, object], variable: str, draw: float, applies_to: str) -> None:
    """Apply a Monte Carlo draw to the relevant section of the configuration."""

    var_key = (variable or "").lower()
    applies = (applies_to or "global").lower()
    factor = 1.0 + float(draw)
    if not np.isfinite(factor):
        return
    factor = max(factor, 0.0)

    def _scale_capex() -> None:
        capex_lines = sample_cfg.get("capex_lines")
        if isinstance(capex_lines, pd.DataFrame):
            df = capex_lines.copy()
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0) * factor
            sample_cfg["capex_lines"] = df
        elif isinstance(capex_lines, list):
            for line in capex_lines:
                if isinstance(line, dict) and "amount" in line and line["amount"] is not None:
                    try:
                        line["amount"] = float(line["amount"]) * factor
                    except (TypeError, ValueError):
                        continue

    def _scale_direct_costs() -> None:
        direct_df = sample_cfg.get("direct_costs_monthly")
        if isinstance(direct_df, pd.DataFrame) and not direct_df.empty:
            df = direct_df.copy()
            mask = pd.Series(True, index=df.index)
            if applies in PRODUCTS and "product_link" in df.columns:
                mask = df["product_link"].astype(str).str.lower() == applies
            for col in ("unit_price", "quantity", "amount"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "unit_price" in df.columns:
                df.loc[mask, "unit_price"] = df.loc[mask, "unit_price"].fillna(0.0) * factor
            if "amount" in df.columns and "unit_price" not in df.columns:
                df.loc[mask, "amount"] = df.loc[mask, "amount"].fillna(0.0) * factor
            sample_cfg["direct_costs_monthly"] = _derive_direct_costs(df)

    def _scale_staff_costs() -> None:
        staff_df = sample_cfg.get("staff_costs_monthly")
        if isinstance(staff_df, pd.DataFrame) and not staff_df.empty:
            df = staff_df.copy()
            mask = pd.Series(True, index=df.index)
            if applies not in {"", "global"} and "dept" in df.columns:
                mask = df["dept"].astype(str).str.lower() == applies
            cost_columns = [
                "gross_pay",
                "benefits",
                "training",
                "other",
                "gross_pay_per_head",
                "benefits_per_head",
                "training_per_head",
                "other_per_head",
            ]
            for col in cost_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            for col in cost_columns:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] * factor
            sample_cfg["staff_costs_monthly"] = df

    def _scale_other_opex() -> None:
        other_df = sample_cfg.get("other_opex_monthly")
        if isinstance(other_df, pd.DataFrame) and not other_df.empty:
            df = other_df.copy()
            mask = pd.Series(True, index=df.index)
            if applies not in {"", "global"} and "category" in df.columns:
                mask = df["category"].astype(str).str.lower() == applies
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
                df.loc[mask, "amount"] = df.loc[mask, "amount"] * factor
            sample_cfg["other_opex_monthly"] = df

    def _scale_production(product: Optional[str]) -> None:
        prod_annual = sample_cfg.get("production_annual")
        if isinstance(prod_annual, pd.DataFrame) and not prod_annual.empty:
            df = prod_annual.copy()
            mask = pd.Series(True, index=df.index)
            if product and "product" in df.columns:
                mask = df["product"].astype(str).str.lower() == product
            if "annual_volume" in df.columns:
                df["annual_volume"] = pd.to_numeric(df["annual_volume"], errors="coerce").fillna(0.0)
                df.loc[mask, "annual_volume"] = df.loc[mask, "annual_volume"] * factor
            if "availability" in df.columns and product is None and var_key == "availability":
                df["availability"] = pd.to_numeric(df["availability"], errors="coerce").fillna(0.0)
                df.loc[:, "availability"] = np.clip(df["availability"] * factor, 0.0, 1.0)
            sample_cfg["production_annual"] = df

        prod_monthly = sample_cfg.get("production_monthly")
        if isinstance(prod_monthly, pd.DataFrame) and not prod_monthly.empty:
            df_m = prod_monthly.copy()
            mask = pd.Series(True, index=df_m.index)
            if product and "product" in df_m.columns:
                mask = df_m["product"].astype(str).str.lower() == product
            if "volume" in df_m.columns:
                df_m["volume"] = pd.to_numeric(df_m["volume"], errors="coerce").fillna(0.0)
                df_m.loc[mask, "volume"] = df_m.loc[mask, "volume"] * factor
            sample_cfg["production_monthly"] = df_m

    def _scale_prices(product: Optional[str]) -> None:
        price_cfg = sample_cfg.setdefault("prices", {})
        if product:
            if product in price_cfg:
                params = price_cfg[product]
            else:
                params = price_cfg.setdefault(product, dict(DEFAULTS["prices"].get(product, {})))
            if isinstance(params, dict):
                params["base_price"] = float(params.get("base_price", 0.0)) * factor
        else:
            for prod, params in price_cfg.items():
                if isinstance(params, dict):
                    params["base_price"] = float(params.get("base_price", 0.0)) * factor

        revenue_df = sample_cfg.get("revenue_params")
        if isinstance(revenue_df, pd.DataFrame) and not revenue_df.empty:
            df = revenue_df.copy()
            mask = pd.Series(True, index=df.index)
            if product and "product" in df.columns:
                mask = df["product"].astype(str).str.lower() == product
            if "base_price" in df.columns:
                df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce")
                df.loc[mask, "base_price"] = df.loc[mask, "base_price"].fillna(0.0) * factor
            sample_cfg["revenue_params"] = df

    def _adjust_debt_rates() -> None:
        debt_df = sample_cfg.get("debt_tranches")
        if isinstance(debt_df, pd.DataFrame) and not debt_df.empty:
            df = debt_df.copy()
            for col in ("interest_rate", "base_rate", "margin"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                    df.loc[:, col] = df[col] * factor
            sample_cfg["debt_tranches"] = df
        debt_cfg = sample_cfg.get("debt", {})
        tranches = debt_cfg.get("tranches")
        if isinstance(tranches, list):
            for tranche in tranches:
                if isinstance(tranche, dict):
                    for col in ("interest_rate", "base_rate", "margin"):
                        if col in tranche and tranche[col] is not None:
                            try:
                                tranche[col] = float(tranche[col]) * factor
                            except (TypeError, ValueError):
                                continue

    def _adjust_debt_shares() -> None:
        debt_df = sample_cfg.get("debt_tranches")
        if isinstance(debt_df, pd.DataFrame) and not debt_df.empty and "share" in debt_df.columns:
            df = debt_df.copy()
            df["share"] = pd.to_numeric(df["share"], errors="coerce").fillna(0.0) * factor
            total = df["share"].sum()
            if total > 0:
                df["share"] = df["share"] / total
            sample_cfg["debt_tranches"] = df
        debt_cfg = sample_cfg.get("debt", {})
        tranches = debt_cfg.get("tranches")
        if isinstance(tranches, list):
            shares = []
            for tranche in tranches:
                share_val = 0.0
                if isinstance(tranche, dict):
                    try:
                        share_val = float(tranche.get("share", 0.0))
                    except (TypeError, ValueError):
                        share_val = 0.0
                shares.append(share_val)
            if shares:
                new_shares = [max(share * factor, 0.0) for share in shares]
                total = sum(new_shares)
                if total > 0:
                    new_shares = [share / total for share in new_shares]
                for tranche, share_val in zip(tranches, new_shares):
                    if isinstance(tranche, dict):
                        tranche["share"] = share_val

    def _adjust_yield() -> None:
        prod_defaults = sample_cfg.setdefault("production", {})
        if "sugarcane_yield_ton_per_ha" in prod_defaults:
            prod_defaults["sugarcane_yield_ton_per_ha"] = float(
                prod_defaults.get("sugarcane_yield_ton_per_ha", 0.0)
            ) * factor
        prod_annual = sample_cfg.get("production_annual")
        if isinstance(prod_annual, pd.DataFrame) and "sugarcane_yield_ton_per_ha" in prod_annual.columns:
            df = prod_annual.copy()
            df["sugarcane_yield_ton_per_ha"] = (
                pd.to_numeric(df["sugarcane_yield_ton_per_ha"], errors="coerce").fillna(0.0) * factor
            )
            sample_cfg["production_annual"] = df

    if var_key in {"capex", "initial_investment"}:
        _scale_capex()
        return
    if var_key in {"opex"}:
        opex_cfg = sample_cfg.setdefault("opex", {})
        for key in ("fixed_opex_per_month", "farm_opex_per_ton", "purchase_price_per_ton"):
            if key in opex_cfg and opex_cfg[key] is not None:
                try:
                    opex_cfg[key] = float(opex_cfg[key]) * factor
                except (TypeError, ValueError):
                    continue
        var_costs = opex_cfg.get("other_variable_cost_per_unit")
        if isinstance(var_costs, dict):
            for prod in list(var_costs.keys()):
                try:
                    var_costs[prod] = float(var_costs[prod]) * factor
                except (TypeError, ValueError):
                    continue
        _scale_direct_costs()
        _scale_staff_costs()
        _scale_other_opex()
        return
    if var_key in {"operating_cost_direct"}:
        _scale_direct_costs()
        return
    if var_key in {"operating_cost_staff", "labour"}:
        _scale_staff_costs()
        return
    if var_key == "operating_cost_other":
        _scale_other_opex()
        return
    if var_key == "interest_rate":
        _adjust_debt_rates()
        return
    if var_key == "debt_schedule":
        _adjust_debt_shares()
        return
    if var_key.startswith("production_"):
        product = var_key.split("_", 1)[1]
        if product in PRODUCTS:
            _scale_production(product)
        return
    if var_key == "production":
        if applies in PRODUCTS:
            _scale_production(applies)
        else:
            _scale_production(None)
        return
    if var_key.startswith("pricing_"):
        product = var_key.split("_", 1)[1]
        if product in PRODUCTS:
            _scale_prices(product)
        return
    if var_key in {"pricing", "revenue"}:
        if applies in PRODUCTS:
            _scale_prices(applies)
        else:
            _scale_prices(None)
        return
    if var_key == "sugarcane_yield":
        _adjust_yield()
        return
    if var_key == "availability":
        production_cfg = sample_cfg.setdefault("production", {})
        current = float(production_cfg.get("plant_availability", DEFAULTS["production"]["plant_availability"]))
        production_cfg["plant_availability"] = float(np.clip(current * factor, 0.0, 1.0))
        _scale_production(None)
        return
    if var_key == "other":
        return


def monte_carlo(cfg: Mapping[str, object], run_model_fn: Callable[[Mapping[str, object]], Dict[str, object]], iterations: int = 2000, random_seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(random_seed)
    risk_params_df = cfg.get("risk_params")
    if not isinstance(risk_params_df, pd.DataFrame) or risk_params_df.empty:
        risk_params_df = DEFAULTS["risk_params"].copy()
    else:
        risk_params_df = risk_params_df.copy()
    active_settings: List[Dict[str, object]] = []
    monte_cfg = cfg.get("monte_carlo_settings")
    if isinstance(monte_cfg, pd.DataFrame) and not monte_cfg.empty:
        for _, row in monte_cfg.iterrows():
            enabled_raw = row.get("enabled", False)
            enabled = False
            if isinstance(enabled_raw, str):
                enabled = enabled_raw.strip().lower() in {"true", "1", "yes", "y"}
            elif isinstance(enabled_raw, (bool, np.bool_)):
                enabled = bool(enabled_raw)
            elif isinstance(enabled_raw, (int, np.integer)):
                enabled = enabled_raw != 0
            elif isinstance(enabled_raw, float):
                enabled = not math.isnan(enabled_raw) and enabled_raw != 0.0
            if not enabled:
                continue
            active_settings.append(
                {
                    "distribution": str(row.get("distribution", "normal")),
                    "variable": str(row.get("variable", "opex")),
                    "applies_to": str(row.get("applies_to", "global")),
                    "p1": _coerce_float(row.get("p1"), 0.0),
                    "p2": _coerce_float(row.get("p2"), 0.0),
                    "p3": _coerce_float(row.get("p3"), 0.0),
                }
            )

    project_npvs: List[float] = []
    equity_irrs: List[float] = []
    unit_margins: List[float] = []
    base_price = cfg["prices"]["ethanol"]["base_price"]

    for _ in range(iterations):
        sample_cfg = copy.deepcopy(cfg)
        risk_df = risk_params_df.copy().reset_index(drop=True)
        driver_draws: List[Tuple[Dict[str, object], float]] = []
        for setting in active_settings:
            draw = _sample_distribution_value(
                rng,
                str(setting.get("distribution", "normal")),
                float(setting.get("p1", 0.0)),
                float(setting.get("p2", 0.0)),
                float(setting.get("p3", 0.0)),
            )
            driver_draws.append((setting, draw))
            _apply_monte_carlo_variable(
                sample_cfg,
                str(setting.get("variable", "")),
                draw,
                str(setting.get("applies_to", "global")),
            )
        for idx, param in risk_df.iterrows():
            dist = param.get("distribution", "normal")
            p1, p2, p3 = param.get("p1", 0.0), param.get("p2", 0.0), param.get("p3", 0.0)
            if dist == "normal":
                draw = rng.normal(p1, p2)
            elif dist == "lognormal":
                draw = rng.lognormal(p1, p2)
            elif dist == "triangular":
                draw = rng.triangular(p1, p2, p3)
            elif dist == "uniform":
                draw = rng.uniform(p1, p2)
            else:
                draw = p1
            target = str(param.get("target", "price")).lower()
            applies_to = param.get("applies_to", "global")
            if target == "risk":
                for column in RISK_MULTIPLIER_COLUMNS:
                    if column in risk_df.columns and pd.notna(param.get(column)):
                        current = float(param.get(column, 1.0))
                        adjustment = max(0.0, 1 + draw)
                        risk_df.at[idx, column] = current * adjustment
            elif target == "price" and applies_to in sample_cfg["prices"]:
                sample_cfg["prices"][applies_to]["base_price"] *= (1 + draw)
            elif target == "availability":
                sample_cfg["production"]["plant_availability"] *= (1 + draw)
            elif target == "opex":
                sample_cfg["opex"]["fixed_opex_per_month"] *= (1 + draw)
            elif target == "capex" and "capex_lines" in sample_cfg:
                if isinstance(sample_cfg["capex_lines"], pd.DataFrame):
                    sample_cfg["capex_lines"] = sample_cfg["capex_lines"].copy()
                    if "amount" in sample_cfg["capex_lines"]:
                        sample_cfg["capex_lines"]["amount"] = sample_cfg["capex_lines"]["amount"].astype(float) * (1 + draw)
                elif isinstance(sample_cfg["capex_lines"], list):
                    for line in sample_cfg["capex_lines"]:
                        line["amount"] *= (1 + draw)
        sample_cfg["risk_params"] = risk_df
        sample_cfg["risk_profile"] = compute_risk_profile(risk_df)
        risk_profile = sample_cfg.get("risk_profile", {})
        if isinstance(risk_profile, dict):
            for setting, draw in driver_draws:
                factor = 1.0 + float(draw)
                if not np.isfinite(factor):
                    continue
                factor = max(factor, 0.0)
                var_name = str(setting.get("variable", "")).lower()
                applies = str(setting.get("applies_to", "global")).lower()
                if var_name in {"operating_cost_staff", "labour"}:
                    risk_profile["labour"] = float(risk_profile.get("labour", 1.0)) * factor
                if var_name in {"pricing", "pricing_ethanol", "pricing_sugar", "pricing_electricity", "pricing_animal_feed"}:
                    risk_profile["price"] = float(risk_profile.get("price", 1.0)) * factor
                    product = None
                    if var_name.startswith("pricing_"):
                        product = var_name.split("_", 1)[1]
                    elif applies in PRODUCTS:
                        product = applies
                    if product in PRODUCTS:
                        price_map: Dict[str, float] = risk_profile.setdefault("price_by_product", {})  # type: ignore[assignment]
                        price_map[product] = price_map.get(product, 1.0) * factor
                if var_name in {"revenue"}:
                    risk_profile["revenue"] = float(risk_profile.get("revenue", 1.0)) * factor
                if var_name in {
                    "production",
                    "production_ethanol",
                    "production_sugar",
                    "production_electricity",
                    "production_animal_feed",
                    "availability",
                }:
                    risk_profile["production"] = float(risk_profile.get("production", 1.0)) * factor
                if var_name in {"sugarcane_yield"}:
                    risk_profile["yield"] = float(risk_profile.get("yield", 1.0)) * factor
        result = run_model_fn(sample_cfg)
        project_npvs.append(result["metrics"].get("Project_NPV", np.nan))
        equity_irrs.append(result["metrics"].get("Equity_IRR", np.nan))
        revenue = result.get("revenue", pd.DataFrame())
        if isinstance(revenue, pd.DataFrame) and not revenue.empty:
            unit_margins.append(revenue["revenue"].sum() / max(revenue["volume"].sum(), 1.0))
        else:
            unit_margins.append(sample_cfg["prices"]["ethanol"]["base_price"] - base_price)

    summary = pd.DataFrame({"Project_NPV": project_npvs, "Equity_IRR": equity_irrs, "Unit_Margin": unit_margins})
    stats = summary.quantile([0.1, 0.5, 0.9]).rename(index={0.1: "P10", 0.5: "P50", 0.9: "P90"})
    return {"samples": summary, "percentiles": stats}


def goal_seek(cfg: Mapping[str, object], run_model_fn: Callable[[Mapping[str, object]], Dict[str, object]], target_metric: str, target_value: float, variable: str, bounds: Tuple[float, float], tol: float = 1e-4, max_iter: int = 100) -> Optional[Dict[str, float]]:
    low, high = bounds
    cfg_low = copy.deepcopy(cfg)
    cfg_high = copy.deepcopy(cfg)

    def apply_value(cfg_mutable: Dict[str, object], value: float) -> None:
        if variable.endswith("_price"):
            prod = variable.split("_")[0]
            cfg_mutable["prices"][prod]["base_price"] = value
        elif variable == "debt_ratio":
            cfg_mutable["debt"]["target_ratio"] = value
        elif variable == "availability":
            cfg_mutable["production"]["plant_availability"] = value
        else:
            cfg_mutable.setdefault("overrides", {})[variable] = value

    for cfg_mutable, val in ((cfg_low, low), (cfg_high, high)):
        apply_value(cfg_mutable, val)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        cfg_mid = copy.deepcopy(cfg)
        apply_value(cfg_mid, mid)
        result = run_model_fn(cfg_mid)
        metric_value = result["metrics"].get(target_metric)
        if metric_value is None:
            return None
        if abs(metric_value - target_value) <= tol:
            return {"value": mid, "metric": metric_value}
        if metric_value < target_value:
            low = mid
        else:
            high = mid
    return None


def _apply_overrides(target: MutableMapping[str, object], overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and key in target and isinstance(target[key], MutableMapping):
            _apply_overrides(target[key], value)
        else:
            target[key] = value


def run_scenarios(cfg: Mapping[str, object], run_model_fn: Callable[[Mapping[str, object]], Dict[str, object]], scenarios: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    results = []
    for name, overrides in scenarios.items():
        scenario_cfg = copy.deepcopy(cfg)
        _apply_overrides(scenario_cfg, overrides)
        result = run_model_fn(scenario_cfg)
        metrics = result["metrics"]
        metrics["scenario"] = name
        results.append(metrics)
    return pd.DataFrame(results)


def _apply_lender_case(base_cfg: Mapping[str, object], case_row: Mapping[str, object]) -> Dict[str, object]:
    cfg_case = copy.deepcopy(dict(base_cfg))
    prod_mult = float(case_row.get("production_multiplier", 1.0) or 1.0)
    price_mult = float(case_row.get("price_multiplier", 1.0) or 1.0)
    opex_mult = float(case_row.get("opex_multiplier", 1.0) or 1.0)
    capex_mult = float(case_row.get("capex_multiplier", 1.0) or 1.0)
    delay_months = max(0, _coerce_int(case_row.get("delay_months"), 0))
    debt_shift = float(case_row.get("debt_rate_shift", 0.0) or 0.0)
    increment_profile = str(case_row.get("increment_profile", "") or "").strip()

    production_monthly = cfg_case.get("production_monthly")
    if isinstance(production_monthly, pd.DataFrame) and "volume" in production_monthly.columns:
        df = production_monthly.copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) * prod_mult
        cfg_case["production_monthly"] = df
    production_annual = cfg_case.get("production_annual")
    if isinstance(production_annual, pd.DataFrame) and "annual_volume" in production_annual.columns:
        df = production_annual.copy()
        df["annual_volume"] = pd.to_numeric(df["annual_volume"], errors="coerce").fillna(0.0) * prod_mult
        cfg_case["production_annual"] = df
    prices = cfg_case.get("prices", {})
    if isinstance(prices, dict):
        for product in PRODUCTS:
            if product in prices and isinstance(prices[product], dict):
                prices[product]["base_price"] = float(prices[product].get("base_price", 0.0)) * price_mult
    opex_cfg = cfg_case.get("opex", {})
    if isinstance(opex_cfg, dict):
        if "fixed_opex_per_month" in opex_cfg:
            opex_cfg["fixed_opex_per_month"] = float(opex_cfg["fixed_opex_per_month"]) * opex_mult
        if isinstance(opex_cfg.get("other_variable_cost_per_unit"), dict):
            for key in list(opex_cfg["other_variable_cost_per_unit"].keys()):
                opex_cfg["other_variable_cost_per_unit"][key] = float(opex_cfg["other_variable_cost_per_unit"][key]) * opex_mult
    capex_lines = cfg_case.get("capex_lines")
    if isinstance(capex_lines, pd.DataFrame) and "amount" in capex_lines.columns:
        df = capex_lines.copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0) * capex_mult
        if delay_months > 0:
            for col in ("start_date", "end_date"):
                if col in df.columns:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    df[col] = (parsed + pd.DateOffset(months=delay_months)).dt.strftime("%Y-%m-%d")
        cfg_case["capex_lines"] = df
    debt_df = cfg_case.get("debt_tranches")
    if isinstance(debt_df, pd.DataFrame):
        df = debt_df.copy()
        for col in ("interest_rate", "base_rate", "margin"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) + debt_shift
        cfg_case["debt_tranches"] = df
    if increment_profile:
        cfg_case["increment_profile"] = increment_profile
    cfg_case["_disable_lender_case_eval"] = True
    return cfg_case
###############################################################################
# Section 15: Break-even analysis
###############################################################################


def break_even_analysis(
    statements: Dict[str, pd.DataFrame],
    revenue_df: pd.DataFrame,
    production_df: pd.DataFrame,
    break_even_inputs: Optional[pd.DataFrame] = None,
    price_config: Optional[Mapping[str, Mapping[str, object]]] = None,
    cfg: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    pnl = statements.get("pnl", pd.DataFrame()).copy()
    if pnl.empty or "COGS" not in pnl:
        return {"overall": {}, "per_product": pd.DataFrame(), "inputs": pd.DataFrame()}

    production_local = production_df.copy() if isinstance(production_df, pd.DataFrame) else pd.DataFrame()
    if not production_local.empty:
        production_local = production_local.copy()
        if "product" in production_local.columns:
            production_local["product"] = (
                production_local["product"].astype(str).str.strip().str.lower()
            )
        if "volume" in production_local.columns:
            production_local["volume"] = pd.to_numeric(
                production_local["volume"], errors="coerce"
            ).fillna(0.0)
    else:
        production_local = pd.DataFrame(columns=["product", "volume"])

    revenue_local = revenue_df.copy() if isinstance(revenue_df, pd.DataFrame) else pd.DataFrame()
    if not revenue_local.empty:
        revenue_local = revenue_local.copy()
        if "product" in revenue_local.columns:
            revenue_local["product"] = (
                revenue_local["product"].astype(str).str.strip().str.lower()
            )
        if "revenue" in revenue_local.columns:
            revenue_local["revenue"] = pd.to_numeric(
                revenue_local["revenue"], errors="coerce"
            ).fillna(0.0)
    else:
        revenue_local = pd.DataFrame(columns=["product", "revenue"])

    total_volume = float(production_local.get("volume", pd.Series(dtype=float)).sum())
    total_periods = max(len(pnl), 1)
    total_fixed_costs = float((pnl.get("Opex", 0.0) + pnl.get("Depreciation", 0.0)).sum())
    total_cogs = float(pnl.get("COGS", 0.0).sum())
    variable_cost_per_unit = (total_cogs - total_fixed_costs) / max(total_volume, 1.0)
    average_price = revenue_local.get("revenue", pd.Series(dtype=float)).sum() / max(total_volume, 1.0)
    contribution_margin = average_price - variable_cost_per_unit
    break_even_volume = total_fixed_costs / max(contribution_margin, 1e-6)
    average_period_volume = total_volume / max(total_periods, 1)
    if average_period_volume > 0:
        margin_of_safety = 1 - break_even_volume / max(average_period_volume, 1e-6)
    else:
        margin_of_safety = np.nan

    default_inputs_df = DEFAULTS["break_even_inputs"].copy()
    default_inputs_df["product"] = (
        default_inputs_df["product"].astype(str).str.strip().str.lower()
    )
    default_map = {
        row["product"]: row
        for _, row in default_inputs_df.iterrows()
    }

    if isinstance(break_even_inputs, pd.DataFrame) and not break_even_inputs.empty:
        inputs_df = break_even_inputs.copy()
    else:
        inputs_df = pd.DataFrame(columns=default_inputs_df.columns)

    inputs_df = inputs_df.replace({"": np.nan})
    if "product" in inputs_df.columns:
        inputs_df["product"] = (
            inputs_df["product"].astype(str).str.strip().str.lower()
        )
    required_cols = [
        "product",
        "unit_price",
        "variable_cost_per_unit",
        "fixed_cost",
        "reference_volume",
    ]
    for col in required_cols:
        if col not in inputs_df.columns:
            inputs_df[col] = np.nan

    sanitized_records: List[Dict[str, object]] = []
    for product in PRODUCTS:
        candidate = inputs_df[inputs_df["product"] == product].tail(1)
        if not candidate.empty:
            record = candidate.iloc[0].to_dict()
        elif product in default_map:
            record = default_map[product].to_dict()
        else:
            record = {
                "product": product,
                "unit_price": np.nan,
                "variable_cost_per_unit": np.nan,
                "fixed_cost": 0.0,
                "reference_volume": np.nan,
            }
        record["product"] = product
        sanitized_records.append(record)

    inputs_used = pd.DataFrame(sanitized_records)
    monthly_dates = pd.to_datetime(production_local.get("date"), errors="coerce") if "date" in production_local.columns else pd.Series(dtype="datetime64[ns]")
    if monthly_dates.empty:
        avg_price_factor = 1.0
        avg_opex_factor = 1.0
    else:
        tmp_timeline = Timeline(
            start_year=int(monthly_dates.dropna().min().year) if not monthly_dates.dropna().empty else DEFAULTS["horizon"]["start_year"],
            end_year=int(monthly_dates.dropna().max().year) if not monthly_dates.dropna().empty else DEFAULTS["horizon"]["end_year"],
            start_month=1,
        )
        avg_price_factor = float(get_increment_series(cfg, tmp_timeline, "price_factor").mean()) if isinstance(cfg, Mapping) else 1.0
        avg_opex_factor = float(get_increment_series(cfg, tmp_timeline, "opex_factor").mean()) if isinstance(cfg, Mapping) else 1.0
    if "unit_price" in inputs_used.columns:
        inputs_used["unit_price"] = pd.to_numeric(inputs_used["unit_price"], errors="coerce") * avg_price_factor
    if "variable_cost_per_unit" in inputs_used.columns:
        inputs_used["variable_cost_per_unit"] = pd.to_numeric(inputs_used["variable_cost_per_unit"], errors="coerce") * avg_opex_factor
    if "fixed_cost" in inputs_used.columns:
        inputs_used["fixed_cost"] = pd.to_numeric(inputs_used["fixed_cost"], errors="coerce") * avg_opex_factor
    for col in ["unit_price", "variable_cost_per_unit", "fixed_cost", "reference_volume"]:
        inputs_used[col] = pd.to_numeric(inputs_used[col], errors="coerce")

    production_totals = (
        production_local.groupby("product")["volume"].sum()
        if "product" in production_local.columns
        else pd.Series(dtype=float)
    )
    revenue_totals = (
        revenue_local.groupby("product")["revenue"].sum()
        if "product" in revenue_local.columns
        else pd.Series(dtype=float)
    )

    unit_map: Dict[str, str] = {}
    if isinstance(price_config, Mapping):
        for product in PRODUCTS:
            price_info = price_config.get(product, {})
            unit_map[product] = (
                str(price_info.get("uom", "")) if isinstance(price_info, Mapping) else ""
            )

    per_product_records: List[Dict[str, object]] = []
    for _, row in inputs_used.iterrows():
        product = str(row.get("product", "")).strip().lower()
        if product not in PRODUCTS:
            continue
        unit_price = float(row.get("unit_price", np.nan))
        variable_cost = float(row.get("variable_cost_per_unit", np.nan))
        fixed_cost = float(row.get("fixed_cost", 0.0) or 0.0)
        reference_volume = float(row.get("reference_volume", np.nan))
        actual_volume = float(production_totals.get(product, 0.0))
        if not np.isfinite(reference_volume) or reference_volume <= 0:
            reference_volume = actual_volume if actual_volume > 0 else np.nan

        contribution = unit_price - variable_cost if np.isfinite(unit_price) and np.isfinite(variable_cost) else np.nan
        if contribution is None or not np.isfinite(contribution) or contribution <= 0:
            break_even_units = np.nan
        else:
            break_even_units = fixed_cost / contribution if contribution != 0 else np.nan

        break_even_revenue = (
            break_even_units * unit_price if np.isfinite(break_even_units) and np.isfinite(unit_price) else np.nan
        )
        actual_revenue = float(revenue_totals.get(product, 0.0))
        actual_average_price = (
            actual_revenue / actual_volume if actual_volume > 0 else np.nan
        )
        margin_units = (
            actual_volume - break_even_units
            if np.isfinite(actual_volume) and np.isfinite(break_even_units)
            else np.nan
        )
        if np.isfinite(margin_units) and actual_volume > 0:
            margin_percent = margin_units / actual_volume
        else:
            margin_percent = np.nan
        if np.isfinite(break_even_units) and reference_volume and np.isfinite(reference_volume) and reference_volume > 0:
            break_even_vs_reference = break_even_units / reference_volume
        else:
            break_even_vs_reference = np.nan

        per_product_records.append(
            {
                "product": product,
                "unit_of_measure": unit_map.get(product, ""),
                "unit_price": unit_price,
                "variable_cost_per_unit": variable_cost,
                "contribution_margin_per_unit": contribution,
                "fixed_cost": fixed_cost,
                "break_even_units": break_even_units,
                "reference_volume": reference_volume,
                "actual_volume": actual_volume,
                "actual_average_price": actual_average_price,
                "break_even_revenue": break_even_revenue,
                "actual_revenue": actual_revenue,
                "margin_of_safety_units": margin_units,
                "margin_of_safety_percent": margin_percent,
                "break_even_vs_reference": break_even_vs_reference,
            }
        )

    per_product_df = pd.DataFrame(per_product_records)
    numeric_columns = [
        "unit_price",
        "variable_cost_per_unit",
        "contribution_margin_per_unit",
        "fixed_cost",
        "break_even_units",
        "reference_volume",
        "actual_volume",
        "actual_average_price",
        "break_even_revenue",
        "actual_revenue",
        "margin_of_safety_units",
        "margin_of_safety_percent",
        "break_even_vs_reference",
    ]
    for col in numeric_columns:
        if col in per_product_df.columns:
            per_product_df[col] = pd.to_numeric(per_product_df[col], errors="coerce")

    per_product_df = per_product_df.sort_values("product").reset_index(drop=True)

    overall = {
        "fixed_costs_total": total_fixed_costs,
        "variable_cost_per_unit": variable_cost_per_unit,
        "average_price": average_price,
        "break_even_volume": break_even_volume,
        "margin_of_safety": margin_of_safety,
        "total_volume": total_volume,
    }

    return {"overall": overall, "per_product": per_product_df, "inputs": inputs_used}
###############################################################################
# Section 16: Model orchestration
###############################################################################


def run_full_model(cfg: Mapping[str, object], export_dir: Optional[Path] = None) -> Dict[str, object]:
    cfg = copy.deepcopy(dict(cfg))
    cfg.setdefault("increment_profile", "base")
    risk_params = cfg.get("risk_params")
    if not isinstance(risk_params, pd.DataFrame) or risk_params.empty:
        risk_params = DEFAULTS["risk_params"].copy()
        cfg["risk_params"] = risk_params
    cfg["risk_profile"] = compute_risk_profile(risk_params)

    timeline = Timeline(
        start_year=int(cfg["projection_horizon"]["start_year"]),
        end_year=int(cfg["projection_horizon"]["end_year"]),
        start_month=int(cfg["projection_horizon"].get("start_month", 1)),
    )
    cfg["_increment_factors"] = build_yearly_increment_factors(cfg, timeline)
    production_monthly, production_annual = build_production_tables(cfg, timeline)
    price_curves = build_price_curves(cfg, timeline)
    revenue_df = build_revenue_stack(cfg, production_monthly, price_curves)
    capex_info = build_capex_depr_monthly(cfg, timeline)
    capex_adjusted, construction_report = apply_construction_risk(cfg, timeline, capex_info["capex"])
    capex_info["capex"] = capex_adjusted
    debt_schedule = build_debt_schedule(cfg, timeline, capex_info["capex"])

    cost_df = cfg.get("direct_costs_monthly") if "direct_costs_monthly" in cfg else pd.DataFrame({"date": timeline.monthly_index(), "amount": 0.0})
    if not isinstance(cost_df, pd.DataFrame) or cost_df.empty:
        cost_df = pd.DataFrame({
            "date": timeline.monthly_index(),
            "unit_price": 0.0,
            "quantity": 0.0,
            "amount": 0.0,
        })
    else:
        cost_df = cost_df.copy()
        cost_df["date"] = pd.to_datetime(cost_df["date"], errors="coerce")
    cost_df = _derive_direct_costs(cost_df)
    wc_df = working_capital_block(revenue_df, cost_df, cfg, timeline)

    statements = statements_monthly(cfg, timeline, revenue_df, production_monthly, capex_info, debt_schedule, wc_df)
    tranches_df = cfg.get("debt_tranches")
    has_sculpted = isinstance(tranches_df, pd.DataFrame) and (
        ("sculpting_enabled" in tranches_df.columns and tranches_df["sculpting_enabled"].fillna(False).astype(bool).any())
        or ("amortization" in tranches_df.columns and tranches_df["amortization"].astype(str).str.lower().eq("sculpted").any())
    )
    if has_sculpted:
        cfads_series = None
        cashflow_for_sculpt = statements.get("cashflow", pd.DataFrame()) if "cashflow" in statements else pd.DataFrame()
        if isinstance(cashflow_for_sculpt, pd.DataFrame) and not cashflow_for_sculpt.empty and "date" in cashflow_for_sculpt.columns:
            cfads_source = cashflow_for_sculpt.copy()
            cfads_source["date"] = pd.to_datetime(cfads_source["date"], errors="coerce")
            cfads_source = cfads_source.dropna(subset=["date"])
            if "CFO" in cfads_source.columns:
                cfads_series = cfads_source.set_index("date")["CFO"]
        debt_schedule = build_debt_schedule(cfg, timeline, capex_info["capex"], cfads_series=cfads_series)
        statements = statements_monthly(cfg, timeline, revenue_df, production_monthly, capex_info, debt_schedule, wc_df)
    staff_detail = statements.get("staff_costs_detail", pd.DataFrame())
    valuation = project_cashflows(statements, cfg, timeline)
    credit_outputs = evaluate_credit_and_waterfall(cfg, timeline, statements, debt_schedule, case_name="base")
    be_inputs_cfg = cfg.get("break_even_inputs") if isinstance(cfg.get("break_even_inputs"), pd.DataFrame) else None
    be = break_even_analysis(
        statements,
        revenue_df,
        production_monthly,
        break_even_inputs=be_inputs_cfg,
        price_config=cfg.get("prices"),
        cfg=cfg,
    )
    dashboard = build_dashboard(
        cfg,
        statements,
        production_monthly,
        revenue_df,
        valuation,
        working_capital=wc_df,
        debt_schedule=debt_schedule,
        staff_detail=staff_detail,
        break_even=be,
        out_dir=export_dir,
    )

    results = {
        "config": cfg,
        "timeline": timeline,
        "production_monthly": production_monthly,
        "production_annual": production_annual,
        "price_curves": price_curves,
        "revenue": revenue_df,
        "capex": capex_info["capex"],
        "depreciation": capex_info["depreciation"],
        "debt_schedule": debt_schedule,
        "working_capital": wc_df,
        "statements_monthly": statements,
        "statements_annual": {k: aggregate_annual(v) for k, v in statements.items()},
        "metrics": valuation["metrics"],
        "dashboard": dashboard,
        "break_even": be,
        "break_even_inputs": (
            be.get("inputs").copy()
            if isinstance(be, dict) and isinstance(be.get("inputs"), pd.DataFrame)
            else be_inputs_cfg,
        ),
        "risk_profile": cfg.get("risk_profile", {}),
        "construction_report": construction_report,
        "credit_metrics_yearly": credit_outputs.get("credit_metrics_yearly", pd.DataFrame()),
        "covenant_monitor": credit_outputs.get("covenant_monitor", pd.DataFrame()),
        "cash_waterfall": credit_outputs.get("cash_waterfall", pd.DataFrame()),
        "completion_tests_result": credit_outputs.get("completion_tests", pd.DataFrame()),
        "tornado_drivers": cfg.get("tornado_drivers").copy()
        if isinstance(cfg.get("tornado_drivers"), pd.DataFrame)
        else cfg.get("tornado_drivers"),
        "monte_carlo_settings": cfg.get("monte_carlo_settings").copy()
        if isinstance(cfg.get("monte_carlo_settings"), pd.DataFrame)
        else cfg.get("monte_carlo_settings"),
        "scenario_comparison": cfg.get("scenario_comparison").copy()
        if isinstance(cfg.get("scenario_comparison"), pd.DataFrame)
        else cfg.get("scenario_comparison"),
        "offtake_terms": cfg.get("offtake_terms").copy() if isinstance(cfg.get("offtake_terms"), pd.DataFrame) else cfg.get("offtake_terms"),
        "dispatch_constraints": cfg.get("dispatch_constraints").copy() if isinstance(cfg.get("dispatch_constraints"), pd.DataFrame) else cfg.get("dispatch_constraints"),
        "construction_schedule": cfg.get("construction_schedule").copy() if isinstance(cfg.get("construction_schedule"), pd.DataFrame) else cfg.get("construction_schedule"),
        "reserve_accounts": cfg.get("reserve_accounts").copy() if isinstance(cfg.get("reserve_accounts"), pd.DataFrame) else cfg.get("reserve_accounts"),
        "covenant_thresholds": cfg.get("covenant_thresholds").copy() if isinstance(cfg.get("covenant_thresholds"), pd.DataFrame) else cfg.get("covenant_thresholds"),
        "lender_cases": cfg.get("lender_cases").copy() if isinstance(cfg.get("lender_cases"), pd.DataFrame) else cfg.get("lender_cases"),
        "yearly_increments": cfg.get("yearly_increments").copy() if isinstance(cfg.get("yearly_increments"), pd.DataFrame) else cfg.get("yearly_increments"),
        "increment_factors": cfg.get("_increment_factors").copy() if isinstance(cfg.get("_increment_factors"), pd.DataFrame) else cfg.get("_increment_factors"),
    }
    if not bool(cfg.get("_disable_lender_case_eval", False)):
        lender_case_rows = cfg.get("lender_cases")
        case_outputs: List[Dict[str, object]] = []
        if isinstance(lender_case_rows, pd.DataFrame) and not lender_case_rows.empty:
            for _, row in lender_case_rows.iterrows():
                if not bool(row.get("enabled", True)):
                    continue
                case_name = str(row.get("case_name", "Case")).strip() or "Case"
                case_cfg = _apply_lender_case(cfg, row.to_dict())
                case_result = run_full_model(case_cfg, export_dir=None)
                case_metrics = dict(case_result.get("metrics", {}))
                case_metrics["case_name"] = case_name
                case_metrics["assumption_version"] = f"{normalize_key(case_name)}-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
                case_metrics["run_timestamp_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
                case_outputs.append(case_metrics)
        results["lender_case_results"] = pd.DataFrame(case_outputs)
    if isinstance(staff_detail, pd.DataFrame):
        results["staff_costs_detail"] = staff_detail
        results["staff_costs_detail_annual"] = aggregate_annual(staff_detail)
    return results
###############################################################################
# Section 17: Export utilities
###############################################################################


def export_csv(results: Mapping[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / f"{key}.csv", index=False)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, pd.DataFrame):
                    subvalue.to_csv(out_dir / f"{key}_{subkey}.csv", index=False)


def _sanitize_sheet_name(name: str, existing: Sequence[str]) -> str:
    base = re.sub(r"[\[\]:\\/?*]", "", str(name)) or "Sheet"
    base = base[:31]
    candidate = base
    counter = 1
    while candidate in existing:
        suffix = f"_{counter}"
        candidate = f"{base[: max(0, 31 - len(suffix))]}{suffix}" or f"Sheet_{counter}"
        counter += 1
    return candidate


def _column_letter(index: int) -> str:
    result = ""
    idx = index
    while idx >= 0:
        idx, rem = divmod(idx, 26)
        result = chr(65 + rem) + result
        idx -= 1
    return result


def write_simple_xlsx(sheets: Mapping[str, pd.DataFrame], handle: BinaryIO) -> None:
    """Write a minimal XLSX workbook without third-party engines."""

    shared_strings: Dict[str, int] = {}
    shared_order: List[str] = []

    def add_shared(text: str) -> int:
        if text in shared_strings:
            return shared_strings[text]
        idx = len(shared_order)
        shared_strings[text] = idx
        shared_order.append(text)
        return idx

    sheet_defs: List[Tuple[str, str]] = []
    existing_names: List[str] = []

    for sheet_name, df in sheets.items():
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        df_local = df.copy()
        sanitized = _sanitize_sheet_name(str(sheet_name), existing_names)
        existing_names.append(sanitized)
        rows_xml: List[str] = []

        headers = list(df_local.columns)
        if headers:
            cells = []
            for col_idx, heading in enumerate(headers):
                text = str(heading)
                ref = f"{_column_letter(col_idx)}1"
                idx = add_shared(text)
                cells.append(f'<c r="{ref}" t="s"><v>{idx}</v></c>')
            rows_xml.append(f"<row r=\"1\">{''.join(cells)}</row>")

        for row_idx, row in enumerate(df_local.itertuples(index=False, name=None), start=2 if headers else 1):
            cells = []
            for col_idx, value in enumerate(row):
                if value is None:
                    continue
                if isinstance(value, (float, np.floating)) and math.isnan(value):
                    continue
                if pd.isna(value):
                    continue
                cell_ref = f"{_column_letter(col_idx)}{row_idx}"
                formatted = None
                cell_type = ""
                if isinstance(value, (int, np.integer)):
                    formatted = f"<v>{int(value)}</v>"
                elif isinstance(value, (float, np.floating)) and not math.isnan(value):
                    formatted = f"<v>{float(value)}</v>"
                elif isinstance(value, (pd.Timestamp, np.datetime64, datetime.date, datetime.datetime)):
                    if isinstance(value, np.datetime64):
                        value = pd.Timestamp(value).to_pydatetime()
                    if isinstance(value, pd.Timestamp):
                        value = value.to_pydatetime()
                    if isinstance(value, datetime.datetime):
                        value = value.isoformat()
                    elif isinstance(value, datetime.date):
                        value = value.isoformat()
                    else:
                        value = str(value)
                    idx = add_shared(str(value))
                    formatted = f"<v>{idx}</v>"
                    cell_type = ' t="s"'
                elif isinstance(value, bool):
                    idx = add_shared("TRUE" if value else "FALSE")
                    formatted = f"<v>{idx}</v>"
                    cell_type = ' t="s"'
                else:
                    idx = add_shared(str(value))
                    formatted = f"<v>{idx}</v>"
                    cell_type = ' t="s"'
                if formatted is not None:
                    cells.append(f'<c r="{cell_ref}"{cell_type}>{formatted}</c>')
            rows_xml.append(f"<row r=\"{row_idx}\">{''.join(cells)}</row>")

        sheet_content = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            f"<sheetData>{''.join(rows_xml)}</sheetData>"
            "</worksheet>"
        )
        sheet_defs.append((sanitized, sheet_content))

    if not sheet_defs:
        sheet_defs.append(
            (
                "Summary",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheetData/></worksheet>",
            )
        )

    shared_entries = ''.join(
        f"<si><t>{xml_escape(text)}</t></si>" for text in shared_order
    )
    shared_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        f"<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"{len(shared_order)}\" "
        f"uniqueCount=\"{len(shared_order)}\">{shared_entries}</sst>"
    )

    sheets_entries = []
    sheets_rels = []
    for idx, (name, _) in enumerate(sheet_defs, start=1):
        sheets_entries.append(
            f"<sheet name=\"{xml_escape(name)}\" sheetId=\"{idx}\" r:id=\"rId{idx}\"/>"
        )
        sheets_rels.append(
            f"<Relationship Id=\"rId{idx}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" "
            f"Target=\"worksheets/sheet{idx}.xml\"/>"
        )

    workbook_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        f"<sheets>{''.join(sheets_entries)}</sheets></workbook>"
    )

    workbook_rels = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        f"{''.join(sheets_rels)}"
        "<Relationship Id=\"rId_styles\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>"
        "<Relationship Id=\"rId_sharedStrings\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings\" Target=\"sharedStrings.xml\"/>"
        "</Relationships>"
    )

    styles_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
        "<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>"
        "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellXfs>"
        "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>"
        "</styleSheet>"
    )

    content_types = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">",
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
        "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>",
        "<Override PartName=\"/xl/sharedStrings.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml\"/>",
        "<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>",
    ]
    for idx, _ in enumerate(sheet_defs, start=1):
        content_types.append(
            f"<Override PartName=\"/xl/worksheets/sheet{idx}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
        )
    content_types.append("</Types>")
    content_types_xml = ''.join(content_types)

    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/></Relationships>")
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/sharedStrings.xml", shared_xml)
        for idx, (_, content) in enumerate(sheet_defs, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", content)


def resolve_excel_engine(preferred: Sequence[str] = ("xlsxwriter", "openpyxl")) -> str:
    """Return the first available Excel writer engine from the preferred list."""

    for engine in preferred:
        module = "openpyxl" if engine == "openpyxl" else engine
        try:
            importlib.import_module(module)
            return engine
        except ImportError:
            continue
    return SIMPLE_XLSX_ENGINE


def build_excel_pack(results: Mapping[str, object], path: Path) -> None:
    engine = resolve_excel_engine()
    sheets: OrderedDict[str, pd.DataFrame] = OrderedDict()
    dashboard = results.get("dashboard") if isinstance(results, Mapping) else None
    if isinstance(dashboard, Mapping):
        snapshot = dashboard.get("assumptions_snapshot")
        if isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
            sheets["Summary"] = snapshot
        overview = dashboard.get("overview_metrics")
        if isinstance(overview, pd.DataFrame) and not overview.empty:
            sheets["Metrics"] = overview
        annual_prod = dashboard.get("annual_production")
        if isinstance(annual_prod, pd.DataFrame) and not annual_prod.empty:
            sheets["Production"] = annual_prod

    statements = results.get("statements_annual") if isinstance(results, Mapping) else None
    if isinstance(statements, Mapping):
        for key in ("pnl", "cashflow", "balancesheet"):
            df = statements.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                sheets[f"Annual_{key}"] = df

    for label, key in (("CAPEX", "capex"), ("Debt", "debt_schedule"), ("WorkingCapital", "working_capital")):
        df = results.get(key) if isinstance(results, Mapping) else None
        if isinstance(df, pd.DataFrame) and not df.empty:
            sheets[label] = df

    if engine == SIMPLE_XLSX_ENGINE:
        with open(path, "wb") as handle:
            write_simple_xlsx(sheets, handle)
        return

    with pd.ExcelWriter(path, engine=engine) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
###############################################################################
# Section 18: CLI entrypoint
###############################################################################


def run_pipeline(cfg: Mapping[str, object]) -> Dict[str, object]:
    return run_full_model(cfg)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sugarcane Bioethanol Multi-Product Project Finance Model")
    parser.add_argument("--excel", type=str, default=None, help="Path to Excel workbook containing assumptions")
    parser.add_argument("--export", type=str, default=None, help="Directory to export CSV outputs")
    parser.add_argument("--excel-pack", dest="excel_pack", type=str, default=None, help="Path to export Excel pack")
    parser.add_argument("--capex-sheet", dest="capex_sheet", type=str, default=None, help="Sheet name containing CAPEX table")
    parser.add_argument("--preview", type=int, default=0, help="Preview N rows per sheet")
    parser.add_argument("--iterations", type=int, default=2000, help="Monte Carlo iterations")
    parser.add_argument("--seed", type=int, default=42, help="Monte Carlo random seed")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_inputs_from_excel(path: Path, preview: int = 0) -> Tuple[InputTables, Dict[str, object]]:
    sheets = load_all(path, header=0)
    if preview:
        print("Workbook preview:")
        for key, df in sheets.items():
            print(f"Sheet {key}: {df.shape[0]} rows x {df.shape[1]} cols")
            print(df.head(preview))
            print()
    assumptions = detect_assumptions(sheets)
    tables = InputTables()
    tables.load_from_workbook(sheets)
    return tables, assumptions


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    tables = InputTables()
    assumptions: Dict[str, object] = {}

    if args.excel:
        excel_path = Path(args.excel)
        if excel_path.exists():
            tables, assumptions = load_inputs_from_excel(excel_path, preview=args.preview)
            if args.capex_sheet:
                try:
                    capex_df = load_sheet(excel_path, args.capex_sheet, header=0)
                    tables.tables["capex_lines"] = capex_df
                except Exception as exc:
                    print(f"Failed to parse CAPEX sheet {args.capex_sheet}: {exc}")
        else:
            print(f"Excel file {excel_path} not found. Using defaults.")

    cfg = build_config(assumptions, tables)

    export_dir = Path(args.export) if args.export else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    results = run_full_model(cfg, export_dir=export_dir / "out_png" if export_dir else None)

    tornado = sensitivity_tornado(cfg, {"metrics": results["metrics"]}, lambda c: run_full_model(c), None)
    monte = monte_carlo(cfg, lambda c: run_full_model(c), iterations=args.iterations, random_seed=args.seed)
    scenarios = {
        "FARM_ONLY": {"production": {"feedstock_scenario": "FARM_ONLY"}},
        "BUY_ONLY": {"production": {"feedstock_scenario": "BUY_ONLY"}},
        "HYBRID": {"production": {"feedstock_scenario": "HYBRID"}},
    }
    scenario_results = run_scenarios(cfg, lambda c: run_full_model(c), scenarios)

    results.update({
        "sensitivities": tornado,
        "monte_carlo": monte,
        "scenarios": scenario_results,
    })

    if export_dir is not None:
        export_csv(results, export_dir)
    if args.excel_pack:
        try:
            build_excel_pack(results, Path(args.excel_pack))
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)

    print("Key Metrics:")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
