"""NumQuants entry point for the Cassava-style Sugar Cane architecture."""
from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

import streamlit as st

from app.plugin.contract import Format, ReportOptions, SubscriptionTier, User
from sugarcane_model import SugarcaneBioethanolInputs, input_from_payload, input_values
from sugarcane_refactored_plugin import MODEL


_INPUT_KEY = "_sugarcane_refactored_inputs"
_RESULT_KEY = "_sugarcane_refactored_result"
_REPORT_CACHE_KEY = "sugarcane_refactored_report_cache"


def _current_user() -> User:
    """Build the plugin user value after the parent shell has enforced access."""
    email = "numquants-user@numquants.com"
    for key in ("user_email", "email", "auth_email"):
        value = st.session_state.get(key)
        if isinstance(value, str) and value.strip():
            email = value.strip()
            break
    return User(id=UUID(int=0), email=email, tier=SubscriptionTier.PRO)


def _input_fingerprint(inputs: SugarcaneBioethanolInputs) -> str:
    payload = json.dumps(input_values(inputs), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_report_downloads(plugin, inputs, results, user) -> None:
    """Render deterministic local downloads without a second plugin registry."""
    fingerprint = _input_fingerprint(inputs)
    cache: dict[str, dict[str, bytes]] = st.session_state.setdefault(
        _REPORT_CACHE_KEY, {}
    )
    reports = cache.get(fingerprint)
    if reports is None:
        generated = plugin.generate_report(
            inputs=inputs,
            results=results,
            formats={Format.XLSX, Format.PDF},
            options=ReportOptions(),
            user=user,
        )
        reports = {fmt.value: data for fmt, data in generated.items()}
        cache[fingerprint] = reports

    st.subheader("Download model reports")
    columns = st.columns(2)
    columns[0].download_button(
        "Download Excel model",
        data=reports[Format.XLSX.value],
        file_name="Sugar_Cane_Bioethanol_Model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    columns[1].download_button(
        "Download PDF summary",
        data=reports[Format.PDF.value],
        file_name="Sugar_Cane_Bioethanol_Summary.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def main() -> None:
    """Render the authoritative modular Sugar Cane application."""
    import app.reports.ui as reports_ui

    reports_ui.render_report_downloads = _render_report_downloads
    st.markdown("## Sugar Cane Bioethanol")
    st.caption(
        "Cassava-style modular architecture for farming, sourcing, multi-product "
        "routing, component financials, and consolidated statements."
    )
    MODEL.render(user=_current_user())


def get_state() -> dict[str, Any]:
    """Return the durable user-editable state consumed by saved cases."""
    state: dict[str, Any] = {}
    inputs = st.session_state.get(_INPUT_KEY)
    if isinstance(inputs, SugarcaneBioethanolInputs):
        state["cassava_architecture_inputs"] = input_values(inputs)
    return state


def set_state(state: dict[str, Any]) -> None:
    """Restore saved inputs and require an explicit recalculation."""
    payload = state.get("cassava_architecture_inputs")
    if isinstance(payload, dict):
        st.session_state[_INPUT_KEY] = input_from_payload(payload)
        st.session_state.pop(_RESULT_KEY, None)
        st.session_state.pop("sugar_active_result", None)
        st.session_state["sugar_model_results_stale"] = True
        st.session_state.pop(_REPORT_CACHE_KEY, None)
