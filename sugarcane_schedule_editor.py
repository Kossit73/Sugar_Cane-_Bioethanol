"""Cassava-style draft editor for Sugar Cane annual driver schedules."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import pandas as pd

from sugarcane_model.driver_schedules import (
    SCHEDULE_DEFINITIONS,
    default_schedule_row,
    schedule_rows,
    validate_driver_schedules,
)
from sugarcane_model.inputs import CapexItem, input_from_payload, input_values
from sugarcane_model.schedule_workspace import (
    ROW_ID,
    add_capex_row,
    add_year_row,
    propagate_capex,
    propagate_yearly,
    remove_row,
    row_label,
    strip_row_ids,
    update_row,
    with_row_ids,
)


_CAPEX_KEY = "capex"
_DRAFTS_KEY = "_sugarcane_schedule_workspace_drafts"
_SOURCES_KEY = "_sugarcane_schedule_workspace_sources"
_DIRTY_KEY = "_sugarcane_schedule_workspace_dirty"
_SELECTED_KEY = "_sugarcane_schedule_workspace_selected"
_REVISION_KEY = "_sugarcane_schedule_workspace_revisions"
_FLASH_KEY = "_sugarcane_schedule_workspace_flash"
_DEFAULT_INCREMENT_FIELDS = {
    "other_assumptions": ["plant_availability"],
    "cycle_planning": ["replant_share_per_cycle"],
    "farming": ["sugarcane_yield_tonnes_per_hectare", "farm_opex_per_tonne", "farm_overhead_per_year", "internal_transfer_price_per_tonne"],
    "sourcing": ["cane_purchase_price_per_tonne", "logistics_cost_per_tonne"],
    "processing_routing": ["annual_cane_capacity_tonnes", "ethanol_litres_per_tonne", "sugar_tonnes_per_tonne", "raw_bagasse_tonnes_per_tonne", "electricity_mwh_per_tonne_bagasse"],
    "commercialization": ["ethanol_price_per_litre", "sugar_price_per_tonne", "electricity_tariff_per_mwh", "bagasse_price_per_tonne", "animal_feed_price_per_tonne"],
    "costs": list(SCHEDULE_DEFINITIONS["costs"]["fields"]),
    "working_capital": list(SCHEDULE_DEFINITIONS["working_capital"]["fields"]),
    "financing": ["interest_rate"],
}



def _signature(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(strip_row_ids(rows), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _state_dict(st, key: str) -> dict[str, Any]:
    value = st.session_state.get(key)
    if not isinstance(value, dict):
        value = {}
        st.session_state[key] = value
    return value


def _source_rows(inputs, schedule_key: str) -> list[dict[str, Any]]:
    if schedule_key == _CAPEX_KEY:
        return with_row_ids(input_values(inputs)["capex"]["items"], "capex")
    return with_row_ids(schedule_rows(inputs, schedule_key), schedule_key)


def _draft(st, inputs, schedule_key: str) -> list[dict[str, Any]]:
    drafts = _state_dict(st, _DRAFTS_KEY)
    sources = _state_dict(st, _SOURCES_KEY)
    dirty = _state_dict(st, _DIRTY_KEY)
    source = _source_rows(inputs, schedule_key)
    source_signature = _signature(source)
    if schedule_key not in drafts or (
        not bool(dirty.get(schedule_key)) and sources.get(schedule_key) != source_signature
    ):
        drafts[schedule_key] = source
        sources[schedule_key] = source_signature
        dirty[schedule_key] = False
    return [dict(row) for row in drafts[schedule_key]]


def _store_draft(st, schedule_key: str, rows: list[dict[str, Any]], *, dirty: bool) -> None:
    _state_dict(st, _DRAFTS_KEY)[schedule_key] = [dict(row) for row in rows]
    _state_dict(st, _DIRTY_KEY)[schedule_key] = bool(dirty)
    revisions = _state_dict(st, _REVISION_KEY)
    revisions[schedule_key] = int(revisions.get(schedule_key, 0)) + 1


def _set_flash(st, schedule_key: str, message: str) -> None:
    _state_dict(st, _FLASH_KEY)[schedule_key] = message


def _field_label(field: str, number_specs: dict[str, list[tuple]]) -> str:
    for specs in number_specs.values():
        for key, label, *_ in specs:
            if key == field:
                return str(label)
    return field.replace("_", " ").title()


def _number_spec(field: str, number_specs: dict[str, list[tuple]]) -> tuple[Any, ...] | None:
    for specs in number_specs.values():
        for spec in specs:
            if spec[0] == field:
                return spec
    return None


def _edit_field(
    st,
    field: str,
    value: Any,
    *,
    widget_key: str,
    number_specs: dict[str, list[tuple]],
) -> Any:
    if field == "component":
        options = [
            "Farming",
            "Shared Processing",
            "Bioethanol",
            "Sugar",
            "Electricity Generation",
            "Bagasse",
            "Animal Feed",
        ]
        current = str(value or "Shared Processing")
        if current not in options:
            options.append(current)
        return st.selectbox(
            "Component", options, index=options.index(current), key=widget_key
        )
    if isinstance(value, bool) or field == "is_farm_capex":
        return st.checkbox(
            _field_label(field, number_specs), value=bool(value), key=widget_key
        )
    if field in {"item", "start_month"} or isinstance(value, str):
        return st.text_input(
            _field_label(field, number_specs), value=str(value or ""), key=widget_key
        )

    spec = _number_spec(field, number_specs)
    if field == "Year":
        return int(
            st.number_input(
                "Year", value=int(value), step=1, format="%d", key=widget_key
            )
        )
    if spec is not None:
        _, label, step, fmt, caster = spec
        return caster(
            st.number_input(
                label,
                value=caster(value),
                step=caster(step),
                format=fmt,
                key=widget_key,
            )
        )
    if isinstance(value, int):
        return int(
            st.number_input(
                _field_label(field, number_specs),
                value=int(value),
                step=1,
                format="%d",
                key=widget_key,
            )
        )
    return float(
        st.number_input(
            _field_label(field, number_specs),
            value=float(value or 0.0),
            step=1.0,
            key=widget_key,
        )
    )


def _validate_row(inputs, schedule_key: str, candidate: dict[str, Any]) -> list[str]:
    clean = {key: value for key, value in candidate.items() if key != ROW_ID}
    if schedule_key == _CAPEX_KEY:
        try:
            validator = getattr(CapexItem, "model_validate", None)
            if callable(validator):
                validator(clean)
            else:
                CapexItem.parse_obj(clean)
        except Exception as exc:
            return [str(exc)]
        return []

    config = SCHEDULE_DEFINITIONS[schedule_key]
    section = getattr(inputs, config["section"])
    section_type = type(section)
    values = input_values(section)
    values.update({field: clean[field] for field in config["fields"]})
    try:
        validator = getattr(section_type, "model_validate", None)
        if callable(validator):
            validator(values)
        else:
            section_type.parse_obj(values)
    except Exception as exc:
        return [str(exc)]
    year = int(clean["Year"])
    start = int(inputs.global_assumptions.start_year)
    end = int(inputs.global_assumptions.end_year)
    if year < start or year > end:
        return [f"Year must be within the projection horizon ({start}-{end})."]
    if schedule_key == "processing_routing":
        routed = sum(
            float(clean[field])
            for field in (
                "bagasse_to_electricity_share",
                "bagasse_to_animal_feed_share",
                "bagasse_to_sale_share",
            )
        )
        if abs(routed - 1.0) > 1e-9:
            return ["Bagasse routing shares must total 1.00 for every scheduled year."]
    return []


def _columns_for(inputs, schedule_key: str) -> list[str]:
    if schedule_key == _CAPEX_KEY:
        return [
            "item",
            "component",
            "amount",
            "start_month",
            "spend_months",
            "life_years",
            "is_farm_capex",
        ]
    return ["Year", *SCHEDULE_DEFINITIONS[schedule_key]["fields"]]


def _commit(st, inputs, schedule_key: str, rows: list[dict[str, Any]]) -> list[str]:
    payload = input_values(inputs)
    cleaned = strip_row_ids(rows)
    if schedule_key == _CAPEX_KEY:
        if not cleaned:
            return ["At least one CAPEX item is required."]
        payload["capex"] = {"items": cleaned}
    else:
        yearly = dict(payload.get("yearly_schedules", {}))
        yearly[schedule_key] = cleaned
        payload["yearly_schedules"] = yearly
    try:
        updated = input_from_payload(payload)
    except Exception as exc:
        return [str(exc)]
    errors = validate_driver_schedules(updated)
    if errors:
        return errors

    st.session_state["_sugarcane_refactored_inputs"] = updated
    st.session_state.pop("_sugarcane_refactored_result", None)
    st.session_state.pop("sugar_active_result", None)
    st.session_state.pop("sugarcane_refactored_report_cache", None)
    st.session_state["sugar_model_results_stale"] = True
    _state_dict(st, _SOURCES_KEY)[schedule_key] = _signature(rows)
    _store_draft(st, schedule_key, rows, dirty=False)
    return []


def render_schedule_workspace(
    st,
    inputs,
    *,
    number_specs: dict[str, list[tuple]],
) -> None:
    """Render row-level editing and propagation for every input schedule."""

    with st.expander("Schedule Edit Workspace", expanded=False):
        st.caption(
            "Select a schedule and row, stage edits with Save Row, then use Save All "
            "Changes to update the model inputs. Run Model remains the calculation step."
        )
        labels = {
            _CAPEX_KEY: "CAPEX",
            **{key: value["label"] for key, value in SCHEDULE_DEFINITIONS.items()},
        }
        schedule_by_label = {label: key for key, label in labels.items()}
        schedule_label = st.selectbox(
            "Schedule",
            list(schedule_by_label),
            key="sugarcane_schedule_workspace_table",
        )
        schedule_key = schedule_by_label[str(schedule_label)]
        draft = _draft(st, inputs, schedule_key)
        revisions = _state_dict(st, _REVISION_KEY)
        revision = int(revisions.get(schedule_key, 0))
        dirty = bool(_state_dict(st, _DIRTY_KEY).get(schedule_key))

        flash = _state_dict(st, _FLASH_KEY).pop(schedule_key, None)
        if flash:
            st.success(str(flash))
        if dirty:
            st.warning("This schedule has staged changes waiting for Save All Changes.")

        if not draft:
            if schedule_key == _CAPEX_KEY:
                st.error("CAPEX cannot be empty.")
                return
            if st.button(
                "Add first row",
                key=f"sugarcane_schedule_add_first_{schedule_key}_r{revision}",
            ):
                row = with_row_ids(
                    [default_schedule_row(inputs, schedule_key)], schedule_key
                )
                _store_draft(st, schedule_key, row, dirty=True)
                _state_dict(st, _SELECTED_KEY)[schedule_key] = row[0][ROW_ID]
                _set_flash(st, schedule_key, "A first schedule row was added to the draft.")
                st.rerun()
            return

        selected_state = _state_dict(st, _SELECTED_KEY)
        row_ids = [str(row[ROW_ID]) for row in draft]
        selected_id = str(selected_state.get(schedule_key, row_ids[0]))
        if selected_id not in row_ids:
            selected_id = row_ids[0]
        label_by_id = {
            str(row[ROW_ID]): row_label(
                row, position, capex=schedule_key == _CAPEX_KEY
            )
            for position, row in enumerate(draft, start=1)
        }
        row_id_by_label = {label: row_id for row_id, label in label_by_id.items()}
        selected_label = label_by_id[selected_id]
        selected_label = st.selectbox(
            "Select row to edit",
            list(row_id_by_label),
            index=list(row_id_by_label).index(selected_label),
            key=f"sugarcane_schedule_row_{schedule_key}_r{revision}",
        )
        selected_id = row_id_by_label[str(selected_label)]
        selected_state[schedule_key] = selected_id
        selected_row = next(row for row in draft if str(row[ROW_ID]) == selected_id)

        add_col, remove_col = st.columns(2)
        if add_col.button(
            "Add",
            key=f"sugarcane_schedule_add_{schedule_key}_r{revision}",
            use_container_width=True,
        ):
            try:
                if schedule_key == _CAPEX_KEY:
                    updated, new_id = add_capex_row(draft, selected_id)
                else:
                    updated, new_id = add_year_row(
                        draft,
                        selected_id,
                        start_year=inputs.global_assumptions.start_year,
                        end_year=inputs.global_assumptions.end_year,
                        prefix=schedule_key,
                    )
                _store_draft(st, schedule_key, updated, dirty=True)
                selected_state[schedule_key] = new_id
                _set_flash(st, schedule_key, "A new row was added to the draft.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if remove_col.button(
            "Remove Row",
            key=f"sugarcane_schedule_remove_{schedule_key}_r{revision}",
            use_container_width=True,
        ):
            if len(draft) == 1:
                st.error("At least one row is required in every schedule.")
            else:
                updated = remove_row(draft, selected_id)
                _store_draft(st, schedule_key, updated, dirty=True)
                if updated:
                    selected_state[schedule_key] = updated[min(len(updated) - 1, 0)][ROW_ID]
                _set_flash(st, schedule_key, "The selected row was removed from the draft.")
                st.rerun()

        columns = _columns_for(inputs, schedule_key)
        with st.form(f"sugarcane_schedule_row_form_{schedule_key}_r{revision}_{selected_id}"):
            st.markdown("**Edit selected row**")
            candidate = {ROW_ID: selected_id}
            editor_columns = st.columns(2)
            for position, field in enumerate(columns):
                with editor_columns[position % 2]:
                    candidate[field] = _edit_field(
                        st,
                        field,
                        selected_row.get(field),
                        widget_key=f"sugarcane_schedule_field_{schedule_key}_{selected_id}_{field}_r{revision}",
                        number_specs=number_specs,
                    )
            save_row = st.form_submit_button(
                "Save Row", type="primary", use_container_width=True
            )

        if save_row:
            errors = _validate_row(inputs, schedule_key, candidate)
            if schedule_key != _CAPEX_KEY:
                year = int(candidate["Year"])
                if any(
                    int(row.get("Year")) == year and str(row[ROW_ID]) != selected_id
                    for row in draft
                ):
                    errors.append(f"Year {year} already exists in this schedule.")
            if errors:
                for error in errors:
                    st.error(error)
            else:
                updated = update_row(draft, selected_id, candidate)
                if schedule_key != _CAPEX_KEY:
                    updated.sort(key=lambda row: int(row["Year"]))
                _store_draft(st, schedule_key, updated, dirty=True)
                _set_flash(
                    st,
                    schedule_key,
                    "Row saved to the draft. Use Save All Changes to update model inputs.",
                )
                st.rerun()

        st.markdown("**Yearly Increment and Propagate**")
        if schedule_key == _CAPEX_KEY:
            increment_fields = ["amount"]
        else:
            increment_fields = [
                field
                for field in SCHEDULE_DEFINITIONS[schedule_key]["fields"]
                if isinstance(selected_row.get(field), (int, float))
                and not isinstance(selected_row.get(field), bool)
            ]
        propagate_columns = st.columns([1.1, 1.8, 1.0])
        annual_percent = propagate_columns[0].number_input(
            "Yearly Increment %",
            min_value=-99.9,
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"sugarcane_schedule_increment_{schedule_key}_{selected_id}_r{revision}",
        )
        default_increment_fields = (
            ["amount"] if schedule_key == _CAPEX_KEY else _DEFAULT_INCREMENT_FIELDS[schedule_key]
        )
        increment_field_by_label = {
            _field_label(field, number_specs): field for field in increment_fields
        }
        selected_field_labels = propagate_columns[1].multiselect(
            "Fields to increment",
            list(increment_field_by_label),
            default=[
                _field_label(field, number_specs)
                for field in default_increment_fields
                if field in increment_fields
            ],
            key=f"sugarcane_schedule_increment_fields_{schedule_key}_{selected_id}_r{revision}",
        )
        selected_fields = [increment_field_by_label[label] for label in selected_field_labels]
        propagate_clicked = propagate_columns[2].button(
            "Propagate",
            key=f"sugarcane_schedule_propagate_{schedule_key}_{selected_id}_r{revision}",
            use_container_width=True,
        )
        if propagate_clicked:
            if not selected_fields:
                st.error("Select at least one numeric field to propagate.")
            else:
                try:
                    if schedule_key == _CAPEX_KEY:
                        updated = propagate_capex(
                            draft,
                            selected_id,
                            annual_rate=float(annual_percent) / 100.0,
                            end_year=inputs.global_assumptions.end_year,
                        )
                    else:
                        updated = propagate_yearly(
                            draft,
                            selected_id,
                            value_fields=selected_fields,
                            annual_rate=float(annual_percent) / 100.0,
                            end_year=inputs.global_assumptions.end_year,
                            prefix=schedule_key,
                        )
                    _store_draft(st, schedule_key, updated, dirty=True)
                    _set_flash(
                        st,
                        schedule_key,
                        "Yearly increment propagated through the remaining projection years.",
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown("**Current schedule draft**")
        st.dataframe(
            pd.DataFrame(strip_row_ids(draft)),
            use_container_width=True,
            hide_index=True,
            height=300,
        )
        save_all_col, discard_col = st.columns(2)
        if save_all_col.button(
            "Save All Changes",
            type="primary",
            disabled=not dirty,
            key=f"sugarcane_schedule_save_all_{schedule_key}_r{revision}",
            use_container_width=True,
        ):
            errors = _commit(st, inputs, schedule_key, draft)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                _set_flash(
                    st,
                    schedule_key,
                    f"Saved all changes to {labels[schedule_key]}. Run Model to recalculate.",
                )
                st.rerun()

        if discard_col.button(
            "Discard Unsaved Changes",
            disabled=not dirty,
            key=f"sugarcane_schedule_discard_{schedule_key}_r{revision}",
            use_container_width=True,
        ):
            source = _source_rows(inputs, schedule_key)
            _state_dict(st, _SOURCES_KEY)[schedule_key] = _signature(source)
            _store_draft(st, schedule_key, source, dirty=False)
            _set_flash(st, schedule_key, "Unsaved schedule changes were discarded.")
            st.rerun()
