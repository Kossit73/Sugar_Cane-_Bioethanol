"""Pure row operations used by the Sugar Cane schedule edit workspace."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


ROW_ID = "_row_id"


def with_row_ids(rows: Iterable[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Attach deterministic UI-only identifiers while preserving existing IDs."""

    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for position, source in enumerate(rows, start=1):
        row = deepcopy(dict(source))
        candidate = str(row.get(ROW_ID) or f"{prefix}-{position}")
        if candidate in used:
            candidate = f"{prefix}-{position}"
        while candidate in used:
            candidate = f"{candidate}-copy"
        row[ROW_ID] = candidate
        used.add(candidate)
        output.append(row)
    return output


def strip_row_ids(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: deepcopy(value) for key, value in row.items() if key != ROW_ID} for row in rows]


def row_label(row: dict[str, Any], position: int, *, capex: bool = False) -> str:
    if capex:
        item = str(row.get("item") or "Unnamed CAPEX item")
        component = str(row.get("component") or "Unassigned")
        return f"{position}. {item} — {component}"
    year = row.get("Year", "Year not set")
    return f"{position}. {year}"


def update_row(
    rows: Iterable[dict[str, Any]], row_id: str, values: dict[str, Any]
) -> list[dict[str, Any]]:
    output = with_row_ids(rows, "row")
    for row in output:
        if row[ROW_ID] == row_id:
            row.update(deepcopy(values))
            row[ROW_ID] = row_id
            return output
    raise KeyError(f"Row {row_id!r} was not found.")


def remove_row(rows: Iterable[dict[str, Any]], row_id: str) -> list[dict[str, Any]]:
    source = list(rows)
    output = [deepcopy(row) for row in source if str(row.get(ROW_ID)) != str(row_id)]
    if len(output) == len(source):
        raise KeyError(f"Row {row_id!r} was not found.")
    return output


def next_available_year(rows: Iterable[dict[str, Any]], start_year: int, end_year: int) -> int | None:
    used = {int(row["Year"]) for row in rows if row.get("Year") is not None}
    return next((year for year in range(int(start_year), int(end_year) + 1) if year not in used), None)


def add_year_row(
    rows: Iterable[dict[str, Any]],
    row_id: str,
    *,
    start_year: int,
    end_year: int,
    prefix: str,
) -> tuple[list[dict[str, Any]], str]:
    output = with_row_ids(rows, prefix)
    source = next((row for row in output if row[ROW_ID] == row_id), None)
    if source is None:
        raise KeyError(f"Row {row_id!r} was not found.")
    year = next_available_year(output, start_year, end_year)
    if year is None:
        raise ValueError("Every projection year already has a row.")
    new_row = deepcopy(source)
    new_id = f"{prefix}-{len(output) + 1}"
    existing = {row[ROW_ID] for row in output}
    while new_id in existing:
        new_id = f"{new_id}-copy"
    new_row[ROW_ID] = new_id
    new_row["Year"] = year
    output.append(new_row)
    output.sort(key=lambda row: int(row["Year"]))
    return output, new_id


def add_capex_row(
    rows: Iterable[dict[str, Any]], row_id: str, *, prefix: str = "capex"
) -> tuple[list[dict[str, Any]], str]:
    output = with_row_ids(rows, prefix)
    source = next((row for row in output if row[ROW_ID] == row_id), None)
    if source is None:
        raise KeyError(f"Row {row_id!r} was not found.")
    new_row = deepcopy(source)
    new_id = f"{prefix}-{len(output) + 1}"
    existing = {row[ROW_ID] for row in output}
    while new_id in existing:
        new_id = f"{new_id}-copy"
    new_row[ROW_ID] = new_id
    new_row["item"] = f"Copy of {source.get('item') or 'CAPEX item'}"
    output.append(new_row)
    return output, new_id


def propagate_yearly(
    rows: Iterable[dict[str, Any]],
    row_id: str,
    *,
    value_fields: Iterable[str],
    annual_rate: float,
    end_year: int,
    prefix: str,
) -> list[dict[str, Any]]:
    """Compound selected fields from the anchor row through remaining years."""

    output = with_row_ids(rows, prefix)
    anchor = next((row for row in output if row[ROW_ID] == row_id), None)
    if anchor is None:
        raise KeyError(f"Row {row_id!r} was not found.")
    base_year = int(anchor["Year"])
    by_year = {int(row["Year"]): row for row in output}
    existing_ids = {row[ROW_ID] for row in output}
    for year in range(base_year + 1, int(end_year) + 1):
        target = by_year.get(year)
        if target is None:
            target = deepcopy(anchor)
            new_id = f"{prefix}-{len(output) + 1}"
            while new_id in existing_ids:
                new_id = f"{new_id}-copy"
            target[ROW_ID] = new_id
            target["Year"] = year
            output.append(target)
            by_year[year] = target
            existing_ids.add(new_id)
        exponent = year - base_year
        for field in value_fields:
            value = float(anchor[field]) * (1.0 + float(annual_rate)) ** exponent
            if isinstance(anchor[field], int) and not isinstance(anchor[field], bool):
                target[field] = int(round(value))
            else:
                target[field] = value
    output.sort(key=lambda row: int(row["Year"]))
    return output


def propagate_capex(
    rows: Iterable[dict[str, Any]],
    row_id: str,
    *,
    annual_rate: float,
    end_year: int,
    prefix: str = "capex",
) -> list[dict[str, Any]]:
    """Repeat a CAPEX item annually with a compounded amount."""

    output = with_row_ids(rows, prefix)
    anchor = next((row for row in output if row[ROW_ID] == row_id), None)
    if anchor is None:
        raise KeyError(f"Row {row_id!r} was not found.")
    start_text = str(anchor.get("start_month") or "")
    if len(start_text) < 7:
        raise ValueError("CAPEX start_month must use YYYY-MM before propagation.")
    base_year = int(start_text[:4])
    month = start_text[5:7]
    for year in range(base_year + 1, int(end_year) + 1):
        new_row = deepcopy(anchor)
        new_id = f"{prefix}-{len(output) + 1}"
        existing = {row[ROW_ID] for row in output}
        while new_id in existing:
            new_id = f"{new_id}-copy"
        new_row[ROW_ID] = new_id
        new_row["item"] = f"{anchor.get('item') or 'CAPEX item'} ({year})"
        new_row["start_month"] = f"{year:04d}-{month}"
        new_row["amount"] = float(anchor["amount"]) * (1.0 + float(annual_rate)) ** (year - base_year)
        output.append(new_row)
    return output
