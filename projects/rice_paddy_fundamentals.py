"""Publication-safe joins for contextual paddy fundamentals evidence.

This module deliberately keeps feature values separate from their audit trail.
Only records that were reviewed and demonstrably available by an example's
origin can enter a feature value.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


IST = ZoneInfo("Asia/Kolkata")
_BLOCKS = {"supply", "demand", "policy"}
_KINDS = {"observation", "state"}
_REVIEW_STATUSES = {"verified", "pending"}


def load_bundle(path: Path) -> dict:
    """Load and validate an evidence bundle without modifying its contents."""
    try:
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load evidence bundle {path}: {exc}") from exc
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict) -> None:
    """Validate the deliberately small, append-only evidence-bundle schema."""
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be an object")
    _require_exact_top_level(bundle)
    if bundle["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for key in ("series", "records", "documents"):
        if not isinstance(bundle[key], list):
            raise ValueError(f"{key} must be a list")

    series_by_id: dict[str, dict[str, Any]] = {}
    for series in bundle["series"]:
        _validate_series(series)
        series_id = series["id"]
        if series_id in series_by_id:
            raise ValueError(f"duplicate series id: {series_id}")
        series_by_id[series_id] = series
    _validate_generated_feature_columns(bundle["series"])

    documents_by_id: dict[str, dict[str, Any]] = {}
    for document in bundle["documents"]:
        _validate_document(document)
        if document["id"] in documents_by_id:
            raise ValueError(f"duplicate document id: {document['id']}")
        documents_by_id[document["id"]] = document

    record_ids: set[str] = set()
    for record in bundle["records"]:
        _validate_record(record, series_by_id)
        if record["id"] in record_ids:
            raise ValueError(f"duplicate record id: {record['id']}")
        if record["source_document_id"] not in documents_by_id:
            raise ValueError("record references an unknown source document")
        source_document = documents_by_id[record["source_document_id"]]
        if record["source_url"] != source_document["url"]:
            raise ValueError("record source_url must match its source document URL")
        _validate_document_timing(record, source_document)
        record_ids.add(record["id"])


def feature_columns(bundle: dict, blocks: tuple[str, ...]) -> list[str]:
    """Return declared feature columns in bundle series order for allowed blocks."""
    validate_bundle(bundle)
    invalid_blocks = set(blocks) - _BLOCKS
    if invalid_blocks:
        raise ValueError(f"unknown blocks: {sorted(invalid_blocks)}")
    columns: list[str] = []
    for series in bundle["series"]:
        if series["block"] not in blocks:
            continue
        base = _feature_base(series)
        columns.extend([base, f"{base}_missing", f"{base}_age_days"])
        if series["kind"] == "state":
            columns.append(f"{base}_known_target")
    return columns


def build_features(examples: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build as-of features and a separate per-case, per-series audit table.

    ``origin`` is a calendar date: its availability cutoff is the end of that
    day in Asia/Kolkata. ``target_month`` is normalized to that month's final
    calendar day for the conditional known-policy scenario.
    """
    validate_bundle(bundle)
    if not isinstance(examples, pd.DataFrame):
        raise ValueError("examples must be a pandas DataFrame")
    missing_columns = {"origin", "target_month"} - set(examples.columns)
    if missing_columns:
        raise ValueError(f"examples missing required columns: {sorted(missing_columns)}")

    normalized = [(_origin_cutoff(origin), _target_month_end(target)) for origin, target in
                  zip(examples["origin"], examples["target_month"])]
    series_by_id = {series["id"]: series for series in bundle["series"]}
    records_by_series: dict[str, list[dict[str, Any]]] = {series_id: [] for series_id in series_by_id}
    for record in bundle["records"]:
        records_by_series[record["series_id"]].append(record)

    output = pd.DataFrame(index=examples.index, columns=feature_columns(bundle, tuple(_BLOCKS)), dtype="float64")
    provenance_rows: list[dict[str, Any]] = []
    seen_provenance: set[tuple[pd.Timestamp, pd.Timestamp, str]] = set()
    selection_cache: dict[tuple[datetime, datetime, str], tuple[list[dict[str, Any]], list[dict[str, Any]] | None, float, int, float, str, float | None]] = {}

    for position, (origin_cutoff, target_end) in enumerate(normalized):
        for series in bundle["series"]:
            base = _feature_base(series)
            records = records_by_series[series["id"]]
            cache_key = (origin_cutoff, target_end, series["id"])
            cached = selection_cache.get(cache_key)
            if cached is None:
                if series["kind"] == "observation":
                    selection = _select_observation(records, origin_cutoff)
                    value, missing, age, status = _observation_value(selection, series, origin_cutoff)
                    known_target = None
                    known_selection = None
                else:
                    selection = _select_state(records, origin_cutoff, origin_cutoff)
                    known_selection = _select_state(records, origin_cutoff, target_end)
                    value, missing, age, status = _state_value(selection, origin_cutoff)
                    known_target = _selected_value(known_selection)
                cached = (selection, known_selection, value, missing, age, status, known_target)
                selection_cache[cache_key] = cached
            selection, known_selection, value, missing, age, status, known_target = cached

            output.iat[position, output.columns.get_loc(base)] = value
            output.iat[position, output.columns.get_loc(f"{base}_missing")] = missing
            output.iat[position, output.columns.get_loc(f"{base}_age_days")] = age
            if series["kind"] == "state":
                output.iat[position, output.columns.get_loc(f"{base}_known_target")] = known_target

            origin_day = pd.Timestamp(origin_cutoff.date())
            target_day = pd.Timestamp(target_end.date())
            key = (origin_day, target_day, series["id"])
            if key not in seen_provenance:
                provenance_rows.append(_provenance_row(
                    origin_day, target_day, series, selection, known_selection, status, value
                ))
                seen_provenance.add(key)

    # Feature flags are categorical integers even when all values are missing.
    for series in bundle["series"]:
        missing_col = f"{_feature_base(series)}_missing"
        output[missing_col] = output[missing_col].astype("int64")
    provenance = pd.DataFrame(provenance_rows, columns=_PROVENANCE_COLUMNS)
    return output, provenance


_PROVENANCE_COLUMNS = [
    "origin", "target_month", "series_id", "block", "record_ids", "source_url",
    "source_urls", "available_at", "period_start", "period_end", "status", "feature_value",
    "known_target_record_ids", "known_target_source_url", "known_target_source_urls",
    "known_target_available_at", "known_target_period_start", "known_target_period_end",
    "known_target_status", "known_target_value",
]


def _require_exact_top_level(bundle: dict) -> None:
    required = {"schema_version", "series", "records", "documents"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"bundle missing required fields: {sorted(missing)}")


def _validate_series(series: Any) -> None:
    if not isinstance(series, dict):
        raise ValueError("series must be objects")
    required = {"id", "block", "unit", "geography", "commodity", "description", "kind", "max_age_days"}
    _require_fields(series, required, "series")
    if not _is_lower_identifier(series["id"]):
        raise ValueError("series id must be a lowercase identifier")
    if series["block"] not in _BLOCKS or series["kind"] not in _KINDS:
        raise ValueError("series has invalid block or kind")
    for field in ("unit", "geography", "commodity", "description"):
        if not isinstance(series[field], str) or not series[field].strip():
            raise ValueError(f"series {field} must be a nonempty string")
    max_age = series["max_age_days"]
    if series["kind"] == "observation":
        if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age <= 0:
            raise ValueError("observation max_age_days must be a positive integer")
    elif max_age is not None:
        raise ValueError("state max_age_days must be null")


def _validate_generated_feature_columns(series_list: list[dict[str, Any]]) -> None:
    """Reject valid-looking IDs that would overwrite another declared feature."""
    owners: dict[str, str] = {}
    for series in series_list:
        base = _feature_base(series)
        names = [base, f"{base}_missing", f"{base}_age_days"]
        if series["kind"] == "state":
            names.append(f"{base}_known_target")
        for name in names:
            previous = owners.get(name)
            if previous is not None:
                raise ValueError(
                    f"generated feature column collision: {name} for series {previous} and {series['id']}"
                )
            owners[name] = series["id"]


def _validate_record(record: Any, series_by_id: dict[str, dict[str, Any]]) -> None:
    if not isinstance(record, dict):
        raise ValueError("records must be objects")
    required = {"id", "series_id", "period_start", "period_end", "value", "available_at", "review_status", "source_document_id", "source_url"}
    _require_fields(record, required, "record")
    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("record id must be a nonempty string")
    if record["series_id"] not in series_by_id:
        raise ValueError("record references an unknown series")
    if record["review_status"] not in _REVIEW_STATUSES:
        raise ValueError("record review_status must be verified or pending")
    if not isinstance(record["source_document_id"], str) or not record["source_document_id"].strip():
        raise ValueError("record source_document_id must be a nonempty string")
    if not isinstance(record["source_url"], str) or not record["source_url"].strip():
        raise ValueError("record source_url must be a nonempty string")
    start, end = _parse_date(record["period_start"], "period_start"), _parse_date(record["period_end"], "period_end")
    if start > end:
        raise ValueError("record period_start must not be after period_end")
    if isinstance(record["value"], bool) or not isinstance(record["value"], (int, float)) or not math.isfinite(record["value"]):
        raise ValueError("record value must be a finite number")
    _parse_aware_optional(record["available_at"], "available_at")
    if series_by_id[record["series_id"]]["kind"] == "state":
        if "effective_at" not in record:
            raise ValueError("state record missing effective_at")
        _parse_aware_optional(record["effective_at"], "effective_at")


def _validate_document(document: Any) -> None:
    if not isinstance(document, dict):
        raise ValueError("documents must be objects")
    required = {"id", "url", "published_at", "publication_evidence", "access_status", "notes"}
    _require_fields(document, required, "document")
    for field in ("id", "url", "publication_evidence", "access_status", "notes"):
        if not isinstance(document[field], str) or not document[field].strip():
            raise ValueError(f"document {field} must be a nonempty string")
    _parse_document_published_at(document["published_at"])
    local_path, sha256 = document.get("local_path"), document.get("sha256")
    if (local_path is None) != (sha256 is None):
        raise ValueError("archived documents require both local_path and sha256")
    if local_path is not None and (not isinstance(local_path, str) or not local_path.strip() or not isinstance(sha256, str) or not sha256.strip()):
        raise ValueError("archived document path and sha256 must be nonempty strings")


def _require_fields(value: dict[str, Any], required: set[str], name: str) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{name} missing required fields: {sorted(missing)}")


def _is_lower_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.lower() and all(c.islower() or c.isdigit() or c == "_" for c in value)


def _parse_date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return parsed


def _parse_aware_optional(value: Any, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp or null") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return parsed


def _parse_document_published_at(value: Any) -> datetime | date | None:
    """Allow a date-only document date, but never treat it as a known time."""
    if value is None:
        return None
    if isinstance(value, str) and "T" not in value and " " not in value:
        return _parse_date(value, "published_at")
    return _parse_aware_optional(value, "published_at")


def _validate_document_timing(record: dict[str, Any], document: dict[str, Any]) -> None:
    """Prevent a verified record from predating the evidence that publishes it."""
    if record["review_status"] != "verified":
        return
    available_at = _parse_aware_optional(record["available_at"], "available_at")
    published_at = _parse_document_published_at(document["published_at"])
    if available_at is None or published_at is None:
        return
    if isinstance(published_at, datetime):
        earliest_available = published_at
    else:
        earliest_available = datetime.combine(published_at + timedelta(days=1), time.min, tzinfo=IST)
    if available_at < earliest_available:
        raise ValueError("verified record available_at predates its document publication evidence")


def _origin_cutoff(value: Any) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("example origin must be a valid calendar date") from exc
    if pd.isna(timestamp):
        raise ValueError("example origin must be a valid calendar date")
    if timestamp.tzinfo is not None:
        local_date = timestamp.tz_convert(IST).date()
    else:
        local_date = timestamp.date()
    return datetime.combine(local_date, time.max, tzinfo=IST)


def _target_month_end(value: Any) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("example target_month must be a valid calendar date") from exc
    if pd.isna(timestamp):
        raise ValueError("example target_month must be a valid calendar date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(IST).tz_localize(None)
    last_day = (timestamp + pd.offsets.MonthEnd(0)).date()
    return datetime.combine(last_day, time.max, tzinfo=IST)


def _feature_base(series: dict[str, Any]) -> str:
    return f"exog_{series['block']}_{series['id']}"


def _eligible_records(records: list[dict[str, Any]], origin_cutoff: datetime) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for record in records:
        available_at = _parse_aware_optional(record["available_at"], "available_at")
        if record["review_status"] == "verified" and available_at is not None and available_at <= origin_cutoff:
            eligible.append(record)
    return eligible


def _select_observation(records: list[dict[str, Any]], origin_cutoff: datetime) -> list[dict[str, Any]]:
    eligible = [record for record in _eligible_records(records, origin_cutoff)
                if _parse_date(record["period_end"], "period_end") <= origin_cutoff.date()]
    if not eligible:
        return []
    latest_end = max(_parse_date(record["period_end"], "period_end") for record in eligible)
    latest_end_records = [record for record in eligible if _parse_date(record["period_end"], "period_end") == latest_end]
    latest_available = max(_parse_aware_optional(record["available_at"], "available_at") for record in latest_end_records)
    selected = [record for record in latest_end_records if _parse_aware_optional(record["available_at"], "available_at") == latest_available]
    _reject_conflicting_tie(selected, "observation")
    return sorted(selected, key=lambda record: record["id"])


def _select_state(records: list[dict[str, Any]], origin_cutoff: datetime, effective_cutoff: datetime) -> list[dict[str, Any]]:
    eligible = []
    for record in _eligible_records(records, origin_cutoff):
        effective_at = _parse_aware_optional(record["effective_at"], "effective_at")
        if effective_at is not None and effective_at <= effective_cutoff:
            eligible.append(record)
    if not eligible:
        return []
    latest_effective = max(_parse_aware_optional(record["effective_at"], "effective_at") for record in eligible)
    effective_records = [record for record in eligible if _parse_aware_optional(record["effective_at"], "effective_at") == latest_effective]
    latest_available = max(_parse_aware_optional(record["available_at"], "available_at") for record in effective_records)
    selected = [record for record in effective_records if _parse_aware_optional(record["available_at"], "available_at") == latest_available]
    _reject_conflicting_tie(selected, "state")
    return sorted(selected, key=lambda record: record["id"])


def _reject_conflicting_tie(records: list[dict[str, Any]], label: str) -> None:
    if len({record["value"] for record in records}) > 1:
        raise ValueError(f"ambiguous conflicting {label} revisions with equal timing")


def _observation_value(selection: list[dict[str, Any]], series: dict[str, Any], origin_cutoff: datetime) -> tuple[float, int, float, str]:
    if not selection:
        return math.nan, 1, math.nan, "missing"
    reference_end = _parse_date(selection[0]["period_end"], "period_end")
    age_days = (origin_cutoff.date() - reference_end).days
    if age_days > series["max_age_days"]:
        return math.nan, 1, age_days, "stale"
    return _selected_value(selection), 0, age_days, "selected"


def _state_value(selection: list[dict[str, Any]], origin_cutoff: datetime) -> tuple[float, int, float, str]:
    """Return current state and its duration at origin; states do not go stale."""
    if not selection:
        return math.nan, 1, math.nan, "missing"
    effective_at = _parse_aware_optional(selection[0]["effective_at"], "effective_at")
    age_days = (origin_cutoff.date() - effective_at.astimezone(IST).date()).days
    return _selected_value(selection), 0, age_days, "selected"


def _selected_value(selection: list[dict[str, Any]] | None) -> float:
    if not selection:
        return math.nan
    return selection[0]["value"]


def _provenance_row(origin: pd.Timestamp, target_month: pd.Timestamp, series: dict[str, Any], selection: list[dict[str, Any]], known_selection: list[dict[str, Any]] | None, status: str, value: float) -> dict[str, Any]:
    primary = selection[0] if selection else None
    known_primary = known_selection[0] if known_selection else None
    return {
        "origin": origin,
        "target_month": target_month,
        "series_id": series["id"],
        "block": series["block"],
        "record_ids": [record["id"] for record in selection],
        "source_url": primary["source_url"] if primary else None,
        "source_urls": [record["source_url"] for record in selection],
        "available_at": primary["available_at"] if primary else None,
        "period_start": primary["period_start"] if primary else None,
        "period_end": primary["period_end"] if primary else None,
        "status": status,
        "feature_value": value,
        "known_target_record_ids": [record["id"] for record in known_selection or []],
        "known_target_source_url": known_primary["source_url"] if known_primary else None,
        "known_target_source_urls": [record["source_url"] for record in known_selection or []],
        "known_target_available_at": known_primary["available_at"] if known_primary else None,
        "known_target_period_start": known_primary["period_start"] if known_primary else None,
        "known_target_period_end": known_primary["period_end"] if known_primary else None,
        "known_target_status": "selected" if known_selection else "missing",
        "known_target_value": _selected_value(known_selection),
    }
