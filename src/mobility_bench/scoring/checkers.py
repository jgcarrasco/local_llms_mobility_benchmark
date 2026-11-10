"""Deterministic task checkers."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdflib.namespace import Namespace
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.term import URIRef, Variable
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mobility_bench.runtime.sandbox import sanitized_environment
from mobility_bench.scoring.base import CheckerResult


def _fail(category: str, errors: list[str], partial: float = 0.0, **metadata: Any) -> CheckerResult:
    return CheckerResult(
        success=False,
        partial_score=partial,
        checker_errors=errors,
        failure_category=category,
        metadata=metadata,
    )


def _ok(**metadata: Any) -> CheckerResult:
    return CheckerResult(success=True, partial_score=1.0, metadata=metadata)


def _read_csv(path: Path) -> tuple[pd.DataFrame | None, CheckerResult | None]:
    if not path.exists():
        return None, _fail("schema_error", [f"missing file: {path}"])
    # A field containing an unquoted comma shifts every following column, and
    # pandas then reports values from the wrong column. Detect the shape before
    # parsing so the failure is attributed to formatting, not to the data source.
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            field_counts = [
                len(row)
                for row in csv.reader(handle)
                if row and any(cell.strip() for cell in row)
            ]
    except OSError as exc:
        return None, _fail("format_error", [f"failed to read CSV {path}: {exc}"])
    if field_counts and len(set(field_counts)) > 1:
        header_fields = field_counts[0]
        observed = ", ".join(
            f"{fields} fields on {rows} row(s)" for fields, rows in sorted(Counter(field_counts).items())
        )
        return None, _fail(
            "format_error",
            [
                f"{path.name} has rows that do not match the {header_fields}-column header "
                f"({observed}); quote fields that contain commas, quotes, or newlines"
            ],
        )
    try:
        return pd.read_csv(path), None
    except Exception as exc:  # pragma: no cover - defensive classifier
        return None, _fail("format_error", [f"failed to read CSV {path}: {exc}"])


def _required_columns(df: pd.DataFrame, columns: set[str]) -> list[str]:
    return sorted(columns.difference(df.columns))


def _load_json(path: Path) -> tuple[Any | None, CheckerResult | None]:
    if not path.exists():
        return None, _fail("schema_error", [f"missing file: {path}"])
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, _fail("format_error", [f"invalid JSON {path}: {exc}"])


def _valid_gtfs_time(value: object) -> bool:
    match = re.fullmatch(r"(\d{1,2}):([0-5]\d):([0-5]\d)", str(value))
    return bool(match)


def _time_to_seconds(value: object) -> int:
    hour, minute, second = [int(part) for part in str(value).split(":")]
    return hour * 3600 + minute * 60 + second


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _id_key(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["city_id"].astype(str), df["year"].astype(str), strict=False))


def _wikidata_expected_records(root: Path) -> dict[str, tuple[float, float]]:
    """Return the latitude and longitude of every valid input record, keyed by qid."""
    source = json.loads((root / "inputs" / "wikidata" / "wikidata_pois_sample.json").read_text(encoding="utf-8"))
    records: dict[str, tuple[float, float]] = {}
    for row in source["results"]["bindings"]:
        qid = str(row.get("qid") or "").strip()
        coord = row.get("coord")
        if isinstance(coord, dict):
            coord = coord.get("value", "")
        match = re.fullmatch(r"Point\((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)", str(coord or ""))
        if not qid or not match:
            continue
        lon, lat = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            records[qid] = (lat, lon)
    return records


def _t05_expected_cleaning(
    root: Path,
) -> tuple[
    set[str],
    dict[str, tuple[Any, float, float]],
    Counter[tuple[str, str, str, str, str]],
    dict[str, int],
]:
    tables = _gtfs_tables(root)
    raw_stops = tables["stops"].copy()
    duplicate_stops_removed = int(raw_stops.duplicated("stop_id").sum())
    unique_stops = raw_stops.drop_duplicates("stop_id", keep="first").copy()
    lat = pd.to_numeric(unique_stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(unique_stops["stop_lon"], errors="coerce")
    valid_coordinates = lat.between(-90, 90) & lon.between(-180, 180)
    invalid_coordinates_removed = int((~valid_coordinates).sum())
    clean_stops = unique_stops.loc[valid_coordinates].copy()
    missing_stop_names_filled = int(clean_stops["stop_name"].fillna("").astype(str).str.strip().eq("").sum())
    clean_stop_ids = set(clean_stops["stop_id"].astype(str))
    expected_stop_times = tables["stop_times"][tables["stop_times"]["stop_id"].astype(str).isin(clean_stop_ids)].copy()
    expected_time_rows = Counter(
        (
            str(row.trip_id),
            str(row.arrival_time),
            str(row.departure_time),
            str(row.stop_id),
            str(row.stop_sequence),
        )
        for row in expected_stop_times.itertuples(index=False)
    )
    expected_report = {
        "invalid_coordinates_removed": invalid_coordinates_removed,
        "duplicate_stops_removed": duplicate_stops_removed,
        "missing_stop_names_filled": missing_stop_names_filled,
        "n_clean_stops": len(clean_stops),
        "n_clean_stop_times": len(expected_stop_times),
    }
    expected_attributes = {
        str(row.stop_id): (row.stop_name, float(str(row.stop_lat)), float(str(row.stop_lon)))
        for row in clean_stops.itertuples(index=False)
    }
    return clean_stop_ids, expected_attributes, expected_time_rows, expected_report


def check_t01(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t01_selected_indicators.csv")
    if error:
        return error
    descriptions, error = _read_csv(root / "inputs" / "eurostat" / "indicator_descriptions.csv")
    if error:
        return error
    assert output is not None and descriptions is not None
    missing = _required_columns(
        output,
        {"indicator_id", "indicator_name", "theme", "justification", "expected_direction"},
    )
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    if not 10 <= len(output) <= 15:
        return _fail("logic_error", ["selected indicator count must be between 10 and 15"])
    valid_ids = set(descriptions["indicator_id"])
    if not set(output["indicator_id"]).issubset(valid_ids):
        return _fail("wrong_data_source", ["selected indicators not present in descriptions"])
    if output["indicator_id"].astype(str).duplicated().any():
        return _fail("logic_error", ["duplicate indicator_id values"])
    required_themes = {"mobility", "population_density", "environment", "living_conditions"}
    if not required_themes.issubset(set(output["theme"])):
        return _fail("logic_error", ["missing required theme coverage"])
    if output["justification"].fillna("").astype(str).str.strip().eq("").any():
        return _fail("logic_error", ["empty justifications found"])
    allowed = {"higher_is_better", "lower_is_better", "contextual"}
    if not set(output["expected_direction"]).issubset(allowed):
        return _fail("schema_error", ["invalid expected_direction value"])
    return _ok(n_selected=len(output))


T02_SCRIPT_TIMEOUT_SECONDS = 60


def check_t02(root: Path) -> CheckerResult:
    script = root / "src" / "t02_load_eurostat.py"
    if not script.exists():
        return _fail("schema_error", ["missing script: src/t02_load_eurostat.py"])
    with tempfile.TemporaryDirectory(prefix="sophia_t02_check_") as temp_dir:
        temp_root = Path(temp_dir) / "workspace"
        shutil.copytree(root, temp_root)
        (temp_root / "outputs" / "t02_eurostat_raw.csv").unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                [sys.executable, "src/t02_load_eurostat.py"],
                cwd=temp_root,
                text=True,
                capture_output=True,
                timeout=T02_SCRIPT_TIMEOUT_SECONDS,
                env=sanitized_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _fail("runtime_error", ["loader script timed out"])
        if completed.returncode != 0:
            return _fail(
                "runtime_error",
                [f"loader script failed with exit code {completed.returncode}: {completed.stderr[-500:]}"],
            )
        output, error = _read_csv(temp_root / "outputs" / "t02_eurostat_raw.csv")
        if error:
            return error
        selected, error = _read_csv(temp_root / "outputs" / "t01_selected_indicators.csv")
        if error:
            return error
    assert output is not None and selected is not None
    required = {"city_id", "city_name", "year", "indicator_id", "indicator_name", "value", "unit"}
    missing = _required_columns(output, required)
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    if output.empty:
        return _fail("schema_error", ["output is empty"])
    if not set(output["indicator_id"]).issubset(set(selected["indicator_id"])):
        return _fail("logic_error", ["output includes indicators not selected in T01"])
    if pd.to_numeric(output["value"], errors="coerce").notna().sum() == 0:
        return _fail("schema_error", ["value column is not numeric"])
    if output.duplicated(["city_id", "year", "indicator_id"]).any():
        return _fail("logic_error", ["duplicated city_id/year/indicator_id rows"])
    return _ok(n_rows=len(output))


MOBILITY_TERMS = ("mobility", "transport", "gtfs", "traffic", "rail", "bike")


def _catalog_records(root: Path) -> dict[str, dict[str, Any]]:
    catalog = json.loads(
        (root / "inputs" / "catalog" / "dataeuropa_mobility_catalog_sample.json").read_text(encoding="utf-8")
    )
    return {str(record["id"]): record for record in catalog["datasets"]}


def _normalized(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    # Trailing punctuation is not a fidelity failure; a missing final period is
    # a formatting difference, not an altered record.
    return " ".join(str(value).split()).lower().rstrip(" .;,")


def _record_distribution_url(record: dict[str, Any]) -> str:
    distributions = record.get("distributions") or [{}]
    return str(distributions[0].get("downloadURL", ""))


def _record_is_mobility(record: dict[str, Any]) -> bool:
    # A dataset about mobility identifies itself in its title or keywords. A
    # catalogue or portal that merely lists mobility among many themes does not,
    # even though the word may appear in its long description.
    identity = f"{record.get('title', '')} {' '.join(record.get('keywords') or [])}".lower()
    return any(term in identity for term in MOBILITY_TERMS)


def check_t03(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t03_mobility_catalog.csv")
    if error:
        return error
    assert output is not None
    required = {"dataset_id", "title", "description", "publisher", "country", "formats", "license", "distribution_url", "keywords"}
    missing = _required_columns(output, required)
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    records = _catalog_records(root)
    if not set(output["dataset_id"].astype(str)).issubset(records):
        return _fail("wrong_data_source", ["dataset_id values must come from the input catalog"])
    # The row must reproduce the catalog record rather than relabel it.
    for row in output.itertuples(index=False):
        record = records[str(row.dataset_id)]
        for field, expected in (
            ("title", record.get("title", "")),
            ("description", record.get("description", "")),
            ("distribution_url", _record_distribution_url(record)),
        ):
            if _normalized(getattr(row, field)) != _normalized(expected):
                return _fail(
                    "logic_error",
                    [f"row {row.dataset_id} does not match the catalog record for {field}"],
                )
    non_mobility = [dataset_id for dataset_id in output["dataset_id"].astype(str) if not _record_is_mobility(records[dataset_id])]
    if non_mobility:
        return _fail("logic_error", [f"records are not mobility datasets: {non_mobility}"])
    if not any("gtfs" in _normalized(records[dataset_id].get("title", "")) for dataset_id in output["dataset_id"].astype(str)):
        return _fail("logic_error", ["GTFS-like dataset missing"])
    if output["distribution_url"].fillna("").astype(str).str.strip().eq("").all():
        return _fail("schema_error", ["distribution_url not extracted"])
    return _ok(n_rows=len(output))


def _gtfs_tables(root: Path) -> dict[str, pd.DataFrame]:
    path = root / "inputs" / "gtfs" / "sample_city_gtfs.zip"
    with zipfile.ZipFile(path) as zip_file:
        return {
            name: pd.read_csv(zip_file.open(f"{name}.txt"))
            for name in ("stops", "routes", "trips", "stop_times")
        }


def check_t04(root: Path) -> CheckerResult:
    output, error = _load_json(root / "outputs" / "t04_gtfs_summary.json")
    if error:
        return error
    try:
        tables = _gtfs_tables(root)
    except Exception as exc:
        return _fail("wrong_data_source", [f"failed to read GTFS zip: {exc}"])
    required_fields = {
        "has_stops",
        "has_routes",
        "has_trips",
        "has_stop_times",
        "n_stops",
        "n_routes",
        "n_trips",
        "n_stop_times",
    }
    missing = sorted(required_fields.difference(output))
    if missing:
        return _fail("schema_error", [f"missing JSON fields: {missing}"])
    expected_counts = {
        "n_stops": len(tables["stops"]),
        "n_routes": len(tables["routes"]),
        "n_trips": len(tables["trips"]),
        "n_stop_times": len(tables["stop_times"]),
    }
    mismatched = [key for key, value in expected_counts.items() if output.get(key) != value]
    if mismatched:
        return _fail("logic_error", [f"count mismatch: {mismatched}"])
    if not all(output.get(key) is True for key in ("has_stops", "has_routes", "has_trips", "has_stop_times")):
        return _fail("logic_error", ["mandatory file detection is incorrect"])
    return _ok(**expected_counts)


def check_t05(root: Path) -> CheckerResult:
    report, error = _load_json(root / "outputs" / "t05_gtfs_cleaning_report.json")
    if error:
        return error
    stops, error = _read_csv(root / "outputs" / "t05_clean_stops.csv")
    if error:
        return error
    stop_times, error = _read_csv(root / "outputs" / "t05_clean_stop_times.csv")
    if error:
        return error
    assert stops is not None and stop_times is not None
    missing_stops = _required_columns(stops, {"stop_id", "stop_name", "stop_lat", "stop_lon"})
    missing_times = _required_columns(stop_times, {"trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"})
    if missing_stops or missing_times:
        return _fail("schema_error", [f"missing columns: stops={missing_stops}, stop_times={missing_times}"])
    if not isinstance(report, dict):
        return _fail("schema_error", ["cleaning report must be a JSON object"])
    expected_stop_ids, expected_attributes, expected_time_rows, expected_report = _t05_expected_cleaning(root)
    missing_report_fields = sorted(set(expected_report).difference(report))
    if missing_report_fields:
        return _fail("schema_error", [f"cleaning report missing fields: {missing_report_fields}"])
    mismatched_report = [
        field
        for field, expected_value in expected_report.items()
        if _safe_int(report.get(field)) != int(expected_value)
    ]
    if mismatched_report:
        return _fail("logic_error", [f"cleaning report count mismatch: {mismatched_report}"])
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    if not lat.between(-90, 90).all() or not lon.between(-180, 180).all():
        return _fail("domain_error", ["invalid coordinates remain in clean stops"])
    if stops["stop_id"].duplicated().any():
        return _fail("logic_error", ["duplicated stop_id rows remain"])
    if set(stops["stop_id"].astype(str)) != expected_stop_ids:
        return _fail("logic_error", ["clean stops do not match expected valid unique stop_ids"])
    mismatched_attributes = []
    for row in stops.itertuples(index=False):
        expected_name, expected_lat, expected_lon = expected_attributes[str(row.stop_id)]
        if isinstance(expected_name, str) and expected_name.strip():
            if str(row.stop_name).strip() != expected_name.strip():
                mismatched_attributes.append(str(row.stop_id))
                continue
        if abs(float(row.stop_lat) - expected_lat) > 1e-6 or abs(float(row.stop_lon) - expected_lon) > 1e-6:
            mismatched_attributes.append(str(row.stop_id))
    if mismatched_attributes:
        return _fail(
            "logic_error",
            [
                "clean stops must keep the first occurrence of each stop_id; "
                f"attributes differ for: {mismatched_attributes[:5]}"
            ],
        )
    if stops["stop_name"].fillna("").astype(str).str.strip().eq("").any():
        return _fail("logic_error", ["missing stop names remain"])
    if not stop_times["departure_time"].apply(_valid_gtfs_time).all():
        return _fail("domain_error", ["invalid GTFS time format"])
    if not any(_time_to_seconds(value) >= 24 * 3600 for value in stop_times["departure_time"]):
        return _fail("domain_error", ["extended service-day times were not preserved"])
    actual_time_rows = Counter(
        (
            str(row.trip_id),
            str(row.arrival_time),
            str(row.departure_time),
            str(row.stop_id),
            str(row.stop_sequence),
        )
        for row in stop_times.itertuples(index=False)
    )
    if actual_time_rows != expected_time_rows:
        return _fail("logic_error", ["clean stop_times rows do not match expected rows for cleaned stops"])
    return _ok(report=report)


def check_t06(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t06_stop_service_intensity.csv")
    if error:
        return error
    clean_times, error = _read_csv(root / "outputs" / "t05_clean_stop_times.csv")
    if error:
        return error
    clean_stops, error = _read_csv(root / "outputs" / "t05_clean_stops.csv")
    if error:
        return error
    assert output is not None and clean_times is not None and clean_stops is not None
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon", "n_departures", "n_routes", "first_departure", "last_departure"}
    missing = _required_columns(output, required)
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    if output["stop_id"].duplicated().any():
        return _fail("logic_error", ["expected one row per stop_id"])
    expected_stops = set(clean_stops["stop_id"].astype(str))
    output_stops = set(output["stop_id"].astype(str))
    if output_stops != expected_stops:
        return _fail("logic_error", ["output must contain exactly one row per cleaned stop"])
    stop_attributes = output.merge(
        clean_stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]], on="stop_id", suffixes=("", "_clean")
    )
    renamed = stop_attributes.loc[
        stop_attributes["stop_name"].astype(str) != stop_attributes["stop_name_clean"].astype(str), "stop_id"
    ].astype(str).tolist()
    if renamed:
        return _fail("logic_error", [f"stop_name does not match the cleaned stop table for: {renamed[:5]}"])
    lat_delta = (
        pd.to_numeric(stop_attributes["stop_lat"], errors="coerce")
        - pd.to_numeric(stop_attributes["stop_lat_clean"], errors="coerce")
    ).abs()
    lon_delta = (
        pd.to_numeric(stop_attributes["stop_lon"], errors="coerce")
        - pd.to_numeric(stop_attributes["stop_lon_clean"], errors="coerce")
    ).abs()
    moved = stop_attributes.loc[(lat_delta > 1e-6) | (lon_delta > 1e-6), "stop_id"].astype(str).tolist()
    if moved:
        return _fail("logic_error", [f"stop coordinates do not match the cleaned stop table for: {moved[:5]}"])
    if (pd.to_numeric(output["n_departures"], errors="coerce") < 0).any():
        return _fail("schema_error", ["negative departures found"])
    if not output["first_departure"].dropna().astype(str).apply(_valid_gtfs_time).all():
        return _fail("domain_error", ["invalid first_departure time"])
    if not output["last_departure"].dropna().astype(str).apply(_valid_gtfs_time).all():
        return _fail("domain_error", ["invalid last_departure time"])
    try:
        trips = _gtfs_tables(root)["trips"][["trip_id", "route_id"]]
    except Exception as exc:
        return _fail("wrong_data_source", [f"failed to read GTFS trips: {exc}"])
    times_with_routes = clean_times.merge(trips, on="trip_id", how="left")
    grouped = times_with_routes.groupby("stop_id")
    output_index = output.set_index("stop_id")
    expected_counts = grouped.size()
    departures = pd.to_numeric(output_index["n_departures"], errors="coerce")
    mismatched = [stop for stop, count in expected_counts.items() if _safe_int(departures.get(stop)) != int(count)]
    if mismatched:
        return _fail("logic_error", [f"n_departures mismatch for stops: {mismatched[:5]}"])
    routes = pd.to_numeric(output_index["n_routes"], errors="coerce")
    expected_routes = grouped["route_id"].nunique()
    mismatched_routes = [stop for stop, count in expected_routes.items() if _safe_int(routes.get(stop)) != int(count)]
    if mismatched_routes:
        return _fail("logic_error", [f"n_routes mismatch for stops: {mismatched_routes[:5]}"])
    for stop_id, group in grouped:
        ordered = sorted(group["departure_time"].dropna().astype(str), key=_time_to_seconds)
        if not ordered:
            continue
        first = str(output_index.loc[stop_id, "first_departure"])
        last = str(output_index.loc[stop_id, "last_departure"])
        if first != ordered[0] or last != ordered[-1]:
            return _fail("logic_error", [f"first/last departure mismatch for stop: {stop_id}"])
    return _ok(n_rows=len(output))


# The Wikidata Query Service defines these prefixes implicitly, so a query may
# use them without declaring them. The parser is given the same set, so that it
# accepts exactly what the endpoint accepts.
WDQS_PREFIXES = {
    "wd": "http://www.wikidata.org/entity/",
    "wds": "http://www.wikidata.org/entity/statement/",
    "wdv": "http://www.wikidata.org/value/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "wikibase": "http://wikiba.se/ontology#",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "bd": "http://www.bigdata.com/rdf#",
    "wdref": "http://www.wikidata.org/reference/",
    "psv": "http://www.wikidata.org/prop/statement/value/",
    "psn": "http://www.wikidata.org/prop/statement/value-normalized/",
    "pqv": "http://www.wikidata.org/prop/qualifier/value/",
    "pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "pr": "http://www.wikidata.org/prop/reference/",
    "prv": "http://www.wikidata.org/prop/reference/value/",
    "prn": "http://www.wikidata.org/prop/reference/value-normalized/",
    "wdno": "http://www.wikidata.org/prop/novalue/",
    "wdata": "http://www.wikidata.org/wiki/Special:EntityData/",
    "schema": "http://schema.org/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov": "http://www.w3.org/ns/prov#",
    "bds": "http://www.bigdata.com/rdf/search#",
    "gas": "http://www.bigdata.com/rdf/gas#",
    "hint": "http://www.bigdata.com/queryHints#",
}
WIKIBASE_LABEL_SERVICE = "http://wikiba.se/ontology#label"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
P625_SUFFIXES = ("/P625", "#P625")


def _compile_sparql(text: str) -> Any:
    """Compile a query to its algebra, supplying the prefixes that WDQS defines."""
    namespaces = {prefix: Namespace(iri) for prefix, iri in WDQS_PREFIXES.items()}
    return translateQuery(parseQuery(text), base="http://example.org/", initNs=namespaces).algebra


def _algebra_nodes(node: Any):
    """Yield every node of a compiled query, including those nested in triples."""
    if hasattr(node, "name"):
        yield node
        for value in node.values():
            yield from _algebra_nodes(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _algebra_nodes(value)


def _triple_patterns(algebra: Any) -> list[tuple[Any, Any, Any]]:
    return [
        triple
        for node in _algebra_nodes(algebra)
        if node.name == "BGP"
        for triple in (node.get("triples") or [])
    ]


def _collect_urirefs(value: object, acc: set[str]) -> None:
    """Collect every URI anywhere in a parsed algebra tree."""
    if isinstance(value, URIRef):
        acc.add(str(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_urirefs(item, acc)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_urirefs(item, acc)


def check_t07(root: Path) -> CheckerResult:
    path = root / "outputs" / "t07_query.sparql"
    if not path.exists():
        return _fail("schema_error", ["missing SPARQL query"])
    try:
        algebra = _compile_sparql(path.read_text(encoding="utf-8"))
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".split("\n")[0]
        return _fail("format_error", [f"query is not valid SPARQL: {message}"])
    if algebra.name != "SelectQuery":
        return _fail("logic_error", [f"query is not a SELECT (found {algebra.name})"])
    projected = {str(variable) for variable in (algebra.get("PV") or [])}
    triples = _triple_patterns(algebra)
    coordinate_triples = [
        triple
        for triple in triples
        if isinstance(triple[1], URIRef) and str(triple[1]).endswith(P625_SUFFIXES)
    ]
    if not coordinate_triples:
        return _fail("logic_error", ["query does not retrieve coordinates (no P625 triple pattern)"])
    if not any(isinstance(subject, Variable) and str(subject) in projected for subject, _, _ in coordinate_triples):
        return _fail("logic_error", ["query does not project the item identifier of the coordinate triple"])
    if not any(isinstance(obj, Variable) and str(obj) in projected for _, _, obj in coordinate_triples):
        return _fail("logic_error", ["query does not project the coordinates"])
    services = {
        str(node.get("term")) for node in _algebra_nodes(algebra) if node.name == "ServiceGraphPattern"
    }
    labelled_by_service = WIKIBASE_LABEL_SERVICE in services and any(
        variable.endswith("Label") for variable in projected
    )
    labelled_by_triple = any(
        isinstance(triple[1], URIRef)
        and str(triple[1]) == RDFS_LABEL
        and isinstance(triple[2], Variable)
        and str(triple[2]) in projected
        for triple in triples
    )
    if not (labelled_by_service or labelled_by_triple):
        return _fail("logic_error", ["query does not retrieve labels"])
    config, error = _load_json(root / "inputs" / "wikidata" / "city_config.json")
    if error:
        return error
    identifiers = {str(config[key]) for key in ("city_qid", "country_qid") if config.get(key)}
    # The identifier may appear as a triple object, in a VALUES clause, or in a
    # FILTER; all three constrain the query, so search the whole algebra.
    terms: set[str] = set()
    _collect_urirefs(algebra, terms)
    if not any(any(term.endswith(f"/{identifier}") for term in terms) for identifier in identifiers):
        return _fail(
            "domain_error",
            [f"query does not constrain to the configured city or country ({', '.join(sorted(identifiers))})"],
        )
    return _ok(projected_variables=sorted(projected), triple_patterns=len(triples))


def check_t08(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t08_pois_clean.csv")
    if error:
        return error
    assert output is not None
    required = {"qid", "label", "poi_type", "latitude", "longitude", "source"}
    missing = _required_columns(output, required)
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    lat = pd.to_numeric(output["latitude"], errors="coerce")
    lon = pd.to_numeric(output["longitude"], errors="coerce")
    if lat.isna().any() or lon.isna().any():
        return _fail("schema_error", ["coordinates are not numeric"])
    if not lat.between(-90, 90).all() or not lon.between(-180, 180).all():
        return _fail("domain_error", ["coordinates out of range"])
    if output["qid"].duplicated().any():
        return _fail("logic_error", ["duplicated qid values remain"])
    expected_records = _wikidata_expected_records(root)
    output_qids = set(output["qid"].astype(str))
    if output_qids != set(expected_records):
        return _fail("logic_error", ["output must contain exactly one row per valid input qid"])
    mismatched_coordinates = [
        str(row.qid)
        for row in output.itertuples(index=False)
        if abs(float(row.latitude) - expected_records[str(row.qid)][0]) > 1e-6
        or abs(float(row.longitude) - expected_records[str(row.qid)][1]) > 1e-6
    ]
    if mismatched_coordinates:
        return _fail(
            "logic_error",
            [f"coordinates do not match the source record for: {mismatched_coordinates[:5]}"],
        )
    if output["label"].fillna("").astype(str).str.strip().eq("").any():
        return _fail("logic_error", ["missing labels remain"])
    if not output["source"].astype(str).str.lower().str.contains("wikidata").all():
        return _fail("schema_error", ["source must be wikidata or equivalent"])
    return _ok(n_rows=len(output))


def check_t09(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t09_poi_nearest_stop.csv")
    if error:
        return error
    pois, error = _read_csv(root / "outputs" / "t08_pois_clean.csv")
    if error:
        return error
    stops, error = _read_csv(root / "outputs" / "t05_clean_stops.csv")
    if error:
        return error
    assert output is not None and pois is not None and stops is not None
    required = {"qid", "label", "poi_type", "nearest_stop_id", "nearest_stop_name", "distance_m"}
    missing = _required_columns(output, required)
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    if len(output) != len(pois):
        return _fail("logic_error", ["expected one row per POI"])
    if output["qid"].duplicated().any():
        return _fail("logic_error", ["duplicated qid values remain"])
    if set(output["qid"].astype(str)) != set(pois["qid"].astype(str)):
        return _fail("logic_error", ["output qids must match cleaned POIs"])
    distances = pd.to_numeric(output["distance_m"], errors="coerce")
    if distances.isna().any() or (distances < 0).any():
        return _fail("schema_error", ["distance_m must be numeric and non-negative"])
    if not set(output["nearest_stop_id"]).issubset(set(stops["stop_id"])):
        return _fail("logic_error", ["nearest_stop_id not found in clean stops"])
    stops_index = stops.set_index("stop_id")
    joined = output.merge(pois[["qid", "latitude", "longitude"]], on="qid")
    for _, row in joined.iterrows():
        nearest = min(
            stops.itertuples(index=False),
            key=lambda stop: (
                _haversine_m(float(row.latitude), float(row.longitude), float(stop.stop_lat), float(stop.stop_lon)),
                str(stop.stop_id),
            ),
        )
        expected_distance = _haversine_m(float(row.latitude), float(row.longitude), float(nearest.stop_lat), float(nearest.stop_lon))
        if row.nearest_stop_id != nearest.stop_id or abs(float(row.distance_m) - expected_distance) > 5:
            return _fail("domain_error", ["nearest stop or distance mismatch"])
    return _ok(n_rows=len(output), n_stops=len(stops_index))


def check_t10(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t10_accessibility_scores.csv")
    if error:
        return error
    nearest, error = _read_csv(root / "outputs" / "t09_poi_nearest_stop.csv")
    if error:
        return error
    assert output is not None and nearest is not None
    required = {"qid", "label", "poi_type", "distance_m", "n_departures", "n_routes", "accessibility_score"}
    missing = _required_columns(output, required)
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    if output["qid"].duplicated().any():
        return _fail("logic_error", ["duplicated qid values remain"])
    if set(output["qid"].astype(str)) != set(nearest["qid"].astype(str)):
        return _fail("logic_error", ["accessibility output must contain exactly one row per nearest-stop POI"])
    service, error = _read_csv(root / "outputs" / "t06_stop_service_intensity.csv")
    if error:
        return error
    assert service is not None
    linked = output.merge(
        nearest[["qid", "nearest_stop_id", "distance_m"]], on="qid", suffixes=("", "_t09")
    )
    if len(linked) != len(output):
        return _fail("logic_error", ["nearest_stop_id is missing from outputs/t09_poi_nearest_stop.csv"])
    linked = linked.merge(
        service[["stop_id", "n_departures", "n_routes"]],
        left_on="nearest_stop_id",
        right_on="stop_id",
        suffixes=("", "_t06"),
    )
    if len(linked) != len(output):
        return _fail("logic_error", ["nearest_stop_id is missing from outputs/t06_stop_service_intensity.csv"])
    linked_distance = pd.to_numeric(linked["distance_m"], errors="coerce")
    linked_distance_expected = pd.to_numeric(linked["distance_m_t09"], errors="coerce")
    bad_distance = linked.loc[(linked_distance - linked_distance_expected).abs() > 1e-6, "qid"].astype(str).tolist()
    if bad_distance:
        return _fail("logic_error", [f"distance_m does not match the nearest-stop table for: {bad_distance[:5]}"])
    linked_departures = pd.to_numeric(linked["n_departures"], errors="coerce")
    linked_departures_expected = pd.to_numeric(linked["n_departures_t06"], errors="coerce")
    bad_departures = linked.loc[linked_departures != linked_departures_expected, "qid"].astype(str).tolist()
    if bad_departures:
        return _fail("logic_error", [f"n_departures does not match the service table for: {bad_departures[:5]}"])
    linked_routes = pd.to_numeric(linked["n_routes"], errors="coerce")
    linked_routes_expected = pd.to_numeric(linked["n_routes_t06"], errors="coerce")
    bad_routes = linked.loc[linked_routes != linked_routes_expected, "qid"].astype(str).tolist()
    if bad_routes:
        return _fail("logic_error", [f"n_routes does not match the service table for: {bad_routes[:5]}"])
    distance = pd.to_numeric(output["distance_m"], errors="coerce")
    departures = pd.to_numeric(output["n_departures"], errors="coerce")
    routes = pd.to_numeric(output["n_routes"], errors="coerce")
    score = pd.to_numeric(output["accessibility_score"], errors="coerce")
    if score.isna().any() or not score.between(0, 1).all():
        return _fail("schema_error", ["accessibility_score must be numeric between 0 and 1"])
    max_dep = np.log1p(departures).max()
    max_routes = np.log1p(routes).max()
    expected = (
        0.50 * np.maximum(0, 1 - np.minimum(distance, 1000) / 1000)
        + 0.35 * (np.log1p(departures) / max_dep if max_dep else 0)
        + 0.15 * (np.log1p(routes) / max_routes if max_routes else 0)
    )
    if not np.allclose(score, expected, atol=1e-5):
        return _fail("logic_error", ["accessibility scores do not match required formula"])
    return _ok(n_rows=len(output))


def check_t11(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t11_city_indicator_matrix.csv")
    if error:
        return error
    raw, error = _read_csv(root / "outputs" / "t02_eurostat_raw.csv")
    if error:
        return error
    assert output is not None and raw is not None
    missing = _required_columns(output, {"city_id", "city_name", "year"})
    if missing:
        return _fail("schema_error", [f"missing identifier columns: {missing}"])
    if output.duplicated(["city_id", "year"]).any():
        return _fail("logic_error", ["duplicated city_id/year rows"])
    if _id_key(output) != _id_key(raw.drop_duplicates(["city_id", "year"])):
        return _fail("logic_error", ["city_id/year rows do not match raw Eurostat input"])
    expected_indicators = set(raw["indicator_id"].dropna())
    if not expected_indicators.issubset(set(output.columns)):
        return _fail("logic_error", ["selected indicators were not pivoted into columns"])
    expected = raw.pivot_table(
        index=["city_id", "year"], columns="indicator_id", values="value", aggfunc="first"
    )
    actual = output.set_index(["city_id", "year"])
    columns = [indicator for indicator in expected.columns if indicator in actual.columns]
    actual_cells = actual[columns].apply(pd.to_numeric, errors="coerce")
    expected_cells = expected[columns].apply(pd.to_numeric, errors="coerce")
    actual_cells, expected_cells = actual_cells.align(expected_cells, join="inner")
    mismatched = ~np.isclose(
        actual_cells.to_numpy(dtype=float),
        expected_cells.to_numpy(dtype=float),
        equal_nan=True,
        atol=1e-9,
    )
    if mismatched.any():
        bad_columns = [columns[index] for index in range(len(columns)) if mismatched[:, index].any()]
        return _fail(
            "logic_error",
            [
                f"{int(mismatched.sum())} matrix cells do not match the long-format values "
                f"(indicators: {bad_columns[:3]})"
            ],
        )
    return _ok(n_rows=len(output), n_indicators=len(expected_indicators))


def _as_bool(value: object) -> bool:
    """Parse a CSV boolean written as true/false, yes/no, or 1/0."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "t"}


def _safe_int(value: object, default: int = -1) -> int:
    """Convert a possibly missing or non-numeric CSV value to an int.

    A checker must never raise on malformed model output; an empty cell is a
    failed count, not a crash.
    """
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def check_t12(root: Path) -> CheckerResult:
    report, error = _read_csv(root / "outputs" / "t12_missingness_report.csv")
    if error:
        return error
    selected, error = _read_csv(root / "outputs" / "t12_selected_matrix.csv")
    if error:
        return error
    matrix, error = _read_csv(root / "outputs" / "t11_city_indicator_matrix.csv")
    if error:
        return error
    assert report is not None and selected is not None and matrix is not None
    missing = _required_columns(report, {"indicator", "missing_count", "missing_rate", "kept"})
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    indicator_cols = [col for col in matrix.columns if col not in {"city_id", "city_name", "year"}]
    expected = matrix[indicator_cols].isna().mean()
    report_index = report.set_index("indicator")
    for indicator, rate in expected.items():
        if indicator not in report_index.index:
            return _fail("logic_error", [f"indicator missing from report: {indicator}"])
        if not math.isclose(float(report_index.loc[indicator, "missing_rate"]), float(rate), abs_tol=1e-6):
            return _fail("logic_error", [f"missing_rate mismatch for {indicator}"])
        try:
            recorded_count = int(float(report_index.loc[indicator, "missing_count"]))
        except (TypeError, ValueError):
            return _fail("schema_error", [f"missing_count is not numeric for {indicator}"])
        if recorded_count != int(matrix[indicator].isna().sum()):
            return _fail("logic_error", [f"missing_count mismatch for {indicator}"])
    kept = {
        indicator
        for indicator, flag in zip(report["indicator"], report["kept"])
        if _as_bool(flag)
    }
    if kept != {indicator for indicator, rate in expected.items() if rate <= 0.30}:
        return _fail("logic_error", ["kept flags do not follow missing_rate <= 0.30"])
    allowed_columns = {"city_id", "city_name", "year", *kept}
    if set(selected.columns) != allowed_columns:
        return _fail("schema_error", ["selected_matrix has wrong columns"])
    if _id_key(selected) != _id_key(matrix):
        return _fail("logic_error", ["selected_matrix must preserve all city_id/year rows"])
    return _ok(n_kept=len(kept))


def check_t13(root: Path) -> CheckerResult:
    clusters, error = _read_csv(root / "outputs" / "t13_city_clusters.csv")
    if error:
        return error
    profiles, error = _read_csv(root / "outputs" / "t13_cluster_profiles.csv")
    if error:
        return error
    matrix, error = _read_csv(root / "outputs" / "t12_selected_matrix.csv")
    if error:
        return error
    assert clusters is not None and profiles is not None and matrix is not None
    missing = _required_columns(clusters, {"city_id", "city_name", "year", "cluster_id"})
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    if len(clusters) != len(matrix):
        return _fail("logic_error", ["every city-year row must appear in clusters"])
    if _id_key(clusters) != _id_key(matrix):
        return _fail("logic_error", ["cluster rows must match selected matrix city_id/year rows"])
    expected_k = min(3, len(matrix))
    if clusters["cluster_id"].nunique() != expected_k:
        return _fail("logic_error", [f"expected {expected_k} clusters"])
    if len(profiles) != expected_k:
        return _fail("logic_error", ["cluster_profiles must contain one row per cluster"])
    if "cluster_id" not in profiles.columns:
        return _fail("schema_error", ["cluster_profiles must include cluster_id"])
    if set(profiles["cluster_id"].astype(str)) != set(clusters["cluster_id"].astype(str)):
        return _fail("logic_error", ["cluster_profiles cluster_id values must match city clusters"])
    indicator_cols = [col for col in matrix.columns if col not in {"city_id", "city_name", "year"}]
    if not indicator_cols:
        return _fail("schema_error", ["selected matrix has no indicator columns"])
    features = matrix[indicator_cols].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    scaled = StandardScaler().fit_transform(imputed)
    expected_labels = KMeans(n_clusters=expected_k, random_state=0, n_init=10).fit_predict(scaled)
    expected = matrix[["city_id", "year"]].copy()
    expected["expected_cluster"] = expected_labels
    observed = clusters[["city_id", "year", "cluster_id"]].merge(expected, on=["city_id", "year"], how="left")
    label_map: dict[str, int] = {}
    for cluster_id, group in observed.groupby(observed["cluster_id"].astype(str)):
        expected_set = set(group["expected_cluster"])
        if len(expected_set) != 1:
            return _fail("logic_error", ["cluster assignments do not match deterministic KMeans partition"])
        label_map[cluster_id] = int(next(iter(expected_set)))
    if len(set(label_map.values())) != expected_k:
        return _fail("logic_error", ["cluster assignments do not match deterministic KMeans partition"])
    profile_index = profiles.set_index(profiles["cluster_id"].astype(str))
    profile_missing = sorted(set(indicator_cols).difference(profiles.columns))
    if profile_missing:
        return _fail("schema_error", [f"cluster_profiles missing indicator columns: {profile_missing[:5]}"])
    profiled = matrix[["city_id", "year", *indicator_cols]].merge(clusters[["city_id", "year", "cluster_id"]], on=["city_id", "year"])
    expected_profiles = profiled.groupby(profiled["cluster_id"].astype(str))[indicator_cols].mean()
    for cluster_id, expected_row in expected_profiles.iterrows():
        actual = pd.to_numeric(profile_index.loc[cluster_id, indicator_cols], errors="coerce")
        if not np.allclose(actual.to_numpy(dtype=float), expected_row.to_numpy(dtype=float), equal_nan=True, atol=1e-6):
            return _fail("logic_error", [f"cluster profile means mismatch for cluster: {cluster_id}"])
    return _ok(n_clusters=expected_k)


T14_UPSTREAM_OUTPUTS = (
    "outputs/t01_selected_indicators.csv",
    "outputs/t05_clean_stops.csv",
    "outputs/t08_pois_clean.csv",
    "outputs/t10_accessibility_scores.csv",
    "outputs/t13_city_clusters.csv",
)
T14_CLEAN_TIMEOUT_SECONDS = 30
T14_CORRUPTION_TIMEOUT_SECONDS = 20


def _run_validation_script(root: Path, corrupt: tuple[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    """Run the T14 validator in a throwaway copy, optionally corrupting one output."""
    with tempfile.TemporaryDirectory(prefix="mobility_t14_check_") as temp_dir:
        temp_root = Path(temp_dir) / "workspace"
        shutil.copytree(root, temp_root)
        summary_path = temp_root / "outputs" / "t14_validation_summary.json"
        if summary_path.exists():
            summary_path.unlink()
        if corrupt is not None:
            kind, relative = corrupt
            target = temp_root / relative
            if kind == "missing":
                target.unlink(missing_ok=True)
            else:
                target.write_text("", encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "src/validate_outputs.py"],
                cwd=temp_root,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=sanitized_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if not summary_path.exists():
            if completed.returncode != 0:
                return {"status": "crashed", "detail": completed.stderr[-300:]}
            return {"status": "no_summary"}
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"status": "unreadable", "detail": str(exc)}
        return {"status": "ok", "summary": summary, "returncode": completed.returncode}


def check_t14(root: Path) -> CheckerResult:
    script = root / "src" / "validate_outputs.py"
    if not script.exists():
        return _fail("schema_error", ["missing validation script"])
    clean = _run_validation_script(root, timeout=T14_CLEAN_TIMEOUT_SECONDS)
    if clean["status"] == "timeout":
        return _fail("runtime_error", ["validation script timed out"])
    if clean["status"] == "crashed":
        return _fail(
            "runtime_error",
            [f"validation script failed on the intact workspace: {clean['detail']}"],
        )
    if clean["status"] != "ok":
        return _fail("schema_error", ["validation script did not write a readable validation summary"])
    summary = clean["summary"]
    if not isinstance(summary, list):
        return _fail("schema_error", ["validation summary must be a list"])
    required = {"check_name", "passed", "message"}
    if not summary:
        return _fail("logic_error", ["validation summary is empty"])
    if len(summary) < 5:
        return _fail("logic_error", ["validation summary must contain at least five checks"])
    for entry in summary:
        if not isinstance(entry, dict) or not required.issubset(entry):
            return _fail("schema_error", ["each validation entry must contain check_name, passed, and message"])
        if not isinstance(entry["passed"], bool):
            return _fail("schema_error", ["passed must be a boolean in every validation entry"])
    if not all(entry["passed"] is True for entry in summary):
        return _fail("logic_error", ["all validation summary checks must pass on the intact workspace"])
    if clean.get("returncode", 0) != 0:
        return _fail(
            "runtime_error",
            [f"validation script exited with code {clean['returncode']} on the intact workspace"],
        )
    for relative in T14_UPSTREAM_OUTPUTS:
        for kind in ("missing", "empty"):
            outcome = _run_validation_script(
                root, corrupt=(kind, relative), timeout=T14_CORRUPTION_TIMEOUT_SECONDS
            )
            if outcome["status"] == "timeout":
                return _fail(
                    "logic_error",
                    [f"validator timed out instead of reporting a failure when {relative} was {kind}"],
                )
            if outcome["status"] == "crashed":
                return _fail(
                    "logic_error",
                    [f"validator crashed instead of reporting a failure when {relative} was {kind}"],
                )
            if outcome["status"] != "ok":
                return _fail(
                    "logic_error",
                    [f"validator wrote no readable summary when {relative} was {kind}"],
                )
            reported = [
                entry
                for entry in outcome["summary"]
                if isinstance(entry, dict) and entry.get("passed") is False
            ]
            if not reported:
                return _fail(
                    "logic_error",
                    [f"validator did not report a failure when {relative} was {kind}"],
                )
    return _ok(n_checks=len(summary), corruptions=len(T14_UPSTREAM_OUTPUTS) * 2)


_LIVE_API_RE = re.compile(r"\b(?:live\s+apis?|queried\s+wikidata|api\s+calls?)\b")
_NEGATION_RE = re.compile(r"\b(?:no|not|never|neither|nor|without|none)\b")


def _claims_live_api_access(text: str) -> bool:
    """True when the report asserts that live API access happened.

    Whitespace is collapsed before sentence splitting so that a negation which
    wraps onto the next line stays attached to its sentence, and the match
    pattern covers plurals explicitly so that a denial such as ``no live APIs
    were queried`` is not scored as a claim. A negation anywhere in the same
    sentence is treated as a denial; that can miss a claim in a sentence that
    denies something else, which is the conservative direction for scoring.
    """
    normalised = re.sub(r"\s+", " ", text.lower())
    for sentence in re.split(r"(?<=[.!?])\s+|;", normalised):
        if not _LIVE_API_RE.search(sentence):
            continue
        if _NEGATION_RE.search(sentence):
            continue
        return True
    return False


MIN_REPORT_SECTION_CHARS = 80
MIN_FIGURE_WIDTH = 400
MIN_FIGURE_HEIGHT = 300
_PLACEHOLDER_TEXT = re.compile(r"\b(todo|tbd|lorem ipsum|placeholder|to be determined)\b", re.IGNORECASE)


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", header[16:24])


def _section_bodies(text: str, headings: list[str]) -> dict[str, str]:
    positions = sorted((text.find(heading), heading) for heading in headings if heading in text)
    bodies: dict[str, str] = {}
    for index, (start, heading) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        bodies[heading] = text[start + len(heading) : end]
    return bodies


def check_t15(root: Path) -> CheckerResult:
    report = root / "reports" / "pipeline_report.md"
    if not report.exists():
        return _fail("schema_error", ["missing pipeline report"])
    text = report.read_text(encoding="utf-8")
    required_sections = ["# Overview", "# Input data", "# Methods", "# Generated outputs", "# Main results", "# Limitations", "# Reproducibility"]
    missing_sections = [section for section in required_sections if section not in text]
    if missing_sections:
        return _fail("format_error", [f"missing report sections: {missing_sections}"])
    # A heading with no text under it is not a report section.
    bodies = _section_bodies(text, required_sections)
    empty_sections = [
        section for section in required_sections if len(bodies.get(section, "").strip()) < MIN_REPORT_SECTION_CHARS
    ]
    if empty_sections:
        return _fail("logic_error", [f"sections without content: {empty_sections}"])
    placeholder = _PLACEHOLDER_TEXT.search(text)
    if placeholder:
        return _fail("logic_error", [f"report contains placeholder text: {placeholder.group(0)}"])
    figures = [
        root / "figures" / "fig_accessibility_distribution.png",
        root / "figures" / "fig_cluster_profiles.png",
    ]
    missing_figures = [str(path) for path in figures if not path.exists() or path.stat().st_size == 0]
    if missing_figures:
        return _fail("schema_error", [f"missing or empty figures: {missing_figures}"])
    for path in figures:
        dimensions = _png_dimensions(path)
        if dimensions is None:
            return _fail("schema_error", [f"{path.name} is not a valid PNG image"])
        width, height = dimensions
        if width < MIN_FIGURE_WIDTH or height < MIN_FIGURE_HEIGHT:
            return _fail(
                "schema_error",
                [f"{path.name} is {width}x{height}; minimum is {MIN_FIGURE_WIDTH}x{MIN_FIGURE_HEIGHT}"],
            )
    if figures[0].read_bytes() == figures[1].read_bytes():
        return _fail("logic_error", ["the two figures are identical"])
    if _claims_live_api_access(text):
        return _fail("hallucinated_claim", ["report claims live API access"])
    if "outputs/" not in text:
        return _fail("incomplete_solution", ["report does not reference generated outputs"])
    return _ok()
