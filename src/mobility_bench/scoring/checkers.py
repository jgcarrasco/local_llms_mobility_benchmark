"""Deterministic task checkers."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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


def _wikidata_expected_qids(root: Path) -> set[str]:
    source = json.loads((root / "inputs" / "wikidata" / "wikidata_pois_sample.json").read_text(encoding="utf-8"))
    qids: set[str] = set()
    for row in source["results"]["bindings"]:
        qid = str(row.get("qid") or "").strip()
        coord = str(row.get("coord") or "")
        match = re.fullmatch(r"Point\((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)", coord)
        if not qid or not match:
            continue
        lon, lat = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            qids.add(qid)
    return qids


def _t05_expected_cleaning(root: Path) -> tuple[set[str], Counter[tuple[str, str, str, str, str]], dict[str, int]]:
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
    return clean_stop_ids, expected_time_rows, expected_report


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
    required_themes = {"mobility", "population_density", "environment", "living_conditions"}
    if not required_themes.issubset(set(output["theme"])):
        return _fail("logic_error", ["missing required theme coverage"])
    if output["justification"].fillna("").astype(str).str.strip().eq("").any():
        return _fail("logic_error", ["empty justifications found"])
    allowed = {"higher_is_better", "lower_is_better", "contextual"}
    if not set(output["expected_direction"]).issubset(allowed):
        return _fail("schema_error", ["invalid expected_direction value"])
    return _ok(n_selected=len(output))


def check_t02(root: Path) -> CheckerResult:
    script = root / "src" / "t02_load_eurostat.py"
    if not script.exists():
        return _fail("schema_error", ["missing script: src/t02_load_eurostat.py"])
    output, error = _read_csv(root / "outputs" / "t02_eurostat_raw.csv")
    if error:
        return error
    selected, error = _read_csv(root / "outputs" / "t01_selected_indicators.csv")
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


def check_t03(root: Path) -> CheckerResult:
    output, error = _read_csv(root / "outputs" / "t03_mobility_catalog.csv")
    if error:
        return error
    assert output is not None
    required = {"dataset_id", "title", "description", "publisher", "country", "formats", "license", "distribution_url", "keywords"}
    missing = _required_columns(output, required)
    if missing:
        return _fail("schema_error", [f"missing columns: {missing}"])
    catalog = json.loads((root / "inputs" / "catalog" / "dataeuropa_mobility_catalog_sample.json").read_text(encoding="utf-8"))
    valid_ids = {str(dataset["id"]) for dataset in catalog["datasets"]}
    if not set(output["dataset_id"].astype(str)).issubset(valid_ids):
        return _fail("wrong_data_source", ["dataset_id values must come from the input catalog"])
    text = (output["title"].fillna("") + " " + output["description"].fillna("") + " " + output["keywords"].fillna("")).str.lower()
    mobility_terms = ("mobility", "transport", "gtfs", "traffic", "rail", "bike")
    if not text.apply(lambda value: any(term in value for term in mobility_terms)).all():
        return _fail("logic_error", ["non-mobility rows included"])
    if not text.str.contains("gtfs").any():
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
        "missing_required_columns",
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
    expected_stop_ids, expected_time_rows, expected_report = _t05_expected_cleaning(root)
    missing_report_fields = sorted(set(expected_report).difference(report))
    if missing_report_fields:
        return _fail("schema_error", [f"cleaning report missing fields: {missing_report_fields}"])
    mismatched_report = [
        field
        for field, expected_value in expected_report.items()
        if int(report.get(field, -1)) != int(expected_value)
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
    mismatched = [stop for stop, count in expected_counts.items() if int(departures.get(stop, -1)) != int(count)]
    if mismatched:
        return _fail("logic_error", [f"n_departures mismatch for stops: {mismatched[:5]}"])
    routes = pd.to_numeric(output_index["n_routes"], errors="coerce")
    expected_routes = grouped["route_id"].nunique()
    mismatched_routes = [stop for stop, count in expected_routes.items() if int(routes.get(stop, -1)) != int(count)]
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


def check_t07(root: Path) -> CheckerResult:
    path = root / "outputs" / "t07_query.sparql"
    if not path.exists():
        return _fail("schema_error", ["missing SPARQL query"])
    query = path.read_text(encoding="utf-8").lower()
    required_terms = ["select", "label", "p625"]
    missing = [term for term in required_terms if term not in query]
    if missing:
        return _fail("logic_error", [f"missing query terms: {missing}"])
    if "wikibase:label" not in query and "rdfs:label" not in query:
        return _fail("logic_error", ["missing label retrieval"])
    config, error = _load_json(root / "inputs" / "wikidata" / "city_config.json")
    if error:
        return error
    allowed_constraints = [
        str(config.get("city_name", "")).lower(),
        str(config.get("country", "")).lower(),
        str(config.get("city_qid", "")).lower(),
        str(config.get("country_qid", "")).lower(),
    ]
    if not any(value and value in query for value in allowed_constraints):
        return _fail("domain_error", ["query does not constrain city or country"])
    if "?item" not in query and "?entity" not in query:
        return _fail("schema_error", ["query does not request item/entity identifier"])
    return _ok()


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
    expected_qids = _wikidata_expected_qids(root)
    output_qids = set(output["qid"].astype(str))
    if output_qids != expected_qids:
        return _fail("logic_error", ["output must contain exactly one row per valid input qid"])
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
            key=lambda stop: _haversine_m(float(row.latitude), float(row.longitude), float(stop.stop_lat), float(stop.stop_lon)),
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
    return _ok(n_rows=len(output), n_indicators=len(expected_indicators))


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
    kept = set(report.loc[report["kept"].astype(bool), "indicator"])
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


def check_t14(root: Path) -> CheckerResult:
    script = root / "src" / "validate_outputs.py"
    if not script.exists():
        return _fail("schema_error", ["missing validation script"])
    with tempfile.TemporaryDirectory(prefix="mobility_t14_check_") as temp_dir:
        temp_root = Path(temp_dir) / "workspace"
        shutil.copytree(root, temp_root)
        temp_summary = temp_root / "outputs" / "t14_validation_summary.json"
        if temp_summary.exists():
            temp_summary.unlink()
        try:
            completed = subprocess.run(
                [sys.executable, "src/validate_outputs.py"],
                cwd=temp_root,
                text=True,
                capture_output=True,
                timeout=30,
                env=sanitized_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _fail("runtime_error", ["validation script timed out"])
        if completed.returncode != 0:
            return _fail(
                "runtime_error",
                [f"validation script failed with exit code {completed.returncode}: {completed.stderr[-500:]}"],
            )
        summary, error = _load_json(temp_summary)
        if error:
            return error
    if not isinstance(summary, list):
        return _fail("schema_error", ["validation summary must be a list"])
    required = {"check_name", "passed", "message"}
    if not summary:
        return _fail("logic_error", ["validation summary is empty"])
    if len(summary) < 5:
        return _fail("logic_error", ["validation summary must contain at least five checks"])
    coverage_text = ""
    for entry in summary:
        if not required.issubset(entry):
            return _fail("schema_error", ["each validation entry must contain check_name, passed, and message"])
        if not isinstance(entry["passed"], bool):
            return _fail("schema_error", ["passed must be a boolean in every validation entry"])
        coverage_text += f" {entry['check_name']} {entry['message']}".lower()
    if not all(entry["passed"] is True for entry in summary):
        return _fail("logic_error", ["all validation summary checks must pass"])
    required_outputs = [
        "t01_selected_indicators",
        "t05_clean_stops",
        "t08_pois_clean",
        "t10_accessibility_scores",
        "t13_city_clusters",
    ]
    covered = [name for name in required_outputs if name in coverage_text]
    if len(covered) < 4:
        return _fail("logic_error", ["validation summary must cover key upstream outputs"])
    return _ok(n_checks=len(summary))


def _claims_live_api_access(text: str) -> bool:
    for sentence in re.split(r"[\n.;]+", text.lower()):
        if "live api" not in sentence and "queried wikidata" not in sentence:
            continue
        negated_before = re.search(r"\b(no|not|never|without)\b.{0,80}\b(live api|queried wikidata|api calls?)\b", sentence)
        negated_after = re.search(r"\b(live api|queried wikidata|api calls?)\b.{0,80}\b(no|not|never|without)\b", sentence)
        if negated_before or negated_after:
            continue
        return True
    return False


def check_t15(root: Path) -> CheckerResult:
    report = root / "reports" / "pipeline_report.md"
    if not report.exists():
        return _fail("schema_error", ["missing pipeline report"])
    text = report.read_text(encoding="utf-8")
    required_sections = ["# Overview", "# Input data", "# Methods", "# Generated outputs", "# Main results", "# Limitations", "# Reproducibility"]
    missing_sections = [section for section in required_sections if section not in text]
    if missing_sections:
        return _fail("format_error", [f"missing report sections: {missing_sections}"])
    figures = [
        root / "figures" / "fig_accessibility_distribution.png",
        root / "figures" / "fig_cluster_profiles.png",
    ]
    missing_figures = [str(path) for path in figures if not path.exists() or path.stat().st_size == 0]
    if missing_figures:
        return _fail("schema_error", [f"missing or empty figures: {missing_figures}"])
    if _claims_live_api_access(text):
        return _fail("hallucinated_claim", ["report claims live API access"])
    if "outputs/" not in text:
        return _fail("incomplete_solution", ["report does not reference generated outputs"])
    return _ok()
