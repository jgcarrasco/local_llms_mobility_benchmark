"""Prepare fixed real-data snapshots for benchmark inputs.

The generated files intentionally match the shapes under ``data/inputs`` so
the existing task prompts and checkers can run unchanged. Public source data is
downloaded first, then a small set of documented controlled corruptions is
injected where deterministic cleaning ground truth is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "mobility-agent-benchmark/0.1 (research fixture preparation)"

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
DATAEUROPA_SEARCH = "https://data.europa.eu/api/hub/search/search"
MOBILITYDATA_FEEDS = "https://files.mobilitydatabase.org/feeds_v2.csv"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

EUROSTAT_DATASETS = {
    "urb_ctran": "mobility",
    "urb_cpop1": "population_density",
    "urb_cenv": "environment",
    "urb_clivcon": "living_conditions",
}

# The study area is the Alicante municipality and its immediate surroundings.
# It is used for two selections: points of interest must fall inside it, and a
# GTFS trip is selected because it stops inside it (the trip is then kept in
# full, so some of its stops lie outside).
STUDY_AREA = {"lat_min": 38.25, "lat_max": 38.47, "lon_min": -0.62, "lon_max": -0.38}

CITY_NAME = "Alicante"
CITY_QID = "Q11959"
COUNTRY_NAME = "Spain"
COUNTRY_QID = "Q29"

# Museums, cultural heritage monuments, libraries, theatres and art galleries.
CULTURAL_POI_TYPES = ["Q33506", "Q4989906", "Q7075", "Q24354", "Q1007870"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fetch_bytes(url: str, *, params: dict[str, Any] | None = None, timeout: int = 90) -> bytes:
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.content


def fetch_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 90) -> Any:
    return json.loads(fetch_bytes(url, params=params, timeout=timeout).decode("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("en", "label", "title", "name", "value"):
            if key in value:
                return coerce_text(value[key])
        return " ".join(coerce_text(item) for item in value.values() if item)
    if isinstance(value, list):
        return " ".join(coerce_text(item) for item in value if item)
    return str(value)


def jsonstat_position_map(category: dict[str, Any]) -> dict[int, str]:
    index = category.get("index", {})
    if isinstance(index, list):
        return {position: code for position, code in enumerate(index)}
    return {int(position): code for code, position in index.items()}


def decode_jsonstat_index(flat_index: int, sizes: list[int]) -> list[int]:
    coords = [0] * len(sizes)
    remaining = flat_index
    for idx in range(len(sizes) - 1, -1, -1):
        size = sizes[idx]
        coords[idx] = remaining % size
        remaining //= size
    return coords


def infer_direction(label: str) -> str:
    lowered = label.lower()
    if any(term in lowered for term in ("pollution", "fatal", "death", "unemploy", "crime", "noise", "car")):
        return "lower_is_better"
    if any(term in lowered for term in ("density", "population", "age", "number of", "total")):
        return "contextual"
    return "higher_is_better"


def infer_unit(label: str) -> str:
    lowered = label.lower()
    if "%" in label or "share" in lowered or "percentage" in lowered:
        return "percent"
    if "minute" in lowered:
        return "minutes"
    if "euro" in lowered:
        return "eur"
    if "density" in lowered:
        return "people_per_km2"
    return "value"


def category_label(payload: dict[str, Any], dimension: str, code: str) -> str:
    category = payload["dimension"][dimension].get("category", {})
    return category.get("label", {}).get(code, code)


def prepare_eurostat(inputs_dir: Path, raw_dir: Path, last_periods: int, max_rows_per_dataset: int) -> dict[str, Any]:
    euro_dir = inputs_dir / "eurostat"
    raw_euro_dir = raw_dir / "eurostat"
    raw_euro_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    indicator_rows: dict[str, dict[str, Any]] = {}
    value_rows: list[dict[str, Any]] = []
    snapshots: dict[str, str] = {}

    for dataset_id, theme in EUROSTAT_DATASETS.items():
        payload = fetch_json(
            f"{EUROSTAT_BASE}/{dataset_id}",
            params={"lang": "EN", "lastTimePeriod": last_periods},
        )
        raw_path = raw_euro_dir / f"{dataset_id}.json"
        write_json(raw_path, payload)
        snapshots[dataset_id] = display_path(raw_path)

        metadata_rows.append(
            {
                "dataset_id": dataset_id,
                "title": payload.get("label", dataset_id),
                "theme": theme,
                "description": f"Eurostat urban dataset {dataset_id}",
            }
        )

        dimension_ids = list(payload.get("id", []))
        sizes = list(payload.get("size", []))
        dimensions = payload.get("dimension", {})
        if "indic_ur" not in dimensions:
            continue
        city_dim = next((name for name in ("cities", "geo") if name in dimensions), "")
        time_dim = "time" if "time" in dimensions else ""

        indicator_positions = jsonstat_position_map(dimensions["indic_ur"]["category"])
        for _, indicator_id in sorted(indicator_positions.items()):
            indicator_name = category_label(payload, "indic_ur", indicator_id)
            indicator_rows[indicator_id] = {
                "indicator_id": indicator_id,
                "indicator_name": indicator_name,
                "theme": theme,
                "description": f"{indicator_name} from Eurostat dataset {dataset_id}",
                "unit": infer_unit(indicator_name),
                "expected_direction": infer_direction(indicator_name),
            }

        values = payload.get("value", {})
        if isinstance(values, list):
            iterator: Iterable[tuple[int, Any]] = enumerate(values)
        else:
            iterator = ((int(index), value) for index, value in values.items())

        dataset_rows = 0
        for flat_index, value in sorted(iterator):
            if value is None:
                continue
            decoded = decode_jsonstat_index(flat_index, sizes)
            decoded_by_dim = dict(zip(dimension_ids, decoded, strict=False))
            indicator_code = indicator_positions.get(decoded_by_dim.get("indic_ur", -1), "")
            if not indicator_code:
                continue

            city_id = ""
            city_name = ""
            if city_dim:
                city_positions = jsonstat_position_map(dimensions[city_dim]["category"])
                city_id = city_positions.get(decoded_by_dim.get(city_dim, -1), "")
                city_name = category_label(payload, city_dim, city_id) if city_id else ""
            time_value = ""
            if time_dim:
                time_positions = jsonstat_position_map(dimensions[time_dim]["category"])
                time_value = time_positions.get(decoded_by_dim.get(time_dim, -1), "")

            indicator_name = indicator_rows[indicator_code]["indicator_name"]
            value_rows.append(
                {
                    "city_id": city_id,
                    "city_name": city_name,
                    "year": time_value,
                    "indicator_id": indicator_code,
                    "indicator_name": indicator_name,
                    "theme": theme,
                    "value": value,
                    "unit": indicator_rows[indicator_code]["unit"],
                }
            )
            dataset_rows += 1
            if dataset_rows >= max_rows_per_dataset:
                break

    write_csv(
        euro_dir / "metadata_tables.csv",
        metadata_rows,
        ["dataset_id", "title", "theme", "description"],
    )
    write_csv(
        euro_dir / "indicator_descriptions.csv",
        list(indicator_rows.values()),
        ["indicator_id", "indicator_name", "theme", "description", "unit", "expected_direction"],
    )
    write_csv(
        euro_dir / "eurostat_city_values_sample.csv",
        value_rows,
        ["city_id", "city_name", "year", "indicator_id", "indicator_name", "theme", "value", "unit"],
    )
    write_json(
        euro_dir / "sample_api_config.json",
        {
            "source": "eurostat_snapshot",
            "base_path": "inputs/eurostat/eurostat_city_values_sample.csv",
            "id_column": "indicator_id",
            "value_column": "value",
            "snapshots": snapshots,
            "note": "Offline fixed snapshot prepared from Eurostat dissemination API.",
        },
    )
    return {"datasets": list(EUROSTAT_DATASETS), "rows": len(value_rows), "indicators": len(indicator_rows)}


def normalize_dataeuropa_dataset(row: dict[str, Any], idx: int) -> dict[str, Any]:
    distributions = row.get("distributions") or row.get("distribution") or []
    if isinstance(distributions, dict):
        distributions = [distributions]
    normalized_distributions = []
    for distribution in distributions:
        if not isinstance(distribution, dict):
            continue
        fmt = coerce_text(distribution.get("format") or distribution.get("media_type") or distribution.get("title"))
        url = coerce_text(
            distribution.get("downloadURL")
            or distribution.get("download_url")
            or distribution.get("accessURL")
            or distribution.get("access_url")
        )
        if fmt or url:
            normalized_distributions.append({"format": fmt, "downloadURL": url})

    if not normalized_distributions:
        normalized_distributions.append({"format": "", "downloadURL": coerce_text(row.get("url"))})

    keywords = row.get("keywords") or row.get("keyword") or row.get("categories") or []
    if isinstance(keywords, str):
        keywords = [keywords]

    publisher = row.get("publisher") or row.get("creator") or row.get("organization") or {}
    country = row.get("country") or row.get("spatial") or {}

    return {
        "id": coerce_text(row.get("id") or row.get("identifier") or f"dataeuropa-{idx:03d}"),
        "title": coerce_text(row.get("title") or row.get("name")),
        "description": coerce_text(row.get("description") or row.get("notes")),
        "publisher": coerce_text(publisher),
        "country": coerce_text(country),
        "distributions": normalized_distributions,
        "license": coerce_text(row.get("license") or row.get("licence")),
        "keywords": [coerce_text(keyword) for keyword in keywords],
    }


def prepare_catalog(inputs_dir: Path, raw_dir: Path, limit: int, selected_gtfs: dict[str, Any]) -> dict[str, Any]:
    payload = fetch_json(
        DATAEUROPA_SEARCH,
        params={"query": "GTFS public transport mobility traffic rail bike", "limit": limit},
    )
    raw_path = raw_dir / "catalog" / "dataeuropa_search.json"
    write_json(raw_path, payload)
    results = payload.get("result", {}).get("results", [])
    datasets = [normalize_dataeuropa_dataset(row, idx) for idx, row in enumerate(results, start=1)]
    datasets.append(
        {
            "id": "mobilitydata-selected-gtfs",
            "title": "Alicante area GTFS public transport feed",
            "description": "Static GTFS feed discovered through the MobilityData catalog and used as the benchmark transit snapshot.",
            "publisher": "MobilityData catalog source",
            "country": "ES",
            "distributions": [{"format": "GTFS ZIP", "downloadURL": selected_gtfs["selected_url"]}],
            "license": "See upstream feed metadata",
            "keywords": ["GTFS", "public transport", "mobility", "Alicante"],
        }
    )
    write_json(
        inputs_dir / "catalog" / "dataeuropa_mobility_catalog_sample.json",
        {
            "source": "data.europa.eu_search_snapshot_with_mobilitydata_gtfs_record",
            "source_url": DATAEUROPA_SEARCH,
            "datasets": datasets,
        },
    )
    return {"rows": len(datasets), "raw_snapshot": display_path(raw_path)}


def csv_text(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def read_gtfs_table(zip_file: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zip_file.open(f"{name}.txt") as handle:
        return pd.read_csv(handle, dtype=str)


def choose_gtfs_urls(feeds: pd.DataFrame) -> list[str]:
    env_url = os.environ.get("MOBILITY_BENCH_GTFS_URL", "").strip()
    if env_url:
        return [env_url]

    url_col = next((column for column in ("urls.direct_download", "static_reference", "urls.latest") if column in feeds.columns), "")
    if not url_col:
        raise RuntimeError("MobilityData feed catalog does not expose a supported GTFS URL column")

    candidates = feeds[feeds[url_col].fillna("").astype(str).str.startswith("http")].copy()
    if "location.country_code" in candidates.columns:
        spanish = candidates[candidates["location.country_code"].fillna("").str.upper().eq("ES")]
        if not spanish.empty:
            candidates = spanish
    if "data_type" in candidates.columns:
        static = candidates[candidates["data_type"].fillna("").str.lower().str.contains("gtfs|static")]
        if not static.empty:
            candidates = static
    city_hints = [
        hint.strip().lower()
        for hint in os.environ.get("MOBILITY_BENCH_GTFS_CITY_HINT", "alicante,alacant").split(",")
        if hint.strip()
    ]
    if city_hints:
        row_text = candidates.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        hinted = candidates[row_text.apply(lambda value: any(hint in value for hint in city_hints))]
        if not hinted.empty:
            candidates = hinted
    if candidates.empty:
        raise RuntimeError("No unauthenticated GTFS download candidate found in MobilityData catalog")
    return list(dict.fromkeys(candidates[url_col].astype(str).tolist()))


def existing_gtfs_url(provenance_path: Path) -> str:
    """Return the GTFS URL recorded by a previous fixture run, if any."""
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        url = payload["sources"]["gtfs"]["selected_url"]
        if isinstance(url, str) and url.startswith("http"):
            return url
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return "https://gvinterbus.gva.es/estatico/gtfs.zip"


def prepare_gtfs(
    inputs_dir: Path,
    raw_dir: Path,
    max_trips: int,
    provenance_path: Path,
    reuse_raw: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    raw_zip_path = raw_dir / "gtfs" / "source_gtfs.zip"
    feed_catalog: str | None = None

    if reuse_raw and raw_zip_path.exists():
        gtfs_bytes = raw_zip_path.read_bytes()
        gtfs_url = existing_gtfs_url(provenance_path)
    else:
        feeds_bytes = fetch_bytes(MOBILITYDATA_FEEDS)
        feeds_path = raw_dir / "gtfs" / "mobilitydata_feeds_v2.csv"
        feeds_path.parent.mkdir(parents=True, exist_ok=True)
        feeds_path.write_bytes(feeds_bytes)
        feed_catalog = display_path(feeds_path)

        feeds = pd.read_csv(BytesIO(feeds_bytes), dtype=str)
        errors: list[str] = []
        gtfs_url = ""
        gtfs_bytes = b""
        for candidate_url in choose_gtfs_urls(feeds)[:25]:
            try:
                gtfs_bytes = fetch_bytes(candidate_url, timeout=180)
            except requests.RequestException as exc:
                errors.append(f"{candidate_url}: {exc}")
                continue
            try:
                with zipfile.ZipFile(BytesIO(gtfs_bytes)):
                    gtfs_url = candidate_url
                    break
            except zipfile.BadZipFile as exc:
                errors.append(f"{candidate_url}: not a zip file ({exc})")
        if not gtfs_url:
            raise RuntimeError("No downloadable GTFS zip found. Tried candidates: " + "; ".join(errors[:5]))
        raw_zip_path.parent.mkdir(parents=True, exist_ok=True)
        raw_zip_path.write_bytes(gtfs_bytes)

    with zipfile.ZipFile(BytesIO(gtfs_bytes)) as source_zip:
        tables = {
            "stops": read_gtfs_table(source_zip, "stops"),
            "routes": read_gtfs_table(source_zip, "routes"),
            "trips": read_gtfs_table(source_zip, "trips"),
            "stop_times": read_gtfs_table(source_zip, "stop_times"),
        }
        agency = source_zip.read("agency.txt").decode("utf-8") if "agency.txt" in source_zip.namelist() else ""
        calendar = source_zip.read("calendar.txt").decode("utf-8") if "calendar.txt" in source_zip.namelist() else ""

    # Sample trips that serve the study area instead of the first rows of the
    # feed. The feed is ordered by location, so taking the head() selects a
    # single region and leaves the points of interest without nearby stops.
    lat_min = STUDY_AREA["lat_min"]
    lat_max = STUDY_AREA["lat_max"]
    lon_min = STUDY_AREA["lon_min"]
    lon_max = STUDY_AREA["lon_max"]

    stops_all = tables["stops"].copy()
    stops_all["stop_id"] = stops_all["stop_id"].astype(str)
    trips_all = tables["trips"].copy()
    trips_all["trip_id"] = trips_all["trip_id"].astype(str)
    routes_all = tables["routes"].copy()
    routes_all["route_id"] = routes_all["route_id"].astype(str)
    stop_times_all = tables["stop_times"].copy()
    stop_times_all["trip_id"] = stop_times_all["trip_id"].astype(str)
    stop_times_all["stop_id"] = stop_times_all["stop_id"].astype(str)

    in_box = stops_all[
        pd.to_numeric(stops_all["stop_lat"], errors="coerce").between(lat_min, lat_max)
        & pd.to_numeric(stops_all["stop_lon"], errors="coerce").between(lon_min, lon_max)
    ]
    if in_box.empty:
        raise RuntimeError("No stops fall inside the POI study area")

    box_stop_ids = set(in_box["stop_id"])
    serving_trips = stop_times_all.loc[stop_times_all["stop_id"].isin(box_stop_ids), "trip_id"].unique().tolist()
    known_trips = set(trips_all["trip_id"])
    candidate_trips = sorted(trip for trip in serving_trips if trip in known_trips)
    if len(candidate_trips) < max_trips:
        raise RuntimeError(f"Only {len(candidate_trips)} trips serve the study area; cannot sample {max_trips}")

    rng = random.Random(seed)
    chosen_trips = sorted(rng.sample(candidate_trips, max_trips))

    # Keep each chosen trip in full, including stops outside the study area: an
    # interurban trip passes through the city and continues elsewhere. Trimming
    # the stop times is what left most trips without a schedule before.
    trips = trips_all[trips_all["trip_id"].isin(chosen_trips)].sort_values("trip_id").copy()
    stop_times = stop_times_all[stop_times_all["trip_id"].isin(chosen_trips)].copy()
    stop_times["_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times = stop_times.sort_values(["trip_id", "_sequence"], kind="stable").drop(columns="_sequence")
    stops = stops_all[stops_all["stop_id"].isin(set(stop_times["stop_id"]))].sort_values("stop_id").copy()
    routes = routes_all[routes_all["route_id"].isin(set(trips["route_id"]))].sort_values("route_id").copy()

    if len(stops) < 3 or stop_times.empty:
        raise RuntimeError("Selected GTFS snapshot is too small for controlled corruptions")

    stops = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].reset_index(drop=True)
    stops.iloc[1, stops.columns.get_loc("stop_name")] = ""
    stops.iloc[2, stops.columns.get_loc("stop_lat")] = "95.0"
    stops.iloc[2, stops.columns.get_loc("stop_lon")] = "-200.0"
    stops = pd.concat([stops, stops.iloc[[0]].assign(stop_name="Duplicate stop id control")], ignore_index=True)

    stop_times = stop_times[["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]].reset_index(drop=True)
    stop_times.iloc[-1, stop_times.columns.get_loc("arrival_time")] = "25:10:00"
    stop_times.iloc[-1, stop_times.columns.get_loc("departure_time")] = "25:10:00"

    output_path = inputs_dir / "gtfs" / "sample_city_gtfs.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        output_zip.writestr("agency.txt", agency or "agency_id,agency_name,agency_url,agency_timezone\nA1,Unknown,https://example.invalid,Europe/Madrid\n")
        output_zip.writestr("stops.txt", csv_text(stops))
        output_zip.writestr("routes.txt", csv_text(routes))
        output_zip.writestr("trips.txt", csv_text(trips))
        output_zip.writestr("stop_times.txt", csv_text(stop_times))
        if calendar:
            output_zip.writestr("calendar.txt", calendar)

    result: dict[str, Any] = {
        "source_gtfs": display_path(raw_zip_path),
        "selected_url": gtfs_url,
        "study_area": STUDY_AREA,
        "sampling": {
            "method": "random trips serving the study area",
            "seed": seed,
            "max_trips": max_trips,
        },
        "n_stops_with_controls": len(stops),
        "n_trips": len(trips),
        "n_stop_times": len(stop_times),
        "n_routes": len(routes),
    }
    if feed_catalog:
        result["feed_catalog"] = feed_catalog
    return result


def prepare_wikidata(inputs_dir: Path, raw_dir: Path) -> dict[str, Any]:
    corners = (
        f'bd:serviceParam wikibase:cornerWest "Point({STUDY_AREA["lon_min"]} {STUDY_AREA["lat_min"]})"^^geo:wktLiteral .\n'
        f'    bd:serviceParam wikibase:cornerEast "Point({STUDY_AREA["lon_max"]} {STUDY_AREA["lat_max"]})"^^geo:wktLiteral .'
    )
    type_values = " ".join(f"wd:{qid}" for qid in CULTURAL_POI_TYPES)
    query = (
        "SELECT DISTINCT ?item ?itemLabel ?coord ?typeLabel WHERE {\n"
        "  SERVICE wikibase:box {\n"
        "    ?item wdt:P625 ?coord .\n"
        f"    {corners}\n"
        "  }\n"
        "  ?item wdt:P31 ?type .\n"
        f"  VALUES ?type {{ {type_values} }}\n"
        '  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,es". }\n'
        "}\n"
        "ORDER BY ?item\n"
        "LIMIT 200\n"
    )
    payload = fetch_json(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
    )
    raw_path = raw_dir / "wikidata" / "alicante_pois_sparql.json"
    write_json(raw_path, payload)

    rows: list[dict[str, Any]] = []
    for binding in payload.get("results", {}).get("bindings", []):
        item = binding.get("item", {}).get("value", "")
        qid = item.rsplit("/", 1)[-1] if item else ""
        coord = binding.get("coord", {}).get("value", "")
        if not qid or not coord:
            continue
        rows.append(
            {
                "item": item,
                "qid": qid,
                "itemLabel": binding.get("itemLabel", {}).get("value", qid),
                "poi_type": binding.get("typeLabel", {}).get("value", "point of interest"),
                "coord": coord,
            }
        )

    if len(rows) < 2:
        raise RuntimeError("Wikidata query returned too few rows for controlled corruptions")

    poi_types = sorted({row["poi_type"] for row in rows if row["poi_type"]})

    rows[1]["itemLabel"] = ""
    rows.append(dict(rows[0]))
    write_json(
        inputs_dir / "wikidata" / "city_config.json",
        {
            "city_name": CITY_NAME,
            "city_qid": CITY_QID,
            "country": COUNTRY_NAME,
            "country_qid": COUNTRY_QID,
            "poi_types": poi_types,
        },
    )
    write_json(
        inputs_dir / "wikidata" / "wikidata_pois_sample.json",
        {
            "source": "wikidata_sparql_snapshot",
            "query": query,
            "study_area": STUDY_AREA,
            "head": {"vars": ["item", "itemLabel", "poi_type", "coord"]},
            "results": {"bindings": rows},
        },
    )
    return {
        "rows_with_controls": len(rows),
        "poi_types": poi_types,
        "study_area": STUDY_AREA,
        "raw_snapshot": display_path(raw_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=ROOT / "data" / "inputs")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "real_sources")
    parser.add_argument(
        "--only",
        choices=["all", "eurostat", "catalog", "gtfs", "wikidata"],
        default="all",
        help="Rebuild a single fixture group instead of all of them.",
    )
    parser.add_argument("--eurostat-last-periods", type=int, default=3)
    parser.add_argument("--eurostat-max-rows", type=int, default=1500)
    parser.add_argument("--catalog-limit", type=int, default=25)
    parser.add_argument("--gtfs-max-trips", type=int, default=80)
    parser.add_argument("--gtfs-seed", type=int, default=0)
    parser.add_argument(
        "--gtfs-redownload",
        dest="gtfs_reuse_raw",
        action="store_false",
        default=True,
        help="Fetch the feed again instead of reusing the committed raw snapshot.",
    )
    parser.add_argument("--provenance-path", type=Path, default=ROOT / "data" / "real_fixture_provenance.json")
    args = parser.parse_args()

    args.inputs_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    def wanted(group: str) -> bool:
        return args.only in ("all", group)

    provenance_path = args.provenance_path
    provenance: dict[str, Any] = {}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    sources = provenance.setdefault("sources", {})

    if wanted("eurostat"):
        sources["eurostat"] = prepare_eurostat(args.inputs_dir, args.raw_dir, args.eurostat_last_periods, args.eurostat_max_rows)
    if wanted("wikidata"):
        sources["wikidata"] = prepare_wikidata(args.inputs_dir, args.raw_dir)
    if wanted("gtfs"):
        sources["gtfs"] = prepare_gtfs(
            args.inputs_dir,
            args.raw_dir,
            args.gtfs_max_trips,
            provenance_path,
            reuse_raw=args.gtfs_reuse_raw,
            seed=args.gtfs_seed,
        )
    if wanted("catalog"):
        if "gtfs" not in sources:
            raise RuntimeError("The catalog fixture needs a GTFS snapshot; run with --only gtfs first.")
        sources["catalog"] = prepare_catalog(args.inputs_dir, args.raw_dir, args.catalog_limit, sources["gtfs"])

    provenance["created_at_utc"] = now_iso()
    provenance["source_policy"] = "real public-data excerpts with documented controlled corruptions"
    provenance.setdefault(
        "controlled_corruptions",
        [
            {
                "file": "inputs/gtfs/sample_city_gtfs.zip/stops.txt",
                "description": "Duplicate first stop_id appended to test duplicate-stop handling.",
            },
            {
                "file": "inputs/gtfs/sample_city_gtfs.zip/stops.txt",
                "description": "Second stop_name blanked to test missing-name handling.",
            },
            {
                "file": "inputs/gtfs/sample_city_gtfs.zip/stops.txt",
                "description": "Third stop coordinates set outside valid latitude/longitude ranges.",
            },
            {
                "file": "inputs/gtfs/sample_city_gtfs.zip/stop_times.txt",
                "description": "Last arrival/departure set to 25:10:00 to test valid extended GTFS service-day times.",
            },
            {
                "file": "inputs/wikidata/wikidata_pois_sample.json",
                "description": "One Wikidata label blanked and one QID duplicated to test POI cleaning.",
            },
        ],
    )
    write_json(provenance_path, provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
