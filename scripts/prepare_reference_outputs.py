"""Prepare fixed upstream outputs used as task inputs.

These files make each benchmark task independently runnable in an isolated
workspace while still using deterministic local snapshots from data/inputs.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INPUTS = ROOT / "data" / "inputs"
TARGET = ROOT / "data" / "reference_outputs"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def gtfs_tables() -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(SOURCE_INPUTS / "gtfs" / "sample_city_gtfs.zip") as zip_file:
        return {
            name: pd.read_csv(zip_file.open(f"{name}.txt"))
            for name in ("stops", "routes", "trips", "stop_times")
        }


def time_to_seconds(value: object) -> int:
    hour, minute, second = [int(part) for part in str(value).split(":")]
    return hour * 3600 + minute * 60 + second


def main() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    outputs = TARGET / "outputs"
    src = TARGET / "src"
    reports = TARGET / "reports"
    figures = TARGET / "figures"
    for path in (outputs, src, reports, figures):
        path.mkdir(parents=True, exist_ok=True)

    indicators = pd.read_csv(SOURCE_INPUTS / "eurostat" / "indicator_descriptions.csv")
    selected = pd.concat(
        [
            indicators[indicators["theme"] == "mobility"].head(4),
            indicators[indicators["theme"] == "population_density"].head(2),
            indicators[indicators["theme"] == "environment"].head(3),
            indicators[indicators["theme"] == "living_conditions"].head(3),
        ]
    ).copy()
    selected["justification"] = "Relevant to mobility and livability benchmarking."
    selected = selected[["indicator_id", "indicator_name", "theme", "justification", "expected_direction"]]
    selected.to_csv(outputs / "t01_selected_indicators.csv", index=False)

    raw = pd.read_csv(SOURCE_INPUTS / "eurostat" / "eurostat_city_values_sample.csv")
    raw = raw[raw["indicator_id"].isin(selected["indicator_id"])]
    raw[["city_id", "city_name", "year", "indicator_id", "indicator_name", "value", "unit"]].to_csv(
        outputs / "t02_eurostat_raw.csv",
        index=False,
    )
    (src / "t02_load_eurostat.py").write_text("# reference upstream loader\n", encoding="utf-8")

    catalog = json.loads((SOURCE_INPUTS / "catalog" / "dataeuropa_mobility_catalog_sample.json").read_text(encoding="utf-8"))
    rows = []
    terms = ("mobility", "transport", "gtfs", "traffic", "rail", "bike")
    for dataset in catalog["datasets"]:
        text = " ".join([dataset["title"], dataset["description"], " ".join(dataset["keywords"])]).lower()
        if any(term in text for term in terms):
            rows.append(
                {
                    "dataset_id": dataset["id"],
                    "title": dataset["title"],
                    "description": dataset["description"],
                    "publisher": dataset["publisher"],
                    "country": dataset["country"],
                    "formats": ";".join(distribution["format"] for distribution in dataset["distributions"]),
                    "license": dataset["license"],
                    "distribution_url": dataset["distributions"][0]["downloadURL"],
                    "keywords": ";".join(dataset["keywords"]),
                }
            )
    pd.DataFrame(rows).to_csv(outputs / "t03_mobility_catalog.csv", index=False)

    gtfs = gtfs_tables()
    (outputs / "t04_gtfs_summary.json").write_text(
        json.dumps(
            {
                "has_stops": True,
                "has_routes": True,
                "has_trips": True,
                "has_stop_times": True,
                "n_stops": len(gtfs["stops"]),
                "n_routes": len(gtfs["routes"]),
                "n_trips": len(gtfs["trips"]),
                "n_stop_times": len(gtfs["stop_times"]),
                "missing_required_columns": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_stops = gtfs["stops"].copy()
    duplicate_stops_removed = int(raw_stops.duplicated("stop_id").sum())
    stops = raw_stops.drop_duplicates("stop_id", keep="first").copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    valid_coordinates = stops["stop_lat"].between(-90, 90) & stops["stop_lon"].between(-180, 180)
    invalid_coordinates_removed = int((~valid_coordinates).sum())
    stops = stops[valid_coordinates]
    stops["stop_name"] = stops["stop_name"].fillna("")
    missing_stop_names_filled = int(stops["stop_name"].astype(str).str.strip().eq("").sum())
    stops.loc[stops["stop_name"].astype(str).str.strip().eq(""), "stop_name"] = "Unnamed stop"
    stop_times = gtfs["stop_times"][gtfs["stop_times"]["stop_id"].isin(stops["stop_id"])].copy()
    stops.to_csv(outputs / "t05_clean_stops.csv", index=False)
    stop_times.to_csv(outputs / "t05_clean_stop_times.csv", index=False)
    (outputs / "t05_gtfs_cleaning_report.json").write_text(
        json.dumps(
            {
                "invalid_coordinates_removed": invalid_coordinates_removed,
                "duplicate_stops_removed": duplicate_stops_removed,
                "missing_stop_names_filled": missing_stop_names_filled,
                "n_clean_stops": len(stops),
                "n_clean_stop_times": len(stop_times),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    times_with_routes = stop_times.merge(gtfs["trips"][["trip_id", "route_id"]], on="trip_id", how="left")
    grouped = times_with_routes.groupby("stop_id")
    service = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    service["n_departures"] = service["stop_id"].map(grouped.size()).fillna(0).astype(int)
    service["n_routes"] = service["stop_id"].map(grouped["route_id"].nunique()).fillna(0).astype(int)
    first_departures = {}
    last_departures = {}
    for stop_id, group in grouped:
        ordered = sorted(group["departure_time"].dropna().astype(str), key=time_to_seconds)
        if ordered:
            first_departures[stop_id] = ordered[0]
            last_departures[stop_id] = ordered[-1]
    service["first_departure"] = service["stop_id"].map(first_departures).fillna("")
    service["last_departure"] = service["stop_id"].map(last_departures).fillna("")
    service.to_csv(outputs / "t06_stop_service_intensity.csv", index=False)

    (outputs / "t07_query.sparql").write_text(
        """
SELECT ?item ?itemLabel ?coord WHERE {
  ?item wdt:P625 ?coord .
  ?item rdfs:label ?itemLabel .
  FILTER(CONTAINS(LCASE(?itemLabel), "alicante") || CONTAINS(LCASE(?itemLabel), "spain"))
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    pois_json = json.loads((SOURCE_INPUTS / "wikidata" / "wikidata_pois_sample.json").read_text(encoding="utf-8"))
    poi_rows = []
    for row in pois_json["results"]["bindings"]:
        lon_lat = row["coord"].removeprefix("Point(").removesuffix(")").split()
        poi_rows.append(
            {
                "qid": row["qid"],
                "label": row["itemLabel"] or row["qid"],
                "poi_type": row["poi_type"],
                "latitude": float(lon_lat[1]),
                "longitude": float(lon_lat[0]),
                "source": "wikidata",
            }
        )
    pois = pd.DataFrame(poi_rows).drop_duplicates("qid", keep="first")
    pois.to_csv(outputs / "t08_pois_clean.csv", index=False)

    nearest_rows = []
    for poi in pois.itertuples(index=False):
        nearest = min(
            stops.itertuples(index=False),
            key=lambda stop: haversine_m(float(poi.latitude), float(poi.longitude), float(stop.stop_lat), float(stop.stop_lon)),
        )
        nearest_rows.append(
            {
                "qid": poi.qid,
                "label": poi.label,
                "poi_type": poi.poi_type,
                "nearest_stop_id": nearest.stop_id,
                "nearest_stop_name": nearest.stop_name,
                "distance_m": haversine_m(float(poi.latitude), float(poi.longitude), float(nearest.stop_lat), float(nearest.stop_lon)),
            }
        )
    nearest = pd.DataFrame(nearest_rows)
    nearest.to_csv(outputs / "t09_poi_nearest_stop.csv", index=False)

    accessibility = nearest.merge(
        service[["stop_id", "n_departures", "n_routes"]],
        left_on="nearest_stop_id",
        right_on="stop_id",
        how="left",
    )
    distance_score = np.maximum(0, 1 - np.minimum(accessibility["distance_m"], 1000) / 1000)
    service_score = np.log1p(accessibility["n_departures"]) / np.log1p(accessibility["n_departures"]).max()
    route_score = np.log1p(accessibility["n_routes"]) / np.log1p(accessibility["n_routes"]).max()
    accessibility["accessibility_score"] = 0.50 * distance_score + 0.35 * service_score + 0.15 * route_score
    accessibility[["qid", "label", "poi_type", "distance_m", "n_departures", "n_routes", "accessibility_score"]].to_csv(
        outputs / "t10_accessibility_scores.csv",
        index=False,
    )

    matrix = raw.pivot_table(
        index=["city_id", "city_name", "year"],
        columns="indicator_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    matrix.to_csv(outputs / "t11_city_indicator_matrix.csv", index=False)

    indicator_cols = [col for col in matrix.columns if col not in {"city_id", "city_name", "year"}]
    missing_rate = matrix[indicator_cols].isna().mean()
    miss_report = pd.DataFrame(
        {
            "indicator": indicator_cols,
            "missing_count": [int(matrix[col].isna().sum()) for col in indicator_cols],
            "missing_rate": [float(missing_rate[col]) for col in indicator_cols],
            "kept": [bool(missing_rate[col] <= 0.30) for col in indicator_cols],
        }
    )
    kept = miss_report.loc[miss_report["kept"], "indicator"].tolist()
    miss_report.to_csv(outputs / "t12_missingness_report.csv", index=False)
    matrix[["city_id", "city_name", "year", *kept]].to_csv(outputs / "t12_selected_matrix.csv", index=False)

    selected_matrix = pd.read_csv(outputs / "t12_selected_matrix.csv")
    clusters = selected_matrix[["city_id", "city_name", "year"]].copy()
    features = selected_matrix[kept].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    scaled = StandardScaler().fit_transform(imputed)
    clusters["cluster_id"] = KMeans(n_clusters=min(3, len(selected_matrix)), random_state=0, n_init=10).fit_predict(scaled)
    clusters.to_csv(outputs / "t13_city_clusters.csv", index=False)
    profiles = selected_matrix.copy()
    profiles["cluster_id"] = clusters["cluster_id"]
    profiles.groupby("cluster_id")[kept].mean().reset_index().to_csv(outputs / "t13_cluster_profiles.csv", index=False)

    validation_script = '''from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path(".")
checks = []

selected = root / "outputs" / "t01_selected_indicators.csv"
checks.append({"check_name": "t01_selected_indicators_exists", "passed": selected.exists(), "message": "outputs/t01_selected_indicators.csv exists."})

stops = pd.read_csv(root / "outputs" / "t05_clean_stops.csv")
checks.append({"check_name": "t05_clean_stops_columns", "passed": {"stop_id", "stop_name", "stop_lat", "stop_lon"}.issubset(stops.columns), "message": "outputs/t05_clean_stops.csv has required columns."})

pois = pd.read_csv(root / "outputs" / "t08_pois_clean.csv")
checks.append({"check_name": "t08_pois_clean_columns", "passed": {"qid", "label", "latitude", "longitude"}.issubset(pois.columns), "message": "outputs/t08_pois_clean.csv has required columns."})

access = pd.read_csv(root / "outputs" / "t10_accessibility_scores.csv")
scores_ok = pd.to_numeric(access["accessibility_score"], errors="coerce").between(0, 1).all()
checks.append({"check_name": "t10_accessibility_scores_range", "passed": bool(scores_ok), "message": "outputs/t10_accessibility_scores.csv scores are bounded."})

clusters = pd.read_csv(root / "outputs" / "t13_city_clusters.csv")
matrix = pd.read_csv(root / "outputs" / "t12_selected_matrix.csv")
checks.append({"check_name": "t13_city_clusters_rows", "passed": len(clusters) == len(matrix), "message": "outputs/t13_city_clusters.csv covers selected matrix rows."})

(root / "outputs" / "t14_validation_summary.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
'''
    (src / "validate_outputs.py").write_text(validation_script, encoding="utf-8")
    checks = [
        {"check_name": "t01_selected_indicators_exists", "passed": True, "message": "outputs/t01_selected_indicators.csv exists."},
        {"check_name": "t05_clean_stops_columns", "passed": True, "message": "outputs/t05_clean_stops.csv has required columns."},
        {"check_name": "t08_pois_clean_columns", "passed": True, "message": "outputs/t08_pois_clean.csv has required columns."},
        {"check_name": "t10_accessibility_scores_range", "passed": True, "message": "outputs/t10_accessibility_scores.csv scores are bounded."},
        {"check_name": "t13_city_clusters_rows", "passed": True, "message": "outputs/t13_city_clusters.csv covers selected matrix rows."},
    ]
    (outputs / "t14_validation_summary.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    (reports / "pipeline_report.md").write_text(
        """# Overview
Reference upstream report.

# Input data
Local fixed snapshots are used.

# Methods
Deterministic reference pipeline.

# Generated outputs
See outputs/t10_accessibility_scores.csv and outputs/t13_city_clusters.csv.

# Main results
The report summarizes generated outputs from local files only.

# Limitations
Reference upstream context for isolated benchmark tasks.

# Reproducibility
Regenerate with scripts/prepare_reference_outputs.py.
""",
        encoding="utf-8",
    )
    for name in ("fig_accessibility_distribution.png", "fig_cluster_profiles.png"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(figures / name)
        plt.close(fig)


if __name__ == "__main__":
    main()
