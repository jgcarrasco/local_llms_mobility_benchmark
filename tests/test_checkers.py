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

from mobility_bench.tasks.registry import get_all_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_gtfs_tables(root: Path) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(root / "inputs" / "gtfs" / "sample_city_gtfs.zip") as zip_file:
        return {
            name: pd.read_csv(zip_file.open(f"{name}.txt"))
            for name in ("stops", "routes", "trips", "stop_times")
        }


def time_to_seconds(value: object) -> int:
    hour, minute, second = [int(part) for part in str(value).split(":")]
    return hour * 3600 + minute * 60 + second


def prepare_inputs(workspace: Path) -> None:
    shutil.copytree(REPO_ROOT / "data" / "inputs", workspace / "inputs")
    (workspace / "outputs").mkdir()
    (workspace / "src").mkdir()
    (workspace / "reports").mkdir()
    (workspace / "figures").mkdir()


def build_valid_outputs(workspace: Path) -> None:
    indicators = pd.read_csv(workspace / "inputs" / "eurostat" / "indicator_descriptions.csv")
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
    selected.to_csv(workspace / "outputs" / "t01_selected_indicators.csv", index=False)

    raw = pd.read_csv(workspace / "inputs" / "eurostat" / "eurostat_city_values_sample.csv")
    raw = raw[raw["indicator_id"].isin(selected["indicator_id"])]
    raw[["city_id", "city_name", "year", "indicator_id", "indicator_name", "value", "unit"]].to_csv(
        workspace / "outputs" / "t02_eurostat_raw.csv",
        index=False,
    )
    (workspace / "src" / "t02_load_eurostat.py").write_text(
        "import pandas as pd\n"
        "\n"
        "selected = pd.read_csv('outputs/t01_selected_indicators.csv')\n"
        "raw = pd.read_csv('inputs/eurostat/eurostat_city_values_sample.csv')\n"
        "raw = raw[raw['indicator_id'].isin(selected['indicator_id'])]\n"
        "columns = ['city_id', 'city_name', 'year', 'indicator_id', 'indicator_name', 'value', 'unit']\n"
        "raw[columns].to_csv('outputs/t02_eurostat_raw.csv', index=False)\n",
        encoding="utf-8",
    )

    catalog = json.loads((workspace / "inputs" / "catalog" / "dataeuropa_mobility_catalog_sample.json").read_text(encoding="utf-8"))
    rows = []
    terms = ("mobility", "transport", "gtfs", "traffic", "rail", "bike")
    for dataset in catalog["datasets"]:
        identity = f"{dataset['title']} {' '.join(dataset['keywords'])}".lower()
        if any(term in identity for term in terms):
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
    pd.DataFrame(rows).to_csv(workspace / "outputs" / "t03_mobility_catalog.csv", index=False)

    gtfs = load_gtfs_tables(workspace)
    (workspace / "outputs" / "t04_gtfs_summary.json").write_text(
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
            }
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
    stops.to_csv(workspace / "outputs" / "t05_clean_stops.csv", index=False)
    stop_times.to_csv(workspace / "outputs" / "t05_clean_stop_times.csv", index=False)
    (workspace / "outputs" / "t05_gtfs_cleaning_report.json").write_text(
        json.dumps(
            {
                "invalid_coordinates_removed": invalid_coordinates_removed,
                "duplicate_stops_removed": duplicate_stops_removed,
                "missing_stop_names_filled": missing_stop_names_filled,
                "n_clean_stops": len(stops),
                "n_clean_stop_times": len(stop_times),
            }
        ),
        encoding="utf-8",
    )

    trip_routes = gtfs["trips"][["trip_id", "route_id"]]
    times_with_routes = stop_times.merge(trip_routes, on="trip_id", how="left")
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
    service.to_csv(workspace / "outputs" / "t06_stop_service_intensity.csv", index=False)

    city_config = json.loads((workspace / "inputs" / "wikidata" / "city_config.json").read_text(encoding="utf-8"))
    (workspace / "outputs" / "t07_query.sparql").write_text(
        f"""
SELECT ?item ?itemLabel ?coord WHERE {{
  ?item wdt:P625 ?coord .
  ?item wdt:P131 wd:{city_config["city_qid"]} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
""",
        encoding="utf-8",
    )

    pois_json = json.loads((workspace / "inputs" / "wikidata" / "wikidata_pois_sample.json").read_text(encoding="utf-8"))
    poi_rows = []
    for row in pois_json["results"]["bindings"]:
        lon_lat = row["coord"].removeprefix("Point(").removesuffix(")").split()
        label = row["itemLabel"] or row["qid"]
        poi_rows.append(
            {
                "qid": row["qid"],
                "label": label,
                "poi_type": row["poi_type"],
                "latitude": float(lon_lat[1]),
                "longitude": float(lon_lat[0]),
                "source": "wikidata",
            }
        )
    pois = pd.DataFrame(poi_rows).drop_duplicates("qid", keep="first")
    pois.to_csv(workspace / "outputs" / "t08_pois_clean.csv", index=False)

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
    nearest.to_csv(workspace / "outputs" / "t09_poi_nearest_stop.csv", index=False)

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
        workspace / "outputs" / "t10_accessibility_scores.csv",
        index=False,
    )

    matrix = raw.pivot_table(
        index=["city_id", "city_name", "year"],
        columns="indicator_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    matrix.to_csv(workspace / "outputs" / "t11_city_indicator_matrix.csv", index=False)

    indicator_cols = [col for col in matrix.columns if col not in {"city_id", "city_name", "year"}]
    missing = matrix[indicator_cols].isna().sum()
    missing_rate = matrix[indicator_cols].isna().mean()
    miss_report = pd.DataFrame(
        {
            "indicator": indicator_cols,
            "missing_count": [int(missing[col]) for col in indicator_cols],
            "missing_rate": [float(missing_rate[col]) for col in indicator_cols],
            "kept": [bool(missing_rate[col] <= 0.30) for col in indicator_cols],
        }
    )
    kept = miss_report.loc[miss_report["kept"], "indicator"].tolist()
    miss_report.to_csv(workspace / "outputs" / "t12_missingness_report.csv", index=False)
    matrix[["city_id", "city_name", "year", *kept]].to_csv(workspace / "outputs" / "t12_selected_matrix.csv", index=False)

    selected_matrix = pd.read_csv(workspace / "outputs" / "t12_selected_matrix.csv")
    clusters = selected_matrix[["city_id", "city_name", "year"]].copy()
    features = selected_matrix[kept].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    scaled = StandardScaler().fit_transform(imputed)
    clusters["cluster_id"] = KMeans(n_clusters=min(3, len(selected_matrix)), random_state=0, n_init=10).fit_predict(scaled)
    clusters.to_csv(workspace / "outputs" / "t13_city_clusters.csv", index=False)
    profiles = selected_matrix.copy()
    profiles["cluster_id"] = clusters["cluster_id"]
    profiles.groupby("cluster_id")[kept].mean().reset_index().to_csv(workspace / "outputs" / "t13_cluster_profiles.csv", index=False)

    validation_script = '''from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path(".")
checks = []

outputs = {
    "t01_selected_indicators": root / "outputs" / "t01_selected_indicators.csv",
    "t05_clean_stops": root / "outputs" / "t05_clean_stops.csv",
    "t08_pois_clean": root / "outputs" / "t08_pois_clean.csv",
    "t10_accessibility_scores": root / "outputs" / "t10_accessibility_scores.csv",
    "t13_city_clusters": root / "outputs" / "t13_city_clusters.csv",
}

readable = {}
for name, path in outputs.items():
    if not path.exists():
        readable[name] = False
        checks.append({"check_name": f"{name}_present", "passed": False, "message": f"{path} is missing."})
    elif path.stat().st_size == 0:
        readable[name] = False
        checks.append({"check_name": f"{name}_present", "passed": False, "message": f"{path} is empty."})
    else:
        readable[name] = True
        checks.append({"check_name": f"{name}_present", "passed": True, "message": f"{path} is present and non-empty."})

if readable["t05_clean_stops"]:
    stops = pd.read_csv(outputs["t05_clean_stops"])
    checks.append({"check_name": "t05_clean_stops_columns", "passed": {"stop_id", "stop_name", "stop_lat", "stop_lon"}.issubset(stops.columns), "message": "outputs/t05_clean_stops.csv has required columns."})

if readable["t08_pois_clean"]:
    pois = pd.read_csv(outputs["t08_pois_clean"])
    checks.append({"check_name": "t08_pois_clean_columns", "passed": {"qid", "label", "latitude", "longitude"}.issubset(pois.columns), "message": "outputs/t08_pois_clean.csv has required columns."})

if readable["t10_accessibility_scores"]:
    access = pd.read_csv(outputs["t10_accessibility_scores"])
    scores_ok = pd.to_numeric(access["accessibility_score"], errors="coerce").between(0, 1).all()
    checks.append({"check_name": "t10_accessibility_scores_range", "passed": bool(scores_ok), "message": "outputs/t10_accessibility_scores.csv scores are bounded."})

if readable["t13_city_clusters"]:
    clusters = pd.read_csv(outputs["t13_city_clusters"])
    matrix = pd.read_csv(root / "outputs" / "t12_selected_matrix.csv")
    checks.append({"check_name": "t13_city_clusters_rows", "passed": len(clusters) == len(matrix), "message": "outputs/t13_city_clusters.csv covers selected matrix rows."})

(root / "outputs" / "t14_validation_summary.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
'''
    (workspace / "src" / "validate_outputs.py").write_text(validation_script, encoding="utf-8")
    checks = [
        {"check_name": "t01_selected_indicators_present", "passed": True, "message": "outputs/t01_selected_indicators.csv is present and non-empty."},
        {"check_name": "t05_clean_stops_present", "passed": True, "message": "outputs/t05_clean_stops.csv is present and non-empty."},
        {"check_name": "t08_pois_clean_present", "passed": True, "message": "outputs/t08_pois_clean.csv is present and non-empty."},
        {"check_name": "t10_accessibility_scores_present", "passed": True, "message": "outputs/t10_accessibility_scores.csv is present and non-empty."},
        {"check_name": "t13_city_clusters_present", "passed": True, "message": "outputs/t13_city_clusters.csv is present and non-empty."},
        {"check_name": "t05_clean_stops_columns", "passed": True, "message": "outputs/t05_clean_stops.csv has required columns."},
        {"check_name": "t08_pois_clean_columns", "passed": True, "message": "outputs/t08_pois_clean.csv has required columns."},
        {"check_name": "t10_accessibility_scores_range", "passed": True, "message": "outputs/t10_accessibility_scores.csv scores are bounded."},
        {"check_name": "t13_city_clusters_rows", "passed": True, "message": "outputs/t13_city_clusters.csv covers selected matrix rows."},
    ]
    (workspace / "outputs" / "t14_validation_summary.json").write_text(json.dumps(checks), encoding="utf-8")

    (workspace / "reports" / "pipeline_report.md").write_text(
        """# Overview
This fixture report documents the deterministic development pipeline used by the checker tests, covering its inputs, processing steps and generated artifacts.

# Input data
The pipeline reads the fixed local snapshots under data/inputs: Eurostat indicators, a catalog sample, a GTFS feed and Wikidata points of interest. No network access is used.

# Methods
Indicators are pivoted into a city-year matrix, GTFS stops are cleaned, points of interest are linked to the nearest stop, and the accessibility score combines proximity and service terms.

# Generated outputs
The pipeline writes outputs/t01_selected_indicators.csv, outputs/t05_clean_stops.csv, outputs/t08_pois_clean.csv, outputs/t10_accessibility_scores.csv and outputs/t13_city_clusters.csv.

# Main results
The accessibility table covers the linked points of interest and the clustering assigns the selected city-year rows to three groups, as recorded in the tables under outputs/.

# Limitations
This is a development fixture used by the checker tests rather than a production report, and it does not describe a full deployment.

# Reproducibility
Regenerate the fixture with scripts/generate_dev_fixtures.py, which reads only data/inputs and uses fixed seeds throughout.
""",
        encoding="utf-8",
    )
    for name, slope in (
        ("fig_accessibility_distribution.png", 1.0),
        ("fig_cluster_profiles.png", -1.0),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([0, 1], [0, slope])
        fig.savefig(workspace / "figures" / name, dpi=120)
        plt.close(fig)


def test_registry_contains_all_task_specs() -> None:
    tasks = get_all_tasks()
    assert list(tasks) == [f"T{i:02d}" for i in range(1, 16)]


def test_all_checkers_pass_known_valid_workspace(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    failures = {}
    for task_id, task in get_all_tasks().items():
        result = task.checker(tmp_path)
        if not result.success:
            failures[task_id] = result
    assert failures == {}


def test_t04_rejects_wrong_counts(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t04_gtfs_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["n_trips"] += 1
    path.write_text(json.dumps(summary), encoding="utf-8")

    result = get_all_tasks()["T04"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


def test_t15_allows_negated_live_api_statement(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    report = tmp_path / "reports" / "pipeline_report.md"
    report.write_text(
        report.read_text(encoding="utf-8")
        + "\nAll analyses are based on local files; no live API access is used.\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T15"].checker(tmp_path)

    assert result.success is True


def test_t08_rejects_invented_qids(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    pois = pd.read_csv(tmp_path / "outputs" / "t08_pois_clean.csv")
    pois.loc[0, "qid"] = "Q_NOT_IN_INPUT"
    pois.to_csv(tmp_path / "outputs" / "t08_pois_clean.csv", index=False)

    result = get_all_tasks()["T08"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


def test_t08_rejects_swapped_coordinates(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    pois = pd.read_csv(tmp_path / "outputs" / "t08_pois_clean.csv")
    pois["latitude"], pois["longitude"] = pois["longitude"], pois["latitude"]
    pois.to_csv(tmp_path / "outputs" / "t08_pois_clean.csv", index=False)

    result = get_all_tasks()["T08"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "coordinates do not match" in result.checker_errors[0]


def test_t09_rejects_stop_that_is_not_nearest(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t09_poi_nearest_stop.csv"
    output = pd.read_csv(path)
    stops = pd.read_csv(tmp_path / "outputs" / "t05_clean_stops.csv")
    current = output.loc[0, "nearest_stop_id"]
    output.loc[0, "nearest_stop_id"] = next(
        stop_id for stop_id in stops["stop_id"] if stop_id != current
    )
    output.to_csv(path, index=False)

    result = get_all_tasks()["T09"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "domain_error"


def test_t10_rejects_service_counts_that_do_not_match(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t10_accessibility_scores.csv"
    scores = pd.read_csv(path)
    scores["n_departures"] = scores["n_departures"] + 1
    scores.to_csv(path, index=False)

    result = get_all_tasks()["T10"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "n_departures does not match" in result.checker_errors[0]


def test_t10_rejects_distances_that_do_not_match(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t10_accessibility_scores.csv"
    scores = pd.read_csv(path)
    scores["distance_m"] = scores["distance_m"] + 25.0
    scores.to_csv(path, index=False)

    result = get_all_tasks()["T10"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "distance_m does not match" in result.checker_errors[0]


def test_t05_rejects_keeping_the_later_duplicate_row(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t05_clean_stops.csv"
    stops = pd.read_csv(path)
    stops.loc[stops["stop_id"].astype(str) == "125101209", "stop_name"] = "Duplicate stop id control"
    stops.to_csv(path, index=False)

    result = get_all_tasks()["T05"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "first occurrence" in result.checker_errors[0]


def test_t06_rejects_stop_attributes_that_do_not_match(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t06_stop_service_intensity.csv"
    service = pd.read_csv(path)
    service.loc[0, "stop_name"] = "Renamed stop"
    service.to_csv(path, index=False)

    result = get_all_tasks()["T06"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "stop_name does not match" in result.checker_errors[0]


def test_t12_rejects_missing_count_that_does_not_match(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t12_missingness_report.csv"
    report = pd.read_csv(path)
    report.loc[0, "missing_count"] = report.loc[0, "missing_count"] + 1
    report.to_csv(path, index=False)

    result = get_all_tasks()["T12"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "missing_count mismatch" in result.checker_errors[0]


def test_t11_rejects_matrix_without_values(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t11_city_indicator_matrix.csv"
    matrix = pd.read_csv(path)
    indicators = [column for column in matrix.columns if column not in {"city_id", "city_name", "year"}]
    matrix[indicators] = np.nan
    matrix.to_csv(path, index=False)

    result = get_all_tasks()["T11"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "do not match the long-format values" in result.checker_errors[0]


def test_t02_rejects_script_that_does_not_produce_the_output(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    (tmp_path / "src" / "t02_load_eurostat.py").write_text("print('done')\n", encoding="utf-8")

    result = get_all_tasks()["T02"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "schema_error"


def test_t02_rejects_script_that_fails(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    (tmp_path / "src" / "t02_load_eurostat.py").write_text("raise SystemExit('boom')\n", encoding="utf-8")

    result = get_all_tasks()["T02"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "runtime_error"


def test_t07_accepts_configured_wikidata_qids(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    city_config = json.loads((tmp_path / "inputs" / "wikidata" / "city_config.json").read_text(encoding="utf-8"))
    city_qid = city_config["city_qid"]
    (tmp_path / "outputs" / "t07_query.sparql").write_text(
        f"""
SELECT ?item ?itemLabel ?coords WHERE {{
  ?item wdt:P31/wdt:P279* wd:Q570119 .
  ?item wdt:P131/wdt:P131* wd:{city_qid} .
  ?item p:P625 ?coordsNode .
  ?coordsNode ps:P625 ?coords .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
""",
        encoding="utf-8",
    )

    result = get_all_tasks()["T07"].checker(tmp_path)

    assert result.success is True


def test_t07_rejects_unparseable_query(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    (tmp_path / "outputs" / "t07_query.sparql").write_text(
        "SELECT ?item ?itemLabel ?coord WHERE { ?item wdt:P625 ?coord . FILTER( }\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T07"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "format_error"


def test_t07_rejects_construct_with_keywords_in_comment(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    (tmp_path / "outputs" / "t07_query.sparql").write_text(
        "# SELECT ?item ?itemLabel wdt:P625 rdfs:label wikibase:label Q11959\n"
        "CONSTRUCT { ?item ?p ?o } WHERE { ?item ?p ?o }\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T07"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


def test_t07_rejects_query_that_does_not_project_the_item(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    city_qid = json.loads((tmp_path / "inputs" / "wikidata" / "city_config.json").read_text(encoding="utf-8"))[
        "city_qid"
    ]
    (tmp_path / "outputs" / "t07_query.sparql").write_text(
        f"""
SELECT ?coord WHERE {{
  ?item wdt:P625 ?coord .
  ?item wdt:P131 wd:{city_qid} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
""",
        encoding="utf-8",
    )

    result = get_all_tasks()["T07"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


def test_t07_rejects_projected_but_unbound_label(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    city_qid = json.loads((tmp_path / "inputs" / "wikidata" / "city_config.json").read_text(encoding="utf-8"))[
        "city_qid"
    ]
    (tmp_path / "outputs" / "t07_query.sparql").write_text(
        f"""
SELECT ?item ?label ?coord WHERE {{
  ?item wdt:P625 ?coord .
  ?item wdt:P131 wd:{city_qid} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
""",
        encoding="utf-8",
    )

    result = get_all_tasks()["T07"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


def test_t07_rejects_constraint_that_only_appears_in_a_comment(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    city_qid = json.loads((tmp_path / "inputs" / "wikidata" / "city_config.json").read_text(encoding="utf-8"))[
        "city_qid"
    ]
    (tmp_path / "outputs" / "t07_query.sparql").write_text(
        f"""
SELECT ?item ?itemLabel ?coord WHERE {{
  # located in wd:{city_qid}
  ?item wdt:P625 ?coord .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
""",
        encoding="utf-8",
    )

    result = get_all_tasks()["T07"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "domain_error"


def test_t05_rejects_incorrect_cleaning_report_counts(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    report = json.loads((tmp_path / "outputs" / "t05_gtfs_cleaning_report.json").read_text(encoding="utf-8"))
    report["invalid_coordinates_removed"] = 0
    (tmp_path / "outputs" / "t05_gtfs_cleaning_report.json").write_text(json.dumps(report), encoding="utf-8")

    result = get_all_tasks()["T05"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


def test_t13_rejects_non_kmeans_partition(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    matrix = pd.read_csv(tmp_path / "outputs" / "t12_selected_matrix.csv")
    clusters = matrix[["city_id", "city_name", "year"]].copy()
    clusters["cluster_id"] = [idx % 3 for idx in range(len(clusters))]
    clusters.to_csv(tmp_path / "outputs" / "t13_city_clusters.csv", index=False)

    result = get_all_tasks()["T13"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


def test_t14_rejects_superficial_validation_summary(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    (tmp_path / "src" / "validate_outputs.py").write_text(
        "from pathlib import Path\nPath('outputs/t14_validation_summary.json').write_text('[]', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T14"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"


T14_PATHS = (
    "outputs/t01_selected_indicators.csv",
    "outputs/t05_clean_stops.csv",
    "outputs/t08_pois_clean.csv",
    "outputs/t10_accessibility_scores.csv",
    "outputs/t13_city_clusters.csv",
)


def test_t14_rejects_validator_that_always_passes(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    (tmp_path / "src" / "validate_outputs.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "checks = [{'check_name': f'check_{i}', 'passed': True, 'message': 'ok'} for i in range(5)]\n"
        "Path('outputs/t14_validation_summary.json').write_text(json.dumps(checks), encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T14"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "did not report a failure" in result.checker_errors[0]


def test_t14_rejects_validator_that_crashes_on_missing_output(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    (tmp_path / "src" / "validate_outputs.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        f"paths = {list(T14_PATHS)!r}\n"
        "checks = []\n"
        "for path in paths:\n"
        "    pd.read_csv(path)\n"
        "    checks.append({'check_name': path, 'passed': True, 'message': f'{path} is readable.'})\n"
        "Path('outputs/t14_validation_summary.json').write_text(json.dumps(checks), encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T14"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "crashed" in result.checker_errors[0]


def test_t14_rejects_validator_that_ignores_empty_outputs(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    (tmp_path / "src" / "validate_outputs.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        f"paths = {list(T14_PATHS)!r}\n"
        "checks = [{'check_name': path, 'passed': Path(path).exists(), 'message': path} for path in paths]\n"
        "Path('outputs/t14_validation_summary.json').write_text(json.dumps(checks), encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T14"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "was empty" in result.checker_errors[0]


def test_t15_rejects_positive_live_api_claim(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    report = tmp_path / "reports" / "pipeline_report.md"
    report.write_text(
        report.read_text(encoding="utf-8")
        + "\nThe report queried Wikidata live to refresh POI labels.\n",
        encoding="utf-8",
    )

    result = get_all_tasks()["T15"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "hallucinated_claim"


def test_checker_fails_known_invalid_workspace(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    (tmp_path / "outputs" / "t01_selected_indicators.csv").write_text(
        "indicator_id,indicator_name,theme,justification,expected_direction\n"
        "NOT_REAL,Made up,mobility,,higher_is_better\n",
        encoding="utf-8",
    )
    result = get_all_tasks()["T01"].checker(tmp_path)
    assert result.success is False
    assert result.failure_category in {"logic_error", "wrong_data_source"}


def test_t15_rejects_empty_sections(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    report = tmp_path / "reports" / "pipeline_report.md"
    head, tail = report.read_text(encoding="utf-8").split("# Limitations", 1)
    _, rest = tail.split("# Reproducibility", 1)
    report.write_text(head + "# Limitations\n\n# Reproducibility" + rest, encoding="utf-8")

    result = get_all_tasks()["T15"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "without content" in result.checker_errors[0]


def test_t15_rejects_placeholder_text(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    report = tmp_path / "reports" / "pipeline_report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\nTODO: fill this in.\n", encoding="utf-8")

    result = get_all_tasks()["T15"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "placeholder" in result.checker_errors[0]


def test_t15_rejects_identical_figures(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    figures = tmp_path / "figures"
    shutil.copy2(figures / "fig_accessibility_distribution.png", figures / "fig_cluster_profiles.png")

    result = get_all_tasks()["T15"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "identical" in result.checker_errors[0]


def test_t15_rejects_tiny_figures(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    fig, ax = plt.subplots(figsize=(0.5, 0.5))
    ax.plot([0, 1], [0, 1])
    fig.savefig(tmp_path / "figures" / "fig_cluster_profiles.png", dpi=20)
    plt.close(fig)

    result = get_all_tasks()["T15"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "schema_error"
    assert "minimum is" in result.checker_errors[0]


def test_t03_rejects_generic_portal_records(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t03_mobility_catalog.csv"
    frame = pd.read_csv(path)
    catalog = json.loads((tmp_path / "inputs" / "catalog" / "dataeuropa_mobility_catalog_sample.json").read_text())
    portal = next(record for record in catalog["datasets"] if record["id"] == "beopen")
    extra = pd.DataFrame(
        [
            {
                "dataset_id": portal["id"],
                "title": portal["title"],
                "description": portal["description"],
                "publisher": portal["publisher"],
                "country": portal["country"],
                "formats": "",
                "license": portal["license"],
                "distribution_url": "",
                "keywords": "",
            }
        ]
    )
    pd.concat([frame, extra], ignore_index=True).to_csv(path, index=False)

    result = get_all_tasks()["T03"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "not mobility datasets" in result.checker_errors[0]


def test_t03_rejects_relabelled_record(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t03_mobility_catalog.csv"
    frame = pd.read_csv(path)
    catalog = json.loads((tmp_path / "inputs" / "catalog" / "dataeuropa_mobility_catalog_sample.json").read_text())
    portal = next(record for record in catalog["datasets"] if record["id"] == "beopen")
    frame.loc[0, "dataset_id"] = portal["id"]

    frame.to_csv(path, index=False)

    result = get_all_tasks()["T03"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "does not match the catalog record" in result.checker_errors[0]


def test_t01_reports_unquoted_comma_as_format_error(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t01_selected_indicators.csv"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("TT1003V,Share of journeys by public transport, rail, bus,mobility,Reason,higher_is_better\n")

    result = get_all_tasks()["T01"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "format_error"
    assert "5-column header" in result.checker_errors[0]


def test_t01_accepts_quoted_commas(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t01_selected_indicators.csv"
    frame = pd.read_csv(path)
    frame["justification"] = "Relevant to mobility, livability, and access."
    frame.to_csv(path, index=False)

    result = get_all_tasks()["T01"].checker(tmp_path)

    assert result.success is True


def test_t01_rejects_duplicate_indicators(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    build_valid_outputs(tmp_path)
    path = tmp_path / "outputs" / "t01_selected_indicators.csv"
    frame = pd.read_csv(path)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(path, index=False)

    result = get_all_tasks()["T01"].checker(tmp_path)

    assert result.success is False
    assert result.failure_category == "logic_error"
    assert "duplicate" in result.checker_errors[0].lower()

def test_t12_accepts_yes_no_kept_flags(tmp_path: Path) -> None:
    from mobility_bench.scoring.checkers import check_t12

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "t11_city_indicator_matrix.csv").write_text(
        "city_id,city_name,year,A,B\nC1,City,2020,1,1\nC2,City,2021,,2\nC3,City,2022,,3\nC4,City,2023,4,\n",
        encoding="utf-8",
    )
    (outputs / "t12_missingness_report.csv").write_text(
        "indicator,missing_count,missing_rate,kept\nA,2,0.5,no\nB,1,0.25,yes\n",
        encoding="utf-8",
    )
    (outputs / "t12_selected_matrix.csv").write_text(
        "city_id,city_name,year,B\nC1,City,2020,1\nC2,City,2021,2\nC3,City,2022,3\nC4,City,2023,\n",
        encoding="utf-8",
    )
    result = check_t12(tmp_path)
    assert result.success is True

def test_t07_accepts_values_constraint(tmp_path: Path) -> None:
    import shutil

    from mobility_bench.scoring.checkers import check_t07

    (tmp_path / "outputs").mkdir()
    (tmp_path / "inputs" / "wikidata").mkdir(parents=True)
    shutil.copy(
        "data/inputs/wikidata/city_config.json",
        tmp_path / "inputs" / "wikidata" / "city_config.json",
    )
    (tmp_path / "outputs" / "t07_query.sparql").write_text(
        """PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
SELECT ?item ?itemLabel ?coord WHERE {
  VALUES ?city { wd:Q11959 }
  ?item wdt:P131 ?city .
  ?item wdt:P625 ?coord .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
        encoding="utf-8",
    )
    assert check_t07(tmp_path).success is True

def test_safe_int_handles_missing_values() -> None:
    from mobility_bench.scoring.checkers import _safe_int

    assert _safe_int(float("nan")) == -1
    assert _safe_int(None) == -1
    assert _safe_int("not a number") == -1
    assert _safe_int(5) == 5
    assert _safe_int(5.0) == 5
    assert _safe_int("7") == 7


def test_t15_live_api_rule_accepts_denials() -> None:
    from mobility_bench.scoring.checkers import _claims_live_api_access

    denials = [
        "No live APIs were queried, so all figures reflect the local snapshot files.",
        "No network calls were made and no live APIs were queried at any point.",
        "The pipeline does not access live APIs and relies on pre-existing data files.",
        "All processing steps are deterministic and do not rely on external live APIs.",
        "Additionally, the analysis does not claim access to live APIs or external sources.",
        "No network\nor live API access is performed at any step of the pipeline.",
        "No live API access was used at any stage.",
        "The pipeline never queried Wikidata; it read the local snapshot.",
    ]
    for text in denials:
        assert _claims_live_api_access(text) is False, text


def test_t15_live_api_rule_rejects_claims() -> None:
    from mobility_bench.scoring.checkers import _claims_live_api_access

    claims = [
        "We used live APIs from Eurostat to obtain the indicator values.",
        "The script queried Wikidata for the points of interest.",
        "Live API calls were made during the run.",
        "The pipeline performs live API requests for each city.",
        "Live APIs were used. No errors were recorded.",
    ]
    for text in claims:
        assert _claims_live_api_access(text) is True, text
