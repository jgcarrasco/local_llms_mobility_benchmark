from __future__ import annotations

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
