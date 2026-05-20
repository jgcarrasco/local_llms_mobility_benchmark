from __future__ import annotations

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
