"""T10 task definition."""

from mobility_bench.scoring.checkers import check_t10
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T10",
    title="Compute cultural accessibility score",
    family="analytics",
    input_files=["outputs/t06_stop_service_intensity.csv", "outputs/t09_poi_nearest_stop.csv"],
    expected_outputs=["outputs/t10_accessibility_scores.csv"],
    prompt_template="Combine nearest-stop distance and service intensity into the required normalized accessibility score.",
    output_requirements={
        "outputs/t10_accessibility_scores.csv": [
            "CSV columns: qid, label, poi_type, distance_m, n_departures, n_routes, accessibility_score",
            "Include exactly one row per qid in outputs/t09_poi_nearest_stop.csv.",
            "distance_m must be taken from outputs/t09_poi_nearest_stop.csv, and n_departures and n_routes from outputs/t06_stop_service_intensity.csv for that nearest stop.",
            "accessibility_score must be numeric between 0 and 1.",
            "Formula: 0.50 * max(0, 1 - min(distance_m, 1000) / 1000) + 0.35 * log1p(n_departures) / max_log_departures + 0.15 * log1p(n_routes) / max_log_routes, where max_log_departures is the maximum of log1p(n_departures) and max_log_routes the maximum of log1p(n_routes) over the rows of this output file (not over the whole stop table).",
        ],
    },
    checker=check_t10,
    timeout_seconds=600,
)
