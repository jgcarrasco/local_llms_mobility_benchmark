"""T05 task definition."""

from mobility_bench.scoring.checkers import check_t05
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T05",
    title="Clean GTFS coordinates and times",
    family="gtfs_processing",
    input_files=["inputs/gtfs/sample_city_gtfs.zip"],
    expected_outputs=[
        "outputs/t05_gtfs_cleaning_report.json",
        "outputs/t05_clean_stops.csv",
        "outputs/t05_clean_stop_times.csv",
    ],
    prompt_template=(
        "Remove stops whose latitude or longitude is outside the valid range, remove duplicate stop rows keeping "
        "the first occurrence, replace blank stop names with a non-empty placeholder, and preserve valid extended "
        "GTFS times."
    ),
    output_requirements={
        "outputs/t05_gtfs_cleaning_report.json": [
            "JSON object with integer fields: invalid_coordinates_removed, duplicate_stops_removed, missing_stop_names_filled, n_clean_stops, n_clean_stop_times.",
        ],
        "outputs/t05_clean_stops.csv": ["CSV columns: stop_id, stop_name, stop_lat, stop_lon"],
        "outputs/t05_clean_stop_times.csv": [
            "CSV columns: trip_id, arrival_time, departure_time, stop_id, stop_sequence",
            "Include exactly the stop_times rows whose stop_id remains in outputs/t05_clean_stops.csv.",
            "Preserve valid GTFS times beyond 24:00:00.",
        ],
    },
    checker=check_t05,
    timeout_seconds=600,
)
