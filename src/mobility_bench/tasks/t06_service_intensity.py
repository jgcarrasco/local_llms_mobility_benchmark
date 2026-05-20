"""T06 task definition."""

from mobility_bench.scoring.checkers import check_t06
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T06",
    title="Compute stop-level service intensity",
    family="gtfs_processing",
    input_files=[
        "outputs/t05_clean_stops.csv",
        "outputs/t05_clean_stop_times.csv",
        "inputs/gtfs/sample_city_gtfs.zip",
    ],
    expected_outputs=["outputs/t06_stop_service_intensity.csv"],
    prompt_template="Calculate stop-level departure counts, route counts, and first/last departure times.",
    output_requirements={
        "outputs/t06_stop_service_intensity.csv": [
            "CSV columns: stop_id, stop_name, stop_lat, stop_lon, n_departures, n_routes, first_departure, last_departure",
            "One row per cleaned stop_id from outputs/t05_clean_stops.csv.",
            "n_departures counts clean stop_times rows per stop_id.",
            "n_routes counts distinct route_id values serving the stop by joining clean stop_times to trips.txt from the GTFS zip.",
            "first_departure and last_departure are the earliest/latest clean departure_time values for the stop.",
        ],
    },
    checker=check_t06,
    timeout_seconds=600,
)
