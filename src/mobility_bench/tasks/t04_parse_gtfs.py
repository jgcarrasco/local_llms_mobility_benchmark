"""T04 task definition."""

from mobility_bench.scoring.checkers import check_t04
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T04",
    title="Validate and parse a GTFS feed",
    family="gtfs_processing",
    input_files=["inputs/gtfs/sample_city_gtfs.zip"],
    expected_outputs=["outputs/t04_gtfs_summary.json"],
    prompt_template="Load a GTFS zip file, report which of the mandatory files are present, and count the rows in each.",
    output_requirements={
        "outputs/t04_gtfs_summary.json": [
            "JSON fields: has_stops, has_routes, has_trips, has_stop_times",
            "JSON fields: n_stops, n_routes, n_trips, n_stop_times",
        ],
    },
    checker=check_t04,
    timeout_seconds=600,
)
