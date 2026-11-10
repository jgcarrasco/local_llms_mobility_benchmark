"""T09 task definition."""

from mobility_bench.scoring.checkers import check_t09
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T09",
    title="Link cultural POIs to nearest public-transport stops",
    family="semantic_enrichment",
    input_files=["outputs/t05_clean_stops.csv", "outputs/t08_pois_clean.csv"],
    expected_outputs=["outputs/t09_poi_nearest_stop.csv"],
    prompt_template=(
        "Match each cultural POI to the nearest cleaned GTFS stop using haversine distances in meters; when two "
        "stops are equidistant, use the lower stop_id."
    ),
    output_requirements={
        "outputs/t09_poi_nearest_stop.csv": [
            "CSV columns: qid, label, poi_type, nearest_stop_id, nearest_stop_name, distance_m",
            "One row per POI with non-negative meter distances computed from latitude/longitude using a haversine or equivalent geodesic formula.",
            "nearest_stop_id must exist in outputs/t05_clean_stops.csv.",
        ],
    },
    checker=check_t09,
    timeout_seconds=600,
)
