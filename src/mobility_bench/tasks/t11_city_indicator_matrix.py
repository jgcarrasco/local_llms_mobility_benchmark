"""T11 task definition."""

from mobility_bench.scoring.checkers import check_t11
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T11",
    title="Create city-level Eurostat analytical matrix",
    family="analytics",
    input_files=["outputs/t02_eurostat_raw.csv"],
    expected_outputs=["outputs/t11_city_indicator_matrix.csv"],
    prompt_template="Pivot Eurostat long-format indicator data into a city-year by indicator matrix.",
    output_requirements={
        "outputs/t11_city_indicator_matrix.csv": [
            "CSV identifier columns: city_id, city_name, year",
            "Include one column for every indicator_id in outputs/t02_eurostat_raw.csv.",
            "Use one row per city_id/year pair.",
        ],
    },
    checker=check_t11,
    timeout_seconds=600,
)
