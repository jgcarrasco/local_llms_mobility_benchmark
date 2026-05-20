"""T01 task definition."""

from mobility_bench.scoring.checkers import check_t01
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T01",
    title="Select relevant Eurostat urban indicators",
    family="dataset_discovery",
    input_files=["inputs/eurostat/metadata_tables.csv", "inputs/eurostat/indicator_descriptions.csv"],
    expected_outputs=["outputs/t01_selected_indicators.csv"],
    prompt_template="Select 10-15 indicators relevant to urban mobility and livability.",
    output_requirements={
        "outputs/t01_selected_indicators.csv": [
            "CSV columns: indicator_id, indicator_name, theme, justification, expected_direction",
            "Select between 10 and 15 rows.",
            "Use indicator_id values from inputs/eurostat/indicator_descriptions.csv.",
            "Cover mobility, population_density, environment, and living_conditions themes.",
            "Fill every justification with a short reason.",
            "expected_direction values must be higher_is_better, lower_is_better, or contextual.",
        ],
    },
    checker=check_t01,
    timeout_seconds=600,
)
