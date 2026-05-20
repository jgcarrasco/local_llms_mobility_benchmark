"""T02 task definition."""

from mobility_bench.scoring.checkers import check_t02
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T02",
    title="Build Eurostat loader",
    family="dataset_discovery",
    input_files=[
        "outputs/t01_selected_indicators.csv",
        "inputs/eurostat/sample_api_config.json",
        "inputs/eurostat/eurostat_city_values_sample.csv",
    ],
    expected_outputs=["src/t02_load_eurostat.py", "outputs/t02_eurostat_raw.csv"],
    prompt_template="Generate a Python script that loads selected Eurostat-style indicators into normalized long format.",
    output_requirements={
        "src/t02_load_eurostat.py": ["Executable Python script that writes outputs/t02_eurostat_raw.csv."],
        "outputs/t02_eurostat_raw.csv": [
            "CSV columns: city_id, city_name, year, indicator_id, indicator_name, value, unit",
            "Use only indicators selected in outputs/t01_selected_indicators.csv.",
            "Include numeric value entries and avoid duplicate city_id/year/indicator_id rows.",
        ],
    },
    checker=check_t02,
    timeout_seconds=600,
)
