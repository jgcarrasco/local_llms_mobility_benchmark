"""T12 task definition."""

from mobility_bench.scoring.checkers import check_t12
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T12",
    title="Select robust indicators based on missingness",
    family="analytics",
    input_files=["outputs/t11_city_indicator_matrix.csv"],
    expected_outputs=["outputs/t12_missingness_report.csv", "outputs/t12_selected_matrix.csv"],
    prompt_template="Compute missingness per indicator and keep indicators with missing_rate <= 0.30.",
    output_requirements={
        "outputs/t12_missingness_report.csv": [
            "CSV columns: indicator, missing_count, missing_rate, kept",
            "missing_count must match the number of missing values for that indicator in outputs/t11_city_indicator_matrix.csv.",
            "missing_rate must match the fraction of missing values in outputs/t11_city_indicator_matrix.csv.",
        ],
        "outputs/t12_selected_matrix.csv": [
            "Keep city_id, city_name, year, and only indicator columns with missing_rate <= 0.30.",
        ],
    },
    checker=check_t12,
    timeout_seconds=600,
)
