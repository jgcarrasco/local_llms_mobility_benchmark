"""T14 task definition."""

from mobility_bench.scoring.checkers import check_t14
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T14",
    title="Generate validation script for pipeline outputs",
    family="validation_reporting",
    input_files=["all previous outputs"],
    expected_outputs=["src/validate_outputs.py", "outputs/t14_validation_summary.json"],
    prompt_template=(
        "Create and run a lightweight validation script for these upstream pipeline outputs: "
        "outputs/t01_selected_indicators.csv, outputs/t05_clean_stops.csv, outputs/t08_pois_clean.csv, "
        "outputs/t10_accessibility_scores.csv, and outputs/t13_city_clusters.csv."
    ),
    output_requirements={
        "src/validate_outputs.py": [
            "Executable Python validation script.",
            "When run from the workspace root as python src/validate_outputs.py, it must write outputs/t14_validation_summary.json.",
            "The script must report a failed check, rather than crash, when an expected output is missing or empty.",
        ],
        "outputs/t14_validation_summary.json": [
            "JSON list of validation entries.",
            "Each entry must contain check_name, passed, and message.",
            "Include at least five checks, with one or more checks for each listed output.",
            "All checks pass on the intact workspace.",
        ],
    },
    checker=check_t14,
    timeout_seconds=600,
)
