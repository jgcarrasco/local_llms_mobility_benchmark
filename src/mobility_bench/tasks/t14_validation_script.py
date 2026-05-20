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
        "Create and run a lightweight validation script for key pipeline outputs, covering selected indicators, "
        "cleaned GTFS stops, cleaned POIs, accessibility scores, and city clusters."
    ),
    output_requirements={
        "src/validate_outputs.py": [
            "Executable Python validation script.",
            "When run from the workspace root as python src/validate_outputs.py, it must write outputs/t14_validation_summary.json.",
        ],
        "outputs/t14_validation_summary.json": [
            "JSON list of validation entries.",
            "Each entry must contain check_name, passed, and message.",
            "Include at least five checks, all with boolean passed=true.",
            "Validation messages or check names must identify the key outputs being checked.",
        ],
    },
    checker=check_t14,
    timeout_seconds=600,
)
