"""T15 task definition."""

from mobility_bench.scoring.checkers import check_t15
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T15",
    title="Produce reproducible report and figures",
    family="validation_reporting",
    input_files=["all previous outputs"],
    expected_outputs=[
        "reports/pipeline_report.md",
        "figures/fig_accessibility_distribution.png",
        "figures/fig_cluster_profiles.png",
    ],
    prompt_template="Generate a reproducible Markdown report and two distinct non-empty figures without unsupported claims.",
    output_requirements={
        "reports/pipeline_report.md": [
            "Markdown sections: # Overview, # Input data, # Methods, # Generated outputs, # Main results, # Limitations, # Reproducibility",
            "Each section must contain at least one sentence (at least 80 characters of text).",
            "Reference the pipeline artifacts by their paths under outputs/.",
            "Do not claim live API access.",
        ],
        "figures/fig_accessibility_distribution.png": ["Non-empty PNG image of at least 400x300 pixels."],
        "figures/fig_cluster_profiles.png": [
            "Non-empty PNG image of at least 400x300 pixels, distinct from the other figure.",
        ],
    },
    checker=check_t15,
    timeout_seconds=600,
)
