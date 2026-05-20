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
    prompt_template="Generate a reproducible Markdown report and two non-empty figures without unsupported claims.",
    output_requirements={
        "reports/pipeline_report.md": [
            "Markdown sections: # Overview, # Input data, # Methods, # Generated outputs, # Main results, # Limitations, # Reproducibility",
            "Reference generated outputs and do not claim live API access.",
        ],
        "figures/fig_accessibility_distribution.png": ["Non-empty image file."],
        "figures/fig_cluster_profiles.png": ["Non-empty image file."],
    },
    checker=check_t15,
    timeout_seconds=600,
)
