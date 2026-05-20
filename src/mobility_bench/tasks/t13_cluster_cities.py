"""T13 task definition."""

from mobility_bench.scoring.checkers import check_t13
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T13",
    title="Cluster cities into mobility/livability profiles",
    family="analytics",
    input_files=["outputs/t12_selected_matrix.csv"],
    expected_outputs=["outputs/t13_city_clusters.csv", "outputs/t13_cluster_profiles.csv"],
    prompt_template=(
        "Cluster city-year rows using numeric indicators, median imputation, standard scaling, "
        "and KMeans with k=3, random_state=0, and n_init=10."
    ),
    output_requirements={
        "outputs/t13_city_clusters.csv": [
            "CSV columns: city_id, city_name, year, cluster_id",
            "Include every city-year row from outputs/t12_selected_matrix.csv.",
        ],
        "outputs/t13_cluster_profiles.csv": [
            "One row per cluster_id with mean profile values for each selected numeric indicator.",
        ],
    },
    checker=check_t13,
    timeout_seconds=600,
)
