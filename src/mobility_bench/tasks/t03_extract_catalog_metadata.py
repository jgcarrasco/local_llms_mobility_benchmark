"""T03 task definition."""

from mobility_bench.scoring.checkers import check_t03
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T03",
    title="Extract mobility catalog metadata",
    family="dataset_discovery",
    input_files=["inputs/catalog/dataeuropa_mobility_catalog_sample.json"],
    expected_outputs=["outputs/t03_mobility_catalog.csv"],
    prompt_template="Extract catalog datasets related to mobility, GTFS, public transport, or urban mobility.",
    output_requirements={
        "outputs/t03_mobility_catalog.csv": [
            "CSV columns: dataset_id, title, description, publisher, country, formats, license, distribution_url, keywords",
            "Include a record only when its title or keywords identify it as mobility data.",
            "Exclude generic catalog, publisher, organization, or portal records even when their description mentions mobility among other themes.",
            "Include at least one GTFS-like dataset.",
            "Copy title, description, and distribution_url from the catalog record without altering them.",
            "Populate distribution_url when available.",
        ],
    },
    checker=check_t03,
    timeout_seconds=600,
)
