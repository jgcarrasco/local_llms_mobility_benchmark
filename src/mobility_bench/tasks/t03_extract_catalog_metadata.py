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
            "Include only individual dataset rows whose title, description, or keywords are mobility-specific.",
            "Exclude generic catalog, publisher, organization, or portal records unless the record itself describes mobility data.",
            "Include at least one GTFS-like dataset.",
            "Populate distribution_url when available.",
        ],
    },
    checker=check_t03,
    timeout_seconds=600,
)
