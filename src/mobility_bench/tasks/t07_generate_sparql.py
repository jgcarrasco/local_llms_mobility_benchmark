"""T07 task definition."""

from mobility_bench.scoring.checkers import check_t07
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T07",
    title="Generate a Wikidata SPARQL query for cultural POIs",
    family="semantic_enrichment",
    input_files=["inputs/wikidata/city_config.json"],
    expected_outputs=["outputs/t07_query.sparql"],
    prompt_template="Generate a syntactically valid SPARQL query for cultural points of interest in the configured city or region.",
    output_requirements={
        "outputs/t07_query.sparql": [
            "SPARQL SELECT query requesting item/entity identifiers and labels.",
            "Include coordinate property p625.",
            "Constrain the query to the configured city or country using its identifier from the configuration file.",
            "The query must be syntactically valid SPARQL.",
        ],
    },
    checker=check_t07,
    timeout_seconds=600,
)
