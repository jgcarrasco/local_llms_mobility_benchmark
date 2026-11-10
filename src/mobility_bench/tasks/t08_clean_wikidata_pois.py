"""T08 task definition."""

from mobility_bench.scoring.checkers import check_t08
from mobility_bench.tasks.base import TaskSpec

TASK_SPEC = TaskSpec(
    task_id="T08",
    title="Clean Wikidata POI table",
    family="semantic_enrichment",
    input_files=["inputs/wikidata/wikidata_pois_sample.json"],
    expected_outputs=["outputs/t08_pois_clean.csv"],
    prompt_template="Parse Wikidata-style coordinates and produce a clean POI table.",
    output_requirements={
        "outputs/t08_pois_clean.csv": [
            "CSV columns: qid, label, poi_type, latitude, longitude, source",
            "Latitude and longitude must be numeric and in valid ranges.",
            "Latitude and longitude must reproduce the coordinate of the input record; the coord value is a WKT Point written as longitude then latitude.",
            "Use one row per unique qid and set source to wikidata or equivalent.",
            "Every label must be non-empty; when itemLabel is missing, use the qid as a deterministic fallback label.",
        ],
    },
    checker=check_t08,
    timeout_seconds=600,
)
