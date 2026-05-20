"""Task registry."""

from __future__ import annotations

from mobility_bench.tasks.base import TaskSpec
from mobility_bench.tasks.t01_select_eurostat_indicators import TASK_SPEC as T01
from mobility_bench.tasks.t02_build_eurostat_loader import TASK_SPEC as T02
from mobility_bench.tasks.t03_extract_catalog_metadata import TASK_SPEC as T03
from mobility_bench.tasks.t04_parse_gtfs import TASK_SPEC as T04
from mobility_bench.tasks.t05_clean_gtfs import TASK_SPEC as T05
from mobility_bench.tasks.t06_service_intensity import TASK_SPEC as T06
from mobility_bench.tasks.t07_generate_sparql import TASK_SPEC as T07
from mobility_bench.tasks.t08_clean_wikidata_pois import TASK_SPEC as T08
from mobility_bench.tasks.t09_link_pois_to_stops import TASK_SPEC as T09
from mobility_bench.tasks.t10_accessibility_score import TASK_SPEC as T10
from mobility_bench.tasks.t11_city_indicator_matrix import TASK_SPEC as T11
from mobility_bench.tasks.t12_missingness_selection import TASK_SPEC as T12
from mobility_bench.tasks.t13_cluster_cities import TASK_SPEC as T13
from mobility_bench.tasks.t14_validation_script import TASK_SPEC as T14
from mobility_bench.tasks.t15_reproducible_report import TASK_SPEC as T15


def get_all_tasks() -> dict[str, TaskSpec]:
    return {task.task_id: task for task in (T01, T02, T03, T04, T05, T06, T07, T08, T09, T10, T11, T12, T13, T14, T15)}
