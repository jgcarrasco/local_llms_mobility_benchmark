from pathlib import Path

from mobility_bench.runtime.runner import _copy_reference_inputs
from mobility_bench.tasks.registry import get_all_tasks


def test_task_prompts_include_output_requirements() -> None:
    tasks = get_all_tasks()
    assert len(tasks) == 15
    for task in tasks.values():
        prompt = task.render_prompt()
        assert "Output requirements:" in prompt
        assert task.output_requirements


def test_t01_prompt_includes_checker_schema() -> None:
    prompt = get_all_tasks()["T01"].render_prompt()
    assert "indicator_id, indicator_name, theme, justification, expected_direction" in prompt
    assert "Cover mobility, population_density, environment, and living_conditions themes." in prompt


def test_all_previous_outputs_do_not_seed_current_expected_outputs(tmp_path: Path) -> None:
    tasks = get_all_tasks()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _copy_reference_inputs(tasks["T15"], Path("data"), workspace)

    assert (workspace / "outputs" / "t14_validation_summary.json").exists()
    assert (workspace / "src" / "validate_outputs.py").exists()
    assert not (workspace / "reports" / "pipeline_report.md").exists()
    assert not (workspace / "figures" / "fig_accessibility_distribution.png").exists()
    assert not (workspace / "figures" / "fig_cluster_profiles.png").exists()


def test_all_previous_outputs_remove_current_task_outputs(tmp_path: Path) -> None:
    tasks = get_all_tasks()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _copy_reference_inputs(tasks["T14"], Path("data"), workspace)

    assert (workspace / "outputs" / "t13_city_clusters.csv").exists()
    assert not (workspace / "outputs" / "t14_validation_summary.json").exists()
    assert not (workspace / "src" / "validate_outputs.py").exists()
