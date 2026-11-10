from pathlib import Path

def test_bash_allows_regex_literals(tmp_path: Path) -> None:
    from mobility_bench.tools.bash import _command_policy_error
    from mobility_bench.runtime.workspace import Workspace

    workspace = Workspace(tmp_path)
    command = (
        "awk -F, '$3 ~ /mobility|population_density|environment/ {print $1}' "
        "inputs/eurostat/indicator_descriptions.csv"
    )
    assert _command_policy_error(workspace, command) is None


def test_bash_rejects_host_paths(tmp_path: Path) -> None:
    from mobility_bench.tools.bash import _command_policy_error
    from mobility_bench.runtime.workspace import Workspace

    workspace = Workspace(tmp_path)
    assert _command_policy_error(workspace, "cat /etc/passwd") is not None
    assert _command_policy_error(workspace, "ls /home/other") is not None
