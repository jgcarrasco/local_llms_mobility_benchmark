from mobility_bench import __version__
from mobility_bench.cli import build_parser


def test_version_is_defined() -> None:
    assert __version__


def test_cli_parser_accepts_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--tasks",
            "T01",
            "--model-id",
            "mock",
            "--mode",
            "chat_only",
        ]
    )
    assert args.command == "run"
