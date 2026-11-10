"""Command-line interface for the mobility data engineering benchmark."""

from __future__ import annotations

import argparse
import json

from mobility_bench.analysis.aggregate_results import aggregate_results
from mobility_bench.analysis.paper_tables import make_paper_tables
from mobility_bench.analysis.plots import make_paper_figures
from mobility_bench.analysis.summary import make_benchmark_summary
from mobility_bench.analysis.sweep_manifest import create_sweep_manifest
from mobility_bench.config import load_benchmark_config, load_models_config
from mobility_bench.models import create_model_client
from mobility_bench.runtime.runner import result_path, run_task_attempt
from mobility_bench.tasks.registry import get_all_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mobility-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run selected benchmark tasks.")
    run.add_argument("--models", default="configs/models.yaml")
    run.add_argument("--benchmark", default="configs/benchmark.yaml")
    run.add_argument("--tasks", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--mode", choices=["chat_only", "chat_schema", "agent_tools"], required=True)
    run.add_argument("--repetitions", type=int, default=None)
    run.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip attempts whose result file already exists.",
    )

    run_all = subparsers.add_parser("run-all", help="Run the configured full benchmark.")
    run_all.add_argument("--models", default="configs/models.yaml")
    run_all.add_argument("--benchmark", default="configs/benchmark.yaml")
    run_all.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip attempts whose result file already exists.",
    )

    smoke = subparsers.add_parser(
        "smoke-model",
        help="Send one small request to a configured model backend.",
    )
    smoke.add_argument("--models", default="configs/models.yaml")
    smoke.add_argument("--model-id", required=True)
    smoke.add_argument("--prompt", default="Reply with exactly: ok")
    smoke.add_argument("--temperature", type=float, default=0.0)
    smoke.add_argument("--max-tokens", type=int, default=16)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate attempt results.")
    aggregate.add_argument("--results-dir", default="runs/results_jsonl")
    aggregate.add_argument("--output-dir", default="runs/aggregated")

    significance = subparsers.add_parser(
        "significance", help="Clustered significance analysis over tasks."
    )
    significance.add_argument("--attempts", default="runs/aggregated/attempts.csv")
    significance.add_argument("--output-dir", default="runs/aggregated/significance")
    significance.add_argument("--iters", type=int, default=10000)
    significance.add_argument("--seed", type=int, default=20261104)
    significance.add_argument("--top", type=int, default=4)

    manifest = subparsers.add_parser("create-sweep-manifest", help="Write a reproducibility manifest for a sweep.")
    manifest.add_argument("--models", default="configs/models.yaml")
    manifest.add_argument("--benchmark", default="configs/benchmark.yaml")
    manifest.add_argument("--output", default="runs/sweep_manifest.json")
    manifest.add_argument("--notes", default=None)

    paper = subparsers.add_parser(
        "make-paper-artifacts",
        help="Generate reports and paper-draft artifacts.",
    )
    paper.add_argument("--aggregated-dir", default="runs/aggregated")
    paper.add_argument("--reports-dir", default="runs/derived_artifacts")
    paper.add_argument("--paper-draft-dir", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        benchmark = load_benchmark_config(args.benchmark)
        models = load_models_config(args.models)
        if args.model_id not in models.models:
            parser.error(f"unknown model id: {args.model_id}")
        model_config = models.models[args.model_id]
        model_client = create_model_client(model_config)
        tasks = get_all_tasks()
        requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        repetitions = args.repetitions or benchmark.runtime.repetitions
        for task_id in requested_tasks:
            if task_id not in tasks:
                parser.error(f"unknown task id: {task_id}")
            for repetition in range(1, repetitions + 1):
                run_id = f"{args.model_id}__{args.mode}__{task_id}__rep{repetition:02d}"
                if args.skip_existing and result_path(benchmark, run_id).exists():
                    continue
                run_task_attempt(
                    task=tasks[task_id],
                    model_client=model_client,
                    model_config=model_config,
                    model_id=args.model_id,
                    mode=args.mode,
                    repetition=repetition,
                    benchmark=benchmark,
                )
        return 0
    if args.command == "run-all":
        benchmark = load_benchmark_config(args.benchmark)
        models = load_models_config(args.models)
        tasks = get_all_tasks()
        for model_id, model_config in models.models.items():
            if model_config.backend == "mock":
                continue
            model_client = create_model_client(model_config)
            for mode in ("chat_only", "chat_schema", "agent_tools"):
                for task in tasks.values():
                    for repetition in range(1, benchmark.runtime.repetitions + 1):
                        run_id = f"{model_id}__{mode}__{task.task_id}__rep{repetition:02d}"
                        if args.skip_existing and result_path(benchmark, run_id).exists():
                            continue
                        run_task_attempt(
                            task=task,
                            model_client=model_client,
                            model_config=model_config,
                            model_id=model_id,
                            mode=mode,
                            repetition=repetition,
                            benchmark=benchmark,
                        )
        return 0
    if args.command == "smoke-model":
        models = load_models_config(args.models)
        if args.model_id not in models.models:
            parser.error(f"unknown model id: {args.model_id}")
        model_client = create_model_client(models.models[args.model_id])
        response = model_client.generate(
            [{"role": "user", "content": args.prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                    "token_counts_estimated": response.token_counts_estimated,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "aggregate":
        aggregate_results(args.results_dir, args.output_dir)
        return 0
    if args.command == "significance":
        from mobility_bench.analysis.significance import run_significance

        paths = run_significance(
            args.attempts,
            args.output_dir,
            iters=args.iters,
            seed=args.seed,
            top=args.top,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0
    if args.command == "create-sweep-manifest":
        create_sweep_manifest(
            benchmark_config=args.benchmark,
            models_config=args.models,
            output_path=args.output,
            notes=args.notes,
        )
        return 0
    if args.command == "make-paper-artifacts":
        make_benchmark_summary(
            aggregated_dir=args.aggregated_dir,
            reports_dir=args.reports_dir,
        )
        make_paper_tables(
            aggregated_dir=args.aggregated_dir,
            reports_dir=args.reports_dir,
            paper_draft_dir=args.paper_draft_dir,
        )
        make_paper_figures(
            aggregated_dir=args.aggregated_dir,
            reports_dir=args.reports_dir,
            paper_draft_dir=args.paper_draft_dir,
        )
        return 0
    parser.error(f"command not implemented yet: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
