# Benchmarking Local Language Model Agents for Reproducible Mobility Data Engineering

This repository contains the benchmark code, fixed data fixtures, and result files for the paper "Benchmarking Local Language Model Agents for Reproducible Mobility Data Engineering".

The benchmark evaluates local language models on fifteen mobility data-engineering tasks. Each task asks a model to produce concrete artifacts such as CSV files, JSON summaries, Python scripts, SPARQL queries, validation reports, figures, or Markdown reports. Outputs are scored by deterministic checkers rather than by free-form textual judgment.

## Repository Structure

- `src/mobility_bench/`: benchmark package, command-line interface, model clients, task definitions, tool loop, runtime isolation, scoring, and analysis code.
- `configs/`: benchmark, model, and task configuration files.
- `data/inputs/`: fixed public-data excerpts used as benchmark inputs.
- `data/reference_outputs/`: deterministic reference artifacts used by task setup and checkers.
- `data/real_fixture_provenance.json`: source provenance and controlled fixture modifications.
- `runs/full_sweep/results_jsonl/`: one JSONL result summary per scored attempt.
- `runs/full_sweep/raw_logs/`: event traces for each attempt, including prompts, model responses, tool calls, tool outputs, and checker events.
- `runs/full_sweep/aggregated/`: canonical aggregate CSV/JSON files used for the paper results.
- `tests/`: unit tests for the benchmark harness, task definitions, scoring logic, and analysis utilities.

## Setup

The project uses Python 3.10+ and `uv`.

```bash
uv sync --extra test
```

Run the test suite:

```bash
uv run --extra test pytest
```

## Running The Benchmark

The CLI entry point is `mobility-bench`.

A small mock run can be used to check the harness without a local model server:

```bash
uv run mobility-bench run \
  --model-id mock \
  --mode agent_tools \
  --tasks T01 \
  --repetitions 1
```

The full sweep configuration expects an OpenAI-compatible local model endpoint, such as a `llama.cpp` server, at `http://127.0.0.1:8081/v1`. The `model_name` served by the endpoint must match the selected model entry in `configs/models_full_sweep_available.yaml`.

Smoke-test one configured model:

```bash
uv run mobility-bench smoke-model \
  --models configs/models_full_sweep_available.yaml \
  --model-id qwen36_27b_q4
```

Run all fifteen tasks for one model in each interaction mode:

```bash
uv run mobility-bench run \
  --models configs/models_full_sweep_available.yaml \
  --benchmark configs/benchmark_full_sweep.yaml \
  --model-id qwen36_27b_q4 \
  --mode chat_only \
  --tasks T01,T02,T03,T04,T05,T06,T07,T08,T09,T10,T11,T12,T13,T14,T15

uv run mobility-bench run \
  --models configs/models_full_sweep_available.yaml \
  --benchmark configs/benchmark_full_sweep.yaml \
  --model-id qwen36_27b_q4 \
  --mode agent_tools \
  --tasks T01,T02,T03,T04,T05,T06,T07,T08,T09,T10,T11,T12,T13,T14,T15
```

Aggregate result summaries:

```bash
uv run mobility-bench aggregate \
  --results-dir runs/full_sweep/results_jsonl \
  --output-dir runs/full_sweep/aggregated
```

## Included Result Files

The paper results are stored in `runs/full_sweep/aggregated/`. The most important files are:

- `attempts.csv`: all 1500 scored attempts, with task, model, mode, repetition, success, failure category, runtime, token counts, and tool-call counts.
- `pass_rate_by_model_mode.csv`: pass rates by model and interaction mode.
- `agentic_gain.csv`: task-level difference between agent-with-tools and chat-only pass rates.
- `pass_rate_by_task.csv`: pass rates by task and mode.
- `efficiency_metrics.csv`: runtime and tool-use summaries.
- `failure_categories.csv`: failure-category counts.

Per-attempt JSONL summaries are stored in `runs/full_sweep/results_jsonl/` for readers who want to inspect individual attempts.

Full event traces are stored in `runs/full_sweep/raw_logs/`. Each filename follows:

```text
MODEL_ID__MODE__TASK_ID__repNN.jsonl
```

For example, the trace for `qwen36_27b_q4` solving `T15` in `agent_tools` mode on repetition 2 is:

```text
runs/full_sweep/raw_logs/qwen36_27b_q4__agent_tools__T15__rep02.jsonl
```

These logs are the best place to inspect how a model approached a task: they record the task prompt, each model response, each tool call, shell output, files written through the tool interface, and the final checker result. Full per-attempt workspaces are not included because they are substantially larger; the raw logs, fixed fixtures, and reference outputs provide the auditable trace needed to follow each attempt.

## Result Summary

The full sweep contains 10 model configurations, 15 tasks, 2 interaction modes, and 5 repetitions per condition, for 1500 scored attempts.

In the aggregate results, tool access substantially improves benchmark success. The best overall configuration is `qwen36_27b_q4` in `agent_tools` mode, with 64 successful attempts out of 75 and an 85.3% pass rate. The same model reaches 44.0% in `chat_only` mode. The compact `qwen35_9b_q4` model reaches 69.3% in `agent_tools` mode and 17.3% in `chat_only` mode.

Across all attempts, the largest unsuccessful categories are logic errors and runtime errors. This reflects the benchmark design: many failures produce parseable artifacts or executable code, but violate task-specific data-engineering rules.

## Data Fixtures

The benchmark uses fixed local snapshots rather than live API calls during evaluation. The fixtures cover Eurostat-style city indicators, data.europa.eu-style catalog records, a sampled GTFS feed, and Wikidata-style point-of-interest records. Source metadata and controlled fixture modifications are documented in `data/real_fixture_provenance.json`.

## Execution Safety

The benchmark may execute model-generated code. The paper sweep was run with the local backend defined in `configs/benchmark_full_sweep.yaml`, with network access disabled by the harness configuration.
