# Data dictionary

Replication package for *Evaluating Local Language Model Agents for Reproducible Data Engineering: An Empirical Software Engineering Study of Mobility Workflows*, version 1.1.1. It describes every directory, file, column, and vocabulary in this repository, so that a value in the paper can be traced to the released data without reading the code.

## Contents at a glance

| Path | Contents |
|---|---|
| `src/mobility_bench/` | Benchmark package: tasks, conditions, tools, runtime, checkers, analysis. |
| `configs/` | Benchmark, model, and task configuration. |
| `data/inputs/` | Fixed public-data excerpts used as task inputs (Section 2 below). |
| `data/reference_outputs/` | Deterministic reference artifacts used by task setup and checkers. |
| `data/real_fixture_provenance.json` | Source and controlled modifications for each fixture. |
| `runs/full_sweep/` | The 3,600-attempt sweep reported in the paper (Section 3). |
| `runs/hosted_reference/` | The 45-attempt hosted calibration run (Section 4). |
| `scripts/` | Fixture preparation, figure generation, and release utilities. |
| `tests/` | Unit tests for tasks, harness, scoring, and analysis. |
| `DATA_DICTIONARY.md` | This file. |

## 1. Conditions and naming

Every attempt identifier follows `{model_id}__{mode}__{task_id}__rep{NN}`, for example `qwen35_9b_q4__agent_tools__T07__rep03`. The same identifier names the row in `attempts.csv`, the result file, and the trace file.

| Condition | Meaning |
|---|---|
| `chat_only` | Task prompt and required outputs; one response; the harness runs the submitted entry script once. |
| `chat_schema` | As `chat_only`, plus a deterministic summary of the declared input files. |
| `agent_tools` | Closed-loop workspace: read, write, edit, bash, submit, with results returned to the model. |

## 2. `data/inputs/`

Fixtures are fixed: the same snapshot is used for every model, condition, and repetition, and no task downloads data at run time. Row counts exclude the header.

| File | Rows | Columns |
|---|---|---|
| `data/inputs/catalog/dataeuropa_mobility_catalog_sample.json` | — | `source`, `source_url`, `datasets` |
| `data/inputs/eurostat/eurostat_city_values_sample.csv` | 5373 | `city_id`, `city_name`, `year`, `indicator_id`, `indicator_name`, `theme`, `value`, `unit` |
| `data/inputs/eurostat/indicator_descriptions.csv` | 160 | `indicator_id`, `indicator_name`, `theme`, `description`, `unit`, `expected_direction` |
| `data/inputs/eurostat/metadata_tables.csv` | 4 | `dataset_id`, `title`, `theme`, `description` |
| `data/inputs/eurostat/sample_api_config.json` | — | `source`, `base_path`, `id_column`, `value_column`, `snapshots`, `note` |
| `data/inputs/gtfs/sample_city_gtfs.zip` | — | `agency.txt` (52 rows), `stops.txt` (409 rows), `routes.txt` (25 rows), `trips.txt` (80 rows), `stop_times.txt` (1893 rows) |
| `data/inputs/wikidata/city_config.json` | — | `city_name`, `city_qid`, `country`, `country_qid`, `poi_types` |
| `data/inputs/wikidata/wikidata_pois_sample.json` | 49 | `results.bindings`: Wikidata SPARQL response |

## 3. `runs/full_sweep/`

The sweep contains 3,600 scored attempts: sixteen local configurations, three conditions, fifteen tasks, five repetitions. Each attempt has one result record and one trace (3,600 trace files in `raw_logs/`).

### 3.1 `results_jsonl/`

One file per attempt, named after the attempt identifier and containing a single JSON object with the attempt-level fields below. The same fields appear as columns of `aggregated/attempts.csv`.

| Field | Meaning |
|---|---|
| `run_id` | Attempt identifier, `{model_id}__{mode}__{task_id}__rep{NN}`. |
| `task_id` | Benchmark task, `T01`--`T15`. |
| `task_family` | Workflow family `A`--`E` (Table 1 of the paper). |
| `model_id` | Configuration key, matching `configs/models_full_sweep_available.yaml`. |
| `model_family` | Model family (`qwen3.5`, `qwen3.6`, `qwen3.8`, `gemma4`, `gptoss`, `llama3.1`, `mistral`, `phi4`, `deepseek`). |
| `model_size_b` | Nominal parameter count in billions. |
| `architecture` | `dense` or `moe`. |
| `quantization` | Weight format of the served model file (`Q4_K_M`, `UD-IQ4_NL`, `UD-Q4_K_M`, `BF16`, or `none` for hosted). |
| `context_length` | Configured context window in tokens. |
| `source_repo` | Upstream model repository or gateway. |
| `artifact_repo` | Path of the served model file, as recorded during the sweep. The home-directory prefix is rewritten in the public release (`local:/models/gguf/...`); the family directory and file name are unchanged. |
| `mode` | Interaction condition: `chat_only`, `chat_schema`, or `agent_tools`. |
| `repetition` | Repetition index, 1--5. |
| `success` | True when the final checker accepted the submitted artifact. |
| `partial_score` | Fraction of the checker's sub-requirements satisfied. |
| `failure_category` | Failure taxonomy label; empty on success. |
| `checker_errors` | List of checker messages for the attempt. |
| `time_to_submission_seconds` | Wall-clock seconds from attempt start to submission or termination. |
| `timeout_reached` | True when the per-task wall-clock limit was hit. |
| `stop_reason` | Why the attempt ended; see the `stop_reason` vocabulary. |
| `cap_exhausted` | True when an iteration or tool-call cap ended the attempt. |
| `n_model_calls` | Model requests issued. |
| `n_tool_calls_total` | Tool calls issued, all tools. |
| `n_read_calls` | `read` tool calls. |
| `n_write_calls` | `write` tool calls. |
| `n_edit_calls` | `edit` tool calls. |
| `n_bash_calls` | `bash` tool calls. |
| `n_submit_calls` | `submit` tool calls. |
| `n_failed_tool_calls` | Tool calls that returned an error. |
| `n_code_executions` | Shell commands that executed a script. |
| `n_error_messages_observed` | Error messages the model was shown in tool results. |
| `input_tokens_total` | Prompt tokens, summed over model calls. |
| `output_tokens_total` | Completion tokens, summed over model calls. |
| `total_tokens` | Sum of input and output tokens. |
| `token_counts_estimated` | True when the endpoint did not report usage and tokens were estimated. |
| `first_silent_checker_success` | Checker success recorded before submission, when the harness could evaluate silently. |
| `final_checker_success` | Checker success at the end of the attempt. |
| `infrastructure_failure` | True when the harness, not the model, caused the failure. |
| `notes` | Free-text harness annotations, such as `unsubmitted_checker_success`. |

### 3.2 `raw_logs/`

One JSONL file per attempt with the full event trace. Each line is one event with an `event_type` key and a `timestamp`.

| Event | Payload |
|---|---|
| `attempt_started` | Opens the trace: prompts, task, mode, and timestamp. |
| `model_request` | A request to the model endpoint (`n_messages`). |
| `model_response` | The completion: text, reasoning text when exposed, finish reason, token counts. |
| `tool_call` | A parsed tool call (`tool`, `args`) and the raw text it was parsed from. |
| `tool_result` | The result returned to the model: `ok`, `stdout`, `stderr`, `metadata`. |
| `final_checker_result` | The checker verdict: `success`, `partial_score`, `failure_category`, `checker_errors`. |
| `attempt_finished` | Closes the trace with the attempt summary (same fields as `attempts.csv`). |

### 3.3 `aggregated/`

Canonical aggregates used for the paper tables and figures. The attempt-level columns of `attempts.csv` are described in Section 3.1; the derived columns that appear in the summary files are listed below. Counts and rates are computed from `attempts.csv`; the `significance/` files apply the task-clustered bootstrap and the paired tests described in the paper.

| File | Columns | Rows | Contents |
|---|---|---|---|
| `agentic_gain.csv` | 8 | 240 | Pass rate by model and task for the three conditions, with the two gains. |
| `attempts.csv` | 39 | 3600 | One row per attempt: the attempt-level record, 39 columns. |
| `efficiency_metrics.csv` | 7 | 48 | Median runtime and tokens per model and condition, with the efficiency ratios. |
| `failure_categories.csv` | 2 | 12 | Failure taxonomy counts over all attempts. |
| `pass_rate_by_model_mode.csv` | 5 | 48 | Pass rate per model and condition. |
| `pass_rate_by_task.csv` | 4 | 15 | Pass rate per task, pooled over models and conditions. |
| `significance/gains_with_ci.csv` | 5 | 48 | Paired condition contrasts with intervals. |
| `significance/pairwise.csv` | 11 | 38 | Pairwise model comparisons with Holm-corrected p-values. |
| `significance/precision.csv` | 6 | 3 | Paired BF16-versus-4-bit contrasts with intervals. |
| `significance/rates_with_ci.csv` | 5 | 48 | Pass rates with task-clustered bootstrap intervals. |
| `significance/regression.csv` | 9 | 130 | Pooled regression of attempt success on condition. |
| `significance/reliability.csv` | 4 | 240 | pass^k for k = 1..5 per model and condition. |
| `significance/significance.json` | — | — | Bootstrap and test configuration for the files above. |
| `sweet_spot_summary.json` | — | — | Thresholds for the deployment sweet-spot search and its outcome. |

### 3.4 Operation logs

`sweep_log.txt` records the sweep schedule; `trunc_log.txt`, `stale_retry_log.txt`, and `prompt_retry_log.txt` record which attempts were repeated, and why; `rescore_log.txt` records the attempts whose checker verdict changed when the shared CSV reader was corrected. Attempts are only repeated when the harness, not the model, caused the termination, and the released set contains no `infrastructure_failure` attempt.

### 3.5 Columns of the summary files

| Column | Meaning |
|---|---|
| `agent_over_schema` | `agent_tools - chat_schema`: value of the loop beyond the input summary. |
| `agent_tools` | Pass rate in the closed-loop workspace condition. |
| `agentic_gain` | `agent_tools - chat_only`: value of the complete workspace condition. |
| `attempts` | Scored attempts in the cell. |
| `chat_only` | Pass rate in the uninformed one-shot condition. |
| `chat_schema` | Pass rate in the informed one-shot condition. |
| `ci_high` | Upper bound of the 95% task-clustered bootstrap interval. |
| `ci_low` | Lower bound of the 95% task-clustered bootstrap interval. |
| `coef` | Estimated coefficient (linear probability model unless `family` says otherwise). |
| `contrast` | Contrast identifier: `input_schema`, `interaction`, or `workspace`. |
| `count` | Number of attempts in the category. |
| `failure_category` | Failure taxonomy label, or `success` when counting all attempts. |
| `family` | Test family used for the Holm correction. |
| `gain` | Paired difference in pass rate, in [0, 1]. |
| `k` | Number of consecutive attempts in the reliability estimate. |
| `median_diff` | Median over tasks of the per-task difference. |
| `median_time_to_submission_seconds` | Median wall-clock seconds per attempt. |
| `median_total_tokens` | Median total tokens per attempt. |
| `mode` | Interaction condition. |
| `model_a` | First member of the pair (or of the precision contrast). |
| `model_b` | Second member of the pair (or of the precision contrast). |
| `model_id` | Configuration key. |
| `n_obs` | Attempts entering the regression. |
| `n_tasks` | Number of task clusters in the comparison. |
| `p_holm` | Holm-corrected p-value within its family. |
| `p_raw` | Uncorrected p-value. |
| `p_value` | Two-sided p-value for the coefficient. |
| `pass_rate` | `successful_attempts / attempts`, in [0, 1]. |
| `schema_gain` | `chat_schema - chat_only`: value of knowing the input structure. |
| `se` | Standard error, clustered on task. |
| `statistic` | Wilcoxon signed-rank statistic. |
| `successful_attempts` | Attempts whose artifact passed the checker. |
| `task_id` | Benchmark task. |
| `term` | Regression term. |
| `time_efficiency` | Pass rate per second of median wall-clock time. |
| `token_control` | True when the model controls for generated tokens. |
| `token_efficiency` | Pass rate per token. |
| `value` | Estimated probability of `k` consecutive successes. |

### 3.6 Vocabularies

| `failure_category` | Meaning |
|---|---|
| `runtime_error` | The generated program crashed. |
| `logic_error` | The program ran and produced a wrong result. |
| `schema_error` | The artifact violates its declared schema. |
| `format_error` | The file cannot be parsed. |
| `domain_error` | The artifact violates a domain rule checked by the task. |
| `wrong_data_source` | The artifact uses records the task excludes. |
| `protocol_error` | The model did not produce a parseable tool call. |
| `tool_error_loop` | Repeated failing tool calls. |
| `tool_loop` | Repeated identical tool calls. |
| `incomplete_solution` | Required outputs are missing. |
| `truncated` | The response ended before an answer was emitted. |
| `success` | Counts over all attempts include successful ones under this label. |

| `stop_reason` | Meaning |
|---|---|
| `submitted` | The model called `submit`. |
| `max_iterations` | The iteration cap ended the attempt. |
| `max_tool_calls` | The tool-call cap ended the attempt. |
| `protocol_stall` | Eight consecutive malformed responses. |
| `repeated_failed_tool_call` | Eight repeated failing tool calls. |
| `repeated_tool_call` | Twenty identical tool calls. |
| `truncated_response` | The completion hit the response budget. |
| `timeout` | The per-task wall-clock limit. |

## 4. `runs/hosted_reference/`

A calibration run of `deepseek_v41_flash` through a hosted API: 45 attempts, one repetition per task and condition, with the same directories and schema as the local sweep. It is not part of the comparison: a single repetition gives no reliability estimate, and the model does not fit on the benchmark hardware.

| File | Columns | Rows | Contents |
|---|---|---|---|
| `agentic_gain.csv` | 8 | 15 | Pass rate by model and task for the three conditions, with the two gains. |
| `attempts.csv` | 39 | 45 | One row per attempt: the attempt-level record, 39 columns. |
| `efficiency_metrics.csv` | 7 | 3 | Median runtime and tokens per model and condition, with the efficiency ratios. |
| `failure_categories.csv` | 2 | 2 | Failure taxonomy counts over all attempts. |
| `pass_rate_by_model_mode.csv` | 5 | 3 | Pass rate per model and condition. |
| `pass_rate_by_task.csv` | 4 | 15 | Pass rate per task, pooled over models and conditions. |
| `sweet_spot_summary.json` | — | — | Thresholds for the deployment sweet-spot search and its outcome. |

## 5. `data/reference_outputs/`

Reference artifacts used by task setup and by checkers that compare against a known good output.

| File | Size |
|---|---|
| `data/reference_outputs/figures/fig_accessibility_distribution.png` | 23,244 B |
| `data/reference_outputs/figures/fig_cluster_profiles.png` | 17,644 B |
| `data/reference_outputs/outputs/t01_selected_indicators.csv` | 1,664 B |
| `data/reference_outputs/outputs/t02_eurostat_raw.csv` | 319,235 B |
| `data/reference_outputs/outputs/t03_mobility_catalog.csv` | 413 B |
| `data/reference_outputs/outputs/t04_gtfs_summary.json` | 169 B |
| `data/reference_outputs/outputs/t05_clean_stop_times.csv` | 80,950 B |
| `data/reference_outputs/outputs/t05_clean_stops.csv` | 26,342 B |
| `data/reference_outputs/outputs/t05_gtfs_cleaning_report.json` | 158 B |
| `data/reference_outputs/outputs/t06_stop_service_intensity.csv` | 35,414 B |
| `data/reference_outputs/outputs/t07_query.sparql` | 165 B |
| `data/reference_outputs/outputs/t08_pois_clean.csv` | 3,849 B |
| `data/reference_outputs/outputs/t09_poi_nearest_stop.csv` | 5,156 B |
| `data/reference_outputs/outputs/t10_accessibility_scores.csv` | 4,432 B |
| `data/reference_outputs/outputs/t11_city_indicator_matrix.csv` | 77,027 B |
| `data/reference_outputs/outputs/t12_missingness_report.csv` | 342 B |
| `data/reference_outputs/outputs/t12_selected_matrix.csv` | 51,454 B |
| `data/reference_outputs/outputs/t13_city_clusters.csv` | 44,193 B |
| `data/reference_outputs/outputs/t13_cluster_profiles.csv` | 66 B |
| `data/reference_outputs/outputs/t14_validation_summary.json` | 1,347 B |
| `data/reference_outputs/reports/pipeline_report.md` | 1,935 B |
| `data/reference_outputs/src/t02_load_eurostat.py` | 373 B |
| `data/reference_outputs/src/validate_outputs.py` | 2,438 B |

## 6. Provenance

| Item | Value |
|---|---|
| Package version | 1.1.1 |
| Benchmark | 15 tasks in five workflow families, three conditions |
| Sweep window | 30 October 2026 to 1 November 2026, with the retry and re-scoring logs released alongside the results |
| Scored attempts | 3,600 local, 45 hosted |
| Hardware | One NVIDIA GeForce RTX 4090, 24 GB, driver 595.91.07, on Ubuntu 26.04.1 |
| Serving | `llama.cpp` build b11756 (commit d80619397), CUDA 12.8 runtime, OpenAI-compatible endpoint at `127.0.0.1:8081/v1` |
| Decoding | Per family, fixed within each model; see `configs/models_full_sweep_available.yaml` |
| Harness caps | 150 model iterations, 150 tool calls, 90 minutes per task, 60 seconds per command |
| Fixtures | `data/real_fixture_provenance.json` records the source and the controlled modifications |
| Sanitisation | Paths under the author's home directory are rewritten in every released text file: the code repository to `/benchmark`, the model directory to `/models/gguf`, and any remaining prefix to `/home/user`. Traces are otherwise unedited. |

The exact model files are recorded in `configs/models_full_sweep_available.yaml`, where `artifact_repo` gives the served file and `generation` gives the decoding settings. Attempts that ended on a harness rather than a model failure were repeated; the released set contains no `infrastructure_failure` attempt.

## 7. Reproducing the paper

```bash
uv sync --extra test
uv run --extra test pytest

# Regenerate the aggregates from the released attempt records
uv run mobility-bench aggregate \
  --results-dir runs/full_sweep/results_jsonl \
  --output-dir runs/full_sweep/aggregated
```

Per-attempt records are authoritative: every aggregate file can be recomputed from `results_jsonl/`, and every number in the paper can be recomputed from `aggregated/`.
