from __future__ import annotations

from typing import Any

from mobility_bench.config import load_benchmark_config, load_models_config, load_tasks_config
from mobility_bench.models.llamacpp_client import LlamaCppClient
from mobility_bench.models.mock_client import MockClient
from mobility_bench.runtime.runner import effective_timeout_seconds
from mobility_bench.tasks.registry import get_all_tasks


def test_config_files_load() -> None:
    benchmark = load_benchmark_config("configs/benchmark.yaml")
    models = load_models_config("configs/models.yaml")
    tasks = load_tasks_config("configs/tasks.yaml")

    assert benchmark.runtime.max_iterations == 1000
    assert benchmark.runtime.max_tool_calls == 1000
    assert benchmark.runtime.max_consecutive_protocol_errors == 8
    assert benchmark.runtime.max_protocol_response_tokens == 8192
    assert benchmark.runtime.max_repeated_tool_calls == 20
    assert benchmark.runtime.max_repeated_failed_tool_calls == 8
    assert benchmark.runtime.timeout_seconds_per_task == 3600
    assert models.models["mock"].backend == "mock"
    assert models.models["qwen35_27b_q4"].source_repo == "Qwen/Qwen3.5-27B"
    assert models.models["qwen36_27b_ud_q4kxl"].source_repo == "Qwen/Qwen3.6-27B"
    assert "T01" in tasks.tasks["enabled"]


def test_runtime_timeout_overrides_task_default() -> None:
    benchmark = load_benchmark_config("configs/benchmark.yaml")
    task = get_all_tasks()["T01"]

    assert task.timeout_seconds == 600
    assert effective_timeout_seconds(task, benchmark) == 3600


def test_mock_client_returns_responses_in_order() -> None:
    client = MockClient(["first", "second"])
    assert client.generate([]).text == "first"
    assert client.generate([]).text == "second"
    assert len(client.calls) == 2


def test_llamacpp_client_request_construction(monkeypatch: Any) -> None:
    observed: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            }

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> FakeResponse:
        observed["url"] = url
        observed["json"] = json
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("mobility_bench.models.llamacpp_client.requests.post", fake_post)

    client = LlamaCppClient(
        "http://localhost:8080/v1/",
        "qwen-test",
        timeout_seconds=9,
        generation_kwargs={"max_tokens": 123, "chat_template_kwargs": {"enable_thinking": False}},
    )
    response = client.generate([{"role": "user", "content": "hello"}], temperature=0.0)

    assert observed["url"] == "http://localhost:8080/v1/chat/completions"
    assert observed["json"]["model"] == "qwen-test"
    assert observed["json"]["max_tokens"] == 123
    assert observed["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert observed["json"]["temperature"] == 0.0
    assert observed["timeout"] == 9
    assert response.text == "answer"
    assert response.total_tokens == 7
    assert response.token_counts_estimated is False
