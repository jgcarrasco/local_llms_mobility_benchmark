from __future__ import annotations

from typing import Any

import pytest

from mobility_bench.config import (
    ModelConfig,
    load_benchmark_config,
    load_models_config,
    load_tasks_config,
)
from mobility_bench.models import create_model_client
from mobility_bench.models.llamacpp_client import LlamaCppClient
from mobility_bench.models.mock_client import MockClient
from mobility_bench.runtime.runner import effective_timeout_seconds
from mobility_bench.tasks.registry import get_all_tasks


def test_config_files_load() -> None:
    benchmark = load_benchmark_config("configs/benchmark.yaml")
    models = load_models_config("configs/models.yaml")
    tasks = load_tasks_config("configs/tasks.yaml")

    assert benchmark.runtime.max_iterations == 150
    assert benchmark.runtime.max_tool_calls == 150
    assert benchmark.runtime.max_consecutive_protocol_errors == 8
    assert benchmark.runtime.max_repeated_tool_calls == 20
    assert benchmark.runtime.max_repeated_failed_tool_calls == 8
    assert benchmark.runtime.timeout_seconds_per_task == 5400
    assert models.models["mock"].backend == "mock"
    assert models.models["qwen35_27b_q4"].source_repo == "Qwen/Qwen3.5-27B"
    assert models.models["qwen36_27b_ud_q4kxl"].source_repo == "Qwen/Qwen3.6-27B"
    assert "T01" in tasks.tasks["enabled"]


def test_runtime_timeout_overrides_task_default() -> None:
    benchmark = load_benchmark_config("configs/benchmark.yaml")
    task = get_all_tasks()["T01"]

    assert task.timeout_seconds == 600
    assert effective_timeout_seconds(task, benchmark) == 5400


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

    def fake_post(
        url: str,
        json: dict[str, Any],
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> FakeResponse:
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


def test_llamacpp_client_handles_reasoning_and_empty_content(monkeypatch: Any) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {"content": None, "reasoning_content": "deliberation"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26},
            }

    monkeypatch.setattr(
        "mobility_bench.models.llamacpp_client.requests.post",
        lambda url, json, timeout, headers=None: FakeResponse(),
    )

    client = LlamaCppClient("http://localhost:8080/v1", "qwen-test")
    response = client.generate([{"role": "user", "content": "hello"}])

    assert response.text == ""
    assert response.reasoning_text == "deliberation"
    assert response.finish_reason == "length"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26}


def test_every_model_declares_a_tool_protocol() -> None:
    models = load_models_config("configs/models_full_sweep_available.yaml")
    for model_id, config in models.models.items():
        assert config.tool_protocol in {"native", "text"}, model_id
    assert models.models["phi4_14b_q4"].tool_protocol == "text"
    assert models.models["gptoss_20b_q4"].tool_protocol == "native"


def test_tool_schemas_match_the_allowed_tools() -> None:
    from mobility_bench.tools import ALLOWED_TOOLS
    from mobility_bench.tools.schemas import TOOL_SCHEMA_NAMES

    assert TOOL_SCHEMA_NAMES == ALLOWED_TOOLS

def test_llamacpp_client_returns_transport_error_instead_of_raising(monkeypatch: Any) -> None:
    import requests

    def failing_post(
        url: str, json: dict[str, Any], timeout: int, headers: dict[str, str] | None = None
    ) -> Any:
        raise requests.ConnectionError("server exploded")

    monkeypatch.setattr("mobility_bench.models.llamacpp_client.requests.post", failing_post)
    client = LlamaCppClient("http://localhost:8080/v1", "qwen-test")
    response = client.generate([{"role": "user", "content": "hello"}])

    assert response.text == ""
    assert response.finish_reason == "transport_error"
    assert "server exploded" in str(response.raw_response)

def test_llamacpp_client_reads_timings(monkeypatch: Any) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                "timings": {"prompt_ms": 10.0, "predicted_ms": 20.0, "predicted_per_second": 200.0},
            }

    monkeypatch.setattr(
        "mobility_bench.models.llamacpp_client.requests.post",
        lambda url, json, timeout, headers=None: FakeResponse(),
    )
    client = LlamaCppClient("http://localhost:8080/v1", "qwen-test")
    response = client.generate([{"role": "user", "content": "hello"}])

    assert response.timings["predicted_per_second"] == 200.0


def _hosted_config(**overrides: Any) -> ModelConfig:
    base = dict(
        family="deepseek",
        size_b=552.0,
        architecture="moe",
        local=False,
        backend="openai_compatible",
        base_url="https://example.test/inference/go/openai/v1",
        model_name="deepseek-v4.1-flash",
        quantization="none",
        context_length=1_000_000,
        api_key_env="TEST_OPENCODE_KEY",
        extra_headers={"x-opencode-session": "ses_test"},
        generation={"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_hosted_client_sends_bearer_token_and_session_header(monkeypatch: Any) -> None:
    monkeypatch.setenv("TEST_OPENCODE_KEY", "oc_sk_test")
    client = create_model_client(_hosted_config())

    assert client.headers["Authorization"] == "Bearer oc_sk_test"
    assert client.headers["x-opencode-session"] == "ses_test"
    assert client.headers["User-Agent"].startswith("mobility-bench")


def test_hosted_client_fails_loudly_without_the_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("TEST_OPENCODE_KEY", raising=False)

    with pytest.raises(ValueError, match="TEST_OPENCODE_KEY"):
        create_model_client(_hosted_config())


def test_local_client_sends_no_authorization_header() -> None:
    client = create_model_client(
        _hosted_config(backend="llamacpp", api_key_env=None, extra_headers={})
    )

    assert "Authorization" not in client.headers


def test_hosted_reference_config_loads() -> None:
    models = load_models_config("configs/models_hosted_reference.yaml")
    entry = models.models["deepseek_v41_flash"]

    assert entry.backend == "openai_compatible"
    assert entry.api_key_env == "OPENCODE_API_KEY"
    assert entry.model_name == "deepseek-v4.1-flash"
    # The client forwards this dict verbatim, so it must hold only fields the
    # OpenAI-compatible endpoint accepts.
    assert set(entry.generation) <= {"temperature", "top_p", "max_tokens"}
