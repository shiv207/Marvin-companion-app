from __future__ import annotations

import io
import json
import logging

import pytest

from marvin_companion.config.logging import JsonFormatter
from marvin_companion.config.settings import AppSettings, get_settings
from marvin_companion.core.exceptions import RetryExhaustedError
from marvin_companion.utils.metrics import correlation_context
from marvin_companion.utils.retry import run_with_retry


def build_settings(**overrides) -> AppSettings:
    base = {
        "groq": {"api_key": "test-key"},
        "features": {"voice_enabled": False},
        "robot": {"mock_mode": True},
    }
    base.update(overrides)
    return AppSettings(**base)


def test_model_alias_resolution() -> None:
    settings = build_settings()
    assert settings.groq.resolved_llm_model == "openai/gpt-oss-20b"
    assert settings.groq.resolved_tts_model == "canopylabs/orpheus-v1-english"


def test_flat_env_settings_loading(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ROBOT_MOCK_MODE", "true")
    monkeypatch.setenv("FEATURES_VOICE_ENABLED", "false")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.groq.api_key == "test-key"
    assert settings.robot.mock_mode is True
    assert settings.features.voice_enabled is False
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_retry_exhaustion() -> None:
    attempts = 0

    async def failing_call() -> object:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    with pytest.raises(RetryExhaustedError):
        await run_with_retry(
            failing_call,
            settings=build_settings(retry={"attempts": 2}).retry,
            retryable=(RuntimeError,),
        )
    assert attempts == 2


def test_json_logging_includes_correlation_id() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test-json-logger")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    with correlation_context("corr-123"):
        logger.info("hello", extra={"component": "unit-test"})

    payload = json.loads(stream.getvalue())
    assert payload["correlation_id"] == "corr-123"
    assert payload["component"] == "unit-test"
