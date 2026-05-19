"""Application configuration models."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LLM_MODEL_ALIASES = {
    "gpt-oss-3": "openai/gpt-oss-20b",
}

TTS_MODEL_ALIASES = {
    "playai-tts": "canopylabs/orpheus-v1-english",
}


class RetrySettings(BaseModel):
    """Retry policy for outbound HTTP requests."""

    attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0
    timeout_seconds: float = 20.0


class GroqSettings(BaseModel):
    """Groq API configuration."""

    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "gpt-oss-3"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 256
    stt_model_primary: str = "whisper-large-v3-turbo"
    stt_model_fallback: str = "whisper-large-v3"
    tts_model: str = "playai-tts"
    tts_voice: str = "tara"
    request_timeout_seconds: float = 20.0

    @property
    def resolved_llm_model(self) -> str:
        return LLM_MODEL_ALIASES.get(self.llm_model, self.llm_model)

    @property
    def resolved_tts_model(self) -> str:
        return TTS_MODEL_ALIASES.get(self.tts_model, self.tts_model)


class AudioSettings(BaseModel):
    """Audio capture and playback configuration."""

    input_sample_rate: int = 48_000
    target_sample_rate: int = 16_000
    channels: int = 1
    block_size: int = 1_024
    silence_threshold: float = 0.015
    silence_duration_seconds: float = 0.8
    max_phrase_seconds: float = 8.0
    min_phrase_seconds: float = 0.35
    preroll_seconds: float = 0.25
    normalize_audio: bool = True
    tts_chunk_max_chars: int = 200
    playback_device: int | None = None
    input_device: int | None = None
    temp_dir: str = ".cache/audio"


class RobotSettings(BaseModel):
    """Marvin robot integration configuration."""

    base_url: str = ""
    timeout_seconds: float = 5.0
    mock_mode: bool = False

    @property
    def is_enabled(self) -> bool:
        return bool(self.base_url) or self.mock_mode


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_output: bool = Field(default=True, alias="json")


class FeatureFlags(BaseModel):
    """Runtime feature toggles."""

    voice_enabled: bool = True
    wake_word_mode: bool = False
    wake_word: str = "hey marvin"
    demo_mode: bool = False
    stream_llm: bool = False


class MetricsSettings(BaseModel):
    """Metrics configuration."""

    enabled: bool = True


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "marvin-assistant-app"
    groq: GroqSettings = Field(default_factory=GroqSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    robot: RobotSettings = Field(default_factory=RobotSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)

    @model_validator(mode="after")
    def validate_settings(self) -> "AppSettings":
        if not self.features.demo_mode and not self.groq.api_key:
            raise ValueError("GROQ API key is required unless demo mode is enabled.")
        if self.audio.channels != 1:
            raise ValueError("Only mono audio is supported for the real-time pipeline.")
        return self

    @property
    def audio_cache_dir(self) -> Path:
        return Path(self.audio.temp_dir)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache application settings."""

    settings_data = _load_env_settings()
    return AppSettings.model_validate(settings_data)


def _load_env_settings() -> dict[str, object]:
    """Load .env and process Marvin/Groq flat keys into nested settings."""

    raw_values: dict[str, str | None] = {}
    raw_values.update(dotenv_values(".env"))
    raw_values.update(os.environ)

    settings: dict[str, object] = {}
    for key, value in raw_values.items():
        if value is None or value == "":
            continue
        path = _env_key_path(key)
        if not path:
            continue
        cursor = settings
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})  # type: ignore[assignment]
        cursor[path[-1]] = value
    return settings


def _env_key_path(key: str) -> tuple[str, ...] | None:
    """Map supported environment keys into model field paths."""

    normalized = key.upper()
    if normalized.startswith("MARVIN_"):
        normalized = normalized.removeprefix("MARVIN_")

    flat_aliases: dict[str, tuple[str, ...]] = {
        "APP_NAME": ("app_name",),
        "GROQ_API_KEY": ("groq", "api_key"),
        "GROQ_BASE_URL": ("groq", "base_url"),
        "GROQ_LLM_MODEL": ("groq", "llm_model"),
        "GROQ_LLM_TEMPERATURE": ("groq", "llm_temperature"),
        "GROQ_LLM_MAX_TOKENS": ("groq", "llm_max_tokens"),
        "GROQ_STT_MODEL_PRIMARY": ("groq", "stt_model_primary"),
        "GROQ_STT_MODEL_FALLBACK": ("groq", "stt_model_fallback"),
        "GROQ_TTS_MODEL": ("groq", "tts_model"),
        "GROQ_TTS_VOICE": ("groq", "tts_voice"),
        "GROQ_REQUEST_TIMEOUT_SECONDS": ("groq", "request_timeout_seconds"),
        "RETRY_ATTEMPTS": ("retry", "attempts"),
        "RETRY_BASE_DELAY_SECONDS": ("retry", "base_delay_seconds"),
        "RETRY_MAX_DELAY_SECONDS": ("retry", "max_delay_seconds"),
        "RETRY_TIMEOUT_SECONDS": ("retry", "timeout_seconds"),
        "AUDIO_INPUT_SAMPLE_RATE": ("audio", "input_sample_rate"),
        "AUDIO_TARGET_SAMPLE_RATE": ("audio", "target_sample_rate"),
        "AUDIO_CHANNELS": ("audio", "channels"),
        "AUDIO_BLOCK_SIZE": ("audio", "block_size"),
        "AUDIO_SILENCE_THRESHOLD": ("audio", "silence_threshold"),
        "AUDIO_SILENCE_DURATION_SECONDS": ("audio", "silence_duration_seconds"),
        "AUDIO_MAX_PHRASE_SECONDS": ("audio", "max_phrase_seconds"),
        "AUDIO_MIN_PHRASE_SECONDS": ("audio", "min_phrase_seconds"),
        "AUDIO_PREROLL_SECONDS": ("audio", "preroll_seconds"),
        "AUDIO_NORMALIZE_AUDIO": ("audio", "normalize_audio"),
        "AUDIO_TTS_CHUNK_MAX_CHARS": ("audio", "tts_chunk_max_chars"),
        "AUDIO_PLAYBACK_DEVICE": ("audio", "playback_device"),
        "AUDIO_INPUT_DEVICE": ("audio", "input_device"),
        "AUDIO_TEMP_DIR": ("audio", "temp_dir"),
        "ROBOT_BASE_URL": ("robot", "base_url"),
        "ROBOT_TIMEOUT_SECONDS": ("robot", "timeout_seconds"),
        "ROBOT_MOCK_MODE": ("robot", "mock_mode"),
        "LOGGING_LEVEL": ("logging", "level"),
        "LOGGING_JSON": ("logging", "json_output"),
        "FEATURES_VOICE_ENABLED": ("features", "voice_enabled"),
        "FEATURES_WAKE_WORD_MODE": ("features", "wake_word_mode"),
        "FEATURES_WAKE_WORD": ("features", "wake_word"),
        "FEATURES_DEMO_MODE": ("features", "demo_mode"),
        "FEATURES_STREAM_LLM": ("features", "stream_llm"),
        "METRICS_ENABLED": ("metrics", "enabled"),
    }
    if normalized in flat_aliases:
        return flat_aliases[normalized]

    if "__" in normalized:
        return tuple(segment.lower() for segment in normalized.split("__") if segment)
    return None
