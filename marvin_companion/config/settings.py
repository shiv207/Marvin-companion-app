"""Application configuration models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    """Sesame robot integration configuration."""

    base_url: str = ""
    timeout_seconds: float = 5.0
    mock_mode: bool = False

    @property
    def is_enabled(self) -> bool:
        return bool(self.base_url) or self.mock_mode


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json: bool = True


class FeatureFlags(BaseModel):
    """Runtime feature toggles."""

    voice_enabled: bool = True
    wake_word_mode: bool = False
    wake_word: str = "hey sesame"
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

    app_name: str = "marvin-companion-app"
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

    return AppSettings()

