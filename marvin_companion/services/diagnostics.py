"""Startup diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.exceptions import HealthCheckError

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency validation
    sd = None


@dataclass(slots=True)
class DiagnosticResult:
    """Single startup diagnostic result."""

    name: str
    status: str
    detail: str


def run_startup_diagnostics(settings: AppSettings) -> list[DiagnosticResult]:
    """Validate runtime prerequisites."""

    results: list[DiagnosticResult] = []
    if settings.features.demo_mode:
        results.append(DiagnosticResult("groq_api_key", "warning", "Demo mode enabled; Groq API key not required."))
    elif settings.groq.api_key:
        results.append(DiagnosticResult("groq_api_key", "ok", "Groq API key configured."))
    else:
        raise HealthCheckError("Missing GROQ API key.")

    if settings.features.voice_enabled:
        if sd is None:
            raise HealthCheckError("sounddevice is required for voice mode.")
        devices = sd.query_devices()
        if not devices:
            raise HealthCheckError("No audio devices available.")
        results.append(DiagnosticResult("audio_devices", "ok", f"{len(devices)} audio devices available."))
    else:
        results.append(DiagnosticResult("audio_devices", "warning", "Voice mode disabled."))

    if settings.robot.mock_mode:
        results.append(DiagnosticResult("robot", "warning", "Robot mock mode enabled."))
    elif settings.robot.base_url:
        results.append(DiagnosticResult("robot", "ok", f"Robot configured at {settings.robot.base_url}."))
    else:
        results.append(DiagnosticResult("robot", "warning", "Robot URL not configured."))
    return results

