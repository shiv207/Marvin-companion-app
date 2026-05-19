from __future__ import annotations

import importlib

import pytest

from marvin_companion.config.settings import AppSettings
from marvin_companion.services import diagnostics
from marvin_companion.services.diagnostics import run_startup_diagnostics


def build_settings(**overrides) -> AppSettings:
    base = {
        "groq": {"api_key": "test-key"},
        "robot": {"mock_mode": True},
        "features": {"voice_enabled": True},
    }
    base.update(overrides)
    return AppSettings(**base)


def test_startup_diagnostics(monkeypatch) -> None:
    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return [{"name": "fake-mic"}]

    monkeypatch.setattr(diagnostics, "sd", FakeSoundDevice)
    results = run_startup_diagnostics(build_settings())
    assert any(result.name == "audio_devices" and result.status == "ok" for result in results)


def test_legacy_wrappers_import() -> None:
    assistant_module = importlib.import_module("marvin_assistant")
    gui_module = importlib.import_module("marvin_gui")
    assert callable(assistant_module.main)
    assert callable(gui_module.main)
