"""Application bootstrap and runtime container."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from marvin_companion.config.logging import configure_logging
from marvin_companion.config.settings import AppSettings, get_settings
from marvin_companion.core.orchestrator import VoiceOrchestrator
from marvin_companion.services.audio_pipeline import MicrophoneAudioPipeline
from marvin_companion.services.diagnostics import run_startup_diagnostics
from marvin_companion.services.http_client import GroqAPIClient
from marvin_companion.services.llm_service import LLMService
from marvin_companion.services.mock_services import MockLLMService, MockRobotController, MockSTTService, MockTTSService
from marvin_companion.services.robot_controller import AsyncRobotController
from marvin_companion.services.stt_service import STTService
from marvin_companion.services.tts_service import TTSService
from marvin_companion.services.websocket_manager import WebSocketManager


@dataclass(slots=True)
class AppRuntime:
    """Runtime container for adapters."""

    settings: AppSettings
    orchestrator: VoiceOrchestrator
    events: WebSocketManager
    groq_client: GroqAPIClient | None = None
    robot_client: AsyncRobotController | None = None

    async def aclose(self) -> None:
        if self.groq_client:
            await self.groq_client.aclose()
        if self.robot_client:
            await self.robot_client.aclose()


def create_runtime(settings: AppSettings | None = None) -> AppRuntime:
    """Build the application runtime."""

    settings = settings or get_settings()
    configure_logging(settings.logging)
    run_startup_diagnostics(settings)
    logger = logging.getLogger(__name__)
    events = WebSocketManager()
    audio_pipeline = MicrophoneAudioPipeline(settings.audio, logger=logger.getChild("audio"))

    if settings.features.demo_mode:
        llm = MockLLMService()
        stt = MockSTTService()
        tts = MockTTSService()
        robot = MockRobotController()
        orchestrator = VoiceOrchestrator(
            settings=settings,
            llm=llm,
            stt=stt,
            tts=tts,
            robot=robot,
            audio_input=audio_pipeline,
            playback=audio_pipeline,
            events=events,
            logger=logger.getChild("orchestrator"),
        )
        return AppRuntime(settings=settings, orchestrator=orchestrator, events=events)

    groq_client = GroqAPIClient(settings, logger=logger.getChild("groq"))
    llm = LLMService(groq_client, settings, events=events, logger=logger.getChild("llm"))
    stt = STTService(groq_client, settings, logger=logger.getChild("stt"))
    tts = TTSService(groq_client, settings, logger=logger.getChild("tts"))
    robot_client = AsyncRobotController(settings.robot, logger=logger.getChild("robot"))
    orchestrator = VoiceOrchestrator(
        settings=settings,
        llm=llm,
        stt=stt,
        tts=tts,
        robot=robot_client,
        audio_input=audio_pipeline,
        playback=audio_pipeline,
        events=events,
        logger=logger.getChild("orchestrator"),
    )
    return AppRuntime(
        settings=settings,
        orchestrator=orchestrator,
        events=events,
        groq_client=groq_client,
        robot_client=robot_client,
    )
