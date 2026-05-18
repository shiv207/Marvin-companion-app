"""Main orchestration layer for voice turns."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.events import HealthEvent, PlaybackEvent, TraceEvent, TranscriptEvent
from marvin_companion.core.exceptions import InvalidResponseError
from marvin_companion.core.interfaces import AudioInput, EventPublisher, LLMProvider, PlaybackSink, RobotController, STTProvider, TTSProvider
from marvin_companion.core.prompts import AssistantDirective, build_messages
from marvin_companion.utils.metrics import Timer, correlation_context


@dataclass(slots=True)
class TurnResult:
    """Result of a user turn."""

    user_text: str
    assistant_text: str
    command: str | None
    face: str | None
    reasoning: str | None
    correlation_id: str


class NullEventPublisher:
    """Default no-op event publisher."""

    async def publish(self, event: object) -> None:
        return None


class VoiceOrchestrator:
    """Coordinates STT, LLM, TTS, playback, and robot control."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        llm: LLMProvider,
        stt: STTProvider,
        tts: TTSProvider,
        robot: RobotController,
        audio_input: AudioInput,
        playback: PlaybackSink,
        events: EventPublisher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.robot = robot
        self.audio_input = audio_input
        self.playback = playback
        self.events = events or NullEventPublisher()
        self.logger = logger or logging.getLogger(__name__)

    async def handle_text(
        self,
        user_text: str,
        *,
        speak_response: bool = True,
        correlation_id: str | None = None,
    ) -> TurnResult:
        """Process a text turn end-to-end."""

        correlation_id = correlation_id or str(uuid.uuid4())
        with correlation_context(correlation_id):
            await self.events.publish(TranscriptEvent(correlation_id, "transcript", user_text, "user"))
            messages = build_messages(user_text)
            with Timer() as llm_timer:
                raw_response = await self.llm.generate(
                    messages,
                    stream=self.settings.features.stream_llm,
                )
            await self.events.publish(
                TraceEvent(correlation_id, "trace", "llm", llm_timer.elapsed_ms, "llm completion")
            )
            directive = self._parse_directive(raw_response)
            if directive.command:
                await self.robot.send_command(directive.command, directive.face)
            elif directive.face:
                await self.robot.send_command("idle", directive.face)
            await self.events.publish(
                TranscriptEvent(correlation_id, "transcript", directive.response, "llm")
            )
            if speak_response and directive.response:
                await self._speak(directive.response, correlation_id)
            return TurnResult(
                user_text=user_text,
                assistant_text=directive.response,
                command=directive.command,
                face=directive.face,
                reasoning=directive.reasoning,
                correlation_id=correlation_id,
            )

    async def handle_audio_turn(
        self,
        *,
        timeout_seconds: float | None = None,
        speak_response: bool = True,
    ) -> TurnResult:
        """Capture audio from the microphone and process a turn."""

        correlation_id = str(uuid.uuid4())
        with correlation_context(correlation_id):
            audio_path = await self.audio_input.record_phrase(timeout_seconds)
            try:
                with Timer() as stt_timer:
                    text = await self.stt.transcribe(str(audio_path))
                await self.events.publish(
                    TraceEvent(correlation_id, "trace", "stt", stt_timer.elapsed_ms, "speech to text")
                )
                await self.events.publish(
                    TranscriptEvent(correlation_id, "transcript", text, "microphone")
                )
                return await self.handle_text(
                    text,
                    speak_response=speak_response,
                    correlation_id=correlation_id,
                )
            finally:
                await self.audio_input.cleanup(audio_path)

    async def health_check(self) -> dict[str, str]:
        """Run a basic health check."""

        status = await self.robot.get_status()
        detail = "robot connected" if "error" not in status else status["error"]
        state = "ok" if "error" not in status else "warning"
        await self.events.publish(HealthEvent("system", "health", state, detail))
        return {"status": state, "detail": detail}

    async def _speak(self, text: str, correlation_id: str) -> None:
        with Timer() as tts_timer:
            wav_audio = await self.tts.synthesize(text)
        await self.events.publish(
            TraceEvent(correlation_id, "trace", "tts", tts_timer.elapsed_ms, "text to speech")
        )
        await self.events.publish(PlaybackEvent(correlation_id, "playback", "started"))
        await self.playback.play_audio(wav_audio)
        await self.events.publish(PlaybackEvent(correlation_id, "playback", "finished"))

    def _parse_directive(self, raw_response: str) -> AssistantDirective:
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            return AssistantDirective.model_validate(json.loads(cleaned.strip()))
        except Exception as exc:
            raise InvalidResponseError(f"Failed to parse LLM response: {raw_response}") from exc

