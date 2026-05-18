"""Async microphone capture and WAV playback."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from marvin_companion.config.settings import AudioSettings
from marvin_companion.core.exceptions import AudioValidationError, PlaybackError
from marvin_companion.core.interfaces import AudioInput, PlaybackSink
from marvin_companion.utils.audio import duration_seconds, load_wav_bytes, normalize, resample, rms_level, silence_frames, write_wav

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled by diagnostics and demo mode
    sd = None


class MicrophoneAudioPipeline(AudioInput, PlaybackSink):
    """Microphone capture and playback service."""

    def __init__(
        self,
        settings: AudioSettings,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self._playback_lock = asyncio.Lock()
        Path(self.settings.temp_dir).mkdir(parents=True, exist_ok=True)

    async def record_phrase(self, timeout_seconds: float | None = None) -> Path:
        if sd is None:
            raise AudioValidationError("sounddevice is not installed.")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        dropped_chunks = 0

        def enqueue(frame: np.ndarray) -> None:
            nonlocal dropped_chunks
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                dropped_chunks += 1
            queue.put_nowait(frame)

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                self.logger.debug("input stream status", extra={"status": str(status)})
            frame = np.copy(indata[:, 0]).astype(np.float32)
            loop.call_soon_threadsafe(enqueue, frame)

        silence_limit = silence_frames(
            self.settings.input_sample_rate,
            self.settings.silence_duration_seconds,
            self.settings.block_size,
        )
        preroll_limit = silence_frames(
            self.settings.input_sample_rate,
            self.settings.preroll_seconds,
            self.settings.block_size,
        )
        timeout_seconds = timeout_seconds or self.settings.max_phrase_seconds
        preroll: deque[np.ndarray] = deque(maxlen=preroll_limit)
        captured: list[np.ndarray] = []
        speech_started = False
        silent_blocks = 0

        stream = sd.InputStream(
            samplerate=self.settings.input_sample_rate,
            blocksize=self.settings.block_size,
            channels=1,
            dtype="float32",
            device=self.settings.input_device,
            callback=callback,
        )

        with stream:
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=timeout_seconds)
                except TimeoutError as exc:
                    raise AudioValidationError("Timed out waiting for microphone input.") from exc

                level = rms_level(frame)
                if not speech_started:
                    preroll.append(frame)
                    if level >= self.settings.silence_threshold:
                        speech_started = True
                        captured.extend(preroll)
                        captured.append(frame)
                    continue

                captured.append(frame)
                if level < self.settings.silence_threshold:
                    silent_blocks += 1
                else:
                    silent_blocks = 0

                total_samples = np.concatenate(captured) if captured else np.array([], dtype=np.float32)
                phrase_duration = duration_seconds(total_samples, self.settings.input_sample_rate)
                if phrase_duration >= self.settings.max_phrase_seconds:
                    break
                if silent_blocks >= silence_limit and phrase_duration >= self.settings.min_phrase_seconds:
                    break

        if not captured:
            raise AudioValidationError("No speech detected.")

        audio = np.concatenate(captured).astype(np.float32)
        if self.settings.normalize_audio:
            audio = normalize(audio)
        if self.settings.input_sample_rate != self.settings.target_sample_rate:
            audio = resample(audio, self.settings.input_sample_rate, self.settings.target_sample_rate)

        output_path = Path(self.settings.temp_dir) / f"{uuid.uuid4()}.wav"
        await asyncio.to_thread(write_wav, output_path, audio, self.settings.target_sample_rate)
        if dropped_chunks:
            self.logger.warning("audio chunks dropped", extra={"dropped_chunks": dropped_chunks})
        return output_path

    async def cleanup(self, audio_path: Path) -> None:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            self.logger.debug("failed to clean temp audio", extra={"path": str(audio_path)})

    async def play_audio(self, wav_audio: bytes) -> None:
        if sd is None:
            raise PlaybackError("sounddevice is not installed.")
        samples, sample_rate = load_wav_bytes(wav_audio)
        async with self._playback_lock:
            await asyncio.to_thread(self._blocking_play, samples, sample_rate)

    def _blocking_play(self, samples: np.ndarray, sample_rate: int) -> None:
        try:
            sd.play(samples, sample_rate, device=self.settings.playback_device, blocking=True)
        except Exception as exc:  # pragma: no cover - depends on host audio stack
            raise PlaybackError(f"Audio playback failed: {exc}") from exc

