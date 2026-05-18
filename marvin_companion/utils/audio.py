"""Audio processing helpers."""

from __future__ import annotations

import io
import math
import re
import wave
from pathlib import Path

import numpy as np

from marvin_companion.core.exceptions import AudioValidationError


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def rms_level(samples: np.ndarray) -> float:
    """Compute RMS level for a float32 PCM buffer."""

    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def normalize(samples: np.ndarray) -> np.ndarray:
    """Peak-normalize a float32 mono signal."""

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0:
        return samples.astype(np.float32)
    return (samples / peak * 0.95).astype(np.float32)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation."""

    if source_rate == target_rate:
        return samples.astype(np.float32)
    if samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.shape[0] / source_rate
    target_length = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 audio to PCM16 WAV."""

    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def load_wav_bytes(wav_audio: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into a float32 mono buffer."""

    try:
        with wave.open(io.BytesIO(wav_audio), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise AudioValidationError("Only mono WAV playback is supported.")
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            pcm16 = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            return pcm16 / 32767.0, sample_rate
    except wave.Error as exc:
        raise AudioValidationError("Invalid WAV payload returned by TTS service.") from exc


def concat_wav_bytes(chunks: list[bytes]) -> bytes:
    """Concatenate mono PCM16 WAV chunks."""

    if not chunks:
        raise AudioValidationError("No WAV chunks were provided.")
    sample_rate: int | None = None
    combined: list[np.ndarray] = []
    for chunk in chunks:
        samples, chunk_rate = load_wav_bytes(chunk)
        if sample_rate is None:
            sample_rate = chunk_rate
        elif chunk_rate != sample_rate:
            raise AudioValidationError("Mismatched WAV sample rates cannot be concatenated.")
        combined.append(samples)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate or 24_000)
        merged = np.concatenate(combined) if combined else np.array([], dtype=np.float32)
        wav_file.writeframes((np.clip(merged, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return output.getvalue()


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into sentence-aware TTS chunks."""

    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return [cleaned]

    segments = SENTENCE_RE.split(cleaned)
    chunks: list[str] = []
    current = ""
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        separator = " " if current else ""
        if len(current) + len(separator) + len(segment) <= max_chars:
            current = f"{current}{separator}{segment}"
            continue
        if current:
            chunks.append(current)
        if len(segment) <= max_chars:
            current = segment
            continue
        words = segment.split()
        current = ""
        for word in words:
            separator = " " if current else ""
            if len(current) + len(separator) + len(word) <= max_chars:
                current = f"{current}{separator}{word}"
            else:
                if current:
                    chunks.append(current)
                current = word
        if current and len(current) > max_chars:
            raise AudioValidationError("Unable to chunk TTS input under model limits.")
    if current:
        chunks.append(current)
    return chunks


def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    """Compute signal duration."""

    return 0.0 if sample_rate <= 0 else samples.shape[0] / sample_rate


def silence_frames(sample_rate: int, silence_seconds: float, block_size: int) -> int:
    """Convert silence duration to block count."""

    return max(1, math.ceil((sample_rate * silence_seconds) / block_size))
