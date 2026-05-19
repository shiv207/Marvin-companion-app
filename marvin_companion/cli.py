"""CLI adapter for the Groq-native runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path

from marvin_companion.app import create_runtime
from marvin_companion.config.settings import get_settings
from marvin_companion.core.prompts import AVAILABLE_COMMANDS, AVAILABLE_FACES


async def _capture_user_text(runtime) -> str:
    path = await runtime.orchestrator.audio_input.record_phrase()
    try:
        return await runtime.orchestrator.stt.transcribe(str(path))
    finally:
        await runtime.orchestrator.audio_input.cleanup(path)


async def interactive_cli() -> None:
    """Run the interactive command-line interface."""

    settings = get_settings()
    runtime = create_runtime(settings)
    voice_mode = settings.features.voice_enabled
    wake_word_mode = settings.features.wake_word_mode

    print("=" * 60)
    print("Marvin Robot Assistant")
    print("Groq-native voice stack")
    print("=" * 60)
    print()
    print("Commands:")
    print("  - Press Enter in voice mode to speak")
    print("  - Type 'voice' to toggle voice capture")
    print("  - Type 'wakeword' to toggle wake-word gating")
    print("  - Type 'status' to view robot status")
    print("  - Type 'help' to view supported robot commands")
    print("  - Type 'quit' or 'exit' to leave")
    print()

    try:
        while True:
            prompt = "[voice] " if voice_mode else "You: "
            user_input = await asyncio.to_thread(input, prompt)
            user_input = user_input.strip()

            if user_input.lower() in {"quit", "exit"}:
                break
            if user_input.lower() == "voice":
                voice_mode = not voice_mode
                print("Voice mode enabled." if voice_mode else "Voice mode disabled.")
                continue
            if user_input.lower() == "wakeword":
                wake_word_mode = not wake_word_mode
                print(
                    f"Wake-word mode {'enabled' if wake_word_mode else 'disabled'} "
                    f"('{settings.features.wake_word}')."
                )
                continue
            if user_input.lower() == "status":
                status = await runtime.orchestrator.robot.get_status()
                print(status)
                continue
            if user_input.lower() == "help":
                print(f"Commands: {', '.join(AVAILABLE_COMMANDS)}")
                print(f"Faces: {', '.join(AVAILABLE_FACES)}")
                continue

            if not user_input and voice_mode:
                try:
                    transcript = await _capture_user_text(runtime)
                except Exception as exc:
                    print(f"Voice capture failed: {exc}")
                    continue
                if wake_word_mode:
                    lowered = transcript.lower()
                    wake_word = settings.features.wake_word.lower()
                    if wake_word not in lowered:
                        print(f"Ignored transcript without wake word: {transcript}")
                        continue
                    transcript = lowered.split(wake_word, 1)[1].strip() or "hello"
                print(f"You said: {transcript}")
                result = await runtime.orchestrator.handle_text(transcript)
            elif user_input:
                result = await runtime.orchestrator.handle_text(user_input)
            else:
                continue

            print(result.assistant_text)
            if result.command:
                print(f"Action: {result.command}")
            if result.face:
                print(f"Face: {result.face}")
    finally:
        await runtime.aclose()


def main() -> None:
    asyncio.run(interactive_cli())


if __name__ == "__main__":
    main()
