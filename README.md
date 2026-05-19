# Marvin Assistant

Groq-native real-time voice assistant built on an async runtime, with CLI and GUI adapters over the same orchestration core.

## What Changed

- The assistant identity has been renamed to Marvin throughout the app surface.
- The runtime stays modular and async, with separate services for LLM, STT, TTS, audio, robot control, and diagnostics.
- Startup entrypoints now live in [`marvin_assistant.py`](/Users/shiv/Developer/Marvin-companion-app/marvin_assistant.py) and [`marvin_gui.py`](/Users/shiv/Developer/Marvin-companion-app/marvin_gui.py).
- Environment variables are standardized to flat, production-ready names like `GROQ_API_KEY`.

## Layout

```text
marvin_companion/
  config/
  core/
  services/
  utils/
tests/
marvin_assistant.py
marvin_gui.py
```

## Requirements

- Python 3.11+
- A Groq API key
- A working microphone and speaker for voice mode
- Robot mock mode enabled, or a reachable robot base URL

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy [`.env.example`](/Users/shiv/Developer/Marvin-companion-app/.env.example) to `.env` and adjust values.

The settings loader accepts both the new flat keys and the older nested `__` form, but the flat form is preferred.

Important variables:

- `MARVIN_APP_NAME`
- `GROQ_API_KEY`
- `GROQ_BASE_URL`
- `GROQ_LLM_MODEL`
- `GROQ_STT_MODEL_PRIMARY`
- `GROQ_STT_MODEL_FALLBACK`
- `GROQ_TTS_MODEL`
- `GROQ_TTS_VOICE`
- `ROBOT_BASE_URL`
- `ROBOT_MOCK_MODE`
- `FEATURES_VOICE_ENABLED`
- `FEATURES_WAKE_WORD_MODE`
- `FEATURES_WAKE_WORD`
- `FEATURES_DEMO_MODE`

Model aliases:

- `GROQ_LLM_MODEL=gpt-oss-3` resolves to `openai/gpt-oss-20b`
- `GROQ_TTS_MODEL=playai-tts` resolves to `canopylabs/orpheus-v1-english`

## Usage

CLI:

```bash
python3 marvin_assistant.py
```

GUI:

```bash
python3 marvin_gui.py
```

Installed entrypoints:

```bash
marvin-assistant
marvin-gui
```

## Voice Flow

1. Microphone audio is captured asynchronously with silence-based segmentation.
2. Audio is normalized, converted to mono 16 kHz, and sent to Groq Whisper.
3. The transcript is turned into a structured Marvin directive through Groq chat completions.
4. Robot actions are dispatched through the async robot controller.
5. The response is synthesized by Orpheus TTS and played back through a serialized audio sink.

## Example

```text
You: hello marvin
Marvin: Hi friend!
Action: wave
Face: happy
```

Expressive TTS example:

```text
[cheerful] Welcome back!
[calm] Let me help you with that.
```

## Development

Run tests:

```bash
pytest
```

The tests are offline by default and use mocked transports and stub services.

## Notes

- Orpheus requests are chunked to stay under the Groq text limit used by the English model.
- TTS output is cached under `.cache/audio`.
- Structured logs include correlation IDs for request tracing.
- Startup diagnostics validate API key presence and local audio device availability.

## Troubleshooting

- `Missing GROQ API key`
  - Set `GROQ_API_KEY`, or enable `FEATURES_DEMO_MODE=true`.
- `sounddevice is required for voice mode`
  - Install dependencies and confirm the OS audio stack is available.
- `Robot base URL is not configured`
  - Set `ROBOT_BASE_URL`, or enable `ROBOT_MOCK_MODE=true`.
- `Audio playback failed`
  - Check default output device access and sample-rate compatibility.
