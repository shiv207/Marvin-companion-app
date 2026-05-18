"""Prompt and schema helpers for Sesame command extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


AVAILABLE_COMMANDS = [
    "walk",
    "rest",
    "swim",
    "dance",
    "wave",
    "point",
    "stand",
    "cute",
    "pushup",
    "freaky",
    "bow",
    "worm",
    "shake",
    "shrug",
    "dead",
    "crab",
    "idle",
    "stop",
]

AVAILABLE_FACES = [
    "default",
    "happy",
    "sad",
    "angry",
    "surprised",
    "sleepy",
    "love",
    "excited",
    "confused",
]


class AssistantDirective(BaseModel):
    """Validated assistant output schema."""

    command: str | None = None
    face: str | None = None
    response: str = Field(min_length=1)
    reasoning: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "AssistantDirective":
        if self.command and self.command not in AVAILABLE_COMMANDS:
            raise ValueError(f"Unsupported command: {self.command}")
        if self.face and self.face not in AVAILABLE_FACES:
            raise ValueError(f"Unsupported face: {self.face}")
        if self.command and self.face and self.command != "wave":
            self.face = None
        return self


SYSTEM_PROMPT = f"""You are Sesame, a tiny robot with a small, funny personality.
Return ONLY valid JSON matching this schema:
{{
  "command": "string or null",
  "face": "string or null",
  "response": "string",
  "reasoning": "string or null"
}}

Rules:
- Speak simply, like a child-sized robot.
- Keep the spoken response short, usually under 15 words.
- Commands are only for direct physical actions.
- Greetings may use the "wave" command.
- If the user is emotional or conversational, prefer a face and response, not a movement command.
- If command is not "wave", do not include a face with the command.

Available commands: {", ".join(AVAILABLE_COMMANDS)}
Available faces: {", ".join(AVAILABLE_FACES)}
"""


def build_messages(user_input: str) -> list[dict]:
    """Create provider-agnostic chat messages."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User input: {user_input}\nRespond with JSON only.",
        },
    ]

