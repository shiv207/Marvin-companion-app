"""Marvin assistant CLI entrypoint."""

from marvin_companion.cli import main
from marvin_companion.core.prompts import AVAILABLE_COMMANDS, AVAILABLE_FACES

__all__ = ["AVAILABLE_COMMANDS", "AVAILABLE_FACES", "main"]


if __name__ == "__main__":
    main()
