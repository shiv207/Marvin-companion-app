"""Tkinter GUI adapter for the Groq-native Marvin runtime."""

from __future__ import annotations

import asyncio
import queue
import threading
import tkinter as tk
from concurrent.futures import Future
from datetime import datetime
from tkinter import scrolledtext, ttk

from marvin_companion.app import AppRuntime, create_runtime
from marvin_companion.config.settings import get_settings
from marvin_companion.core.events import BaseEvent, PlaybackEvent, TraceEvent, TranscriptEvent
from marvin_companion.core.prompts import AVAILABLE_COMMANDS


class MarvinGUI:
    """Desktop adapter for the async runtime."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Marvin Robot Assistant")
        self.root.geometry("900x700")
        self.root.minsize(820, 600)
        self.settings = get_settings()
        self.voice_enabled = tk.BooleanVar(value=self.settings.features.voice_enabled)
        self.wake_word_mode = tk.BooleanVar(value=self.settings.features.wake_word_mode)
        self.message_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.runtime: AppRuntime | None = None
        self.event_task: Future | None = None
        self.wake_task: Future | None = None
        self.is_listening = False

        self.bg_color = "#16181c"
        self.secondary_bg = "#23262d"
        self.accent_color = "#ff7a18"
        self.text_color = "#edf2f7"
        self.success_color = "#55d187"
        self.error_color = "#ff5f57"

        self._setup_ui()
        self._start_async_runtime()
        self._process_queue()

    def _setup_ui(self) -> None:
        self.root.configure(bg=self.bg_color)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Avenir Next", 10))
        style.configure("TButton", font=("Avenir Next", 10))

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X)
        ttk.Label(header, text="MARVIN DESKTOP INTERFACE", font=("Avenir Next", 20, "bold")).pack(side=tk.LEFT)
        self.connection_label = ttk.Label(header, text="Booting...", font=("Avenir Next", 10))
        self.connection_label.pack(side=tk.RIGHT)

        content = ttk.Frame(main_frame)
        content.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self._build_actions(left)

        center = ttk.Frame(content)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_chat(center)

        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))
        self._build_settings(right)

    def _build_actions(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Quick Actions", font=("Avenir Next", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))
        for label, command in [
            ("Wave", "wave"),
            ("Dance", "dance"),
            ("Walk", "walk"),
            ("Rest", "rest"),
            ("Pushup", "pushup"),
            ("Bow", "bow"),
            ("Stop", "stop"),
        ]:
            tk.Button(
                parent,
                text=label,
                command=lambda cmd=command: self._send_quick_command(cmd),
                bg=self.secondary_bg,
                fg=self.text_color,
                activebackground=self.accent_color,
                relief=tk.FLAT,
                padx=10,
                pady=8,
                cursor="hand2",
            ).pack(fill=tk.X, pady=2)

    def _build_chat(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Conversation", font=("Avenir Next", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.chat_display = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=("Menlo", 10),
            bg=self.secondary_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.tag_config("user", foreground="#7dd3fc", font=("Menlo", 10, "bold"))
        self.chat_display.tag_config("marvin", foreground=self.accent_color, font=("Menlo", 10, "bold"))
        self.chat_display.tag_config("system", foreground="#a0aec0")
        self.chat_display.tag_config("error", foreground=self.error_color)

        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        self.input_entry = tk.Entry(
            input_frame,
            font=("Avenir Next", 11),
            bg=self.secondary_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT,
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 6))
        self.input_entry.bind("<Return>", lambda _: self.send_message())

        self.mic_button = tk.Button(
            input_frame,
            text="MIC",
            command=self.toggle_listening,
            bg=self.accent_color,
            fg="white",
            relief=tk.FLAT,
            width=5,
        )
        self.mic_button.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg=self.accent_color,
            fg="white",
            relief=tk.FLAT,
            padx=15,
        ).pack(side=tk.LEFT)

    def _build_settings(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Settings", font=("Avenir Next", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))
        tk.Checkbutton(
            parent,
            text="Voice Mode",
            variable=self.voice_enabled,
            command=self._toggle_voice_mode,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.secondary_bg,
            activebackground=self.bg_color,
            activeforeground=self.text_color,
        ).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(
            parent,
            text=f"Wake Word ({self.settings.features.wake_word})",
            variable=self.wake_word_mode,
            command=self._toggle_wake_word_mode,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.secondary_bg,
            activebackground=self.bg_color,
            activeforeground=self.text_color,
        ).pack(anchor=tk.W, pady=4)
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=14)
        ttk.Label(parent, text="Available Commands", font=("Avenir Next", 10, "bold")).pack(anchor=tk.W, pady=(0, 6))
        commands_text = scrolledtext.ScrolledText(
            parent,
            height=10,
            wrap=tk.WORD,
            font=("Menlo", 8),
            bg=self.secondary_bg,
            fg=self.text_color,
            relief=tk.FLAT,
        )
        commands_text.pack(fill=tk.BOTH)
        commands_text.insert(tk.END, ", ".join(AVAILABLE_COMMANDS))
        commands_text.config(state=tk.DISABLED)

    def _start_async_runtime(self) -> None:
        def runner() -> None:
            asyncio.set_event_loop(self.loop)
            self.runtime = create_runtime(self.settings)
            self.connection_label.after(0, lambda: self.connection_label.config(text="Connected", foreground=self.success_color))
            self.event_task = asyncio.run_coroutine_threadsafe(self._bridge_events(), self.loop)
            self.loop.run_forever()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    async def _bridge_events(self) -> None:
        if not self.runtime:
            return
        subscriber = self.runtime.events.subscribe()
        try:
            while True:
                event = await subscriber.get()
                self.message_queue.put(("event", self._render_event(event)))
        finally:
            self.runtime.events.unsubscribe(subscriber)

    def _render_event(self, event: BaseEvent) -> str:
        if isinstance(event, TranscriptEvent):
            return f"{event.source}: {event.text}"
        if isinstance(event, TraceEvent):
            return f"{event.stage}: {event.latency_ms:.1f} ms"
        if isinstance(event, PlaybackEvent):
            return f"playback {event.status}"
        return event.type

    def send_message(self) -> None:
        message = self.input_entry.get().strip()
        if not message or not self.runtime:
            return
        self.input_entry.delete(0, tk.END)
        self._add_message("user", message)

        async def task() -> None:
            try:
                result = await self.runtime.orchestrator.handle_text(message)
                self.message_queue.put(("marvin", result.assistant_text))
            except Exception as exc:
                self.message_queue.put(("error", str(exc)))

        asyncio.run_coroutine_threadsafe(task(), self.loop)

    def toggle_listening(self) -> None:
        if self.is_listening or not self.runtime:
            return
        self.is_listening = True
        self.mic_button.config(bg=self.error_color, text="...")
        self._add_message("system", "Listening...")

        async def task() -> None:
            try:
                transcript = await self._capture_user_text()
                if self.wake_word_mode.get():
                    wake_word = self.settings.features.wake_word.lower()
                    lowered = transcript.lower()
                    if wake_word not in lowered:
                        self.message_queue.put(("system", f"Ignored transcript without wake word: {transcript}"))
                        return
                    transcript = lowered.split(wake_word, 1)[1].strip() or "hello"
                self.message_queue.put(("user", f"[VOICE] {transcript}"))
                result = await self.runtime.orchestrator.handle_text(transcript)
                self.message_queue.put(("marvin", result.assistant_text))
            except Exception as exc:
                self.message_queue.put(("error", str(exc)))
            finally:
                self.message_queue.put(("listening_done", ""))

        asyncio.run_coroutine_threadsafe(task(), self.loop)

    async def _capture_user_text(self) -> str:
        if not self.runtime:
            raise RuntimeError("Runtime is not ready.")
        path = await self.runtime.orchestrator.audio_input.record_phrase()
        try:
            return await self.runtime.orchestrator.stt.transcribe(str(path))
        finally:
            await self.runtime.orchestrator.audio_input.cleanup(path)

    def _toggle_voice_mode(self) -> None:
        state = "enabled" if self.voice_enabled.get() else "disabled"
        self._add_message("system", f"Voice mode {state}")

    def _toggle_wake_word_mode(self) -> None:
        state = "enabled" if self.wake_word_mode.get() else "disabled"
        self._add_message("system", f"Wake-word mode {state}")
        if self.wake_word_mode.get() and not self.wake_task:
            self.wake_task = asyncio.run_coroutine_threadsafe(self._wake_word_loop(), self.loop)
        elif not self.wake_word_mode.get() and self.wake_task:
            self.wake_task.cancel()
            self.wake_task = None

    async def _wake_word_loop(self) -> None:
        while self.wake_word_mode.get() and self.runtime:
            try:
                transcript = await self._capture_user_text()
                wake_word = self.settings.features.wake_word.lower()
                lowered = transcript.lower()
                if wake_word not in lowered:
                    continue
                cleaned = lowered.split(wake_word, 1)[1].strip() or "hello"
                self.message_queue.put(("user", f"[WAKE] {cleaned}"))
                result = await self.runtime.orchestrator.handle_text(cleaned)
                self.message_queue.put(("marvin", result.assistant_text))
            except Exception as exc:
                self.message_queue.put(("error", str(exc)))
                return

    def _send_quick_command(self, command: str) -> None:
        self._add_message("system", f"Executing: {command}")
        if not self.runtime:
            return

        async def task() -> None:
            result = await self.runtime.orchestrator.robot.send_command(command)
            if "error" in result:
                self.message_queue.put(("error", result["error"]))
            else:
                self.message_queue.put(("system", f"Executed: {command}"))

        asyncio.run_coroutine_threadsafe(task(), self.loop)

    def _add_message(self, sender: str, message: str) -> None:
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        tag = "system"
        label = ""
        if sender == "user":
            tag = "user"
            label = "You: "
        elif sender == "marvin":
            tag = "marvin"
            label = "Marvin: "
        elif sender == "error":
            tag = "error"
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
        if label:
            self.chat_display.insert(tk.END, label, tag)
        self.chat_display.insert(tk.END, f"{message}\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _process_queue(self) -> None:
        try:
            while True:
                kind, message = self.message_queue.get_nowait()
                if kind == "listening_done":
                    self.is_listening = False
                    self.mic_button.config(bg=self.accent_color, text="MIC")
                    continue
                self._add_message("system" if kind == "event" else kind, message)
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    def _on_close(self) -> None:
        async def shutdown() -> None:
            if self.runtime:
                await self.runtime.aclose()

        if self.wake_task:
            self.wake_task.cancel()
        asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MarvinGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
