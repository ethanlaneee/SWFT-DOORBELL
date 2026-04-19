"""Text-to-speech: converts text to spoken audio on the default speaker."""

import threading
from typing import Optional

import pyttsx3
from loguru import logger

from config import cfg


class TextToSpeech:
    def __init__(self) -> None:
        self._engine_name: str = cfg.get("tts", "engine", default="pyttsx3")
        self._rate: int = cfg.get("tts", "rate", default=175)
        self._volume: float = cfg.get("tts", "volume", default=0.9)
        self._voice_id: Optional[str] = cfg.get("tts", "voice_id")
        self._lock = threading.Lock()
        self._engine = self._init_engine()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def speak(self, text: str) -> None:
        """Synthesise and play `text` synchronously (blocks until done)."""
        if not text.strip():
            return
        logger.info(f"TTS: '{text[:80]}{'…' if len(text) > 80 else ''}'")
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                logger.error(f"TTS error: {exc}")
                # Re-initialise engine in case it crashed
                try:
                    self._engine = self._init_engine()
                except Exception:
                    pass

    def speak_async(self, text: str) -> threading.Thread:
        """Fire-and-forget version; returns the background thread."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _init_engine(self) -> pyttsx3.Engine:
        engine = pyttsx3.init(self._engine_name if self._engine_name != "pyttsx3" else None)
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        else:
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)

        return engine
