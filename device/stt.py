"""Speech-to-text: listens via microphone and returns transcribed text."""

import io
import wave
from typing import Optional

import speech_recognition as sr
from loguru import logger

from config import cfg


class SpeechToText:
    def __init__(self) -> None:
        self._recognizer = sr.Recognizer()
        self._engine: str = cfg.get("stt", "engine", default="google")
        self._language: str = cfg.get("stt", "language", default="en-US")
        self._timeout: float = cfg.get("stt", "timeout_seconds", default=5.0)
        self._phrase_limit: float = cfg.get("stt", "phrase_time_limit", default=10.0)

        silence_threshold: int = cfg.get("audio", "silence_threshold", default=500)
        self._recognizer.energy_threshold = silence_threshold
        self._recognizer.dynamic_energy_threshold = True

    def listen(self) -> Optional[str]:
        """Block until a phrase is captured, then return transcription or None."""
        device_index: Optional[int] = cfg.get("audio", "input_device")
        if isinstance(device_index, str):
            device_index = None

        with sr.Microphone(device_index=device_index, sample_rate=16000) as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
            logger.debug("Listening for speech …")
            try:
                audio = self._recognizer.listen(
                    source,
                    timeout=self._timeout,
                    phrase_time_limit=self._phrase_limit,
                )
            except sr.WaitTimeoutError:
                logger.debug("STT listen timeout")
                return None

        return self._transcribe(audio)

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> Optional[str]:
        audio = sr.AudioData(wav_bytes, sample_rate=16000, sample_width=2)
        return self._transcribe(audio)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _transcribe(self, audio: sr.AudioData) -> Optional[str]:
        try:
            if self._engine == "google":
                text = self._recognizer.recognize_google(audio, language=self._language)
            elif self._engine == "sphinx":
                text = self._recognizer.recognize_sphinx(audio)
            else:
                logger.error(f"Unknown STT engine: {self._engine}")
                return None
            logger.info(f"Transcribed: '{text}'")
            return text
        except sr.UnknownValueError:
            logger.debug("STT could not understand audio")
            return None
        except sr.RequestError as exc:
            logger.error(f"STT request error: {exc}")
            return None
