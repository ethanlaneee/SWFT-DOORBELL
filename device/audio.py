"""Manages microphone input and speaker output via PyAudio."""

import queue
import threading
from typing import Callable, Optional

import pyaudio
from loguru import logger

from config import cfg


class AudioIO:
    def __init__(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._sample_rate: int = cfg.get("audio", "sample_rate", default=16000)
        self._chunk: int = cfg.get("audio", "chunk_size", default=1024)
        self._input_device: Optional[int] = self._resolve_device(
            cfg.get("audio", "input_device"), input=True
        )
        self._output_device: Optional[int] = self._resolve_device(
            cfg.get("audio", "output_device"), input=False
        )

        self._in_stream: Optional[pyaudio.Stream] = None
        self._out_stream: Optional[pyaudio.Stream] = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._listeners: list[Callable[[bytes], None]] = []
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def on_audio_chunk(self, callback: Callable[[bytes], None]) -> None:
        self._listeners.append(callback)

    def start_capture(self) -> None:
        self._in_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._input_device,
            frames_per_buffer=self._chunk,
        )
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("Audio capture started")

    def stop_capture(self) -> None:
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._in_stream:
            self._in_stream.stop_stream()
            self._in_stream.close()
        logger.info("Audio capture stopped")

    def play_bytes(self, audio_bytes: bytes, sample_rate: int = 22050) -> None:
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=self._output_device,
        )
        try:
            stream.write(audio_bytes)
        finally:
            stream.stop_stream()
            stream.close()

    def close(self) -> None:
        self.stop_capture()
        self._pa.terminate()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _capture_loop(self) -> None:
        while self._running:
            try:
                data = self._in_stream.read(self._chunk, exception_on_overflow=False)
            except OSError as exc:
                logger.warning(f"Audio read error: {exc}")
                continue
            for cb in self._listeners:
                try:
                    cb(data)
                except Exception as exc:
                    logger.error(f"Audio listener error: {exc}")

    def _resolve_device(self, name_or_index, *, input: bool) -> Optional[int]:
        if name_or_index is None:
            return None
        if isinstance(name_or_index, int):
            return name_or_index
        # Search by name substring
        direction = "input" if input else "output"
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if name_or_index.lower() in info["name"].lower():
                logger.info(f"Using {direction} device #{i}: {info['name']}")
                return i
        logger.warning(f"Audio {direction} device '{name_or_index}' not found — using default")
        return None
