"""Orchestrates camera, vision, audio, STT, TTS, and AI brain."""

import io
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
from loguru import logger

from ai_brain import AIBrain
from audio import AudioIO
from camera import Camera
from config import cfg
from stt import SpeechToText
from tts import TextToSpeech
from vision import VisitorInfo, VisionProcessor


class Doorbell:
    def __init__(self) -> None:
        self._camera = Camera()
        self._vision = VisionProcessor()
        self._audio = AudioIO()
        self._stt = SpeechToText()
        self._tts = TextToSpeech()
        self._brain = AIBrain()

        self._server_url: str = cfg.get("server", "url", default="http://localhost:5000")
        self._server_key: str = cfg.get("server", "api_key", default="")
        self._greeting: str = cfg.get(
            "doorbell", "greeting_message",
            default="Hello! I'm the SWFT doorbell assistant.",
        )
        self._motion_cooldown: float = cfg.get(
            "doorbell", "motion_cooldown_seconds", default=30.0
        )
        self._convo_timeout: float = cfg.get(
            "doorbell", "conversation_timeout_seconds", default=60.0
        )

        self._last_motion_time: float = 0.0
        self._in_conversation = False
        self._convo_lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._running = True
        self._camera.on_motion(self._on_motion)
        self._camera.on_frame(self._on_frame_snapshot)
        self._camera.start()
        logger.info("Doorbell ready — watching for visitors")
        while self._running:
            time.sleep(1)

    def stop(self) -> None:
        self._running = False
        self._camera.stop()
        self._audio.close()
        logger.info("Doorbell stopped")

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_motion(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_motion_time < self._motion_cooldown:
            return
        self._last_motion_time = now

        if self._in_conversation:
            return

        logger.info("Motion detected — analysing visitor")
        threading.Thread(
            target=self._handle_visitor, args=(frame,), daemon=True
        ).start()

    def _on_frame_snapshot(self, frame: np.ndarray) -> None:
        jpeg = self._frame_to_jpeg(frame)
        if jpeg:
            self._post_to_server("/api/snapshot", data=jpeg, content_type="image/jpeg")

    # ------------------------------------------------------------------ #
    # Conversation flow
    # ------------------------------------------------------------------ #

    def _handle_visitor(self, frame: np.ndarray) -> None:
        with self._convo_lock:
            if self._in_conversation:
                return
            self._in_conversation = True

        try:
            visitor_info = self._vision.analyze(frame)
            logger.info(
                f"Visitor: known={visitor_info.known}, name={visitor_info.name}, "
                f"objects={[d.label for d in visitor_info.objects]}"
            )
            self._post_visitor_event(visitor_info, frame)

            greeting = self._brain.greet(visitor_info, frame)
            self._tts.speak(greeting)

            deadline = time.monotonic() + self._convo_timeout
            while time.monotonic() < deadline and self._running:
                visitor_text = self._stt.listen()
                if not visitor_text:
                    break
                logger.info(f"Visitor said: '{visitor_text}'")
                reply = self._brain.respond(visitor_text)
                self._tts.speak(reply)

            self._brain.reset()
        finally:
            self._in_conversation = False

    # ------------------------------------------------------------------ #
    # Server integration
    # ------------------------------------------------------------------ #

    def _post_visitor_event(self, info: VisitorInfo, frame: np.ndarray) -> None:
        payload = {
            "event": "visitor",
            "visitor_name": info.name or "Unknown",
            "known": info.known,
            "objects": [d.label for d in info.objects],
        }
        self._post_to_server("/api/event", json=payload)

    def _post_to_server(
        self,
        path: str,
        *,
        json: Optional[dict] = None,
        data: Optional[bytes] = None,
        content_type: str = "application/json",
    ) -> None:
        if not self._server_url:
            return
        url = self._server_url.rstrip("/") + path
        headers = {"X-API-Key": self._server_key, "Content-Type": content_type}
        try:
            if json is not None:
                requests.post(url, json=json, headers=headers, timeout=3)
            elif data is not None:
                requests.post(url, data=data, headers=headers, timeout=3)
        except requests.RequestException as exc:
            logger.debug(f"Server post failed ({path}): {exc}")

    @staticmethod
    def _frame_to_jpeg(frame: np.ndarray) -> Optional[bytes]:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes() if buf is not None else None
