"""Captures frames from a local camera and detects motion."""

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger

from config import cfg


class Camera:
    def __init__(self) -> None:
        self._device_index: int = cfg.get("camera", "device_index", default=0)
        self._width: int = cfg.get("camera", "width", default=1280)
        self._height: int = cfg.get("camera", "height", default=720)
        self._fps: int = cfg.get("camera", "fps", default=30)
        self._snapshot_interval: float = cfg.get(
            "camera", "snapshot_interval_seconds", default=1.0
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._motion_callbacks: list[Callable[[np.ndarray], None]] = []
        self._frame_callbacks: list[Callable[[np.ndarray], None]] = []

        self._motion_threshold: float = 5000.0  # sum of diff pixels

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def on_motion(self, callback: Callable[[np.ndarray], None]) -> None:
        self._motion_callbacks.append(callback)

    def on_frame(self, callback: Callable[[np.ndarray], None]) -> None:
        self._frame_callbacks.append(callback)

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self._device_index}")

        logger.info(f"Camera opened: {self._width}x{self._height} @ {self._fps}fps")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self.get_frame()
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _capture_loop(self) -> None:
        last_snapshot = 0.0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera read failed — retrying")
                time.sleep(0.1)
                continue

            with self._lock:
                self._latest_frame = frame

            now = time.monotonic()
            if now - last_snapshot >= self._snapshot_interval:
                last_snapshot = now
                for cb in self._frame_callbacks:
                    try:
                        cb(frame.copy())
                    except Exception as exc:
                        logger.error(f"Frame callback error: {exc}")

            if self._detect_motion(frame):
                for cb in self._motion_callbacks:
                    try:
                        cb(frame.copy())
                    except Exception as exc:
                        logger.error(f"Motion callback error: {exc}")

    def _detect_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        delta = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        motion_score = float(thresh.sum())
        self._prev_gray = gray
        return motion_score > self._motion_threshold
