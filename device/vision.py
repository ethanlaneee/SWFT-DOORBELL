"""YOLO object detection + face recognition on camera frames."""

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np
from loguru import logger
from ultralytics import YOLO

from config import cfg


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class VisitorInfo:
    known: bool
    name: Optional[str]
    objects: list[Detection] = field(default_factory=list)
    face_image: Optional[np.ndarray] = None


class VisionProcessor:
    def __init__(self) -> None:
        model_path: str = cfg.get("vision", "yolo_model", default="yolov8n.pt")
        self._yolo = YOLO(model_path)
        logger.info(f"YOLO model loaded: {model_path}")

        self._confidence: float = cfg.get("vision", "confidence_threshold", default=0.5)
        self._classes: list[str] = cfg.get(
            "vision", "detection_classes", default=["person"]
        )

        face_db_path: str = cfg.get("vision", "face_db_path", default="known_faces/")
        self._face_db_dir = Path(face_db_path)
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._load_known_faces()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, frame: np.ndarray) -> VisitorInfo:
        objects = self._detect_objects(frame)
        name, face_img = self._identify_face(frame)
        known = name is not None
        return VisitorInfo(known=known, name=name, objects=objects, face_image=face_img)

    def register_face(self, name: str, frame: np.ndarray) -> bool:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            logger.warning(f"No face found in frame for '{name}'")
            return False

        encoding = face_recognition.face_encodings(rgb, locations)[0]
        self._known_encodings.append(encoding)
        self._known_names.append(name)
        self._save_known_faces()
        logger.info(f"Registered face for '{name}'")
        return True

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _detect_objects(self, frame: np.ndarray) -> list[Detection]:
        results = self._yolo(frame, conf=self._confidence, verbose=False)
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                label = self._yolo.names[int(box.cls)]
                if label not in self._classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections

    def _identify_face(
        self, frame: np.ndarray
    ) -> tuple[Optional[str], Optional[np.ndarray]]:
        if not self._known_encodings:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            return None, None

        encodings = face_recognition.face_encodings(rgb, locations)
        for enc, loc in zip(encodings, locations):
            matches = face_recognition.compare_faces(self._known_encodings, enc)
            if not any(matches):
                continue
            distances = face_recognition.face_distance(self._known_encodings, enc)
            best_idx = int(np.argmin(distances))
            if matches[best_idx]:
                top, right, bottom, left = loc
                face_crop = frame[top:bottom, left:right]
                return self._known_names[best_idx], face_crop

        return None, None

    def _load_known_faces(self) -> None:
        db_file = self._face_db_dir / "encodings.pkl"
        if not db_file.exists():
            logger.info("No known-faces database found — starting fresh")
            return
        with db_file.open("rb") as f:
            data = pickle.load(f)
        self._known_encodings = data.get("encodings", [])
        self._known_names = data.get("names", [])
        logger.info(f"Loaded {len(self._known_names)} known face(s)")

    def _save_known_faces(self) -> None:
        self._face_db_dir.mkdir(parents=True, exist_ok=True)
        db_file = self._face_db_dir / "encodings.pkl"
        with db_file.open("wb") as f:
            pickle.dump(
                {"encodings": self._known_encodings, "names": self._known_names}, f
            )
