"""Claude-powered conversation brain for the doorbell."""

import base64
from typing import Optional

import anthropic
import cv2
import numpy as np
from loguru import logger

from config import cfg
from vision import VisitorInfo

_SYSTEM_PROMPT = """\
You are SWFT, a friendly and helpful AI doorbell assistant. Your role is to:

1. Greet visitors warmly — use their name if they are a known resident or recognised guest.
2. Relay messages to the homeowner (e.g. package delivery, visitor waiting).
3. Ask for visitor name and purpose if they are unknown.
4. Answer basic questions (is the homeowner home, estimated wait times, etc.) using only
   information explicitly provided in the conversation context.
5. Keep responses concise — this is a doorbell, not a phone call. 2-3 sentences max.
6. Never reveal sensitive home information (alarm codes, schedules, etc.).
7. Be polite but firm if a visitor seems suspicious.

Tone: warm, professional, efficient.
"""


class AIBrain:
    def __init__(self) -> None:
        api_key: str = cfg.get("anthropic", "api_key", default="")
        if not api_key or api_key.startswith("YOUR_"):
            raise ValueError(
                "Anthropic API key not configured. Set 'anthropic.api_key' in config.yaml."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model: str = cfg.get("anthropic", "model", default="claude-opus-4-7")
        self._max_turns: int = cfg.get("doorbell", "max_conversation_turns", default=6)
        self._history: list[dict] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def greet(self, visitor_info: VisitorInfo, frame: Optional[np.ndarray] = None) -> str:
        """Generate an opening greeting based on what the camera sees."""
        context = self._build_visitor_context(visitor_info)
        content: list = [{"type": "text", "text": context}]

        if frame is not None:
            img_b64 = self._frame_to_b64(frame)
            if img_b64:
                content.insert(
                    0,
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                )

        self._history = [{"role": "user", "content": content}]
        return self._complete()

    def respond(self, visitor_text: str) -> str:
        """Continue an ongoing conversation with the visitor's spoken input."""
        if len(self._history) >= self._max_turns * 2:
            self._history = self._history[-(self._max_turns * 2 - 2):]

        self._history.append({"role": "user", "content": visitor_text})
        return self._complete()

    def reset(self) -> None:
        self._history = []

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _complete(self) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=self._history,
            )
            reply = response.content[0].text.strip()
            self._history.append({"role": "assistant", "content": reply})
            logger.debug(f"AI reply: {reply}")
            return reply
        except anthropic.APIError as exc:
            logger.error(f"Claude API error: {exc}")
            return "I'm having trouble connecting right now. Please knock or ring the bell."

    @staticmethod
    def _build_visitor_context(info: VisitorInfo) -> str:
        parts = ["A visitor has arrived at the door."]
        if info.known and info.name:
            parts.append(f"Face recognition identified them as: {info.name}.")
        else:
            parts.append("The visitor is not in the known-faces database.")
        if info.objects:
            labels = [f"{d.label} ({d.confidence:.0%})" for d in info.objects]
            parts.append(f"Detected objects: {', '.join(labels)}.")
        parts.append("Please greet the visitor appropriately.")
        return " ".join(parts)

    @staticmethod
    def _frame_to_b64(frame: np.ndarray) -> Optional[str]:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buf.tobytes()).decode()
        except Exception as exc:
            logger.warning(f"Could not encode frame for vision: {exc}")
            return None
