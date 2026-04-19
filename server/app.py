"""Flask server — receives events from the device and serves the dashboard."""

import base64
import io
import os
from collections import deque
from datetime import datetime
from functools import wraps
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from loguru import logger
from PIL import Image

load_dotenv()

app = Flask(
    __name__,
    static_folder="../dashboard/static",
    template_folder="../dashboard/templates",
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

API_KEY: str = os.environ.get("SERVER_API_KEY", "")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "data", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

_event_log: deque[dict] = deque(maxlen=200)
_latest_snapshot_b64: Optional[str] = None


# ------------------------------------------------------------------ #
# Auth
# ------------------------------------------------------------------ #

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ------------------------------------------------------------------ #
# Device API endpoints
# ------------------------------------------------------------------ #

@app.route("/api/event", methods=["POST"])
@require_api_key
def receive_event():
    data = request.get_json(silent=True) or {}
    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    _event_log.appendleft(data)
    logger.info(f"Event: {data.get('event')} — {data.get('visitor_name')}")
    socketio.emit("new_event", data)

    webhook_url = os.environ.get("NOTIFICATION_WEBHOOK_URL")
    if webhook_url:
        try:
            import requests as req
            req.post(webhook_url, json=data, timeout=5)
        except Exception as exc:
            logger.warning(f"Webhook delivery failed: {exc}")

    return jsonify({"status": "ok"}), 200


@app.route("/api/snapshot", methods=["POST"])
@require_api_key
def receive_snapshot():
    global _latest_snapshot_b64
    image_bytes = request.data
    if not image_bytes:
        return jsonify({"error": "empty body"}), 400

    _latest_snapshot_b64 = base64.b64encode(image_bytes).decode()
    socketio.emit("snapshot", {"data": _latest_snapshot_b64})
    return jsonify({"status": "ok"}), 200


# ------------------------------------------------------------------ #
# Dashboard API endpoints
# ------------------------------------------------------------------ #

@app.route("/api/events")
def get_events():
    return jsonify(list(_event_log))


@app.route("/api/latest-snapshot")
def get_latest_snapshot():
    if _latest_snapshot_b64 is None:
        return jsonify({"snapshot": None})
    return jsonify({"snapshot": _latest_snapshot_b64})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "events": len(_event_log)})


# ------------------------------------------------------------------ #
# Dashboard
# ------------------------------------------------------------------ #

@app.route("/")
@app.route("/<path:path>")
def dashboard(path="index.html"):
    return send_from_directory(app.template_folder, "index.html")


# ------------------------------------------------------------------ #
# WebSocket
# ------------------------------------------------------------------ #

@socketio.on("connect")
def on_connect():
    logger.debug(f"Dashboard client connected: {request.sid}")
    if _latest_snapshot_b64:
        emit("snapshot", {"data": _latest_snapshot_b64})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"SWFT server starting on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
