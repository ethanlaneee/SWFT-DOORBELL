# SWFT Doorbell — AI-Powered Smart Doorbell System

An intelligent doorbell that uses computer vision, face recognition, and a Claude AI brain to greet visitors, identify known guests, and notify homeowners in real time.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi Device                │
│  camera.py → vision.py → ai_brain.py → tts.py      │
│  audio.py  → stt.py   ↗              → speaker      │
│                doorbell.py (orchestrator)           │
└───────────────────────┬─────────────────────────────┘
                        │ WebSocket / REST
┌───────────────────────▼─────────────────────────────┐
│                  Flask Server (server/)              │
│  Events, snapshots, known-faces DB, notifications   │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│               Web Dashboard (dashboard/)             │
│  Live feed, visitor log, face management, settings  │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Device

```bash
cd device
pip install -r requirements.txt
cp config.example.yaml config.yaml   # fill in API keys & settings
python main.py
```

### Server

```bash
cd server
pip install -r requirements.txt
python app.py
```

### Dashboard

Open `http://<server-host>:5000` in your browser.

## Configuration

Copy `device/config.example.yaml` to `device/config.yaml` and set:

| Key | Description |
|-----|-------------|
| `anthropic.api_key` | Claude API key from console.anthropic.com |
| `server.url` | URL of the Flask server |
| `camera.device_index` | Camera device index (default `0`) |
| `audio.input_device` | Microphone device name or index |
| `notifications.webhook_url` | Optional webhook for push notifications |

## Features

- **Face recognition** — greets known visitors by name
- **YOLO object detection** — detects packages, vehicles, and more  
- **Two-way audio** — visitors can speak; AI responds via speaker
- **Claude AI brain** — context-aware, natural conversation
- **Real-time dashboard** — live snapshots and visitor log
- **Push notifications** — webhook / SMS alerts for homeowners

## Requirements

- Python 3.10+
- Raspberry Pi 4 (or any Linux device with USB camera + speaker)
- Anthropic API key
