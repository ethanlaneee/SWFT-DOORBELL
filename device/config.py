"""Loads and validates config.yaml, exposing a single `cfg` singleton."""

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_EXAMPLE_PATH = Path(__file__).parent / "config.example.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        if _EXAMPLE_PATH.exists():
            logger.warning(
                "config.yaml not found — using config.example.yaml. "
                "Copy it to config.yaml and fill in your secrets."
            )
            path = _EXAMPLE_PATH
        else:
            raise FileNotFoundError(
                f"Neither {_CONFIG_PATH} nor {_EXAMPLE_PATH} found."
            )
    else:
        path = _CONFIG_PATH

    with path.open() as f:
        data: dict = yaml.safe_load(f) or {}

    # Allow environment variable overrides for secrets
    env_overrides = {
        "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
        "SERVER_API_KEY": ("server", "api_key"),
        "SERVER_URL": ("server", "url"),
    }
    for env_var, key_path in env_overrides.items():
        val = os.environ.get(env_var)
        if val:
            section, key = key_path
            data.setdefault(section, {})[key] = val

    return data


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"<Config loaded from {_CONFIG_PATH}>"


cfg = _Config(_load())
