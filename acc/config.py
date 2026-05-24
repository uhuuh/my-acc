"""Centralized config for ACC: init() overrides env vars, getters read from env."""

import os
from typing import Optional


def _set_with_warning(key: str, value: Optional[str]):
    if value is None:
        return
    old = os.environ.get(key)
    if old is not None and old != value:
        print(f"[CONFIG WARN] env var {key}={old!r} conflicts with init value {value!r}, using init value")
    os.environ[key] = value


def init(**kwargs):
    """Initialize config from kwargs (e.g. dump_path=..., max_tensor_size_mb=...)."""
    for raw_key, value in kwargs.items():
        env_key = f"ACC_{raw_key.upper()}"
        _set_with_warning(env_key, str(value) if value is not None else None)
    for key in sorted(os.environ):
        if key.startswith('ACC_'):
            print(f"[CONFIG] {key}={os.environ[key]}")


def get_dump_path() -> str:
    return os.environ.get('ACC_DUMP_PATH', '')


def get_dump_enabled() -> bool:
    val = os.environ.get('ACC_DUMP_ENABLED', '1')
    return val.lower() not in ('0', 'false', 'no', 'off')


def get_max_tensor_size_mb(default: int = 10240) -> int:
    return int(os.environ.get('ACC_MAX_TENSOR_SIZE_MB', str(default)))
