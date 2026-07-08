"""Per-session configuration for acc_dump."""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger


DEFAULTS = {
    "dump_path": "",
    "dump_enabled": True,
    "capturer_backends": "ops,module",
    "hash_workers": None,
    "cache_write_workers": None,
}

_CONFIG_FIELDS = frozenset(DEFAULTS)
_AUTO_FIELDS = {
    "hash_workers",
    "cache_write_workers",
}


@dataclass(init=False)
class Config:
    """Config resolved once per acc_dump session.

    Expensive performance knobs default to ``auto``. Callers can still override
    them through kwargs or ACC_* environment variables.
    """

    dump_path: str
    dump_enabled: bool
    capturer_backends: str
    hash_workers: int
    cache_write_workers: int
    dump_dir: str

    def __init__(self, dump_path=None, **kwargs):
        values = dict(DEFAULTS)

        for name, default in DEFAULTS.items():
            env_value = os.environ.get(f"ACC_{name.upper()}")
            if env_value is not None:
                values[name] = _coerce_value(env_value, default, name)

        if dump_path is not None:
            values["dump_path"] = dump_path

        for name, value in kwargs.items():
            if name not in DEFAULTS:
                logger.warning(f"[Config] unknown option '{name}', ignoring")
                continue
            values[name] = value

        _resolve_auto_values(values)

        for name, value in values.items():
            setattr(self, name, value)

        self.dump_dir = _make_dump_dir(self.dump_path)
        self._log(values)

    def _log(self, values: dict):
        logger.info("=== ACC Config ===")
        for name in DEFAULTS:
            logger.info(f"  {name} = {values[name]}")
        logger.info(f"  dump_dir = {self.dump_dir}")


def _coerce_value(value: str, default: Any, name: str):
    if name in _AUTO_FIELDS and value.lower() in ("", "auto", "none"):
        return None
    if isinstance(default, bool):
        return value.lower() not in ("0", "false", "no", "off")
    if isinstance(default, int) or (default is None and name in _AUTO_FIELDS):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _resolve_auto_values(values: dict):
    cpu_count = os.cpu_count() or 4

    auto_workers = min(4, max(1, cpu_count // 2))
    for name in _AUTO_FIELDS:
        values[name] = int(
            auto_workers if values[name] is None else values[name]
        )


def _make_dump_dir(base_path: str) -> str:
    base = base_path or "_acc_dump"
    rank = "None"
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        pass

    pid = os.getpid()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sid = uuid.uuid4().hex[:8]
    return os.path.join(base, f"{rank}-{pid}-{ts}-{sid}")
