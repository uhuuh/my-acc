"""Centralized config: dataclass singleton, init from env, update via kwargs."""
from dataclasses import dataclass
import os

@dataclass
class Config:
    dump_path: str = "."
    dump_enabled: bool = True
    max_tensor_size_mb: int = 10240
    io_monitor_interval: float = 5.0
    cache_monitor_interval: float = 5.0
    pool_monitor_interval: float = 5.0
    async_io: bool = True
    async_serialization: bool = True
    io_flush_mode: str = "atexit"

    def __post_init__(self):
        annotations = type(self).__annotations__
        for name, var_type in annotations.items():
            key = f"ACC_{name.upper()}"
            val = os.environ.get(key)
            if val is None:
                continue
            if var_type is bool:
                setattr(self, name, val.lower() not in ('0', 'false', 'no', 'off'))
            elif var_type is int:
                setattr(self, name, int(val))
            elif var_type is float:
                setattr(self, name, float(val))
            else:
                setattr(self, name, val)

    def update(self, **kwargs):
        annotations = type(self).__annotations__
        for name, value in kwargs.items():
            if name not in annotations:
                print(f"[CONFIG WARN] Unknown config '{name}', skipping")
                continue
            old_value = getattr(self, name, None)
            if old_value != value:
                print(f"[CONFIG] {name}: {old_value} -> {value}")
                setattr(self, name, value)
                os.environ[f"ACC_{name.upper()}"] = str(value)

config = Config()
