"""Centralized config: dataclass singleton, init from env, update via kwargs."""

from dataclasses import dataclass
import os


@dataclass
class Config:
    dump_path: str = ""
    dump_enabled: bool = True
    max_tensor_size_mb: int = 10240

    def __post_init__(self):
        self._load_from_env([])

    def _load_from_env(self, changed: list):
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
            changed.append(f"{key}={val}")

    def _reload_env(self, existing: set, changed: list):
        annotations = type(self).__annotations__
        cls = type(self)
        for name, var_type in annotations.items():
            if name in existing:
                continue
            setattr(self, name, getattr(cls, name))
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
            changed.append(f"{key}={val}")

    def update(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        annotations = type(self).__annotations__
        changed = []
        self._reload_env(set(kwargs), changed)
        for name, value in kwargs.items():
            if name not in annotations:
                print(f"[CONFIG WARN] Unknown config '{name}', skipping")
                continue
            env_key = f"ACC_{name.upper()}"
            old = os.environ.get(env_key)
            if old is not None and old != str(value):
                print(f"[CONFIG WARN] {env_key}={old!r} conflicts with {value!r}, using update")
            os.environ[env_key] = str(value)
            setattr(self, name, value)
            changed.append(f"{name}={value}")
        if changed:
            print(f"[CONFIG] {' | '.join(changed)}")


config = Config()
