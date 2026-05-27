"""Content-addressable cache for tensor/numpy deduplication."""

import os
import time as _time_module
from dataclasses import dataclass
from typing import Any, List, Set, Optional, Dict

import numpy as np
import torch

from .memory import Storage, MemoryAllocator
from .io import IOWriter


@dataclass
class CacheEntry:
    """Metadata for a cached tensor/numpy array."""
    cache_id: str
    type: str
    dtype: str
    shape: List[int]


def _tensor_size_mb(obj) -> float:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size() / (1024 * 1024)
    return obj.size * obj.itemsize / (1024 * 1024)


class CacheManager:
    def __init__(self):
        self.cache_dir = None
        self._io = None
        self._save_cached: Set[str] = set()
        self._load_cached: Dict[str, torch.Tensor] = {}
        self._pool = None
        self._max_tensor_size_mb = None
        self._writes_this_interval = 0
        self._writes_total = 0
        self._hits_this_interval = 0
        self._hits_total = 0
        self._last_monitor_time = 0.0
        self._started = False

    def start(self, session_dir):
        from .config import config
        self._started = True
        self.cache_dir = os.path.join(session_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=False)
        self._pool = MemoryAllocator.create("pin")
        self._max_tensor_size_mb = config.max_tensor_size_mb
        self._io = IOWriter(name="cache", on_done=self._on_io_done)
        self._io.start()
        self._save_cached = set()
        self._load_cached = {}
        self._writes_this_interval = 0
        self._writes_total = 0
        self._hits_this_interval = 0
        self._hits_total = 0
        self._last_monitor_time = _time_module.time()
        print(f"[CACHE] started: {self.cache_dir}")

    def stop(self):
        if self._io is not None:
            self._io.stop()
        self._started = False
        print(f"[CACHE] stopped")

    def _on_io_done(self, content):
        if isinstance(content, torch.Tensor) and self._pool is not None:
            self._pool.release(content)

    def _check_monitor(self):
        from .config import config
        now = _time_module.time()
        elapsed = now - self._last_monitor_time
        if elapsed >= config.cache_monitor_interval:
            i_hits = self._hits_this_interval
            i_writes = self._writes_this_interval
            i_rate = i_hits / i_writes * 100 if i_writes > 0 else 0
            h_hits = self._hits_total
            h_writes = self._writes_total
            h_rate = h_hits / h_writes * 100 if h_writes > 0 else 0

            pool_stats = ""
            if hasattr(self._pool, 'pool_stats'):
                s = self._pool.pool_stats()
                if s:
                    pool_bytes, pool_delta = s
                    delta_sign = "+" if pool_delta >= 0 else ""
                    pool_stats = f" | Pool: {self._format_bytes(pool_bytes)} (Δ {delta_sign}{self._format_bytes(abs(pool_delta))})"

            print(f"[CACHE MONITOR] Interval: Hits={i_hits} Writes={i_writes} Rate={i_rate:.1f}%"
                  f" | Historical: Hits={h_hits} Writes={h_writes} Rate={h_rate:.1f}%"
                  f"{pool_stats}")

            self._writes_this_interval = 0
            self._hits_this_interval = 0
            self._last_monitor_time = now

    def _format_bytes(self, num_bytes):
        units = ['B', 'KB', 'MB', 'GB']
        value = float(num_bytes)
        unit_idx = 0
        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1
        return f"{value:,.2f} {units[unit_idx]}"

    def save(self, data):
        if not self._started:
            raise RuntimeError("CacheManager not started")
        def processor(obj):
            if not isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj
            self._writes_this_interval += 1
            self._writes_total += 1
            if self._max_tensor_size_mb is not None:
                size_mb = _tensor_size_mb(obj)
                if size_mb > self._max_tensor_size_mb:
                    print(f"[DUMP WARN] Tensor size {size_mb:.2f} MB exceeds limit, replacing with None")
                    return None
            storage = Storage(obj)
            cache_id = storage.cache_id
            entry = CacheEntry(
                cache_id=cache_id,
                type='tensor' if isinstance(obj, torch.Tensor) else 'numpy',
                dtype=str(obj.dtype).replace('torch.', ''),
                shape=list(obj.shape)
            )
            if cache_id not in self._save_cached:
                storage_tensor = storage.materialize(self._pool)
                filepath = os.path.join(self.cache_dir, f"{cache_id}.pt")
                self._io.save(filepath, storage_tensor)
                self._save_cached.add(cache_id)
            else:
                self._hits_this_interval += 1
                self._hits_total += 1
            self._check_monitor()
            return entry
        return self._traverse(data, processor)

    def load(self, data: Any) -> Any:
        """Traverse data, resolve CacheEntry back to tensors/numpy, cache in memory."""
        def processor(obj: Any) -> Any:
            if not isinstance(obj, CacheEntry):
                return obj
            if obj.cache_id not in self._load_cached:
                filepath = os.path.join(self.cache_dir, f"{obj.cache_id}.pt")
                self._load_cached[obj.cache_id] = torch.load(filepath, weights_only=False)
            storage = self._load_cached[obj.cache_id]
            if obj.type == 'tensor':
                t = storage
                if list(t.shape) != obj.shape:
                    t = t.reshape(obj.shape)
                target_dtype = getattr(torch, obj.dtype.replace('torch.', ''))
                if t.dtype != target_dtype:
                    t = t.to(target_dtype)
                return t
            arr = storage.numpy()
            if list(arr.shape) != obj.shape:
                arr = arr.reshape(obj.shape)
            return arr
        return self._traverse(data, processor)

    @staticmethod
    def _traverse(data: Any, processor) -> Any:
        if isinstance(data, dict):
            return {k: CacheManager._traverse(v, processor) for k, v in data.items()}
        if isinstance(data, list):
            return [CacheManager._traverse(item, processor) for item in data]
        if isinstance(data, tuple):
            return tuple(CacheManager._traverse(item, processor) for item in data)
        if isinstance(data, (type(None), int, float, str, bool, bytes)):
            return data
        result = processor(data)
        if result is data:
            return None
        return result
