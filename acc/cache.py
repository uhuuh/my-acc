"""Content-addressable cache for tensor/numpy deduplication."""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, List, Dict, Callable, Optional

import numpy as np
import torch

from .io import IOWriter


@dataclass
class CacheEntry:
    """Metadata for a cached tensor/numpy array."""
    cache_id: str    # 存储块 hash (blake2b)
    type: str        # 'tensor' or 'numpy'
    dtype: str       # e.g. 'float32', 'int64'
    shape: List[int] # tensor/array shape

    @classmethod
    def from_obj(cls, obj: Any, mode: str = 'fast') -> 'CacheEntry':
        """从 tensor/numpy 创建 CacheEntry"""
        storage = _extract_storage(obj)
        return cls(
            cache_id=_compute_hash(storage, mode),
            type='tensor' if isinstance(obj, torch.Tensor) else 'numpy',
            dtype=str(obj.dtype).replace('torch.', ''),
            shape=list(obj.shape)
        )

    def to_obj(self, storage: torch.Tensor) -> Any:
        """从 tensor 存储块重建 tensor/numpy"""
        if self.type == 'tensor':
            t = storage
            if list(t.shape) != self.shape:
                t = t.reshape(self.shape)
            return t
        else:
            # numpy 类型
            arr = storage.numpy()
            if list(arr.shape) != self.shape:
                arr = arr.reshape(self.shape)
            return arr


def _extract_storage(obj: Any) -> torch.Tensor:
    """从 tensor/numpy 中提取存储块（tensor）"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().contiguous().cpu()
    # numpy 转 tensor
    return torch.from_numpy(obj).contiguous().cpu()


def _compute_hash(storage: torch.Tensor, mode: str = 'fast') -> str:
    if mode == 'fast':
        return f"ptr_{storage.data_ptr()}_{storage.numel()}_{storage._version}"

    storage = storage.cpu()
    if storage.dtype == torch.bfloat16:
        storage = storage.float()
    return hashlib.blake2b(storage.numpy().tobytes(), digest_size=32).hexdigest()


class CacheManager:
    """Content-addressable tensor/numpy cache."""

    def __init__(self, cache_dir: str, io_writer: Optional[IOWriter] = None, mode: str = 'fast', monitor_interval: float = 5.0):
        self.cache_dir = cache_dir
        self._io_writer = io_writer
        self._mode = mode
        self._monitor_interval = monitor_interval
        self._save_cache_map: Dict[str, bool] = {}  # cache_id -> 是否已写入
        self._load_cache_map: Dict[str, torch.Tensor] = {}  # cache_id -> 存储块
        # 监控统计
        self._save_total = 0      # save 总请求数
        self._save_hits = 0       # save 命中数（已存在 cache_id）
        self._load_total = 0      # load 总请求数
        self._load_hits = 0       # load 命中数（已存在内存缓存）
        self._last_monitor_time = time.time()
        # 历史累计
        self._save_total_history = 0
        self._save_hits_history = 0
        self._load_total_history = 0
        self._load_hits_history = 0

    def _check_monitor(self):
        """检查并打印监控日志"""
        now = time.time()
        elapsed = now - self._last_monitor_time
        if elapsed >= self._monitor_interval:
            # 当前时段统计
            save_rate = self._save_hits / self._save_total if self._save_total > 0 else 0
            load_rate = self._load_hits / self._load_total if self._load_total > 0 else 0
            # 历史累计统计
            save_rate_history = self._save_hits_history / self._save_total_history if self._save_total_history > 0 else 0
            load_rate_history = self._load_hits_history / self._load_total_history if self._load_total_history > 0 else 0
            print(f"[CACHE MONITOR] Save: {self._save_hits}/{self._save_total} ({save_rate:.1%}) | "
                  f"History: {self._save_hits_history}/{self._save_total_history} ({save_rate_history:.1%})")
            print(f"[CACHE MONITOR] Load: {self._load_hits}/{self._load_total} ({load_rate:.1%}) | "
                  f"History: {self._load_hits_history}/{self._load_total_history} ({load_rate_history:.1%})")
            # 累加到历史
            self._save_total_history += self._save_total
            self._save_hits_history += self._save_hits
            self._load_total_history += self._load_total
            self._load_hits_history += self._load_hits
            # 重置当前时段
            self._save_total = 0
            self._save_hits = 0
            self._load_total = 0
            self._load_hits = 0
            self._last_monitor_time = now

    def save(self, data: Any) -> Any:
        """遍历 data，将 tensor/numpy 替换为 CacheEntry，首次写入文件"""
        def processor(obj: Any) -> Any:
            if not isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj
            self._save_total += 1
            t = obj.detach().contiguous() if isinstance(obj, torch.Tensor) else torch.from_numpy(obj)
            cache_id = _compute_hash(t, self._mode)
            entry = CacheEntry(
                cache_id=cache_id,
                type='tensor' if isinstance(obj, torch.Tensor) else 'numpy',
                dtype=str(obj.dtype).replace('torch.', ''),
                shape=list(obj.shape)
            )
            if entry.cache_id not in self._save_cache_map:
                filepath = os.path.join(self.cache_dir, f"{entry.cache_id}.pt")
                if t.device.type == 'cpu':
                    write_storage = t
                else:
                    write_storage = torch.empty(t.size(), dtype=t.dtype, device='cpu', pin_memory=True)
                    write_storage.copy_(t, non_blocking=False)
                self._io_writer.write(filepath, write_storage)
                self._save_cache_map[entry.cache_id] = True
            else:
                self._save_hits += 1
            self._check_monitor()
            return entry
        return self._traverse(data, processor)

    def load(self, data: Any) -> Any:
        """遍历 data，将 CacheEntry 还原为 tensor/numpy，首次从文件加载"""
        def processor(obj: Any) -> Any:
            if not isinstance(obj, CacheEntry):
                return obj
            self._load_total += 1
            if obj.cache_id not in self._load_cache_map:
                filepath = os.path.join(self.cache_dir, f"{obj.cache_id}.pt")
                self._load_cache_map[obj.cache_id] = self._io_writer.read(filepath)
            else:
                self._load_hits += 1
            self._check_monitor()
            return obj.to_obj(self._load_cache_map[obj.cache_id])
        return self._traverse(data, processor)

    def _traverse(self, data: Any, processor: Callable[[Any], Any]) -> Any:
        """递归遍历 dict/list/tuple，对非容器元素调用 processor"""
        if isinstance(data, dict):
            return {k: self._traverse(v, processor) for k, v in data.items()}
        if isinstance(data, list):
            return [self._traverse(item, processor) for item in data]
        if isinstance(data, tuple):
            return tuple(self._traverse(item, processor) for item in data)
        return processor(data)