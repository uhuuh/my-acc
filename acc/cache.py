"""Content-addressable cache for tensor/numpy deduplication."""

import hashlib
import os
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
    def from_obj(cls, obj: Any) -> 'CacheEntry':
        """从 tensor/numpy 创建 CacheEntry"""
        storage = _extract_storage(obj)
        return cls(
            cache_id=_compute_hash(storage),
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


def _compute_hash(storage: torch.Tensor) -> str:
    """计算 tensor 存储块的 BLAKE2b 哈希"""
    # BFloat16 不被 numpy 支持，转为 float32 计算 hash
    if storage.dtype == torch.bfloat16:
        storage = storage.float()
    return hashlib.blake2b(storage.numpy().tobytes(), digest_size=32).hexdigest()


class CacheManager:
    """Content-addressable tensor/numpy cache."""

    def __init__(self, cache_dir: str, io_writer: Optional[IOWriter] = None):
        self.cache_dir = cache_dir
        self._io_writer = io_writer
        self._save_cache_map: Dict[str, bool] = {}  # cache_id -> 是否已写入
        self._load_cache_map: Dict[str, torch.Tensor] = {}  # cache_id -> 存储块

    def save(self, data: Any) -> Any:
        """遍历 data，将 tensor/numpy 替换为 CacheEntry，首次写入文件"""
        def processor(obj: Any) -> Any:
            if not isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj
            entry = CacheEntry.from_obj(obj)
            if entry.cache_id not in self._save_cache_map:
                filepath = os.path.join(self.cache_dir, f"{entry.cache_id}.pkl")
                self._io_writer.write(filepath, _extract_storage(obj))
                self._save_cache_map[entry.cache_id] = True
            return entry
        return self._traverse(data, processor)

    def load(self, data: Any) -> Any:
        """遍历 data，将 CacheEntry 还原为 tensor/numpy，首次从文件加载"""
        def processor(obj: Any) -> Any:
            if not isinstance(obj, CacheEntry):
                return obj
            if obj.cache_id not in self._load_cache_map:
                filepath = os.path.join(self.cache_dir, f"{obj.cache_id}.pkl")
                self._load_cache_map[obj.cache_id] = self._io_writer.read(filepath)
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