"""
AsyncIO-based IO Layer for PyTorch Operator Dump Tool.
Provides async file writing to avoid blocking main thread.
"""

import asyncio
import json
import os
import pickle
import time
import torch
from typing import Any, Set, Optional, Callable, Dict, Tuple
import threading


class FileHandler:
    """File format handler with registered read/write methods per extension."""

    def __init__(self):
        def _write_json(p, c):
            with open(p, 'w') as f:
                json.dump(c, f, indent=2)
        def _read_json(p):
            with open(p, 'r') as f:
                return json.load(f)
        def _write_pkl(p, c):
            with open(p, 'wb') as f:
                pickle.dump(c, f)
        def _read_pkl(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
        def _write_pt(p, c):
            with open(p, 'wb') as f:
                torch.save(c, f)
        def _read_pt(p):
            with open(p, 'rb') as f:
                return torch.load(f, weights_only=False)

        self._handlers: Dict[str, Tuple[Callable, Callable]] = {
            '.json': (_write_json, _read_json),
            '.pkl': (_write_pkl, _read_pkl),
            '.pt': (_write_pt, _read_pt),
        }

    def write(self, file_path: str, content):
        """Write content to file using the registered handler for the extension."""
        ext = os.path.splitext(file_path)[1]
        if ext not in self._handlers:
            raise ValueError(f"Unsupported file extension '{ext}' for {file_path}")
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        write_fn, _ = self._handlers[ext]
        write_fn(file_path, content)

    def read(self, file_path: str):
        """Read file using the registered handler for the extension."""
        ext = os.path.splitext(file_path)[1]
        if ext not in self._handlers:
            raise ValueError(f"Unsupported file extension '{ext}' for {file_path}")
        _, read_fn = self._handlers[ext]
        return read_fn(file_path)


class IOWriter:
    """IO Writer with async loop for concurrent file writes."""

    def __init__(self, enable_async: bool = True, monitor_interval: float = 5.0):
        self.enable_async = enable_async
        self._pending_files: Set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._monitor_interval = monitor_interval
        self._bytes_written = 0
        self._last_monitor_time = time.time()

        self._handler = FileHandler()

        if enable_async:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop).result()

    @property
    def handler(self) -> FileHandler:
        """Expose the underlying FileHandler for custom registration."""
        return self._handler

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()

    def wait_complete(self, timeout: float = 5.0):
        """Wait for pending writes, then stop loop."""
        if not self.enable_async:
            return
        while timeout > 0 and len(self._pending_files) > 0:
            time.sleep(0.05)
            timeout -= 0.05
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)

    def write(self, file_path: str, content):
        """Write file. Async if enabled, otherwise sync."""
        if self.enable_async:
            self._pending_files.add(file_path)
            asyncio.run_coroutine_threadsafe(self._write_async(file_path, content), self._loop)
        else:
            self._write_sync(file_path, content)

    def _write_sync(self, file_path: str, content):
        """Synchronous write via FileHandler."""
        self._handler.write(file_path, content)
        self._bytes_written += os.path.getsize(file_path)

    async def _write_async(self, file_path: str, content):
        """Async write task."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._write_sync, file_path, content)
            self._check_monitor()
        except Exception as e:
            print(f"[IO ERROR] Failed to write {file_path}: {e}")
        finally:
            self._pending_files.discard(file_path)

    def read(self, file_path: str):
        """Read file via FileHandler extension dispatch."""
        return self._handler.read(file_path)

    def _check_monitor(self):
        """检查并打印监控日志"""
        now = time.time()
        elapsed = now - self._last_monitor_time
        if elapsed >= self._monitor_interval:
            pending_count = len(self._pending_files)
            throughput = self._bytes_written / elapsed if elapsed > 0 else 0
            throughput_str = self._format_bytes(throughput)
            print(f"[IO MONITOR] Pending: {pending_count} files | Throughput: {throughput_str}/s")
            self._bytes_written = 0
            self._last_monitor_time = now

    def _format_bytes(self, bytes_per_sec: float) -> str:
        """格式化字节吞吐，带千分位符号"""
        units = ['B', 'KB', 'MB', 'GB']
        value = bytes_per_sec
        unit_idx = 0
        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1
        return f"{value:,.2f} {units[unit_idx]}"