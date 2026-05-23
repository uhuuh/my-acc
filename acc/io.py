"""
AsyncIO-based IO Layer for PyTorch Operator Dump Tool.
Provides async file writing to avoid blocking main thread.
"""

import asyncio
import os
import pickle
import time
from typing import Set, Optional
import threading
import torch


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

        if enable_async:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop).result()

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
        """Synchronous write."""
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if isinstance(content, str):
            with open(file_path, 'w') as f:
                f.write(content)
        elif isinstance(content, torch.Tensor):
            # 使用 torch.save 对 tensor 更高效
            torch.save(content, file_path)
            self._bytes_written += content.element_size() * content.numel()
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(content, f)

    async def _write_async(self, file_path: str, content):
        """Async write task."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._write_sync, file_path, content)
            self._check_monitor()
        except Exception as e:
            print(f"[IO ERROR] Failed to write {file_path}: {e}")
        finally:
            self._pending_files.discard(file_path)

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

    def read(self, file_path: str):
        """Read file. Returns str for text, tensor or object for pickle."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except UnicodeDecodeError:
            # 先尝试 torch.load（对 tensor 文件更高效）
            try:
                return torch.load(file_path, weights_only=True)
            except Exception:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)