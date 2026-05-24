"""
Simple IO Layer for PyTorch Operator Dump Tool.
Uses a queue-based worker thread for non-blocking file writes.
"""

import json
import os
import pickle
import queue
import time
import torch
import threading
from typing import Any, Set, Optional, Dict, Tuple, Callable


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
    """IO Writer with a background thread for async file writes."""

    def __init__(self, name: str = "", enable_async: bool = True, monitor_interval: float = 5.0):
        self.name = name
        self.enable_async = enable_async
        self._pending_files: Set[str] = set()
        self._monitor_interval = monitor_interval
        self._bytes_written = 0
        self._last_monitor_time = time.time()
        self._handler = FileHandler()

        if enable_async:
            self._queue: queue.Queue = queue.Queue()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        else:
            self._thread = None

    @property
    def handler(self) -> FileHandler:
        return self._handler

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            file_path, content = item
            try:
                self._handler.write(file_path, content)
                self._bytes_written += os.path.getsize(file_path)
                self._check_monitor()
            except Exception as e:
                print(f"[IO ERROR] Failed to write {file_path}: {e}")
            finally:
                self._pending_files.discard(file_path)
                self._queue.task_done()

    def wait_complete(self, timeout: float = 1.0):
        if not self.enable_async:
            print("[IO] All writes completed")
            return
        self._queue.put(None)
        while self._thread.is_alive():
            self._thread.join(timeout=timeout)
            print(f"[IO] Waited {timeout}s for worker thread")
        print("[IO] All writes completed")

    def write(self, file_path: str, content):
        if self.enable_async:
            self._pending_files.add(file_path)
            self._queue.put((file_path, content))
        else:
            self._handler.write(file_path, content)
            self._bytes_written += os.path.getsize(file_path)

    def read(self, file_path: str):
        return self._handler.read(file_path)

    def _check_monitor(self):
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
        units = ['B', 'KB', 'MB', 'GB']
        value = bytes_per_sec
        unit_idx = 0
        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1
        return f"{value:,.2f} {units[unit_idx]}"
