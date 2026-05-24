"""
Simple IO Layer for PyTorch Operator Dump Tool.
Uses a queue-based worker thread for non-blocking file writes.
"""

import json
import os
import pickle
import queue
import time
import atexit
import torch
import threading
from typing import Dict, Tuple, Callable


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
    def __init__(self, name: str = "", enable_async: bool = None, on_done=None):
        self.name = name
        self._on_done = on_done
        if enable_async is None:
            from .config import config
            enable_async = config.async_io
        self.enable_async = enable_async
        self._pending_files = set()
        self._bytes_written = 0
        self._files_written = 0
        self._last_monitor_time = 0.0
        self._handler = FileHandler()
        self._thread = None
        self._stopped = False
        if enable_async:
            self._queue = queue.Queue()

    def start(self):
        from .config import config
        print(f"[IO] {self.name} started" + (" (async)" if self.enable_async else " (sync)"))
        if self.enable_async:
            self._last_monitor_time = time.time()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            if config.io_flush_mode == "atexit":
                atexit.register(self._atexit_flush)

    def stop(self):
        if not self.enable_async:
            print(f"[IO] {self.name} stopped")
            return
        self._queue.put(None)
        self._stopped = True
        from .config import config
        if config.io_flush_mode == "stop":
            self._flush()
        print(f"[IO] {self.name} stopped")

    def _atexit_flush(self):
        if not self.enable_async:
            return
        if not self._stopped:
            self._queue.put(None)
        self._flush()

    def _flush(self):
        while True:
            pending = len(self._pending_files)
            if pending == 0:
                break
            print(f"[IO] {self.name} remaining: {pending} files")
            time.sleep(1)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def save(self, file_path, content):
        if self.enable_async:
            self._pending_files.add(file_path)
            self._queue.put_nowait((file_path, content))
        else:
            self._handler.write(file_path, content)
            self._bytes_written += os.path.getsize(file_path)
            self._files_written += 1
            self._check_on_done(content)

    def load(self, file_path):
        return self._handler.read(file_path)

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            file_path, content = item
            try:
                self._handler.write(file_path, content)
                self._bytes_written += os.path.getsize(file_path)
                self._files_written += 1
                self._check_monitor()
            except Exception as e:
                print(f"[IO ERROR] Failed to write {file_path}: {e}")
            finally:
                self._pending_files.discard(file_path)
                self._check_on_done(content)

    def _check_on_done(self, content):
        if self._on_done:
            self._on_done(content)

    def _check_monitor(self):
        from .config import config
        now = time.time()
        elapsed = now - self._last_monitor_time
        if elapsed >= config.io_monitor_interval:
            pending_count = len(self._pending_files)
            throughput = self._bytes_written / elapsed if elapsed > 0 else 0
            throughput_str = self._format_bytes(throughput)
            print(f"[IO MONITOR] {self.name}: Written: {self._files_written} files ({self._format_bytes(self._bytes_written)}) | Pending: {pending_count} files | Throughput: {throughput_str}/s")
            self._last_monitor_time = now

    def _format_bytes(self, bytes_per_sec):
        units = ['B', 'KB', 'MB', 'GB']
        value = bytes_per_sec
        unit_idx = 0
        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1
        return f"{value:,.2f} {units[unit_idx]}"
