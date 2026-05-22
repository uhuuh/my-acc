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


class IOWriter:
    """IO Writer with async loop for concurrent file writes."""

    def __init__(self, enable_async: bool = True):
        self.enable_async = enable_async
        self._pending_files: Set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

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
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(content, f)

    async def _write_async(self, file_path: str, content):
        """Async write task."""
        self._pending_files.add(file_path)
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._write_sync, file_path, content)
        except Exception as e:
            print(f"[IO ERROR] Failed to write {file_path}: {e}")
        finally:
            self._pending_files.discard(file_path)

    def read(self, file_path: str):
        """Read file. Returns str for text, object for pickle."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'rb') as f:
                return pickle.load(f)