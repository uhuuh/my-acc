"""
AsyncIO-based IO Layer for PyTorch Operator Dump Tool.
Provides async file writing to avoid blocking main thread.
"""

import asyncio
import threading
import atexit
import signal
import os
import json
import pickle
import time
from typing import Set, Optional


class IOWriter:
    """
    IO Writer manager with async loop for concurrent file writes.
    Simple interface: write(file_path, content)
    """

    def __init__(self, enable_async: bool = True):
        """
        Initialize IOWriter.

        Args:
            enable_async: Global config for async/sync mode (default: True)
        """
        self.enable_async = enable_async
        self._pending_files: Set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if enable_async:
            self._start_loop()
            self._register_exit_handlers()

    def _start_loop(self):
        """Start asyncio event loop in dedicated thread"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AsyncIO-Loop"
        )
        self._thread.start()

        # Wait for loop to start
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop).result()

    def _run_loop(self):
        """Thread target: run event loop"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()

    def _register_exit_handlers(self):
        """Register atexit and signal handlers"""
        atexit.register(self._on_exit)

        # Signal handlers (Unix/Linux, Windows only supports SIGINT)
        if os.name != 'nt':  # Unix/Linux
            signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_exit(self):
        """atexit callback: check pending files"""
        if not self.enable_async:
            return

        with self._lock:
            pending = list(self._pending_files)

        if pending:
            print(f"[IO WARN] Pending write tasks on exit ({len(pending)} files):")
            for file_path in pending:
                print(f"  - {file_path}")

    def _on_signal(self, signum, frame):
        """Signal handler callback"""
        self._on_exit()
        # Re-trigger default handler
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def shutdown(self):
        """Shutdown async loop and wait for pending tasks"""
        if not self.enable_async:
            return

        if self._loop and self._thread:
            # Wait for pending files to complete (with timeout)
            timeout = 5.0
            start = time.time()
            while self._pending_files and (time.time() - start) < timeout:
                time.sleep(0.01)

            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def write(self, file_path: str, content):
        """
        Write file asynchronously (or synchronously if enable_async=False).

        Args:
            file_path: File path (keep original format, no suffix)
            content: File content (auto-serialize based on extension)
        """
        if not self.enable_async:
            self._write_sync(file_path, content)
        else:
            self._write_async(file_path, content)

    def _write_sync(self, file_path: str, content):
        """Synchronous file write"""
        self._serialize_and_write(file_path, content)

    def _write_async(self, file_path: str, content):
        """Asynchronous file write via loop"""
        with self._lock:
            self._pending_files.add(file_path)

        coro = self._async_write_task(file_path, content)
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _async_write_task(self, file_path: str, content):
        """Async write task"""
        try:
            # Use asyncio-friendly file I/O (thread pool fallback)
            await asyncio.get_event_loop().run_in_executor(
                None, self._serialize_and_write, file_path, content
            )
        except Exception as e:
            print(f"[IO ERROR] Failed to write {file_path}: {e}")
        finally:
            with self._lock:
                self._pending_files.discard(file_path)

    def _serialize_and_write(self, file_path: str, content):
        """Serialize and write file (sync operation)"""
        # Create parent directory if needed
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Auto-serialize based on extension
        if file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(content, f)
        else:
            # Default: write as text
            with open(file_path, 'w') as f:
                f.write(str(content))