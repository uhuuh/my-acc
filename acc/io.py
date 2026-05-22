"""
AsyncIO-based IO Layer for PyTorch Operator Dump Tool.
Provides async file writing to avoid blocking main thread.
"""

import asyncio
import threading
import atexit
import signal
import os
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
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)