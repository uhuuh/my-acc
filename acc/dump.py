"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump context manager for capturing operator calls.
Environment variable ACC_DUMP_ENABLED controls global dump behavior:
- Not set or "1"/"true": dump enabled (default)
- "0"/"false": dump disabled
"""

import os
import traceback
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .serialization import SerializationSession


# Module-level reference for impl patch to access active session
_active_session = None


# ============================================================
# Patch torch.library.impl for custom operator internal capture
# ============================================================

_original_impl = None
_patch_installed = False


def _install_impl_patch():
    global _original_impl, _patch_installed
    if _patch_installed:
        return
    if not hasattr(torch.library, 'impl'):
        return
    _original_impl = torch.library.impl
    _patch_installed = True
    print("[DUMP PATCH] Installing torch.library.impl patch")

    def patched_impl(qualname, types, func=None, *, lib=None):
        def wrap(f):
            def wrapped(*args, **kwargs):
                class NestedMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        result = func(*args, **(kwargs or {}))
                        if _active_session is not None:
                            stack = traceback.extract_stack()
                            filepath, filename, func_name, lineno = "", "", "", 0
                            for frame_info in reversed(stack):
                                if not frame_info.filename.endswith('dump.py'):
                                    filepath = frame_info.filename
                                    filename = os.path.basename(frame_info.filename)
                                    func_name = frame_info.name
                                    lineno = frame_info.lineno
                                    break
                            _active_session.save_operation(
                                func, filepath, filename, func_name, lineno,
                                args, kwargs or {}, result
                            )
                        return result
                mode = NestedMode()
                mode.__enter__()
                try:
                    return f(*args, **kwargs)
                finally:
                    mode.__exit__(None, None, None)
            return wrapped
        if func is None:
            def decorator(f):
                return _original_impl(qualname, types, wrap(f), lib=lib)
            return decorator
        else:
            return _original_impl(qualname, types, wrap(func), lib=lib)

    torch.library.impl = patched_impl


# Install patch on import
_install_impl_patch()


# ============================================================
# ops_dump context manager (entry point)
# ============================================================

class ops_dump(TorchDispatchMode):
    """
    Context manager and decorator for dumping PyTorch operator calls.

    Usage:
        with ops_dump('/path/to/dump'):
            model(input)
    """

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240, enable_async_io: bool = True):
        self.dump_path = dump_path
        self.max_tensor_size_mb = max_tensor_size_mb
        self.enable_async_io = enable_async_io
        self.session = SerializationSession(dump_path, max_tensor_size_mb, enable_async_io)
        self.enabled = os.environ.get('ACC_DUMP_ENABLED', '1').lower() not in ('0', 'false', 'no', 'off')
        self._is_nested = False  # Flag to track if this is a nested dump

    def __enter__(self):
        global _active_session
        if not self.enabled:
            return super().__enter__()
        # Check if there's already an active session - nested dump should be ignored
        if _active_session is not None:
            print(f"[DUMP WARN] Nested ops_dump ignored, all operators will be captured by outer session")
            self._is_nested = True
            return self  # Don't enter TorchDispatchMode for nested dump
        self.session.start()
        _active_session = self.session
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_session
        if self._is_nested:
            # Nested dump - don't do anything, outer session handles everything
            return False
        if self.enabled:
            self.session.end()
            _active_session = None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self.enabled:
            return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        stack = traceback.extract_stack()
        filepath, filename, func_name, lineno = "", "", "", 0
        for frame_info in reversed(stack):
            if not frame_info.filename.endswith('dump.py'):
                filepath = frame_info.filename
                filename = os.path.basename(frame_info.filename)
                func_name = frame_info.name
                lineno = frame_info.lineno
                break
        self.session.save_operation(func, filepath, filename, func_name, lineno, args, kwargs, result)
        return result

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper