"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump context manager for capturing operator calls.
Environment variable ACC_DUMP_ENABLED controls global dump behavior:
- Not set or "1"/"true": dump enabled (default)
- "0"/"false": dump disabled
"""

import os
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .serialization import SerializationSession


# Module-level reference for impl patch to access active session
_active_session = None


# ============================================================
# Patch torch.library.impl — wrap impls with a nested
# TorchDispatchMode so internal operator calls are captured.
# This is needed because TorchDispatchMode is temporarily
# disabled while inside __torch_dispatch__, so the outer
# ops_dump cannot see dispatches from within a kernel.
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
                if _active_session is None:
                    return f(*args, **kwargs)
                with ops_dump():
                    return f(*args, **kwargs)
            return wrapped

        # torch.library.impl(f) 直接注册 vs @impl 装饰器语法
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

    def __init__(self, dump_path: str = None):
        self.enabled = os.environ.get('ACC_DUMP_ENABLED', '1').lower() not in ('0', 'false', 'no', 'off')
        self._in_dispatch = False
        if not self.enabled:
            self._owns_session = False
            return
        self._owns_session = _active_session is None
        self.session = SerializationSession(dump_path) if self._owns_session else _active_session

    @property
    def max_tensor_size_mb(self):
        return self.session.max_tensor_size_mb if self.enabled else 10240

    def __enter__(self):
        global _active_session
        if self._owns_session and self.enabled:
            self.session.start()
            _active_session = self.session
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_session
        if self._owns_session and self.enabled:
            self.session.end()
            _active_session = None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self.enabled or self._in_dispatch:
            return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self._in_dispatch = True
        try:
            self.session.save_operation(str(func), args, kwargs, result)
        finally:
            self._in_dispatch = False
        return result

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper