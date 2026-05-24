"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump context manager for capturing operator calls.
Uses a two-process pipeline: SerializationSender (main) + SerializationReceiver (subprocess).
"""

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .config import config
from .serialization import create_pipeline


_active_session = None


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

        if func is None:
            def decorator(f):
                return _original_impl(qualname, types, wrap(f), lib=lib)
            return decorator
        else:
            return _original_impl(qualname, types, wrap(func), lib=lib)

    torch.library.impl = patched_impl


_install_impl_patch()


class ops_dump(TorchDispatchMode):
    """
    Context manager and decorator for dumping PyTorch operator calls.

    Usage:
        with ops_dump('/path/to/dump'):
            model(input)
    """

    def __init__(self, dump_path: str = None, max_tensor_size_mb: int = None):
        kwargs = {}
        if dump_path is not None:
            kwargs['dump_path'] = dump_path
        if max_tensor_size_mb is not None:
            kwargs['max_tensor_size_mb'] = max_tensor_size_mb
        config.update(**kwargs)
        self.enabled = config.dump_enabled
        self._in_dispatch = False
        self._sender = None
        self._dump_path = dump_path
        if not self.enabled:
            self._owns_session = False
            return
        self._owns_session = _active_session is None

    def __enter__(self):
        global _active_session
        if self._owns_session and self.enabled:
            self._sender = create_pipeline(self._dump_path)
            _active_session = self._sender
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_session
        if self._owns_session and self.enabled:
            self._sender.stop()
            _active_session = None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        global _active_session
        if not self.enabled or self._in_dispatch:
            return func(*args, **(kwargs or {}))
        sender = self._sender if self._owns_session else _active_session
        if sender is None:
            return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self._in_dispatch = True
        try:
            sender.save_operation(str(func), args, kwargs, result)
        finally:
            self._in_dispatch = False
        return result

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
