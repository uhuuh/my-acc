import torch
from torch.utils._python_dispatch import TorchDispatchMode


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
                with _active_session:
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


class Capturer(TorchDispatchMode):
    def __init__(self):
        self._handler = None
        self._in_dispatch = False
        self._enabled = True

    def start(self, handler):
        from .config import config
        global _active_session
        self._handler = handler
        self._enabled = config.dump_enabled
        if self._enabled:
            _active_session = self
            self.__enter__()

    def stop(self):
        global _active_session
        _active_session = None
        self.__exit__(None, None, None)
        self._handler = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self._enabled or self._in_dispatch or self._handler is None:
            return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self._in_dispatch = True
        try:
            self._handler(str(func), args, kwargs, result)
        finally:
            self._in_dispatch = False
        return result
