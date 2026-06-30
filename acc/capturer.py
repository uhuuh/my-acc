"""Capturer with pluggable backends: OpsCapturer (TorchDispatchMode) and ModuleCapturer (forward hooks)."""

import functools
import torch
from torch.utils._python_dispatch import TorchDispatchMode


# ═══════════════════════════════════════════════════════════════════════
# Module-level: patch Library.__getattribute__ so that impl / _register_fake
# are intercepted on EVERY Library instance, regardless of when it was
# created (before or after this module is imported).
#
# On each access, a wrapper is returned that auto-wraps the kernel function
# to re-enter TorchDispatchMode. Without this, the C++ dispatcher pops
# TorchDispatchMode before calling CPU/CUDA kernels.
#
# The wrapper is a no-op when _active_instance is None, so there is zero
# overhead when no acc_dump is active.
#
# Covers both APIs with a single patch:
#   my_lib.impl(op, fn, "CPU")          → Library.__getattribute__ → wrapped
#   torch.library.impl("ns::op", "CPU") → _impl → use_lib.impl()  → same path
# ═══════════════════════════════════════════════════════════════════════

def _kernel_wrapper(fn):
    """Wrap a kernel so it re-enters the active TorchDispatchMode."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if OpsCapturer._active_instance is None:
            return fn(*args, **kwargs)
        with OpsCapturer._active_instance:
            return fn(*args, **kwargs)
    return wrapped


_WRAPPED_METHODS = frozenset({'impl', '_register_fake'})

if hasattr(torch.library, 'Library'):
    _original_library_getattribute = torch.library.Library.__getattribute__
    print("[DUMP PATCH] Installing Library.__getattribute__ patch")

    def _patched_getattribute(self, name):
        attr = _original_library_getattribute(self, name)
        if name in _WRAPPED_METHODS:
            original_bound = attr  # already bound to self
            @functools.wraps(original_bound)
            def wrapper(op_name, fn, *args, **kwargs):
                return original_bound(op_name, _kernel_wrapper(fn), *args, **kwargs)
            return wrapper
        return attr

    torch.library.Library.__getattribute__ = _patched_getattribute


# ═══════════════════════════════════════════════════════════════════════

class Capturer:
    """Facade: manages multiple capture backends internally.

    Usage::
        capturer = Capturer(model=model)
        capturer.start(handler)
        ...
        capturer.stop()
    """

    @staticmethod
    def _handler_wrapper(backend_name, handler):
        if handler is None:
            return None
        return lambda key, args, kwargs, outputs: handler(backend_name, key, args, kwargs, outputs)

    def __init__(self, model=None):
        self._backends: dict[str, object] = {}
        self._model = model

    def start(self, handler):
        from .config import config

        backends = [b.strip() for b in config.capturer_backends.split(",")]

        for name in backends:
            if name == "ops":
                backend = OpsCapturer()
            elif name == "module":
                if self._model is None:
                    print("[WARN] module backend requires a model. Pass model= to acc_dump(). Skipping module backend.")
                    continue
                backend = ModuleCapturer(self._model)
            else:
                raise ValueError(
                    f"Unknown capturer backend: '{name}'. Expected 'ops' or 'module'."
                )
            backend.start(Capturer._handler_wrapper(name, handler))
            self._backends[name] = backend
            print(f"[CAPTURER] {name} backend started")

    def stop(self):
        for name, backend in list(self._backends.items()):
            backend.stop()
            print(f"[CAPTURER] {name} backend stopped")
        self._backends.clear()


class OpsCapturer(TorchDispatchMode):
    """Captures PyTorch operators via TorchDispatchMode."""

    _active_instance = None

    def __init__(self):
        super().__init__()
        self._handler = None
        self._in_dispatch = False
        self._enabled = True

    def start(self, handler):
        from .config import config

        self._handler = handler
        self._enabled = config.dump_enabled
        if self._enabled:
            self.__class__._active_instance = self
            self.__enter__()
            print(f"[OPS CAPTURER] started (dump_enabled={self._enabled})")
        else:
            print(f"[OPS CAPTURER] not started (dump_enabled={self._enabled})")

    def stop(self):
        self.__class__._active_instance = None
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


class ModuleCapturer:
    """Captures module forward calls via forward hooks."""

    def __init__(self, model):
        self._model = model
        self._handler = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._enabled = True

    def start(self, handler):
        from .config import config

        self._handler = handler
        self._enabled = config.dump_enabled
        if not self._enabled:
            print(f"[MODULE CAPTURER] not started (dump_enabled={self._enabled})")
            return

        self._handles.clear()
        for name, module in self._model.named_modules():
            if not name:
                continue  # skip root module
            handle = module.register_forward_hook(
                self._make_hook(name), with_kwargs=True
            )
            self._handles.append(handle)

        print(f"[MODULE CAPTURER] started: {len(self._handles)} modules hooked")

    def stop(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._handler = None

    def _make_hook(self, module_name):
        def hook(_module, args, kwargs, output):
            if not self._enabled or self._handler is None:
                return None
            self._handler(module_name, args, kwargs, output)
            return None

        return hook
