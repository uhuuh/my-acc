"""Capturer facade + OpsCapturer + ModuleCapturer + Library.__getattribute__ patch."""

import functools
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from loguru import logger


# ═══════════════════════════════════════════════════════════════════
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
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Backend factory
# ═══════════════════════════════════════════════════════════════════

def create_backends(config, model=None):
    """Create capture backend instances from config.capturer_backends.

    Returns:
        dict[str, object]: backend name → instance mapping.
    """
    backends = {}
    names = [b.strip() for b in config.capturer_backends.split(",")]
    for name in names:
        if name == "ops":
            backends["ops"] = OpsCapturer()
        elif name == "module":
            if model is None:
                logger.warning(
                    "module backend requires a model. "
                    "Pass model= to acc_dump(). Skipping."
                )
                continue
            backends["module"] = ModuleCapturer(model)
        else:
            raise ValueError(
                f"Unknown capturer backend: '{name}'. Expected 'ops' or 'module'."
            )
    return backends


# ═══════════════════════════════════════════════════════════════════
# Capturer facade
# ═══════════════════════════════════════════════════════════════════

class Capturer:
    """Facade managing multiple capture backends.

    __init__ receives a handler and starts all backends immediately.
    Only stop() is exposed publicly.

    The handler signature is::

        handler(capturer_type: str, capturer_key: str,
                args, kwargs, outputs)
    """

    def __init__(self, config, model=None, handler=None):
        self._backends = {}
        for name, backend in create_backends(config, model).items():
            wrapped = functools.partial(handler, name)
            backend.start(wrapped)
            self._backends[name] = backend
            logger.info(f"[Capturer] {name} backend started")

    def stop(self):
        for name, backend in list(self._backends.items()):
            backend.stop()
            logger.info(f"[Capturer] {name} backend stopped")
        self._backends.clear()


# ═══════════════════════════════════════════════════════════════════
# OpsCapturer (TorchDispatchMode)
# ═══════════════════════════════════════════════════════════════════

class OpsCapturer(TorchDispatchMode):
    """Captures PyTorch operators via TorchDispatchMode."""

    _active_instance = None

    def __init__(self):
        super().__init__()
        self._handler = None
        self._in_dispatch = False

    def start(self, handler):
        self._handler = handler
        self.__class__._active_instance = self
        self.__enter__()

    def stop(self):
        self.__class__._active_instance = None
        self.__exit__(None, None, None)
        self._handler = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self._in_dispatch or self._handler is None:
            return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self._in_dispatch = True
        try:
            self._handler(str(func), args, kwargs, result)
        finally:
            self._in_dispatch = False
        return result


# ═══════════════════════════════════════════════════════════════════
# ModuleCapturer (forward hooks)
# ═══════════════════════════════════════════════════════════════════

class ModuleCapturer:
    """Captures module forward calls via forward hooks."""

    def __init__(self, model):
        self._model = model
        self._handler = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def start(self, handler):
        self._handler = handler
        self._handles.clear()
        for name, module in self._model.named_modules():
            if not name:
                continue  # skip root module
            handle = module.register_forward_hook(
                self._make_hook(name), with_kwargs=True
            )
            self._handles.append(handle)

    def stop(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._handler = None

    def _make_hook(self, module_name):
        def hook(_module, args, kwargs, output):
            if self._handler is None:
                return None
            self._handler(module_name, args, kwargs, output)
            return None

        return hook
