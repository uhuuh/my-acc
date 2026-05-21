"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump context manager for capturing operator calls.
Uses a global DumperManager for state management.
"""

import os
import pickle
import traceback
import uuid
import json
from datetime import datetime
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from .serialization import OperatorDump, _serialize_inputs, _serialize_outputs, _sanitize_filename, _sanitize_opname


# ============================================================
# Global state management
# ============================================================

class DumperManager:
    """
    Global singleton manager for operator dump state.
    Maintains session directory, sequence counter, and active status.
    """

    def __init__(self):
        self.session_dir = None
        self.sequence = 0
        self.active = False
        self.max_tensor_size_mb = 10240
        self.enabled = True  # 全局开关，控制所有 dump 行为

    def start_session(self, dump_path: str, max_tensor_size_mb: int = 10240):
        """
        Start a dump session.

        Returns True if this is a new session, False if already in a session.
        """
        if not self.enabled:
            return False

        if self.active:
            # 已经在一个 session 中，不允许嵌套
            print(f"[DUMP WARN] ops_dump called while already in active session")
            print(f"[DUMP WARN] Continuing with existing session: {self.session_dir}")
            return False

        # 创建新的 session
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = "None"

        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_id = uuid.uuid4().hex[:8]

        self.session_dir = os.path.join(
            dump_path,
            f"{rank}-{pid}-{timestamp}-{session_id}"
        )

        os.makedirs(self.session_dir, exist_ok=False)
        self.sequence = 0
        self.max_tensor_size_mb = max_tensor_size_mb
        self.active = True

        print(f"[DUMP] Created session directory: {self.session_dir}")
        return True

    def end_session(self):
        """End the current dump session."""
        if not self.enabled:
            return

        if self.active:
            print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir}")
            self.active = False
            self.session_dir = None

    def dump_operation(self, func, args, kwargs, result):
        """Dump a single operator call."""
        if not self.enabled or not self.active:
            return

        stack = traceback.extract_stack()

        filepath = ""
        filename = ""
        func_name = ""
        lineno = 0

        for frame_info in reversed(stack):
            if not frame_info.filename.endswith('dump.py'):
                filepath = frame_info.filename
                filename = os.path.basename(frame_info.filename)
                func_name = frame_info.name
                lineno = frame_info.lineno
                break

        call_stack = [
            {
                'filepath': frame.filename,
                'lineno': frame.lineno,
                'line': frame.line
            }
            for frame in stack
        ]

        filename_safe = _sanitize_filename(filename)
        opname_safe = _sanitize_opname(str(func))

        inputs = _serialize_inputs(args, kwargs, self.max_tensor_size_mb)
        outputs = _serialize_outputs(result, self.max_tensor_size_mb)

        op_dump = OperatorDump(
            sequence=self.sequence,
            filepath=filepath,
            filename=filename,
            function=func_name,
            lineno=lineno,
            opname=str(func),
            call_stack=call_stack,
            inputs=inputs,
            outputs=outputs
        )

        json_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.json"
        pkl_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.pkl"

        json_path = os.path.join(self.session_dir, json_filename)
        pkl_path = os.path.join(self.session_dir, pkl_filename)

        try:
            with open(json_path, 'w') as f:
                json.dump(op_dump.to_dict(), f, indent=2)
            with open(pkl_path, 'wb') as f:
                pkl_data = inputs + [{'outputs': outputs}]
                pickle.dump(pkl_data, f)
        except Exception as e:
            print(f"[DUMP ERROR] {self.sequence:06d} | {filename}:{lineno} | {func} | {e}")
            self.sequence += 1
            return

        print(f"[DUMP] {self.sequence:06d} | {filename}:{lineno} | {func} | saved to {json_filename}")
        self.sequence += 1


# Initialize global dumper manager
dumper_manager = DumperManager()

# Global flag for easy access (alias to dumper_manager.enabled)
DUMP_ENABLED = True


def set_dump_enabled(enabled: bool):
    """Set global dump enabled flag."""
    global DUMP_ENABLED
    DUMP_ENABLED = enabled
    dumper_manager.enabled = enabled


def get_dump_enabled() -> bool:
    """Get global dump enabled flag."""
    return dumper_manager.enabled


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

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240):
        self.dump_path = dump_path
        self.max_tensor_size_mb = max_tensor_size_mb

    def __enter__(self):
        """Enter context manager."""
        dumper_manager.start_session(self.dump_path, self.max_tensor_size_mb)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        dumper_manager.end_session()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Intercept torch operator calls."""
        if not dumper_manager.enabled or not DUMP_ENABLED:
            return func(*args, **(kwargs or {}))

        kwargs = kwargs or {}

        # Execute the operation
        result = func(*args, **kwargs)

        # Dump this operation
        dumper_manager.dump_operation(func, args, kwargs, result)

        return result

    def __call__(self, func):
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# ============================================================
# Patch torch.library.impl for custom operator internal capture
# ============================================================

_original_impl = None
_patch_installed = False


def _install_impl_patch():
    """
    Patch torch.library.impl to capture internal operators in custom implementations.

    When a custom operator implementation is registered, wrap it with nested dispatch mode
    to capture internal operator calls.
    """
    global _original_impl, _patch_installed

    if _patch_installed:
        return

    if not hasattr(torch.library, 'impl'):
        return

    _original_impl = torch.library.impl
    _patch_installed = True
    print("[DUMP PATCH] Installing torch.library.impl patch for custom operator capture")

    def patched_impl(op_name_or_def, dispatch_key='CompositeExplicitAutograd'):
        """
        Patched impl decorator that wraps implementations with nested dispatch mode.
        """
        def decorator(func):
            # Apply original impl registration
            original_result = _original_impl(op_name_or_def, dispatch_key)(func)

            # Check if we need to wrap for internal capture
            # CompositeExplicitAutograd and CPU backend don't capture internal ops
            # We wrap the implementation to enable nested dispatch mode

            # Note: We cannot modify the registered implementation after registration
            # The patch is for informational purposes only
            print(f"[DUMP PATCH] Registered custom operator: {op_name_or_def} with dispatch_key={dispatch_key}")

            if dispatch_key in ('CompositeExplicitAutograd', 'CompositeExplicitAutogradNonFunctional', 'CPU', 'CUDA'):
                print(f"[DUMP PATCH] Note: dispatch_key={dispatch_key} will not capture internal operators")
                print(f"[DUMP PATCH] Use CompositeImplicitAutograd for decomposition-based capture")

            return original_result

        return decorator

    torch.library.impl = patched_impl


def _uninstall_impl_patch():
    """Remove the torch.library.impl patch."""
    global _original_impl, _patch_installed

    if _patch_installed and _original_impl:
        torch.library.impl = _original_impl
        _patch_installed = False
        print("[DUMP PATCH] Removed torch.library.impl patch")


# Install patch on module import
_install_impl_patch()