"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump class for capturing operator calls.
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


class ops_dump(TorchDispatchMode):
    """Context manager and decorator for dumping PyTorch operator calls."""

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240):
        """
        Initialize OperatorDumper.

        Args:
            dump_path: Base path for dump output
            max_tensor_size_mb: Maximum tensor size in MB (default 10GB)
        """
        self.dump_path = dump_path
        self.max_tensor_size_mb = max_tensor_size_mb
        self.session_dir = None
        self.sequence = 0
        self._active = False

    def __enter__(self):
        """Enter context manager, create session directory."""
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = "None"

        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_id = uuid.uuid4().hex[:8]

        self.session_dir = os.path.join(
            self.dump_path,
            f"{rank}-{pid}-{timestamp}-{session_id}"
        )

        os.makedirs(self.session_dir, exist_ok=False)
        self.sequence = 0
        self._active = True
        print(f"[DUMP] Created session directory: {self.session_dir}")
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._active = False
        print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir}")
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Intercept torch operator calls."""
        if not self._active:
            return func(*args, **(kwargs or {}))

        kwargs = kwargs or {}

        # Execute the operation
        result = func(*args, **kwargs)

        # Dump this operation (including outputs)
        self._dump_operation(func, args, kwargs, result)

        return result

    def _dump_operation(self, func, args, kwargs, result):
        """Dump a single operator call."""
        stack = traceback.extract_stack()

        filepath = ""
        filename = "<global>"
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

    def __call__(self, func):
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper