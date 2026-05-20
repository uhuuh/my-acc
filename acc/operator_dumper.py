"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump class for capturing operator calls.
"""

import os
import pickle
import traceback
import inspect
import uuid
from datetime import datetime
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from .serialization import _serialize_value, _sanitize_filename, _make_pickle_safe


class ops_dump(TorchDispatchMode):
    """Context manager and decorator for dumping PyTorch operator calls."""
    
    def __init__(self, dump_path: str):
        """
        Initialize OperatorDumper.
        
        Args:
            dump_path: Base path for dump output
        """
        self.dump_path = dump_path
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
        
        # Dump this operation
        self._dump_operation(func, args, kwargs, result)
        
        return result
    
    def _dump_operation(self, func, args, kwargs, result):
        """Dump a single operator call."""
        # Get full call stack (no frame skipping)
        call_stack = ''.join(traceback.format_stack())
        
        # Extract caller info from call stack (use last frame before operator_dumper.py)
        stack = traceback.extract_stack()
        filename = "unknown"
        lineno = 0
        
        # Find first frame that's not operator_dumper.py (from the end)
        for frame_info in reversed(stack):
            if not frame_info.filename.endswith('operator_dumper.py'):
                filename = os.path.basename(frame_info.filename)
                lineno = frame_info.lineno
                break
        
        # Get operator name
        opname = str(func)
        
        # Serialize inputs (no outputs)
        inputs = [_serialize_value(arg) for arg in args]
        if kwargs:
            inputs.append(_serialize_value(kwargs))
        
        # Create dump data (no outputs)
        dump_data = {
            'sequence': self.sequence,
            'filename': filename,
            'lineno': lineno,
            'opname': opname,
            'call_stack': call_stack,
            'inputs': inputs
        }
        
        # Create dump filename (6-digit sequence)
        sanitized_name = _sanitize_filename(filename)
        opname_safe = opname.replace('.', '_').replace('::', '_')
        dump_filename = f"{self.sequence:06d}_{sanitized_name}_{opname_safe}.pkl"
        dump_path = os.path.join(self.session_dir, dump_filename)
        
        # Write dump file with error handling
        try:
            with open(dump_path, 'wb') as f:
                pickle.dump(dump_data, f)
        except Exception as e:
            print(f"[DUMP ERROR] {self.sequence:06d} | {filename}:{lineno} | {opname} | {e}")
            self.sequence += 1
            return
        
        # Log the dump (6-digit sequence)
        print(f"[DUMP] {self.sequence:06d} | {filename}:{lineno} | {opname} | saved to {dump_filename}")
        
        self.sequence += 1
    
    def __call__(self, func):
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper