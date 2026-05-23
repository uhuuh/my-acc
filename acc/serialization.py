"""
Serialization helpers and data structures for PyTorch Operator Dump Tool.
"""

import os
import json
import pickle
import time
import uuid
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional

import torch

from .cache import CacheEntry, CacheManager
from .io import IOWriter


@dataclass
class OperatorRecord:
    """Data structure for a single operator dump."""
    sequence: int
    filepath: str
    filename: str
    function: str
    lineno: int
    opname: str
    call_stack: List[Dict]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'OperatorRecord':
        return cls(
            sequence=data['sequence'],
            filepath=data.get('filepath', ''),
            filename=data['filename'],
            function=data['function'],
            lineno=data.get('lineno', 0),
            opname=data['opname'],
            call_stack=data.get('call_stack', []),
            args=data.get('args', []),
            kwargs=data.get('kwargs', {}),
            outputs=data.get('outputs', [])
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON (metadata only, no tensor data)."""
        return {
            'sequence': self.sequence,
            'filepath': self.filepath,
            'filename': self.filename,
            'function': self.function,
            'lineno': self.lineno,
            'opname': self.opname,
            'call_stack': self.call_stack
        }


OperatorDump = OperatorRecord


def _sanitize_filename(filename: str) -> str:
    if filename.startswith('<') and filename.endswith('>'):
        filename = filename[1:-1]
    result = filename.replace('/', '_').replace('\\', '_').replace('.py', '')
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in invalid_chars:
        result = result.replace(char, '_')
    return result


def _sanitize_opname(opname: str) -> str:
    return opname.replace('.', '_').replace('::', '_')


def _serialize_tensor(tensor, max_tensor_size_mb: int):
    """Prepare a single tensor for serialization with size check."""
    tensor_size_bytes = tensor.numel() * tensor.element_size()
    tensor_size_mb = tensor_size_bytes / (1024 * 1024)
    if tensor_size_mb > max_tensor_size_mb:
        print(f"[DUMP WARN] Tensor size {tensor_size_mb:.2f} MB exceeds limit {max_tensor_size_mb} MB, replacing with None")
        return None
    try:
        if tensor.device.type == 'cpu':
            return tensor.detach().contiguous()
        # 尝试 pin_memory 加速，如果不支持则回退
        try:
            return tensor.detach().contiguous().cpu(pin_memory=True)
        except TypeError:
            # pin_memory 参数不支持，使用普通 cpu()
            return tensor.detach().contiguous().cpu()
    except Exception as e:
        print(f"[DUMP WARN] Failed to make tensor contiguous: {e}, replacing with None")
        return None


def _serialize_value(value, max_tensor_size_mb: int, cache_mgr: CacheManager):
    """Serialize a single value: tensors get prepared + cached, everything else passes through."""
    if isinstance(value, torch.Tensor):
        prepared = _serialize_tensor(value, max_tensor_size_mb)
        if prepared is None:
            return None
        return cache_mgr.save(prepared)
    return value


def _serialize_outputs(result, max_tensor_size_mb: int, cache_mgr: CacheManager) -> list:
    """Serialize operator outputs to a list with tensor preparation and caching."""
    outputs_list = []
    if result is None:
        return outputs_list
    if isinstance(result, torch.Tensor):
        prepared = _serialize_tensor(result, max_tensor_size_mb)
        if prepared is None:
            outputs_list.append(None)
        else:
            outputs_list.append(cache_mgr.save(prepared))
    elif isinstance(result, (tuple, list)):
        for item in result:
            outputs_list.append(_serialize_value(item, max_tensor_size_mb, cache_mgr))
    else:
        outputs_list.append(result)
    return outputs_list


class SerializationSession:
    """Manages a single serialization session, integrating CacheManager."""

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240, enable_async_io: Optional[bool] = None):
        self.dump_path = dump_path
        self.session_dir: Optional[str] = None
        self.sequence: int = 0
        self.max_tensor_size_mb: int = max_tensor_size_mb
        self._enable_async_io: Optional[bool] = enable_async_io
        self._cache_manager: Optional[CacheManager] = None
        self._io_writer: Optional[IOWriter] = None
        self._start_time: Optional[float] = None

    def start(self) -> str:
        """Create session directory and storage subdirectory, initialize CacheManager."""
        import torch.distributed as dist
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
        storage_dir = os.path.join(self.session_dir, 'storage')
        os.makedirs(storage_dir, exist_ok=False)
        # Initialize IOWriter (default: async enabled)
        async_enabled = self._enable_async_io if self._enable_async_io is not None else True
        self._io_writer = IOWriter(enable_async=async_enabled)
        self._cache_manager = CacheManager(storage_dir, io_writer=self._io_writer)
        self.sequence = 0
        self._start_time = time.time()
        print(f"[DUMP] Created session directory: {self.session_dir}")
        return self.session_dir

    def save_operation(
        self, func, filepath: str, filename: str, function: str,
        lineno: int, args: tuple, kwargs: dict, outputs
    ) -> int:
        """Save a single operator dump. Returns sequence number."""
        if self._cache_manager is None:
            raise RuntimeError("Session not started")
        filename_safe = _sanitize_filename(filename)
        function_safe = _sanitize_filename(function)
        opname_safe = _sanitize_opname(str(func))
        serialized_args = [
            _serialize_value(arg, self.max_tensor_size_mb, self._cache_manager)
            for arg in args
        ]
        serialized_kwargs = {}
        for key, val in (kwargs or {}).items():
            serialized_kwargs[key] = _serialize_value(val, self.max_tensor_size_mb, self._cache_manager)
        serialized_outputs = _serialize_outputs(outputs, self.max_tensor_size_mb, self._cache_manager)
        stack = traceback.extract_stack()
        call_stack = [
            {'filepath': frame.filename, 'lineno': frame.lineno, 'line': frame.line}
            for frame in stack
        ]
        seq = self.sequence
        json_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.json"
        pkl_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.pkl"
        json_path = os.path.join(self.session_dir, json_filename)
        pkl_path = os.path.join(self.session_dir, pkl_filename)
        try:
            self._io_writer.write(json_path, json.dumps({
                'sequence': seq, 'filepath': filepath, 'filename': filename,
                'function': function, 'lineno': lineno, 'opname': str(func),
                'call_stack': call_stack
            }, indent=2))
            self._io_writer.write(pkl_path, {
                'inputs': {'args': serialized_args, 'kwargs': serialized_kwargs},
                'outputs': serialized_outputs,
            })
        except Exception as e:
            print(f"[DUMP ERROR] {seq:06d} | {filename}:{lineno} | {func} | {e}")
            self.sequence += 1
            return seq
        print(f"[DUMP] {seq:06d} | {filename}:{lineno} | {func} | saved to {json_filename}")
        self.sequence += 1
        return seq

    def end(self):
        """End the session and print summary."""
        if self._io_writer is not None:
            self._io_writer.wait_complete()
        if self.session_dir:
            elapsed = time.time() - self._start_time if self._start_time else 0
            print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir} in {elapsed:.1f}s")

    @staticmethod
    def load_metadata(json_path: str) -> OperatorRecord:
        """Load JSON metadata (no tensor data)."""
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        return OperatorRecord(
            sequence=metadata['sequence'],
            filepath=metadata.get('filepath', ''),
            filename=metadata['filename'],
            function=metadata['function'],
            lineno=metadata.get('lineno', 0),
            opname=metadata['opname'],
            call_stack=metadata.get('call_stack', []),
        )

    @staticmethod
    def load_data(pkl_path: str, storage_dir: str) -> Tuple[Dict, List]:
        """Load PKL data and resolve CacheEntry references to actual tensors.

        Returns: (inputs_dict, outputs_list) where inputs_dict has 'args' and 'kwargs' keys.
        """
        io_writer = IOWriter(enable_async=False)
        cache_mgr = CacheManager(storage_dir, io_writer=io_writer)
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        inputs = pkl_data['inputs']
        outputs = pkl_data['outputs']
        resolved_args = cache_mgr.load(inputs['args'])
        resolved_kwargs = cache_mgr.load(inputs['kwargs'])
        resolved_outputs = cache_mgr.load(outputs)
        return {'args': resolved_args, 'kwargs': resolved_kwargs}, resolved_outputs

    @property
    def storage_dir(self) -> Optional[str]:
        if self.session_dir:
            return os.path.join(self.session_dir, 'storage')
        return None