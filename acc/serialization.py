"""
Serialization helpers and data structures for PyTorch Operator Dump Tool.

Implements two-process pipeline:
- SerializationSender: collects frames + transforms tensors via CacheManager, queues
- SerializationReceiver: subprocess, holds SerializationManager, writes .json/.pkl
- SerializationManager: processes frames + writes files (called by receiver)
"""

import os
import json
import pickle
import time
import uuid
import multiprocessing as mp
from datetime import datetime


_MP_CONTEXT = mp.get_context('fork')
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional
import sys
import linecache

import torch

from . import config
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


def _process_frames(frames: List[Dict]) -> Tuple[str, str, str, int]:
    """From frame dicts list, find first non-acc frame info."""
    _acc_dir = os.path.dirname(os.path.abspath(__file__))
    for entry in frames:
        if not entry['filepath'].startswith(_acc_dir):
            return (
                entry['filepath'],
                os.path.basename(entry['filepath']),
                entry.get('function', ''),
                entry['lineno'],
            )
    return "", "", "", 0


def _wrap_outputs(data: Any) -> list:
    if data is None:
        return []
    if isinstance(data, (tuple, list)):
        return list(data)
    return [data]


class SerializationManager:
    """Integrates save (frame processing + file writes) and load (metadata/data).

    Created and used by SerializationReceiver (subprocess). Does NOT do tensor
    caching — that is handled by CacheManager in SerializationSender.
    """

    def __init__(self, session_dir: str, io: IOWriter):
        self.session_dir = session_dir
        self._io = io

    def save(self, item: dict):
        """Process frames, then write .json and .pkl files via _io."""
        seq = item['sequence']
        opname = item['opname']
        frames = item['frames']

        filepath, filename, function, lineno = _process_frames(frames)
        call_stack = frames

        filename_safe = _sanitize_filename(filename)
        function_safe = _sanitize_filename(function)
        opname_safe = _sanitize_opname(opname)

        json_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.json"
        pkl_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.pkl"
        json_path = os.path.join(self.session_dir, json_filename)
        pkl_path = os.path.join(self.session_dir, pkl_filename)

        self._io.write(json_path, {
            'sequence': seq, 'filepath': filepath, 'filename': filename,
            'function': function, 'lineno': lineno, 'opname': opname,
            'call_stack': call_stack,
        })
        self._io.write(pkl_path, {
            'inputs': item['inputs'],
            'outputs': item['outputs'],
        })

    @staticmethod
    def load_metadata(json_path: str) -> OperatorRecord:
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
        io_writer = IOWriter(enable_async=False)
        cache_mgr = CacheManager(storage_dir, cache_io=io_writer)
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        inputs = pkl_data['inputs']
        outputs = pkl_data['outputs']
        resolved_args = cache_mgr.load(inputs['args'])
        resolved_kwargs = cache_mgr.load(inputs['kwargs'])
        resolved_outputs = cache_mgr.load(outputs)
        return {'args': resolved_args, 'kwargs': resolved_kwargs}, resolved_outputs


class SerializationSender:
    """Runs in main process: collects raw frames, caches tensors, queues for receiver."""

    def __init__(self, dump_path: Optional[str] = None, max_tensor_size_mb: Optional[int] = None):
        config.init(dump_path=dump_path, max_tensor_size_mb=max_tensor_size_mb)
        self.dump_path = config.get_dump_path()
        if not self.dump_path:
            raise ValueError("dump_path is not set. Provide dump_path or set ACC_DUMP_PATH env var.")
        self.max_tensor_size_mb = config.get_max_tensor_size_mb()
        self.session_dir: Optional[str] = None
        self.sequence: int = 0
        self._cache_mgr: Optional[CacheManager] = None
        self._cache_io: Optional[IOWriter] = None
        self._start_time: Optional[float] = None
        self.queue: mp.Queue = _MP_CONTEXT.Queue()
        self._process: Optional[mp.Process] = None

    def start(self) -> str:
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
        self._cache_io = IOWriter(name="cache", enable_async=True)
        self._cache_mgr = CacheManager(
            storage_dir, cache_io=self._cache_io, max_tensor_size_mb=self.max_tensor_size_mb
        )
        self.sequence = 0
        self._start_time = time.time()
        print(f"[DUMP] Created session directory: {self.session_dir}")
        return self.session_dir

    def _ensure_process(self):
        if self._process is None:
            self._process = _MP_CONTEXT.Process(target=_receiver_main, args=(self.session_dir, self.queue))
            self._process.start()

    def save_operation(self, opname: str, args: tuple, kwargs: dict, outputs) -> int:
        """Collect raw frames, cache tensors, queue for receiver."""
        if self._cache_mgr is None:
            raise RuntimeError("Sender not started")
        self._ensure_process()

        frames = []
        f = sys._getframe(0)
        while f:
            frames.append(f)
            f = f.f_back

        frame_dicts = [
            {
                'filepath': f.f_code.co_filename,
                'lineno': f.f_lineno,
                'function': f.f_code.co_name,
                'line': linecache.getline(f.f_code.co_filename, f.f_lineno).rstrip('\n'),
            }
            for f in reversed(frames)
        ]

        serialized_args = self._cache_mgr.save(args)
        serialized_kwargs = self._cache_mgr.save(kwargs or {})
        serialized_outputs = self._cache_mgr.save(_wrap_outputs(outputs))

        seq = self.sequence
        item = {
            'sequence': seq,
            'opname': opname,
            'frames': frame_dicts,
            'inputs': {'args': serialized_args, 'kwargs': serialized_kwargs},
            'outputs': serialized_outputs,
        }
        try:
            self.queue.put(item)
        except Exception as e:
            print(f"[DUMP ERROR] {seq:06d} | {opname} | queue put failed: {e}")
            self.sequence += 1
            return seq
        self.sequence += 1
        return seq

    def stop(self):
        if self._cache_io is not None:
            self._cache_io.wait_complete()
        self.queue.put(None)
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
        elapsed = time.time() - self._start_time if self._start_time else 0
        print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir} in {elapsed:.1f}s")


class SerializationReceiver:
    """Runs in subprocess: holds SerializationManager, calls manager.save(item)."""

    def __init__(self, session_dir: str, queue: mp.Queue):
        self.session_dir = session_dir
        self._queue = queue

    def run(self):
        io = IOWriter(name="seq", enable_async=True)
        mgr = SerializationManager(self.session_dir, io)
        while True:
            item = self._queue.get()

            if item is None:
                break
            try:
                mgr.save(item)
            except Exception as e:
                seq = item.get('sequence', '?')
                opname = item.get('opname', '?')
                print(f"[DUMP ERROR] {seq:06d} | {opname} | manager.save failed: {e}")
        io.wait_complete()


def _receiver_main(session_dir, queue):
    """Module-level target for multiprocessing spawn."""
    receiver = SerializationReceiver(session_dir, queue)
    receiver.run()


def create_pipeline(dump_path: str, max_tensor_size_mb: Optional[int] = None) -> SerializationSender:
    """Factory: create sender + start receiver subprocess, return sender."""
    sender = SerializationSender(dump_path, max_tensor_size_mb=max_tensor_size_mb)
    sender.start()
    sender._ensure_process()
    return sender
