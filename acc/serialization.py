"""
Serialization helpers and data structures for PyTorch Operator Dump Tool.

Provides sync and async serializers that process operator frames and write
.json metadata and .pkl data files via the IOWriter.
"""

import os
import json
import pickle
import time
import uuid
import multiprocessing as mp

_MP_CONTEXT = mp.get_context('fork')
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional

from .config import config
from .cache import CacheEntry, CacheManager
from .io import IOWriter


@dataclass
class OperatorRecord:
    """Data structure for a single operator dump."""
    seq_id: int
    save_id: str = ''
    filepath: str = ''
    filename: str = ''
    function: str = ''
    lineno: int = 0
    capturer: str = ''
    key: str = ''
    call_stack: List[Dict] = field(default_factory=list)
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'OperatorRecord':
        return cls(
            seq_id=data['seq_id'],
            save_id=data.get('save_id', ''),
            filepath=data.get('filepath', ''),
            filename=data['filename'],
            function=data['function'],
            lineno=data.get('lineno', 0),
            capturer=data.get('capturer', 'ops'),
            key=data['key'],
            call_stack=data.get('call_stack', []),
            args=data.get('args', []),
            kwargs=data.get('kwargs', {}),
            outputs=data.get('outputs', [])
        )

    def to_dict(self) -> Dict:
        return {
            'seq_id': self.seq_id,
            'save_id': self.save_id,
            'filepath': self.filepath,
            'filename': self.filename,
            'function': self.function,
            'lineno': self.lineno,
            'capturer': self.capturer,
            'key': self.key,
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


def _sanitize_key(key: str) -> str:
    return key.replace('.', '_').replace('::', '_')


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


class Serializer:
    """Base serializer with factory method."""

    def __init__(self):
        self.session_dir = None

    @classmethod
    def create(cls, kind: str = "sync") -> 'Serializer':
        if kind == "sync":
            return SyncSerializer()
        if kind == "async":
            return AsyncSerializer()
        raise ValueError(f"Unknown serializer kind: '{kind}'. Expected 'sync' or 'async'.")

    def start(self, session_dir):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def save(self, item):
        raise NotImplementedError

    @staticmethod
    def load_metadata(json_path):
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        return OperatorRecord(
            seq_id=metadata['seq_id'],
            save_id=metadata.get('save_id', ''),
            filepath=metadata.get('filepath', ''),
            filename=metadata['filename'],
            function=metadata['function'],
            lineno=metadata.get('lineno', 0),
            capturer=metadata.get('capturer', 'ops'),
            key=metadata['key'],
            call_stack=metadata.get('call_stack', []),
        )

    @staticmethod
    def load_data(pkl_path, storage_dir):
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        inputs = pkl_data['inputs']
        outputs = pkl_data['outputs']
        cache_io = IOWriter(enable_async=False)
        cache_dir = storage_dir
        if not os.path.isdir(cache_dir):
            cache_dir = os.path.join(os.path.dirname(storage_dir), 'cache')
        cache_mgr = CacheManager()
        cache_mgr.cache_dir = cache_dir
        cache_mgr._io = cache_io
        cache_mgr._load_cached = {}
        cache_mgr._started = True
        resolved_args = cache_mgr.load(inputs['args'])
        resolved_kwargs = cache_mgr.load(inputs['kwargs'])
        resolved_outputs = cache_mgr.load(outputs)
        return {'args': resolved_args, 'kwargs': resolved_kwargs}, resolved_outputs


class SyncSerializer(Serializer):
    """Sync serializer using IOWriter directly."""

    def __init__(self):
        super().__init__()
        self._io = None

    def start(self, session_dir):
        self.session_dir = session_dir
        self._io = IOWriter(name="seq")
        self._io.start()
        print(f"[SERIALIZER] started: {self.session_dir}")

    def stop(self):
        if self._io is not None:
            self._io.stop()
        print(f"[SERIALIZER] stopped")

    def save(self, item):
        seq = item['seq_id']
        key = item['key']
        capturer = item['capturer']
        frames = item['frames']

        filepath, filename, function, lineno = _process_frames(frames)
        call_stack = frames

        filename_safe = _sanitize_filename(filename)
        function_safe = _sanitize_filename(function)
        key_safe = _sanitize_key(key)

        save_id = f"{seq:06d}__{capturer}__{filename_safe}__{function_safe}__{key_safe}"
        json_filename = f"{save_id}.json"
        pkl_filename = f"{save_id}.pkl"
        json_path = os.path.join(self.session_dir, json_filename)
        pkl_path = os.path.join(self.session_dir, pkl_filename)

        record = OperatorRecord(
            seq_id=seq,
            save_id=save_id,
            filepath=filepath,
            filename=filename,
            function=function,
            lineno=lineno,
            capturer=capturer,
            key=key,
            call_stack=call_stack,
        )
        self._io.save(json_path, record.to_dict())
        self._io.save(pkl_path, {
            'inputs': item['inputs'],
            'outputs': item['outputs'],
        })


class AsyncSerializer(Serializer):
    def __init__(self):
        self.session_dir = None
        self._process = None
        self.queue = None

    def start(self, session_dir):
        self.session_dir = session_dir
        self.queue = _MP_CONTEXT.Queue()
        self._process = _MP_CONTEXT.Process(
            target=_serializer_subprocess,
            args=(self.session_dir, self.queue)
        )
        self._process.start()
        print(f"[ASYNC SERIALIZER] started (subprocess): {self.session_dir}")

    def stop(self):
        self.queue.put(None)
        if self._process is not None:
            while self._process.is_alive():
                print(f"[ASYNC SERIALIZER] waiting for subprocess...")
                self._process.join(timeout=1)
        print(f"[ASYNC SERIALIZER] stopped")

    def save(self, item):
        self.queue.put_nowait(item)


def _serializer_subprocess(session_dir, queue):
    serializer = SyncSerializer()
    serializer.start(session_dir)
    while True:
        item = queue.get()
        if item is None:
            break
        try:
            serializer.save(item)
        except Exception as e:
            seq = item.get('seq_id', '?')
            key = item.get('key', '?')
            print(f"[DUMP ERROR] {seq:06d} | {key} | serializer.save failed: {e}")
    serializer.stop()

