import os
import sys
import linecache
import uuid
from datetime import datetime

from .config import config
from .cache import CacheManager


class Manager:
    def __init__(self, capturer):
        self.session_dir = None
        self._cache_mgr = CacheManager()
        self._capturer = capturer
        self._serializer = None
        self._sequence = 0

    def start(self):
        from .serialization import Serializer

        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = "None"
        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_id = uuid.uuid4().hex[:8]
        self.session_dir = os.path.join(
            config.dump_path,
            f"{rank}-{pid}-{timestamp}-{session_id}"
        )
        os.makedirs(self.session_dir, exist_ok=False)
        print(f"[MANAGER] started: {self.session_dir}")

        self._cache_mgr.start(self.session_dir)

        self._serializer = Serializer.create(config.serializer_kind)
        print(f"[MANAGER] using {config.serializer_kind} serializer")
        self._serializer.start(self.session_dir)

        self._sequence = 0
        self._capturer.start(self._handler)

        return self.session_dir

    def stop(self):
        self._capturer.stop()
        if self._serializer is not None:
            self._serializer.stop()
        self._cache_mgr.stop()
        print(f"[MANAGER] stopped")

    def _handler(self, capturer, key, args, kwargs, outputs):
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

        from .serialization import _wrap_outputs
        serialized_args = self._cache_mgr.save(args)
        serialized_kwargs = self._cache_mgr.save(kwargs or {})
        serialized_outputs = self._cache_mgr.save(_wrap_outputs(outputs))

        seq = self._sequence
        item = {
            'seq_id': seq,
            'capturer': capturer,
            'key': key,
            'frames': frame_dicts,
            'inputs': {'args': serialized_args, 'kwargs': serialized_kwargs},
            'outputs': serialized_outputs,
        }
        try:
            self._serializer.save(item)
        except Exception as e:
            print(f"[DUMP ERROR] {seq:06d} | {key} | serializer.save failed: {e}")
        self._sequence += 1
