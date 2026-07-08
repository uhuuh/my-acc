"""Threaded sharded tensor data writer."""

import os
import queue
import threading
import time
from dataclasses import dataclass

import torch

from .dump_format import TENSOR_FILE_TEMPLATE
from .tensor_utils import tensor_bytes, tensor_nbytes, write_all


@dataclass
class Location:
    file: int
    offset: int
    nbytes: int


class WriterStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.bytes = 0
        self.items = 0
        self.pending_bytes = 0
        self.pending_items = 0
        self.view_seconds = 0.0
        self.write_seconds = 0.0
        self.callback_seconds = 0.0

    def queued(self, nbytes: int):
        with self._lock:
            self.pending_bytes += nbytes
            self.pending_items += 1

    def written(
        self,
        nbytes: int,
        view_seconds: float,
        write_seconds: float,
        callback_seconds: float,
    ):
        with self._lock:
            self.bytes += nbytes
            self.items += 1
            self.pending_bytes -= nbytes
            self.pending_items -= 1
            self.view_seconds += view_seconds
            self.write_seconds += write_seconds
            self.callback_seconds += callback_seconds

    def snapshot(self):
        with self._lock:
            return {
                "bytes": self.bytes,
                "items": self.items,
                "pending_bytes": self.pending_bytes,
                "pending_items": self.pending_items,
                "view_seconds": self.view_seconds,
                "write_seconds": self.write_seconds,
                "callback_seconds": self.callback_seconds,
            }


class ShardedTensorWriter:
    def __init__(self, dump_dir: str, shard_count: int, on_done):
        self._next_shard = 0
        self._lock = threading.Lock()
        self.stats = WriterStats()
        self._shards = [
            _ShardWriter(dump_dir, i, on_done, self.stats)
            for i in range(max(1, int(shard_count or 1)))
        ]

    def submit(self, tensor_id: int, content_hash: str, tensor: torch.Tensor):
        self.stats.queued(tensor_nbytes(tensor))
        with self._lock:
            shard = self._shards[self._next_shard]
            self._next_shard = (self._next_shard + 1) % len(self._shards)
        shard.submit(tensor_id, content_hash, tensor)

    def close(self):
        for shard in self._shards:
            shard.close()


class _ShardWriter:
    def __init__(self, dump_dir: str, file_id: int, on_done, stats: WriterStats):
        self.file_id = file_id
        self._on_done = on_done
        self._stats = stats
        self._queue = queue.Queue()
        self._error = None
        self._path = os.path.join(
            dump_dir, TENSOR_FILE_TEMPLATE.format(file_id=file_id)
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"acc-tensor-writer-{file_id}",
            daemon=False,
        )
        self._thread.start()

    def submit(self, tensor_id: int, content_hash: str, tensor: torch.Tensor):
        self._queue.put((tensor_id, content_hash, tensor))

    def close(self):
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def _run(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            fd = os.open(self._path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o666)
            offset = os.path.getsize(self._path)
            try:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    tensor_id, content_hash, tensor = item
                    t0 = time.perf_counter()
                    data = tensor_bytes(tensor)
                    t1 = time.perf_counter()
                    nbytes = len(data)
                    start = offset
                    write_all(fd, data)
                    t2 = time.perf_counter()
                    offset += nbytes
                    self._on_done(
                        tensor_id,
                        content_hash,
                        tensor,
                        Location(self.file_id, start, nbytes),
                    )
                    t3 = time.perf_counter()
                    self._stats.written(nbytes, t1 - t0, t2 - t1, t3 - t2)
            finally:
                os.close(fd)
        except BaseException as exc:
            self._error = exc
