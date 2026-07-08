"""Dump pipeline orchestration: split, hash, deduplicate, write, release."""

import os
import queue
import threading
import time

import torch
import torch.multiprocessing as mp
from loguru import logger

from .dump_format import (
    JsonlWriter,
    LOCATIONS_FILE,
    RECORDS_FILE,
)
from .record_splitter import RecordSplitter
from .shared import SharedTensorManager
from .tensor_utils import hash_tensor, tensor_nbytes
from .tensor_writer import ShardedTensorWriter

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.to_cpu_bytes = 0
        self.hash_bytes = 0
        self.write_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def add(self, name: str, value: int = 1):
        with self._lock:
            setattr(self, name, getattr(self, name) + value)

    def snapshot(self):
        with self._lock:
            return {
                "to_cpu_bytes": self.to_cpu_bytes,
                "hash_bytes": self.hash_bytes,
                "write_bytes": self.write_bytes,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            }


class DumpPipeline:
    def __init__(self, config):
        os.makedirs(config.dump_dir, exist_ok=True)
        self._stats = Stats()
        self._shared = SharedTensorManager()
        self._splitter = RecordSplitter(self._shared)
        self._records = JsonlWriter(os.path.join(config.dump_dir, RECORDS_FILE))
        self._locations = JsonlWriter(os.path.join(config.dump_dir, LOCATIONS_FILE))

        self._hash_to_location = {}
        self._pending_by_hash = {}
        self._save_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._errors = []
        self._errors_lock = threading.Lock()
        self._stopping = threading.Event()
        self._started_at = time.perf_counter()

        self._hash_worker_count = max(1, int(config.hash_workers or 1))
        self._hash_use_processes = self._shared.shared_enabled
        self._hash_in, self._hash_out, self._hash_workers = self._start_hash_workers()

        self._writer = ShardedTensorWriter(
            config.dump_dir,
            config.cache_write_workers,
            self._on_tensor_written,
        )
        self._cache_thread = threading.Thread(
            target=self._cache_loop,
            name="acc-cache-check",
            daemon=False,
        )
        self._cache_thread.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="acc-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def save(self, record):
        self._raise_if_failed()
        try:
            with self._save_lock:
                info, tensors = self._splitter.split(record)
                self._records.write(info)
                for tensor_id, tensor in tensors:
                    self._stats.add("to_cpu_bytes", tensor_nbytes(tensor))
                    self._hash_in.put((tensor_id, tensor))
        except BaseException as exc:
            self._record_error(f"record save failed: {exc}")
            self._raise_if_failed()

    def close(self):
        for _ in self._hash_workers:
            self._hash_in.put(None)
        for worker in self._hash_workers:
            worker.join()
            if self._hash_use_processes and worker.exitcode != 0:
                self._record_error(f"hash worker failed: exitcode={worker.exitcode}")

        self._cache_thread.join()
        self._writer.close()
        self._stopping.set()
        self._records.close()
        self._locations.close()
        self._log_summary()
        self._raise_if_failed()

    def _start_hash_workers(self):
        if self._hash_use_processes:
            ctx = mp.get_context("fork")
            hash_in = ctx.Queue()
            hash_out = ctx.Queue()
            workers = [
                ctx.Process(target=_hash_worker, args=(hash_in, hash_out))
                for _ in range(self._hash_worker_count)
            ]
        else:
            hash_in = queue.Queue()
            hash_out = queue.Queue()
            workers = [
                threading.Thread(
                    target=_hash_worker,
                    args=(hash_in, hash_out),
                    name=f"acc-hash-{i}",
                    daemon=False,
                )
                for i in range(self._hash_worker_count)
            ]
        for worker in workers:
            worker.start()
        return hash_in, hash_out, workers

    def _cache_loop(self):
        done_workers = 0
        try:
            while done_workers < len(self._hash_workers):
                item = self._hash_out.get()
                if item is None:
                    done_workers += 1
                    continue
                tensor_id, content_hash, tensor = item
                self._stats.add("hash_bytes", tensor_nbytes(tensor))
                self._handle_hashed_tensor(tensor_id, content_hash, tensor)
        except BaseException as exc:
            self._record_error(f"cache loop failed: {exc}")

    def _handle_hashed_tensor(self, tensor_id, content_hash, tensor):
        with self._cache_lock:
            location = self._hash_to_location.get(content_hash)
            if location is not None:
                self._stats.add("cache_hits")
                self._record_tensor_location(tensor_id, location)
                self._shared.release(tensor)
                return

            pending = self._pending_by_hash.get(content_hash)
            if pending is not None:
                self._stats.add("cache_hits")
                pending.append((tensor_id, tensor))
                return

            self._stats.add("cache_misses")
            self._pending_by_hash[content_hash] = [(tensor_id, tensor)]

        self._writer.submit(tensor_id, content_hash, tensor)

    def _on_tensor_written(self, tensor_id, content_hash, tensor, location):
        with self._cache_lock:
            pending = self._pending_by_hash.pop(content_hash, [(tensor_id, tensor)])
            self._hash_to_location[content_hash] = location
            for pending_id, pending_tensor in pending:
                self._record_tensor_location(pending_id, location)
                self._shared.release(pending_tensor)
            self._stats.add("write_bytes", location.nbytes)

    def _record_tensor_location(self, tensor_id: int, location):
        self._locations.write({
            "id": tensor_id,
            "file": location.file,
            "offset": location.offset,
            "nbytes": location.nbytes,
        })

    def _log_summary(self):
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        stats = self._stats.snapshot()
        writer = self._writer.stats.snapshot()
        logger.info(
            "[acc summary] "
            f"elapsed={elapsed:.2f}s "
            f"to_cpu={stats['to_cpu_bytes'] / 1024**3:.2f}GB/"
            f"{_gbps(stats['to_cpu_bytes'] / elapsed)} "
            f"hash={stats['hash_bytes'] / 1024**3:.2f}GB/"
            f"{_gbps(stats['hash_bytes'] / elapsed)} "
            f"write={stats['write_bytes'] / 1024**3:.2f}GB/"
            f"{_gbps(stats['write_bytes'] / elapsed)} "
            f"writer_items={writer['items']} "
            f"writer_sys={writer['write_seconds']:.2f}s "
            f"writer_view={writer['view_seconds']:.2f}s "
            f"writer_cb={writer['callback_seconds']:.2f}s "
            f"cache_hit={stats['cache_hits']} "
            f"cache_miss={stats['cache_misses']}"
        )

    def _monitor_loop(self):
        last = self._stats.snapshot()
        last_writer = self._writer.stats.snapshot()
        while not self._stopping.wait(1.0):
            current = self._stats.snapshot()
            dt = {key: current[key] - last.get(key, 0) for key in current}
            last = current
            writer = self._writer.stats.snapshot()
            writer_dt = {
                key: writer[key] - last_writer.get(key, 0)
                for key in writer
                if key not in ("pending_bytes", "pending_items")
            }
            last_writer = writer
            logger.info(
                "[acc stats] "
                f"to_cpu={_gbps(dt['to_cpu_bytes'])} "
                f"hash={_gbps(dt['hash_bytes'])} "
                f"write={_gbps(dt['write_bytes'])} "
                f"writer_items={writer_dt['items']} "
                f"writer_pending={writer['pending_bytes'] / 1024**3:.2f}GB/"
                f"{writer['pending_items']} "
                f"writer_view={writer_dt['view_seconds'] * 1000:.1f}ms "
                f"writer_sys={writer_dt['write_seconds'] * 1000:.1f}ms "
                f"writer_cb={writer_dt['callback_seconds'] * 1000:.1f}ms "
                f"cache_hit={current['cache_hits']} "
                f"cache_miss={current['cache_misses']} "
                f"shared={self._shared.used_bytes / 1024**3:.2f}GB"
            )

    def _record_error(self, message: str):
        logger.error(message)
        with self._errors_lock:
            self._errors.append(message)

    def _raise_if_failed(self):
        with self._errors_lock:
            if self._errors:
                raise RuntimeError(self._errors[0])


def _hash_worker(input_queue, output_queue):
    while True:
        item = input_queue.get()
        if item is None:
            output_queue.put(None)
            return
        tensor_id, tensor = item
        output_queue.put((tensor_id, hash_tensor(tensor), tensor))


def _gbps(nbytes: int) -> str:
    return f"{nbytes / 1024**3:.2f}GB/s"
