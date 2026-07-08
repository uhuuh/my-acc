"""Benchmark dump bottlenecks: device-to-CPU copy, hashing, and append writes.

This is intentionally standalone. It does not import acc internals, so the
numbers reflect primitive operation costs instead of the current pipeline shape.

Examples:
    python util/bench_dump_costs.py --device cuda --mb 256 --iters 20
    python util/bench_dump_costs.py --device cpu --mb 1024 --iters 10
    python util/bench_dump_costs.py --sweep-mb 1,16,256,1024 --iters 5
"""

import argparse
import mmap
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List

import torch
import xxhash


GB = 1024 ** 3
MB = 1024 ** 2


@dataclass
class Result:
    name: str
    nbytes: int
    times: List[float]

    @property
    def best(self) -> float:
        return min(self.times)

    @property
    def avg(self) -> float:
        return sum(self.times) / len(self.times)

    @property
    def gbps_best(self) -> float:
        return self.nbytes / self.best / GB if self.best > 0 else 0.0

    @property
    def gbps_avg(self) -> float:
        return self.nbytes / self.avg / GB if self.avg > 0 else 0.0


def main():
    args = parse_args()
    device = resolve_device(args.device)
    sizes_mb = parse_sizes(args)

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    print(f"device={device} dtype={args.dtype} iters={args.iters} warmup={args.warmup}")
    print()

    for mb in sizes_mb:
        nbytes = mb * MB
        print(f"== {mb} MB ==")
        tensor = make_tensor(nbytes, args.dtype, device)
        cpu_tensor = make_tensor(nbytes, args.dtype, "cpu")
        cpu_buffer = tensor_to_bytes(cpu_tensor)
        if args.write_direct:
            cpu_buffer = make_direct_buffer(len(cpu_buffer))

        results = []
        if not args.only_write:
            results.extend(bench_to_cpu(tensor, args.iters, args.warmup))
            results.append(bench_memcpy(cpu_buffer, args.iters, args.warmup))
            results.append(bench_hash(cpu_buffer, args.iters, args.warmup))
        results.extend(
            bench_write(
                cpu_buffer,
                args.iters,
                args.warmup,
                args.output_dir,
                args.shards,
                args.write_mode,
                args.write_workers,
                args.write_repeat,
                args.write_direct,
                args.fsync,
            )
        )
        print_results(results)
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda, cpu, cuda:0, etc.")
    parser.add_argument("--dtype", default="float32", help="torch dtype name")
    parser.add_argument("--mb", type=int, default=256, help="single tensor size in MB")
    parser.add_argument(
        "--sweep-mb",
        default="",
        help="comma-separated tensor sizes in MB, overrides --mb",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--only-write", action="store_true", help="benchmark writes only")
    parser.add_argument(
        "--output-dir",
        default="",
        help="write benchmark directory; default is a temporary directory",
    )
    parser.add_argument("--shards", type=int, default=1, help="append files to rotate over")
    parser.add_argument(
        "--write-mode",
        choices=("append", "pwrite", "threaded-pwrite", "threaded-append"),
        default="append",
        help="write benchmark mode",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=4,
        help="worker threads for threaded-pwrite mode",
    )
    parser.add_argument(
        "--write-repeat",
        type=int,
        default=1,
        help="copies of the buffer written per timed write iteration",
    )
    parser.add_argument(
        "--write-direct",
        action="store_true",
        help="open output files with O_DIRECT and use an aligned mmap buffer",
    )
    parser.add_argument("--fsync", action="store_true", help="fsync after each write")
    return parser.parse_args()


def parse_sizes(args) -> List[int]:
    if not args.sweep_mb:
        return [args.mb]
    return [int(item.strip()) for item in args.sweep_mb.split(",") if item.strip()]


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device)


def make_tensor(nbytes: int, dtype_name: str, device) -> torch.Tensor:
    dtype = getattr(torch, dtype_name)
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = max(1, nbytes // element_size)
    return torch.empty(numel, dtype=dtype, device=device).normal_()


def bench_to_cpu(tensor: torch.Tensor, iters: int, warmup: int) -> List[Result]:
    nbytes = tensor.numel() * tensor.element_size()
    if tensor.device.type == "cpu":
        return [
            time_result(
                "cpu clone",
                nbytes,
                lambda: tensor.detach().contiguous().clone(),
                iters,
                warmup,
                sync=None,
            ),
            time_result(
                "cpu share_memory_ clone",
                nbytes,
                lambda: tensor.detach().contiguous().clone().share_memory_(),
                iters,
                warmup,
                sync=None,
            ),
        ]

    def sync():
        torch.cuda.synchronize(tensor.device)

    results = [
        time_result(
            "cuda to cpu",
            nbytes,
            lambda: tensor.detach().to("cpu"),
            iters,
            warmup,
            sync=sync,
        ),
        time_result(
            "cuda to cpu pinned",
            nbytes,
            lambda: copy_to_preallocated_cpu(tensor, pin_memory=True),
            iters,
            warmup,
            sync=sync,
        ),
        time_result(
            "cuda to cpu shared",
            nbytes,
            lambda: copy_to_preallocated_cpu(tensor, share_memory=True),
            iters,
            warmup,
            sync=sync,
        ),
    ]
    return results


def copy_to_preallocated_cpu(
    tensor: torch.Tensor,
    pin_memory: bool = False,
    share_memory: bool = False,
) -> torch.Tensor:
    target = torch.empty_like(tensor, device="cpu", pin_memory=pin_memory)
    if share_memory:
        target.share_memory_()
    target.copy_(tensor, non_blocking=pin_memory)
    return target


def bench_hash(data, iters: int, warmup: int) -> Result:
    return time_result(
        "xxh3_128 hash",
        len(data),
        lambda: xxhash.xxh3_128_hexdigest(data),
        iters,
        warmup,
        sync=None,
    )


def bench_memcpy(data, iters: int, warmup: int) -> Result:
    src = memoryview(data)
    dst = bytearray(len(src))
    dst_view = memoryview(dst)

    return time_result(
        "memcpy bytearray",
        len(src),
        lambda: copy_memory(dst_view, src),
        iters,
        warmup,
        sync=None,
    )


def copy_memory(dst, src):
    dst[:] = src
    return dst


def bench_write(
    data,
    iters: int,
    warmup: int,
    output_dir: str,
    shards: int,
    mode: str,
    workers: int,
    repeat: int,
    direct: bool,
    fsync: bool,
) -> List[Result]:
    shards = max(1, shards)
    workers = max(1, workers)
    repeat = max(1, repeat)
    temp_ctx = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_dir = output_dir
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="acc-bench-write-")
        base_dir = temp_ctx.name

    try:
        paths = [os.path.join(base_dir, f"shard-{i:03d}.data") for i in range(shards)]
        if mode == "append":
            flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
        else:
            flags = os.O_CREAT | os.O_WRONLY
        if direct:
            flags |= getattr(os, "O_DIRECT", 0)
        fds = [os.open(path, flags, 0o666) for path in paths]
        next_fd = 0
        offsets = [0] * len(fds)

        def append_one():
            nonlocal next_fd
            for _ in range(repeat):
                fd = fds[next_fd]
                next_fd = (next_fd + 1) % len(fds)
                write_all(fd, data)
            if fsync:
                for fd in fds:
                    os.fsync(fd)

        def pwrite_one():
            nonlocal next_fd
            for _ in range(repeat):
                shard = next_fd
                next_fd = (next_fd + 1) % len(fds)
                offset = offsets[shard]
                offsets[shard] += len(data)
                pwrite_all(fds[shard], data, offset)
            if fsync:
                for fd in fds:
                    os.fsync(fd)

        def threaded_pwrite_one():
            write_threaded_pwrite(fds, offsets, data, repeat, workers)
            if fsync:
                for fd in fds:
                    os.fsync(fd)

        def threaded_append_one():
            write_threaded_append(fds, data, repeat, workers)
            if fsync:
                for fd in fds:
                    os.fsync(fd)

        try:
            if mode == "append":
                fn = append_one
                label = f"append write x{shards}"
            elif mode == "pwrite":
                fn = pwrite_one
                label = f"pwrite x{shards}"
            elif mode == "threaded-pwrite":
                fn = threaded_pwrite_one
                label = f"threaded pwrite w{workers} x{shards}"
            else:
                fn = threaded_append_one
                label = f"threaded append w{workers} x{shards}"
            if direct:
                label += " direct"
            if repeat > 1:
                label += f" r{repeat}"
            if fsync:
                label += " fsync"
            return [time_result(label, len(data) * repeat, fn, iters, warmup, sync=None)]
        finally:
            for fd in fds:
                os.close(fd)
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def time_result(
    name: str,
    nbytes: int,
    fn: Callable[[], object],
    iters: int,
    warmup: int,
    sync: Callable[[], None] | None,
) -> Result:
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - t0)
        keep_alive(out)
    return Result(name=name, nbytes=nbytes, times=times)


def tensor_to_bytes(tensor: torch.Tensor):
    tensor = tensor.detach().contiguous()
    if tensor.numel() == 0:
        return memoryview(b"")
    return memoryview(tensor.numpy()).cast("B")


def make_direct_buffer(nbytes: int):
    page_size = mmap.PAGESIZE
    aligned_nbytes = ((nbytes + page_size - 1) // page_size) * page_size
    mm = mmap.mmap(-1, aligned_nbytes)
    chunk = (b"accbench" * (1024 * 1024 // 8))
    remaining = aligned_nbytes
    while remaining > 0:
        piece = chunk[: min(len(chunk), remaining)]
        mm.write(piece)
        remaining -= len(piece)
    mm.seek(0)
    keep_alive(mm)
    return memoryview(mm)


def write_all(fd: int, data):
    offset = 0
    total = len(data)
    while offset < total:
        written = os.write(fd, data[offset:])
        if written == 0:
            raise OSError("os.write returned 0 bytes")
        offset += written


def pwrite_all(fd: int, data, file_offset: int):
    data_offset = 0
    total = len(data)
    while data_offset < total:
        written = os.pwrite(fd, data[data_offset:], file_offset + data_offset)
        if written == 0:
            raise OSError("os.pwrite returned 0 bytes")
        data_offset += written


def write_threaded_pwrite(fds, offsets, data, repeat: int, workers: int):
    tasks = queue.Queue()
    errors = queue.Queue()
    next_shard = 0

    for _ in range(repeat):
        shard = next_shard
        next_shard = (next_shard + 1) % len(fds)
        offset = offsets[shard]
        offsets[shard] += len(data)
        tasks.put((fds[shard], offset))

    def worker():
        while True:
            item = tasks.get()
            if item is None:
                tasks.task_done()
                return
            fd, offset = item
            try:
                pwrite_all(fd, data, offset)
            except BaseException as exc:
                errors.put(exc)
            finally:
                tasks.task_done()

    threads = [
        threading.Thread(target=worker, name=f"bench-pwrite-{i}", daemon=False)
        for i in range(min(workers, repeat))
    ]
    for thread in threads:
        thread.start()
    for _ in threads:
        tasks.put(None)
    tasks.join()
    for thread in threads:
        thread.join()
    if not errors.empty():
        raise errors.get()


def write_threaded_append(fds, data, repeat: int, workers: int):
    worker_count = min(workers, len(fds), repeat)
    queues = [queue.Queue() for _ in range(worker_count)]
    errors = queue.Queue()

    for task_id in range(repeat):
        queues[task_id % worker_count].put(fds[task_id % len(fds)])

    def worker(q):
        while True:
            fd = q.get()
            if fd is None:
                q.task_done()
                return
            try:
                write_all(fd, data)
            except BaseException as exc:
                errors.put(exc)
            finally:
                q.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(queues[i],),
            name=f"bench-append-{i}",
            daemon=False,
        )
        for i in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for q in queues:
        q.put(None)
    for q in queues:
        q.join()
    for thread in threads:
        thread.join()
    if not errors.empty():
        raise errors.get()


_KEEP_ALIVE = []


def keep_alive(value):
    _KEEP_ALIVE.append(value)
    if len(_KEEP_ALIVE) > 2:
        del _KEEP_ALIVE[:-2]


def print_results(results: Iterable[Result]):
    print(f"{'stage':<24} {'best ms':>10} {'avg ms':>10} {'best GB/s':>12} {'avg GB/s':>11}")
    print("-" * 72)
    for item in results:
        print(
            f"{item.name:<24} "
            f"{item.best * 1000:>10.3f} "
            f"{item.avg * 1000:>10.3f} "
            f"{item.gbps_best:>12.2f} "
            f"{item.gbps_avg:>11.2f}"
        )


if __name__ == "__main__":
    main()
