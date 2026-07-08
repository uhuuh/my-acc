"""Benchmark XXH3_64 vs MD5 on 1 KB and 1 GB."""

import timeit
import time
import xxhash
import hashlib

REPEAT = 5

CASES = [
    ("1 KB", 1024, 1000),
    ("1 GB", 1024**3, 100),
]

VARIANTS = [
    ("xxh3_64", lambda d: xxhash.xxh3_64(d).digest()),
    # ("hashlib.md5", lambda d: hashlib.md5(d).digest()),
]


def bench_one(name, fn, data, number):
    timer = timeit.Timer(lambda: fn(data))
    raw = timer.repeat(REPEAT, number=number)
    best = min(raw)
    return best / number * 1e9  # ns per call


def main():
    print(f"{'size':>6s}  {'variant':>12s}  {'ns/call':>12s}  {'GB/s':>10s}  {'total':>10s}  {'runs':>8s}")
    print("-" * 70)

    for label, size, runs in CASES:
        data = bytes(size)
        for name, fn in VARIANTS:
            t0 = time.perf_counter()
            ns = bench_one(name, fn, data, runs)
            t1 = time.perf_counter()
            gb_s = size / ns
            total_s = t1 - t0
            print(
                f"{label:>6s}  {name:>12s}  {ns:>12.1f}  {gb_s:>10.2f}  "
                f"{total_s:>8.1f}s  {runs:>6d}"
            )


if __name__ == "__main__":
    main()
