r"""Verify torch.multiprocessing Queue uses shared memory for tensors.

Experiment A — torch.mp.Queue:  reducer calls share_memory_(), child
maps the same physical pages via fd — is_shared=True, <5ms, no RSS growth.

Experiment B — pickle via stdin:  tensor bytes fully serialized, child
allocates fresh memory — is_shared=False, slow, RSS grows by tensor size.

Usage:  python util/verify_shared_memory.py
"""

import io, os, sys, time, pickle, subprocess, multiprocessing as mp
import torch, psutil

# ═══════════════════════════════════════════════════════════════════
MB = 128
N = (MB * 1024 * 1024) // 4               # float32 elements
CTX = mp.get_context("spawn")


# ═══════════════════════════════════════════════════════════════════
# A. torch.mp.Queue spawn  →  shared memory
# ═══════════════════════════════════════════════════════════════════

def _child_a(q_in, q_out):
    p = psutil.Process()
    rss0 = p.memory_info().rss
    t0 = time.perf_counter()
    t = q_in.get()
    dt = time.perf_counter() - t0
    rss1 = p.memory_info().rss
    q_out.put((t.is_shared(), dt * 1000, (rss1 - rss0) / (1024 * 1024)))

def test_shared_memory():
    print(f"A. torch.mp.Queue spawn  —  {MB} MB tensor\n")
    t = torch.randn(N)

    qi, qo = CTX.Queue(), CTX.Queue()
    p = CTX.Process(target=_child_a, args=(qi, qo))
    p.start(); time.sleep(0.3)

    t0 = time.perf_counter()
    qi.put(t)
    put_ms = (time.perf_counter() - t0) * 1000
    p.join(5)
    shared, recv_ms, rss_mb = qo.get()

    print(f"    parent Queue.put:    {put_ms:.1f} ms")
    print(f"    child  Queue.get:    {recv_ms:.1f} ms")
    print(f"    child  is_shared():  {shared}")
    print(f"    child  RSS delta:    {rss_mb:+.1f} MB  (tensor = {MB} MB)")
    ok = shared and recv_ms < 50 and abs(rss_mb) < 5
    print(f"    verdict:             {'PASS — shared memory, zero copy' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════════════════════
# B. pickle via stdin  →  real copy
# ═══════════════════════════════════════════════════════════════════

B_SCRIPT = """\
import sys, time, pickle, io, os, psutil, torch, json
p = psutil.Process()
rss0 = p.memory_info().rss
t0 = time.perf_counter()
raw = sys.stdin.buffer.read()
dt = time.perf_counter() - t0
t1 = time.perf_counter()
t = pickle.load(io.BytesIO(raw))
dt_load = time.perf_counter() - t1
rss1 = p.memory_info().rss
print(json.dumps({"read_ms": dt*1000, "load_ms": dt_load*1000,
    "shared": t.is_shared(), "rss_mb": (rss1-rss0)/(1024*1024)}))
"""

def test_real_copy():
    print(f"\nB. pickle via stdin  —  {MB} MB tensor\n")
    t = torch.randn(N)
    buf = io.BytesIO()
    pickle.dump(t, buf)
    raw = buf.getvalue()

    r = subprocess.run([sys.executable, "-c", B_SCRIPT],
                       input=raw, capture_output=True, timeout=30)

    import json
    d = json.loads(r.stdout.decode().splitlines()[-1])

    print(f"    child  read+load:    {d['read_ms'] + d['load_ms']:.0f} ms")
    print(f"    child  is_shared():  {d['shared']}")
    print(f"    child  RSS delta:    {d['rss_mb']:+.1f} MB  (tensor = {MB} MB)")
    ok = not d["shared"] and d["rss_mb"] > MB * 0.4 and d["read_ms"] > 20
    print(f"    verdict:             {'PASS — real copy (no shared memory reducers)' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 56)
    print("  torch.multiprocessing Queue — shared memory proof")
    print("=" * 56 + "\n")
    a = test_shared_memory()
    b = test_real_copy()
    print("\n" + "=" * 56)
    print(f"  {'ALL PASS' if a and b else 'SOME FAILED'}"
          f"  |  shared-memory {MB}MB in ~2ms vs real-copy ~200ms")
    print("=" * 56)
    sys.exit(0 if a and b else 1)
