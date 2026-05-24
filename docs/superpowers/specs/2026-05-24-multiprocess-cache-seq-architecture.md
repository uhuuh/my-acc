# Multi-process Cache+Seq Architecture

## Problem
Current single-process architecture does synchronous tensor processing + disk I/O in the main process, blocking the model execution. Need multi-process pipeline with pinned memory pool and clean abstraction boundaries.

## Architecture

```
ops_dump (thin layer in dump.py)
  __enter__ → SerializationSender + start SerializationReceiver (subprocess)
  __torch_dispatch__ → sender.save_operation(opname, args, kwargs, outputs)
  __exit__ → sender.stop() → receiver joins

SerializationSender + SerializationReceiver (in serialization.py)
  - Sender: holds CacheManager + cache IOWriter + queue
  - Receiver: subprocess, holds seq IOWriter, consumes from queue

CacheManager (in cache.py)
  - Owns PinMemoryPool
  - track cache_id set (no load support)
  - tensor → cache_id → if miss: materialize via PinMemoryPool → write .pt via cache_io

PinMemoryPool + Storage (in memory.py)
  - Base allocator + NaiveAllocator + AdvancedAllocator (free-list buckets by size)
  - Storage: compute cache_id on create, materialize via allocator

IOWriter (in io.py)
  - add `name` attribute ("cache" / "seq")

## Data Flow
1. ops_dump intercepts op → calls sender.save_operation()
2. CacheManager traverses args/kwargs/outputs
3. For each tensor: compute cache_id (fast mode only, no strict)
4. If cache_id not in set: Storage.materialize → PinMemoryPool.acquire → to(contiguous, cpu, pin) → cache_io.write(.pt) → release → mark cached
5. Replace tensor with CacheEntry(cache_id, dtype, shape, type)
6. Transformed data → multiprocessing.Queue
7. Receiver subprocess: queue.get() → seq_io.write(.json) + seq_io.write(.pkl)
8. queue.get() == None → receiver exits

## Files Changed
- NEW: acc/memory.py — PinMemoryPool base + NaiveAllocator + AdvancedAllocator, Storage
- MOD: acc/io.py — IOWriter.name
- MOD: acc/cache.py — CacheManager simplified, owns PinMemoryPool
- MOD: acc/serialization.py — SerializationSender + SerializationReceiver
- MOD: acc/dump.py — ops_dump uses sender/receiver
- MOD: tests/ — adapt to new API, remove strict mode tests
