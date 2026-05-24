# Component Refactor Design

Date: 2026-05-24

## Overview

Refactor `my-acc`'s operator dump pipeline into well-separated components with clear start/stop/save lifecycles. Replaces the monolithic `SerializationSender`/`SerializationReceiver` with composable Capturer, Manager, CacheManager, IOWriter, and Serializer components.

## Component Design

### 1. IOWriter (`io.py`)

Responsible for async/sync file writes with periodic throughput monitoring.

**Changes:**
- Rename `write`/`read` to `save`/`load`
- Add explicit start/stop lifecycle
- stop() blocks until all writes complete, prints "remaining: N files" every 1s
- No atexit registration
- Monitor prints pending count + throughput every `config.io_monitor_interval` seconds

**Interface:**

```
class IOWriter:
    __init__(name="", enable_async=None)  # reads config if enable_async not explicitly set
    start()                                # start worker thread (async) or enable writes (sync)
    stop(timeout=None)                     # blocking wait, print pending count every 1s
    save(file_path, content)               # async queue or direct write
    load(file_path)                        # synchronous read via FileHandler
```

**Internal state:** worker thread, queue, pending_files set, bytes_written counter, monitor timer.

**Monitor behavior:** After each file write completes (in the async worker), check if `cache_monitor_interval` elapsed. If so, print pending count and throughput in bytes/sec, reset counters.

### 2. CacheManager (`cache.py`)

Content-addressable tensor cache. Owns its own IOWriter.

**Changes:**
- Constructor takes no args (reads config)
- Create cache dir (`<session>/cache/`) in start()
- Owns IOWriter internally, creates it in start()
- Periodic monitoring: bytes added this interval + total bytes cached, every `config.cache_monitor_interval`
- Previous `save`/`load` logic for tensor <-> CacheEntry conversion unchanged

**Monitor behavior:** After each `save()` call, check if monitor interval elapsed. Print bytes added in this interval and total bytes cached across all time.

**Interface:**

```
class CacheManager:
    __init__()                             # no args
    start(session_dir)                     # creates <session_dir>/cache/, starts IOWriter
    stop()                                 # calls io.stop()
    save(data) -> Any                     # traverse, replace tensors with CacheEntry
    load(data) -> Any                     # resolve CacheEntry back to tensors
```

### 3. Serializer / AsyncSerializer (`serialization.py`)

Splits into two classes. Writes .json (metadata) and .pkl (data) files.

**Serializer (sync):**
- `start(session_dir)`: creates IOWriter for seq writes
- `save(item)`: processes frames, writes .json + .pkl
- `stop()`: calls io.stop()
- `load_metadata(json_path)`: classmethod, reads .json -> OperatorRecord
- `load_data(pkl_path, storage_dir)`: classmethod, reads .pkl, resolves CacheEntry via CacheManager
- Constructor: no args

**AsyncSerializer (subprocess):**
- `start(session_dir)`: spawns subprocess holding Serializer
- `save(item)`: queues item via mp.Queue to subprocess
- `stop()`: sends sentinel, joins subprocess
- Constructor: no args

**OperatorRecord** stays in serialization.py, unchanged.

**SerializationManager compatibility shim:** `SerializationManager` is kept with `load_metadata` and `load_data` static methods that delegate to `Serializer.load_metadata`/`Serializer.load_data`, so comp.py (which imports `SerializationManager` directly) continues to work.

### 4. Capturer (`capturer.py`) — NEW FILE

Extracts hook mechanism from dump.py. Responsible for PyTorch operator interception.

**Behavior:**
- `start(handler)`: activates TorchDispatchMode, sets _active_session global
- `stop()`: deactivates TorchDispatchMode, clears _active_session
- `__torch_dispatch__`: calls handler(opname, args, kwargs, result)
- `torch.library.impl` patch: installed on import (same as current)
- Constructor: no args

**Interface:**

```
class Capturer:
    __init__()
    start(handler)     # handler(opname, args, kwargs, outputs)
    stop()
```

Internal: TorchDispatchMode subclass, _active_session global, impl patch.

### 5. Manager (`manager.py`) — NEW FILE

Orchestrates all components. Creates session directory.

**Behavior:**
- `start()`: creates session dir, starts cache, starts serializer, starts capturer with self._handler
- `_handler(opname, args, kwargs, outputs)`: captures call stack via sys._getframe, calls cache.save(), calls serializer.save()
- `stop()`: stops capturer, stops serializer, stops cache
- Constructor: no args (reads config)
- Chooses Serializer vs AsyncSerializer based on config.async_serialization

**Interface:**

```
class Manager:
    __init__()
    start()
    stop()
    _handler(opname, args, kwargs, outputs)
```

### 6. Config (`config.py`)

Add four new fields:

```python
io_monitor_interval: float = 5.0       # ACC_IO_MONITOR_INTERVAL
cache_monitor_interval: float = 5.0    # ACC_CACHE_MONITOR_INTERVAL
async_io: bool = True                  # ACC_ASYNC_IO
async_serialization: bool = True       # ACC_ASYNC_SERIALIZATION
```

### 7. Entry Point (`main.py`) — NEW FILE

Single public function `ops_dump`:

```python
def ops_dump(**kwargs):
    config.update(**{k: v for k, v in kwargs.items() if v is not None})
    mgr = Manager()
    return _OpsDumpContext(mgr)
```

`_OpsDumpContext` supports `__enter__`/`__exit__` (context manager) and `__call__` (decorator).

### 8. `__init__.py`

Exports from main.py + remaining modules:

```python
from .main import ops_dump
from .comp import ops_comp
from .cache import CacheEntry
from .config import config
from .io import IOWriter, FileHandler
from .serialization import SerializationManager, Serializer, AsyncSerializer, OperatorRecord
```

Note: `SerializationManager` is kept as a thin wrapper in serialization.py for comp.py backward compatibility (comp.py imports it directly).

### 9. Files to remove

- `dump.py` — logic moved to capturer.py + manager.py + main.py

## Directory Structure After Refactor

```
acc/
  __init__.py
  config.py          # modified: 4 new fields
  io.py              # modified: rename methods, add start/stop
  cache.py           # modified: owns IOWriter, add start/stop, monitoring
  serialization.py   # modified: Serializer + AsyncSerializer classes
  capturer.py        # NEW: TorchDispatchMode hook
  manager.py         # NEW: orchestrates components
  main.py            # NEW: ops_dump entry point
  memory.py          # unchanged
  comp.py            # unchanged
  comparators.py     # unchanged
  formatting.py      # unchanged
```

## Key Design Decisions

1. Components are **no-arg constructed** and read config directly (except ops_dump kwargs which call config.update first)
2. **No atexit** — cleanup is blocking in stop()
3. **stop()** is blocking for IOWriter (waits all writes, prints pending count every 1s)
4. IO **renamed**: write->save, read->load
5. **Cache dir** renamed from `storage/` to `cache/`
6. Call stack capture in **Manager's handler**, not Capturer
7. **comp.py** unchanged — `SerializationManager` class kept as thin wrapper with `load_metadata`/`load_data` static methods that delegate to Serializer's classmethods
8. **OperatorRecord** stays in serialization.py, unchanged
