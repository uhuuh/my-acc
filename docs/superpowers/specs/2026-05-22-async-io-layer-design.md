---
name: async-io-layer
description: Design for asyncio-based IO layer with async/sync write modes, callback support, and exit handling
metadata:
  type: project
---

# AsyncIO-based IO Layer Design

## Overview

Add an IO layer to the PyTorch Operator Dump Tool that provides async file writing capabilities to avoid blocking the main thread during model execution.

## Requirements

1. **IO Layer Interface**: Provide a `write(file_path, content)` interface (simple interface, no callback parameter)
2. **Async Write Mode**: Internal async loop for concurrent file writes (default mode)
3. **Sync Mode Option**: Support synchronous write mode as an alternative
4. **Exit Handling**: Log warning when program exits with pending write tasks (using atexit and signal handlers)
5. **Pending Task Tracking**: IOWriter internally tracks pending task count for exit warning
6. **Serialization Integration**: Replace seq and cache file write operations with async IO

## Architecture

```
┌─────────────────────────────────────────────────┐
│   External Components (SerializationSession)    │
│   - 调用 io.write(path, content)                │
│   - seq文件写入和cache文件写入都替换为async IO   │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│              IOWriter (Manager)                  │
│   - write(path, content)                        │
│   - 内置asyncio loop在独立线程运行               │
│   - 内部维护pending_files集合                   │
│   - 多个write任务并发执行                        │
│   - 文件名保持原始格式                           │
│   - shutdown()关闭loop                           │
└─────────────────────────────────────────────────┘
```

## Components

### 1. IOWriter (Manager)

**Purpose**: Single manager class providing `write()` interface with built-in async loop management and pending task tracking

**Constructor**:
- `IOWriter(enable_async=True)`: Initialize IOWriter with async/sync mode configuration
  - `enable_async`: 全局配置，决定所有write操作是否异步（默认True）
  - 当`enable_async=False`时，所有write操作都是同步的，不启动async loop
  - 当`enable_async=True`时，启动async loop，所有write操作异步执行

**Key Methods**:
- `write(file_path, content)`: Write file
  - `file_path`: 文件路径（保持原始格式，不添加任何标记）
  - `content`: 文件内容（根据文件扩展名自动选择序列化方式）
  - **Simple interface** - caller doesn't need to care about async/sync mode or completion

- `shutdown()`: Close async loop and wait for all pending tasks to complete
  - 在enable_async=True时调用，确保所有pending任务完成
  - 在enable_async=False时为空操作

**Internal Pending File Tracking**:
- Maintains `_pending_files` set (thread-safe): tracks file paths currently being written
- Adds file_path on write task submission
- Removes file_path on task completion (internal callback in async loop)
- Used by exit handlers to print warning with specific file paths:
  ```
  [IO WARN] Pending write tasks on exit:
    - /path/to/file1.json
    - /path/to/file2.pkl
  ```

**Internal Async Loop Management**:
- Maintains asyncio event loop in a dedicated thread
- `_start_loop()`: Start event loop thread on IOWriter initialization
- `_submit(coro)`: Submit coroutine via `run_coroutine_threadsafe`
- `_stop_loop()`: Stop loop and wait for thread to finish
- `_on_task_complete()`: Internal callback to decrement pending_count

**Write Behavior**:
- **Async mode** (default): Submit write task to loop, return immediately, execute concurrently
- **Sync mode**: Direct file write, block until complete (no pending count tracking)

**Content Serialization** (automatic based on file extension):
- `.json` files: use `json.dump()` with text mode
- `.pkl` files: use `pickle.dump()` with binary mode
- Other extensions: write as text or binary based on content type

**Concurrency**: Multiple `write()` calls execute simultaneously in the async loop

### 2. Exit Handlers

**atexit Handler**:
- Registered via `atexit.register()` in IOWriter initialization
- On normal exit: check `_pending_files` set, log warning with file list if not empty
- Attempt quick shutdown of pending tasks

**Signal Handlers**:
- Capture `SIGTERM` and `SIGINT`
- Log pending files list
- Re-trigger default signal handler after cleanup attempt
- **Windows Note**: SIGTERM is not available on Windows; only SIGINT (Ctrl+C) works. On Windows, atexit is the primary mechanism for handling program termination. Signal handlers should be registered conditionally based on platform.

**Why both**: atexit handles normal exit; signals handle forced termination (Ctrl+C, kill command on Unix)

### 3. SerializationSession Integration

**Changes**:
- Add `enable_async_io` parameter to constructor
- Initialize `IOWriter` in `start()` method
- Replace ALL file writes with `io_writer.write()` calls:
  ```python
  # Seq JSON file write (replace existing direct write)
  io_writer.write(json_path, json.dumps(metadata, indent=2))

  # Seq PKL file write (replace existing direct write)
  io_writer.write(pkl_path, pickle.dumps(data))

  # Cache file write (replace CacheManager direct writes)
  io_writer.write(cache_path, pickle.dumps(tensor_data))
  ```
- Remove `_on_json_written` and `_on_pkl_written` callbacks (no longer needed)
- Call `io_writer.shutdown()` in `end()` method to wait for all pending writes

**Integration Points**:
- **Seq file writes**: Replace `save_operation()` method's direct file writes (lines 185-197 in serialization.py)
- **Cache file writes**: Replace `CacheManager._write_cache_file()` direct writes (need to add method in cache.py)

**CacheManager Changes**:
- Pass `IOWriter` instance to `CacheManager`
- Replace direct pickle.dump in cache storage with `io_writer.write(cache_path, data)`
- Keep cache lookup logic unchanged (still in-memory hash map)

## Data Flow

1. **Write Request**: SerializationSession/CacheManager calls `io_writer.write(path, data)`
2. **Mode Selection**: IOWriter checks `enable_async` config
   - Sync mode (enable_async=False): direct file write → return
   - Async mode (enable_async=True): submit coroutine to loop → add path to `_pending_files` → return immediately
3. **Async Execution**: Loop executes write task concurrently with other tasks
4. **Task Completion**: Async task completes → remove path from `_pending_files` (internal tracking)
5. **Exit Handling**: On program exit, check `_pending_files` set, log warning with file list if not empty

## Error Handling

**File Write Failure**: Catch exception, log `[IO ERROR]`, remove file from `_pending_files` set

**Disk Space**: Pre-check available space before write

**Invalid Path**: Validate and create parent directories

**Permission**: Catch `PermissionError`, log and fail gracefully

**Pending Files on Exit**: Log `[IO WARN]` with file list, attempt quick shutdown

## Testing Strategy

**Unit Tests**:
- Single file async write
- Concurrent multiple file writes
- Sync mode write
- Pending files tracking (verify files added/removed from set correctly)
- Shutdown with pending files (verify all files complete)
- Exit handler warning (verify warning printed with file list)

**Integration Tests**:
- SerializationSession integration (seq file writes)
- CacheManager integration (cache file writes)
- High-frequency operator calls
- Large file writes

**Performance Tests**:
- Async vs sync benchmark
- Concurrent write throughput

## Implementation Notes

- Use `aiofiles` for async file I/O (or implement with thread pool executor as fallback)
- Track pending files with `Set[str]` (file paths) to support exit checks and warnings
- Ensure thread safety for `_pending_files` set operations (use threading.Lock or asyncio-safe operations)
- No callback mechanism needed - simple fire-and-forget write interface

## Dependencies

- `asyncio` (standard library)
- `threading` (standard library)
- `atexit` (standard library)
- `signal` (standard library)
- Optional: `aiofiles` for async file operations (can use `ThreadPoolExecutor` as alternative)

## File Structure

```
acc/
  io.py          # New module: IOWriter only
  serialization.py  # Modified: integrate IOWriter
  dump.py        # Modified: pass enable_async_io parameter
```