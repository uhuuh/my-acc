---
title: Remove io_flush_mode Config Option
date: 2026-05-25
---

## Summary

Remove the `io_flush_mode` configuration option entirely. The IOWriter will always use "stop" mode behavior (explicit `stop()` call required to flush pending writes).

## Motivation

Simplify the codebase by removing an unnecessary configuration option. The "atexit" mode was unreliable and added complexity. Explicit `stop()` calls are clearer and already used in tests.

## Changes

### acc/config.py
- Remove `io_flush_mode: str = "atexit"` field from Config dataclass

### acc/io.py
- Remove `import atexit`
- Remove atexit registration logic from `start()` (lines that check `io_flush_mode == "atexit"`)
- Remove flush-on-stop logic that checks `io_flush_mode == "stop"` (always flush on stop now)
- Remove `_atexit_flush()` method entirely
- Simplify `stop()` to always flush when async mode is enabled

### tests/conftest.py
- Remove `os.environ["ACC_IO_FLUSH_MODE"] = "stop"` line
- Remove `config.update(io_flush_mode="stop")` line

## Behavior After Change

- IOWriter always requires explicit `stop()` call to flush pending async writes
- No atexit handlers registered
- Simpler, more predictable behavior