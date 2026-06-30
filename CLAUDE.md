# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package (editable)
pip install --system -e .

# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_io.py::test_iowriter_read_pickle -v
```

No lint / typecheck / formatter config exists.

## Public API

Three functions exposed from `acc`:

- `acc_dump(dump_path, model=None, **kwargs)` — context manager that captures PyTorch operator calls during the wrapped block. Uses `TorchDispatchMode` (ops backend) and optionally `register_forward_hook` (module backend, requires `model=`).
- `acc_info(dump_dir, filter_fn=None)` — prints operator info from a dump session. Shows kwargs and outputs; tensors display shape, dtype, max, min, mean, std, and quartiles.
- `acc_comp(dump_dir_a, dump_dir_b, key_fn=None, filter_fn=None)` — compares two dump sessions using LCS matching between operator sequences, then per-operator tensor/arg comparison.

## Architecture

```
acc_dump()  →  Manager  →  Capturer (backends: OpsCapturer / ModuleCapturer)
                          →  CacheManager (dedup tensors by content, writes .pt files)
                          →  Serializer (writes .json + .pkl per operator via IOWriter)

acc_comp()  →  LCS match operator sequences
            →  per-pair: comparators.py (Tensor, Numpy, Int, Float, …)
```

**Capture pipeline** (`acc_dump`):
1. `Manager.start()` creates a session dir with UUID-based name
2. Starts `CacheManager` (content-addressable tensor dedup via `Storage.cache_id` — hash by ptr+version+numel)
3. Starts `Serializer` (sync thread or async subprocess) that writes `.json` metadata + `.pkl` (CacheEntry-wrapped args)
4. Starts `Capturer` which activates `TorchDispatchMode` (all ops) + optional forward hooks (module names)
5. Each captured call → `Manager._handler()` → capture frames → `CacheManager.save()` (dedup, write `.pt`) → `Serializer.save()` (write `.json` + `.pkl`)

**File format per operator** — three files:
- `<save_id>.json` — metadata (seq_id, filepath, function, lineno, key, capturer, call_stack)
- `<save_id>.pkl` — inputs/outputs as `CacheEntry` references (lightweight, no tensor data)
- `cache/<cache_id>.pt` — raw tensor storage (one per unique tensor across all ops)

**Comparison pipeline** (`acc_comp`):
1. Load all `.json` metadata from two dump dirs, sort by `seq_id`
2. Compute LCS between operator sequences using configurable `key_fn(is_left, record) -> str`
3. Print matched/skipped operator pairs
4. For each matched pair: load tensor data from `.pkl` + `.pt`, compare using `comparators.py`

## Memory Allocators

- `"native"` (default): `torch.empty_like` for each tensor, no pooling
- `"pin"`: power-of-2 free-list pool with pinned memory, view-based splitting. Use for GPU training where CPU→GPU transfer speed matters.

Config: `config.update(memory_allocator="pin")` or env `ACC_MEMORY_ALLOCATOR=pin`.

## Config

`acc/config.py` has a `Config` dataclass singleton. All fields settable via kwargs or `ACC_<NAME>` env vars:

| Field | Default | Description |
|---|---|---|
| `dump_path` | `"."` | Output directory |
| `dump_enabled` | `True` | Set `0` to globally disable capture |
| `max_tensor_size_mb` | `10240` | Tensors above this are replaced with `None` |
| `capturer_backends` | `"ops,module"` | Comma-separated: `ops`, `module` |
| `memory_allocator` | `"native"` | `"native"` or `"pin"` |
| `serializer_kind` | `"sync"` | `"sync"` or `"async"` (subprocess) |
| `async_io` | `True` | Queue-based async file writes |
| `*_monitor_interval` | `1.0` | Print stats interval in seconds |

## Gotchas

- AGENTS.md is the authoritative dev reference (out of date relative to current code — verifies against source).
- `torch.load(weights_only=False)` is required throughout because `.pt` files contain custom dataclasses (`CacheEntry`) — PyTorch 2.6+ safe-mode blocks this by default.
- `torch.library.Library.impl` and `torch.library.impl` are patched at module level (on `import acc`) via `_kernel_wrapper` to re-enter `__torch_dispatch__` from custom ops. The wrapper is a no-op when no `acc_dump` is active.
- The `my/` directory is gitignored for user-specific output/analysis.
- Tests write to `tempfile.TemporaryDirectory` — no artifacts persist.
- Tests import via `sys.path.insert(0, ...)` hack (in each test file header), not via the installed package.
- Fork-based multiprocessing (`mp.get_context('fork')`) is used for async serializer — safe as child doesn't touch parent's daemon threads or CUDA.
