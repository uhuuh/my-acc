# AGENTS.md — my-acc (PyTorch Operator Dump Tool)

## Quick start
- `pyproject.toml` — install via `pip install -e .`
- No lint/typecheck/formatter config — run only `pytest tests/`
- Dependencies: torch 2.9.1+, numpy, pytest

## Tests
- All tests: `python -m pytest tests/ -v` (48 tests, all pass)
- Single test: `python -m pytest tests/test_io.py::test_iowriter_read_pickle -v`
- Tests use `sys.path.insert(0, ...)` hack for import — not a proper editable install
- No fixtures, no conftest.py, no markers

## File format
- `.json`: metadata (sequence, filename, function, lineno, opname, call_stack)
- `.pkl`: CacheEntry-wrapped data, saved/loaded via `pickle` (fast path, small objects)
- `.pt`: raw tensor storage in `storage/`, saved/loaded via `torch.save`/`torch.load(weights_only=False)`
- Naming: `{sequence:06d}__{filename}__{function}__{opname}.json`
- `weights_only=False` is required for `torch.load` (PyTorch 2.6+ safe-mode default blocks custom classes)

## Architecture
- `acc/__init__.py` exports: `acc_dump`, `acc_comp`, `OperatorRecord`, `SerializationSender`, `SerializationReceiver`, `load_metadata`, `load_data`, `IOWriter`, `CacheEntry`, `CacheManager`, `resolve_cache_entry`, `resolve_cache_entries`
- `main.py`: `acc_dump` context manager → creates Capturer + Manager
- `memory.py`: `PinMemoryAllocator` base, `NaiveAllocator`, `AdvancedAllocator` (free-list buckets by size), `Storage` (compute cache_id + materialize via allocator)
- `cache.py`: `CacheEntry` dataclass, `CacheManager` (tracks cache_id set, owns PinMemoryPool, writes .pt via cache IOWriter), `resolve_cache_entry` / `resolve_cache_entries` for loading
- `serialization.py`: `SerializationSender` (transforms tensors via CacheManager, queues data), `SerializationReceiver` (subprocess target, writes .json/.pkl via seq IOWriter), `OperatorRecord`, `load_metadata`, `load_data`
- `io.py`: `IOWriter` with `name` attribute ("cache"/"seq"), `FileHandler` with module-level handler functions (pickle-safe for spawn)
- `comp.py`: `acc_comp` → LCS matching between two dump sessions, uses `load_metadata`/`load_data`
- `formatting.py`: display helpers
- `comparators.py`: per-operator comparison logic

## Multiprocessing
- Uses `mp.get_context('fork')` for subprocess — faster, no `if __name__` requirement
- Queue is created with fork context in `SerializationSender.__init__`
- `_receiver_main(session_dir, queue)` is the module-level target for Process
- WARNING: `fork` from multi-threaded process produces a deprecation warning but is safe — child process does not touch parent's daemon threads or CUDA contexts

## Config
- `acc/config.py` — centralized config via `config.init()` and `config.get_*()` functions
- `config.init()` sets env vars (`ACC_DUMP_PATH`, `ACC_DUMP_ENABLED`, `ACC_MAX_TENSOR_SIZE_MB`); conflicts print a warning
- `acc_dump.__init__` calls `config.init()`; all modules read via config getters
- `ACC_DUMP_PATH=<path>`: required (via arg or env var)
- `ACC_DUMP_ENABLED=0` disables all dump capture globally
- `my/` directory is gitignored — user-specific output/analysis files

## Gotchas
- No type annotations enforced — code uses minimal typing
- Custom classes like `CacheEntry` are serialized in `.pkl` files — `torch.load` must use `weights_only=False`
- Tests write to temp dirs via `tempfile.TemporaryDirectory` — no test artifacts persist
- `torch.library.impl` patching is done globally on import of `capturer.py` (OpsCapturer) — side-effect if imported twice
