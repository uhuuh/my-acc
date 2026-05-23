# AGENTS.md — my-acc (PyTorch Operator Dump Tool)

## Quick start
- No setup.py/pyproject.toml — install via `pip install -e .` (uses implicit discovery)
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
- `acc/__init__.py` exports: `ops_dump`, `ops_comp`, `OperatorRecord`, `SerializationSession`, `IOWriter`
- `dump.py`: `ops_dump` context manager / decorator → patches `torch.library.impl` + wraps operators via `TorchDispatchMode`
- `comp.py`: `ops_comp` → LCS matching between two dump sessions
- `serialization.py`: `SerializationSession` — manages dump sessions, `OperatorRecord` dataclass
- `cache.py`: `CacheEntry` / `CacheManager` — content-addressable tensor dedup via blake2b hash
- `io.py`: `IOWriter` — async or sync file writes
- `formatting.py`: display helpers
- `comparators.py`: per-operator comparison logic

## Environment
- `ACC_DUMP_ENABLED=0` disables all dump capture globally
- `my/` directory is gitignored — user-specific output/analysis files

## Gotchas
- No type annotations enforced — code uses minimal typing
- Custom classes like `CacheEntry` are serialized in `.pkl` files — `torch.load` must use `weights_only=False`
- Tests write to temp dirs via `tempfile.TemporaryDirectory` — no test artifacts persist
- `torch.library.impl` patching is done globally on import of `dump.py` — side-effect if imported twice
