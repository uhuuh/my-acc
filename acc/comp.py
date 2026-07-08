"""
Operator Dumps Comparison for PyTorch Operator Dump Tool.

Provides acc_comp (compare two dumps), acc_info (print dump info),
and acc_get (read dump into structured data).
"""

import functools
import os
import time
from typing import Callable, List, Tuple

import numpy as np
import torch
from loguru import logger

from .record import Record
from .dump_format import has_record_store, load_data, load_info, read_all_infos
from .comparators import MISSING, compare_values


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m{int(seconds % 60)}s"
    else:
        return f"{int(seconds / 3600)}h{int((seconds % 3600) / 60)}m"


def default_key_fn(_is_left: bool, record: Record) -> str:
    return f"{record.capturer_type}::{record.capturer_key}"


def _lcs_length(
    a: List[str], b: List[str]
) -> Tuple[int, List[Tuple[int, int]]]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    matched_pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            matched_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matched_pairs.reverse()
    return dp[m][n], matched_pairs


def get_tensor_info(obj) -> str:
    """Return formatted tensor stats string for a tensor or numpy array."""
    if isinstance(obj, np.ndarray):
        t = torch.from_numpy(obj)
    else:
        t = obj
    if t.numel() == 0:
        return f"tensor(dtype={t.dtype}, shape={list(t.shape)}, empty)"
    t_flat = t.float().flatten()
    max_val = t_flat.max().item()
    min_val = t_flat.min().item()
    mean_val = t_flat.mean().item()
    std_val = t_flat.std(unbiased=False).item()
    q = torch.quantile(t_flat, torch.tensor([0.25, 0.5, 0.75])).tolist()
    return (
        f"tensor(dtype={t.dtype}, shape={list(t.shape)}, "
        f"max={max_val:.4f}, min={min_val:.4f}, mean={mean_val:.4f}, "
        f"std={std_val:.4f}, "
        f"q25={q[0]:.4f}, q50={q[1]:.4f}, q75={q[2]:.4f})"
    )


def _format_val(v):
    """Format a single value for info display."""
    if isinstance(v, (torch.Tensor, np.ndarray)):
        return get_tensor_info(v)
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return "None"
    return repr(v)


# ═══════════════════════════════════════════════════════════════════
# Loading helpers
# ═══════════════════════════════════════════════════════════════════

def _record_from_info(info: dict) -> Record:
    """Create a Record from a load_info dict, filtering non-field keys."""
    field_names = {f.name for f in Record.__dataclass_fields__.values()}
    filtered = {k: v for k, v in info.items() if k in field_names}
    return Record(**filtered)


def _load_single(json_path: str) -> dict:
    """Load a single .json (+ its .pkl + cache) into a structured dict."""
    base = os.path.dirname(json_path)
    info = load_info(json_path)
    record = _record_from_info(info)
    pkl_path = os.path.join(base, f"{record.save_id}.pkl")
    inputs, outputs = load_data(pkl_path, base)
    return {"record": record, "inputs": inputs, "outputs": outputs}


def _load_all_metadata(
    dump_dir: str,
    filter_fn: Callable | None = None,
) -> list:
    """Load all metadata from dump directory (Record objects, no tensor data)."""
    records = []
    if has_record_store(dump_dir):
        infos = read_all_infos(dump_dir)
        for info in infos:
            try:
                record = _record_from_info(info)
                if filter_fn is not None and filter_fn(record):
                    continue
                records.append(record)
            except Exception as e:
                logger.warning(f"Failed to load merged record: {e}")
        records.sort(key=lambda x: x.seq_id)
        return records

    for fname in sorted(os.listdir(dump_dir)):
        if not fname.endswith(".json"):
            continue
        json_path = os.path.join(dump_dir, fname)
        try:
            info = load_info(json_path)
            record = _record_from_info(info)
            if filter_fn is not None and filter_fn(record):
                continue
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to load {fname}: {e}")
    records.sort(key=lambda x: x.seq_id)
    return records


# ═══════════════════════════════════════════════════════════════════
# Comparison helpers
# ═══════════════════════════════════════════════════════════════════

def _compare_lists(list_a: List, list_b: List, label: str):
    """Compare two lists of elements and print results."""
    max_len = max(len(list_a), len(list_b))
    for i in range(max_len):
        a = list_a[i] if i < len(list_a) else MISSING
        b = list_b[i] if i < len(list_b) else MISSING
        left_info, right_info, log = compare_values(a, b)
        print(f"  {label}[{i}] | {left_info} | {right_info} | {log}")


def _compare_kwargs(kwargs_a: dict, kwargs_b: dict, label: str):
    """Compare two kwargs dicts by key and print results."""
    all_keys = sorted(set(list(kwargs_a.keys()) + list(kwargs_b.keys())))
    for key in all_keys:
        key_label = f"{label}[{key}]"
        left_info, right_info, log = compare_values(
            kwargs_a.get(key, MISSING),
            kwargs_b.get(key, MISSING),
        )
        print(f"  {key_label} | {left_info} | {right_info} | {log}")


def _find_lcs_matches(records_a, records_b, key_fn):
    sigs_a = [key_fn(True, d) for d in records_a]
    sigs_b = [key_fn(False, d) for d in records_b]
    info_a = [(r.save_id, s) for r, s in zip(records_a, sigs_a)]
    info_b = [(r.save_id, s) for r, s in zip(records_b, sigs_b)]

    lcs_len, matched_pairs = _lcs_length(sigs_a, sigs_b)
    a_only = len(records_a) - lcs_len
    b_only = len(records_b) - lcs_len
    print(
        f"[LCS] Matched: {lcs_len} operators "
        f"| A-only: {a_only} | B-only: {b_only}"
    )

    i = j = 0
    for idx_a, idx_b in matched_pairs:
        while i < idx_a:
            print(f"[SKIP] A[{i}] {info_a[i]} <-> <empty>")
            i += 1
        while j < idx_b:
            print(f"[SKIP] <empty> <-> B[{j}] {info_b[j]}")
            j += 1
        print(f"[MATCH] A[{i}] {info_a[i]} <-> B[{j}] {info_b[j]}")
        i += 1
        j += 1
    while i < len(records_a):
        print(f"[SKIP] A[{i}] {info_a[i]} <-> <empty>")
        i += 1
    while j < len(records_b):
        print(f"[SKIP] <empty> <-> B[{j}] {info_b[j]}")
        j += 1
    return matched_pairs


def _compare_matched_pairs(
    records_a, records_b, matched_pairs, dump_a, dump_b
):
    total = len(matched_pairs)
    start_time = time.time()
    print(f"[COMPARE] Starting detailed comparison of {total} matched pairs...")
    for idx, (idx_a, idx_b) in enumerate(matched_pairs, 1):
        elapsed = time.time() - start_time
        avg_time = elapsed / idx if idx > 0 else 0
        eta_seconds = avg_time * (total - idx)
        print(f"[COMPARE {idx}/{total} | ETA: {format_eta(eta_seconds)}]")
        rec_a = records_a[idx_a]
        rec_b = records_b[idx_b]
        print(f"{rec_a.save_id}.json <-> {rec_b.save_id}.json")

        pkl_a = os.path.join(dump_a, f"{rec_a.save_id}.pkl")
        inputs_a, outputs_a = load_data(pkl_a, dump_a)

        pkl_b = os.path.join(dump_b, f"{rec_b.save_id}.pkl")
        inputs_b, outputs_b = load_data(pkl_b, dump_b)

        _compare_lists(inputs_a["args"], inputs_b["args"], "Inputs.args")
        _compare_kwargs(
            inputs_a["kwargs"], inputs_b["kwargs"], "Inputs.kwargs"
        )
        _compare_lists(outputs_a, outputs_b, "Outputs")


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def acc_get(path: str) -> list:
    """Read dump data from a directory or single .json file.

    Args:
        path: Session directory path, or a single .json file path.

    Returns:
        List of dicts, each containing:
            {"record": Record, "inputs": {...}, "outputs": [...]}
        Tensor placeholders are resolved through tensor_locations.jsonl.
    """
    if os.path.isfile(path):
        if not path.endswith(".json"):
            raise ValueError(
                f"Expected a .json file or directory, got: {path}"
            )
        return [_load_single(path)]

    results = []
    if has_record_store(path):
        for record in _load_all_metadata(path):
            pkl_path = os.path.join(path, f"{record.save_id}.pkl")
            inputs, outputs = load_data(pkl_path, path)
            results.append({
                "record": record,
                "inputs": inputs,
                "outputs": outputs,
            })
        return results

    for fname in sorted(os.listdir(path)):
        if fname.endswith(".json"):
            json_path = os.path.join(path, fname)
            try:
                results.append(_load_single(json_path))
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
    return results


def acc_info(dump_dir, filter_fn=None):
    """Print operator info from a dump session.

    Args:
        dump_dir: Path to the dump session directory, or a single .json file.
        filter_fn: Optional callable(record) -> bool.
            Return True to skip (filter out) the record.
    """
    if os.path.isfile(dump_dir):
        if not dump_dir.endswith(".json"):
            print(f"[ERROR] expected a .json file, got: {dump_dir}")
            return
        base = os.path.dirname(dump_dir)
        try:
            result = _load_single(dump_dir)
            records = [result["record"]]
            # We'll load inline below
        except Exception as e:
            print(f"[ERROR] failed to load {dump_dir}: {e}")
            return
    else:
        records = _load_all_metadata(dump_dir, filter_fn=filter_fn)
        base = dump_dir

    total = len(records)
    if total == 0:
        print(f"[INFO] No operators found in {dump_dir}")
        return

    print(f"[INFO] {total} operators loaded from {dump_dir}")

    for idx, record in enumerate(records, 1):
        print(f"[INFO {idx}/{total}] {record.save_id}")

        pkl_path = os.path.join(base, f"{record.save_id}.pkl")
        try:
            inputs, outputs = load_data(pkl_path, base)
        except Exception as e:
            print(f"  [ERROR] failed to load data: {e}")
            continue

        args = inputs.get("args", [])
        if args:
            for i, v in enumerate(args):
                print(f"args [{i}]: {_format_val(v)}")

        kwargs = inputs.get("kwargs", {})
        if kwargs:
            for k, v in kwargs.items():
                print(f"kwargs [{k}]: {_format_val(v)}")

        if outputs:
            for i, v in enumerate(outputs):
                print(f"outputs [{i}]: {_format_val(v)}")


def acc_comp(
    dump_dir_a: str,
    dump_dir_b: str,
    key_fn: Callable | None = None,
    filter_fn: Callable | None = None,
):
    """Compare two operator dump sessions.

    Args:
        dump_dir_a: Path to first dump session directory.
        dump_dir_b: Path to second dump session directory.
        key_fn: Optional callable(is_left: bool, record: Record) -> str
            to compute LCS matching key.
        filter_fn: Optional callable(is_left: bool, record: Record) -> bool.
            Return True to skip (filter out) the record before comparison.
    """
    if key_fn is None:
        key_fn = default_key_fn
    records_a = _load_all_metadata(
        dump_dir_a,
        filter_fn=(
            functools.partial(filter_fn, True)
            if filter_fn is not None
            else None
        ),
    )
    print(
        f"[LCS] Loading dump A: {len(records_a)} operators from {dump_dir_a}"
    )
    records_b = _load_all_metadata(
        dump_dir_b,
        filter_fn=(
            functools.partial(filter_fn, False)
            if filter_fn is not None
            else None
        ),
    )
    print(
        f"[LCS] Loading dump B: {len(records_b)} operators from {dump_dir_b}"
    )
    print("\n" * 4)
    matched_pairs = _find_lcs_matches(records_a, records_b, key_fn)
    print("\n" * 4)
    _compare_matched_pairs(
        records_a, records_b, matched_pairs, dump_dir_a, dump_dir_b
    )
