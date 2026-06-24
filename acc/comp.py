"""
Operator Dumps Comparison for PyTorch Operator Dump Tool.

Provides ops_comp function for comparing two dump sessions.
"""

import json
import os
import time
from typing import Callable, List, Tuple
from .serialization import Serializer, OperatorRecord
from .comparators import (
    create_comparator,
    MissingInAComparator,
    MissingInBComparator
)


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds/3600)}h{int(seconds%3600/60)}m"


def default_key_fn(_is_left: bool, record: OperatorRecord) -> str:
    return f"{record.capturer}:{record.filename.replace('.py', '')}::{record.key}"


def _sep():
    for _ in range(5):
        print()


def _lcs_length(a: List[str], b: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    matched_pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i-1] == b[j-1]:
            matched_pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    matched_pairs.reverse()
    return dp[m][n], matched_pairs


def _load_all_metadata(
    dump_dir: str,
    is_left: bool,
    filter_fn: Callable[[bool, OperatorRecord], bool] | None = None,
) -> list:
    """Load all metadata from dump directory (JSON only, no tensor data).

    Args:
        dump_dir: Path to the dump session directory.
        is_left: True when loading the left-side (A) dump, False for right (B).
        filter_fn: Optional callable(is_left, record) -> bool.
            Return True to skip (filter out) the record.
    """
    records = []
    for filename in os.listdir(dump_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(dump_dir, filename)
            try:
                record = Serializer.load_metadata(json_path)
                if filter_fn is not None and filter_fn(is_left, record):
                    continue
                records.append(record)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"[COMP WARN] Failed to load metadata {filename}: {e}")
            except Exception as e:
                print(f"[COMP WARN] Unexpected error loading {filename}: {e}")
    records.sort(key=lambda x: x.seq_id)
    return records


def _compare_lists(list_a: List, list_b: List, label: str):
    """Compare two lists of elements and print results."""
    max_len = max(len(list_a), len(list_b))
    for i in range(max_len):
        if i >= len(list_a):
            comparator = MissingInAComparator(list_b[i])
        elif i >= len(list_b):
            comparator = MissingInBComparator(list_a[i])
        else:
            comparator = create_comparator(list_a[i], list_b[i])
        left_info, right_info = comparator.get_type_info()
        log = comparator.compare()
        print(f"  {label}[{i}] | {left_info} | {right_info} | {log}")


def _compare_kwargs(kwargs_a: dict, kwargs_b: dict, label: str):
    """Compare two kwargs dicts by key and print results."""
    all_keys = sorted(set(list(kwargs_a.keys()) + list(kwargs_b.keys())))
    for key in all_keys:
        key_label = f"{label}[{key}]"
        if key not in kwargs_a:
            comparator = MissingInAComparator(kwargs_b[key])
        elif key not in kwargs_b:
            comparator = MissingInBComparator(kwargs_a[key])
        else:
            comparator = create_comparator(kwargs_a[key], kwargs_b[key])
        left_info, right_info = comparator.get_type_info()
        log = comparator.compare()
        print(f"  {key_label} | {left_info} | {right_info} | {log}")


def _find_lcs_matches(records_a: list, records_b: list, key_fn) -> List[Tuple[int, int]]:
    sigs_a = [key_fn(True, d) for d in records_a]
    sigs_b = [key_fn(False, d) for d in records_b]
    info_a = [(r.save_id, s) for r, s in zip(records_a, sigs_a)]
    info_b = [(r.save_id, s) for r, s in zip(records_b, sigs_b)]

    lcs_len, matched_pairs = _lcs_length(sigs_a, sigs_b)
    a_only = len(records_a) - lcs_len
    b_only = len(records_b) - lcs_len
    print(f"[LCS] Matched: {lcs_len} operators | A-only: {a_only} | B-only: {b_only}")

    i = j = 0
    for idx_a, idx_b in matched_pairs:
        while i < idx_a:
            print(f"[SKIP] A[{i}] {info_a[i]} <-> <empty>"); i += 1
        while j < idx_b:
            print(f"[SKIP] <empty> <-> B[{j}] {info_b[j]}"); j += 1
        print(f"[MATCH] A[{i}] {info_a[i]} <-> B[{j}] {info_b[j]}")
        i += 1; j += 1
    while i < len(records_a):
        print(f"[SKIP] A[{i}] {info_a[i]} <-> <empty>"); i += 1
    while j < len(records_b):
        print(f"[SKIP] <empty> <-> B[{j}] {info_b[j]}"); j += 1
    return matched_pairs


def _compare_matched_pairs(records_a: list, records_b: list, matched_pairs, storage_a: str, storage_b: str):
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
        session_dir_a = os.path.dirname(storage_a)
        session_dir_b = os.path.dirname(storage_b)
        pkl_path_a = os.path.join(session_dir_a, f"{rec_a.save_id}.pkl")
        pkl_path_b = os.path.join(session_dir_b, f"{rec_b.save_id}.pkl")
        inputs_a, outputs_a = Serializer.load_data(pkl_path_a, storage_a)
        inputs_b, outputs_b = Serializer.load_data(pkl_path_b, storage_b)
        _compare_lists(inputs_a['args'], inputs_b['args'], "Inputs.args")
        _compare_kwargs(inputs_a['kwargs'], inputs_b['kwargs'], "Inputs.kwargs")
        _compare_lists(outputs_a, outputs_b, "Outputs")


def ops_comp(
    dump_dir_a: str,
    dump_dir_b: str,
    key_fn: Callable[[bool, OperatorRecord], str] | None = None,
    filter_fn: Callable[[bool, OperatorRecord], bool] | None = None,
):
    """Compare two operator dump sessions.

    Args:
        dump_dir_a: Path to first dump session directory.
        dump_dir_b: Path to second dump session directory.
        key_fn: Optional callable(is_left: bool, record: OperatorRecord) -> str
            to compute LCS matching key.
        filter_fn: Optional callable(is_left: bool, record: OperatorRecord) -> bool.
            Return True to skip (filter out) the record before comparison.
    """
    if key_fn is None:
        key_fn = default_key_fn
    records_a = _load_all_metadata(dump_dir_a, is_left=True, filter_fn=filter_fn)
    print(f"[LCS] Loading dump A: {len(records_a)} operators from {dump_dir_a}")
    records_b = _load_all_metadata(dump_dir_b, is_left=False, filter_fn=filter_fn)
    print(f"[LCS] Loading dump B: {len(records_b)} operators from {dump_dir_b}")
    _sep()
    matched_pairs = _find_lcs_matches(records_a, records_b, key_fn)
    _sep()
    storage_a = os.path.join(dump_dir_a, 'storage')
    storage_b = os.path.join(dump_dir_b, 'storage')
    _compare_matched_pairs(records_a, records_b, matched_pairs, storage_a, storage_b)