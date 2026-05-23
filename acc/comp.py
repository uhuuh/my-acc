"""
Operator Dumps Comparison for PyTorch Operator Dump Tool.

Provides ops_comp function for comparing two dump sessions.
"""

import json
import os
import time
from typing import List, Tuple
from .serialization import SerializationSession
from .formatting import (
    format_signature,
    format_display_key,
    format_dump_filename,
    format_eta,
)
from .comparators import (
    create_comparator,
    MissingInAComparator,
    MissingInBComparator
)


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


def _load_all_metadata(dump_dir: str) -> list:
    """Load all metadata from dump directory (JSON only, no tensor data)."""
    records = []
    for filename in os.listdir(dump_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(dump_dir, filename)
            try:
                record = SerializationSession.load_metadata(json_path)
                records.append(record)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"[COMP WARN] Failed to load metadata {filename}: {e}")
            except Exception as e:
                print(f"[COMP WARN] Unexpected error loading {filename}: {e}")
    records.sort(key=lambda x: x.sequence)
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


def _find_lcs_matches(records_a: list, records_b: list) -> List[Tuple[int, int]]:
    sigs_a = [format_signature(d) for d in records_a]
    sigs_b = [format_signature(d) for d in records_b]
    lcs_len, matched_pairs = _lcs_length(sigs_a, sigs_b)
    a_only = len(records_a) - lcs_len
    b_only = len(records_b) - lcs_len
    print(f"[LCS] Matched: {lcs_len} operators | A-only: {a_only} | B-only: {b_only}")
    prev_a, prev_b = 0, 0
    for idx_a, idx_b in matched_pairs:
        for i in range(prev_a, idx_a):
            key_a = format_display_key(records_a[i])
            print(f"[SKIP] A[{i}] {key_a} <-> <empty>")
        for j in range(prev_b, idx_b):
            key_b = format_display_key(records_b[j])
            print(f"[SKIP] <empty> <-> B[{j}] {key_b}")
        key_a = format_display_key(records_a[idx_a])
        key_b = format_display_key(records_b[idx_b])
        print(f"[MATCH] A[{idx_a}] {key_a} <-> B[{idx_b}] {key_b}")
        prev_a = idx_a + 1
        prev_b = idx_b + 1
    for i in range(prev_a, len(records_a)):
        key_a = format_display_key(records_a[i])
        print(f"[SKIP] A[{i}] {key_a} <-> <empty>")
    for j in range(prev_b, len(records_b)):
        key_b = format_display_key(records_b[j])
        print(f"[SKIP] <empty> <-> B[{j}] {key_b}")
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
        filename_a = format_dump_filename(rec_a)
        filename_b = format_dump_filename(rec_b)
        print(f"{filename_a} <-> {filename_b}")
        session_dir_a = os.path.dirname(storage_a)
        session_dir_b = os.path.dirname(storage_b)
        pkl_path_a = os.path.join(session_dir_a, filename_a.replace('.json', '.pkl'))
        pkl_path_b = os.path.join(session_dir_b, filename_b.replace('.json', '.pkl'))
        inputs_a, outputs_a = SerializationSession.load_data(pkl_path_a, storage_a)
        inputs_b, outputs_b = SerializationSession.load_data(pkl_path_b, storage_b)
        _compare_lists(inputs_a['args'], inputs_b['args'], "Inputs.args")
        _compare_kwargs(inputs_a['kwargs'], inputs_b['kwargs'], "Inputs.kwargs")
        _compare_lists(outputs_a, outputs_b, "Outputs")


def ops_comp(dump_dir_a: str, dump_dir_b: str):
    """Compare two operator dump sessions."""
    records_a = _load_all_metadata(dump_dir_a)
    print(f"[LCS] Loading dump A: {len(records_a)} operators from {dump_dir_a}")
    records_b = _load_all_metadata(dump_dir_b)
    print(f"[LCS] Loading dump B: {len(records_b)} operators from {dump_dir_b}")
    matched_pairs = _find_lcs_matches(records_a, records_b)
    storage_a = os.path.join(dump_dir_a, 'storage')
    storage_b = os.path.join(dump_dir_b, 'storage')
    _compare_matched_pairs(records_a, records_b, matched_pairs, storage_a, storage_b)