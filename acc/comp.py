"""
Operator Dumps Comparison for PyTorch Operator Dump Tool.

Provides ops_comp function for comparing two dump sessions.
"""

import json
import os
import pickle
from typing import List, Tuple
from .serialization import OperatorDump
from .formatting import (
    format_signature,
    format_display_key,
    format_dump_filename,
    format_comparison_result
)
from .comparators import (
    create_comparator,
    MissingInAComparator,
    MissingInBComparator
)


def _lcs_length(a: List[str], b: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Find longest common subsequence and return matched pairs.

    Args:
        a: First sequence of signatures
        b: Second sequence of signatures

    Returns:
        Tuple of (lcs_length, matched_pairs as list of (idx_a, idx_b))
    """
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


def _load_dumps(dump_dir: str) -> List[OperatorDump]:
    """
    Load all dump files from directory.
    Each dump has .json (metadata) and .pkl (input data list).
    PKL structure: [input1, input2, ..., {'outputs': [output1, ...]}]
    """
    dumps = []

    for filename in os.listdir(dump_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(dump_dir, filename)
            pkl_path = json_path.replace('.json', '.pkl')

            with open(json_path, 'r') as f:
                metadata = json.load(f)

            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)

            inputs = pkl_data[:-1]
            outputs = pkl_data[-1]['outputs']

            dumps.append(OperatorDump(
                sequence=metadata['sequence'],
                filepath=metadata.get('filepath', ''),
                filename=metadata['filename'],
                function=metadata['function'],
                lineno=metadata.get('lineno', 0),
                opname=metadata['opname'],
                call_stack=metadata.get('call_stack', []),
                inputs=inputs,
                outputs=outputs
            ))

    dumps.sort(key=lambda x: x.sequence)
    return dumps


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
        result = comparator.compare()
        log = format_comparison_result(result)
        print(f"  {label}[{i}] | {left_info} | {right_info} | {log}")


def _find_lcs_matches(dumps_a: List[OperatorDump], dumps_b: List[OperatorDump]) -> List[Tuple[int, int]]:
    """
    Find LCS matches between two dump lists and print matching info.

    Args:
        dumps_a: First dump list
        dumps_b: Second dump list

    Returns:
        List of matched pairs (idx_a, idx_b)
    """
    sigs_a = [format_signature(d) for d in dumps_a]
    sigs_b = [format_signature(d) for d in dumps_b]

    lcs_len, matched_pairs = _lcs_length(sigs_a, sigs_b)

    a_only = len(dumps_a) - lcs_len
    b_only = len(dumps_b) - lcs_len
    print(f"[LCS] Matched: {lcs_len} operators | A-only: {a_only} | B-only: {b_only}")

    prev_a, prev_b = 0, 0
    for idx_a, idx_b in matched_pairs:
        # 不匹配部分：先打印 A 信息，再打印 B 信息
        for i in range(prev_a, idx_a):
            key_a = format_display_key(dumps_a[i])
            print(f"[SKIP] A[{i}] {key_a} <-> <empty>")

        for j in range(prev_b, idx_b):
            key_b = format_display_key(dumps_b[j])
            print(f"[SKIP] <empty> <-> B[{j}] {key_b}")

        # 配对部分：打印配对信息
        key_a = format_display_key(dumps_a[idx_a])
        key_b = format_display_key(dumps_b[idx_b])
        print(f"[MATCH] A[{idx_a}] {key_a} <-> B[{idx_b}] {key_b}")

        prev_a = idx_a + 1
        prev_b = idx_b + 1

    # 剩余不匹配部分
    for i in range(prev_a, len(dumps_a)):
        key_a = format_display_key(dumps_a[i])
        print(f"[SKIP] A[{i}] {key_a} <-> <empty>")

    for j in range(prev_b, len(dumps_b)):
        key_b = format_display_key(dumps_b[j])
        print(f"[SKIP] <empty> <-> B[{j}] {key_b}")

    return matched_pairs


def _compare_matched_pairs(dumps_a: List[OperatorDump], dumps_b: List[OperatorDump], matched_pairs: List[Tuple[int, int]]):
    """
    Compare matched dump pairs and print detailed results.

    Args:
        dumps_a: First dump list
        dumps_b: Second dump list
        matched_pairs: List of matched index pairs
    """
    print(f"[COMPARE] Starting detailed comparison of {len(matched_pairs)} matched pairs...")

    for idx_a, idx_b in matched_pairs:
        dump_a = dumps_a[idx_a]
        dump_b = dumps_b[idx_b]

        filename_a = format_dump_filename(dump_a)
        filename_b = format_dump_filename(dump_b)
        print(f"{filename_a} <-> {filename_b}")

        _compare_lists(dump_a.inputs, dump_b.inputs, "Inputs")
        _compare_lists(dump_a.outputs, dump_b.outputs, "Outputs")


def ops_comp(dump_dir_a: str, dump_dir_b: str):
    """
    Compare two operator dump sessions.

    Args:
        dump_dir_a: Path to first dump directory
        dump_dir_b: Path to second dump directory
    """
    dumps_a = _load_dumps(dump_dir_a)
    print(f"[LCS] Loading dump A: {len(dumps_a)} operators from {dump_dir_a}")
    dumps_b = _load_dumps(dump_dir_b)
    print(f"[LCS] Loading dump B: {len(dumps_b)} operators from {dump_dir_b}")

    matched_pairs = _find_lcs_matches(dumps_a, dumps_b)

    _compare_matched_pairs(dumps_a, dumps_b, matched_pairs)