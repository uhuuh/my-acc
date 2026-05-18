"""
Operator Dumps Comparison for PyTorch Operator Dump Tool.

Provides compare_operator_dumps function for comparing two dump sessions.
"""

import os
import pickle
from typing import List, Dict
from .comparison_utils import _lcs_length, _format_type_info, _compare_element


def _load_dumps(dump_dir: str) -> List[Dict]:
    """
    Load all dump files from directory.
    
    Args:
        dump_dir: Path to dump directory
    
    Returns:
        List of dump data sorted by sequence
    """
    dumps = []
    for filename in os.listdir(dump_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(dump_dir, filename)
            with open(filepath, 'rb') as f:
                dumps.append(pickle.load(f))
    
    dumps.sort(key=lambda x: x['sequence'])
    return dumps


def compare_operator_dumps(dump_dir_a: str, dump_dir_b: str):
    """
    Compare two operator dump sessions.
    
    Args:
        dump_dir_a: Path to first dump directory
        dump_dir_b: Path to second dump directory
    """
    # Phase 1: Load dumps
    dumps_a = _load_dumps(dump_dir_a)
    print(f"[LCS] Loading dump A: {len(dumps_a)} operators from {dump_dir_a}")
    dumps_b = _load_dumps(dump_dir_b)
    print(f"[LCS] Loading dump B: {len(dumps_b)} operators from {dump_dir_b}")
    
    # Build signatures
    print("[LCS] Building operator signatures...")
    sigs_a = [f"{d['filename']}::{d['opname']}" for d in dumps_a]
    sigs_b = [f"{d['filename']}::{d['opname']}" for d in dumps_b]
    
    # Find LCS
    print("[LCS] Finding longest common subsequence...")
    lcs_len, matched_pairs = _lcs_length(sigs_a, sigs_b)
    
    a_only = len(dumps_a) - lcs_len
    b_only = len(dumps_b) - lcs_len
    print(f"[LCS] Matched: {lcs_len} operators | A-only: {a_only} | B-only: {b_only}")
    
    # Log matches and skips
    matched_a_indices = set(idx_a for idx_a, idx_b in matched_pairs)
    matched_b_indices = set(idx_b for idx_a, idx_b in matched_pairs)
    
    for i, dump in enumerate(dumps_a):
        if i in matched_a_indices:
            # Find the matching pair
            for idx_a, idx_b in matched_pairs:
                if idx_a == i:
                    match_idx = idx_b
                    break
            print(f"[MATCH] A:{dump['sequence']:04d}_{dump['filename']}_{dump['opname']} <-> B:{dumps_b[match_idx]['sequence']:04d}_{dumps_b[match_idx]['filename']}_{dumps_b[match_idx]['opname']}")
        else:
            print(f"[SKIP] A:{dump['sequence']:04d}_{dump['filename']}_{dump['opname']} (no match in B)")
    
    for j, dump in enumerate(dumps_b):
        if j not in matched_b_indices:
            print(f"[SKIP] B:{dump['sequence']:04d}_{dump['filename']}_{dump['opname']} (no match in A)")
    
    # Phase 2: Detailed comparison
    print(f"[COMPARE] Starting detailed comparison of {lcs_len} matched pairs...")
    
    # Statistics counters
    stats = {
        'exact_match': 0,
        'precision_diff': 0,
        'dtype_mismatch': 0,
        'shape_mismatch': 0,
        'input_count_mismatch': 0,
        'output_count_mismatch': 0
    }
    
    for idx_a, idx_b in matched_pairs:
        dump_a = dumps_a[idx_a]
        dump_b = dumps_b[idx_b]
        
        op_id = f"{dump_a['sequence']:04d}_{dump_a['filename']}_{dump_a['opname']}"
        print(f"[COMPARE] {op_id}:")
        
        # Compare inputs
        inputs_a = dump_a['inputs']
        inputs_b = dump_b['inputs']
        
        max_inputs = max(len(inputs_a), len(inputs_b))
        if len(inputs_a) != len(inputs_b):
            stats['input_count_mismatch'] += 1
        
        for i in range(max_inputs):
            if i >= len(inputs_a):
                b_info = _format_type_info(inputs_b[i])
                print(f"  Inputs[{i}] | <missing> | {b_info} | missing_in_A")
            elif i >= len(inputs_b):
                a_info = _format_type_info(inputs_a[i])
                print(f"  Inputs[{i}] | {a_info} | <missing> | missing_in_B")
            else:
                log, element_stats = _compare_element(inputs_a[i], inputs_b[i])
                print(f"  Inputs[{i}] | {log}")
                # Update stats
                if element_stats['exact_match']:
                    stats['exact_match'] += 1
                if element_stats['precision_diff']:
                    stats['precision_diff'] += 1
                if element_stats['dtype_mismatch']:
                    stats['dtype_mismatch'] += 1
                if element_stats['shape_mismatch']:
                    stats['shape_mismatch'] += 1
        
        # Compare outputs
        outputs_a = dump_a['outputs']
        outputs_b = dump_b['outputs']
        
        max_outputs = max(len(outputs_a), len(outputs_b))
        if len(outputs_a) != len(outputs_b):
            stats['output_count_mismatch'] += 1
        
        for i in range(max_outputs):
            if i >= len(outputs_a):
                b_info = _format_type_info(outputs_b[i])
                print(f"  Outputs[{i}] | <missing> | {b_info} | missing_in_A")
            elif i >= len(outputs_b):
                a_info = _format_type_info(outputs_a[i])
                print(f"  Outputs[{i}] | {a_info} | <missing> | missing_in_B")
            else:
                log, element_stats = _compare_element(outputs_a[i], outputs_b[i])
                print(f"  Outputs[{i}] | {log}")
                # Update stats
                if element_stats['exact_match']:
                    stats['exact_match'] += 1
                if element_stats['precision_diff']:
                    stats['precision_diff'] += 1
                if element_stats['dtype_mismatch']:
                    stats['dtype_mismatch'] += 1
                if element_stats['shape_mismatch']:
                    stats['shape_mismatch'] += 1
    
    # Summary
    print(f"[SUMMARY] Total: {lcs_len} | Exact match: {stats['exact_match']} | Precision diff: {stats['precision_diff']} | Dtype mismatch: {stats['dtype_mismatch']} | Shape mismatch: {stats['shape_mismatch']} | Input count mismatch: {stats['input_count_mismatch']} | Output count mismatch: {stats['output_count_mismatch']}")