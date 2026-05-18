"""
Operator Dumps Comparison for PyTorch Operator Dump Tool.

Provides ops_comp function for comparing two dump sessions.
"""

import os
import pickle
from typing import List, Dict
from .comparison_utils import (
    _lcs_length, 
    create_comparator, 
    MissingInAComparator, 
    MissingInBComparator
)


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


def ops_comp(dump_dir_a: str, dump_dir_b: str):
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
    
    # Build lookup map for efficiency
    matched_map_a = {idx_a: idx_b for idx_a, idx_b in matched_pairs}
    matched_map_b = {idx_b: idx_a for idx_a, idx_b in matched_pairs}
    
    # Log matches and skips
    for i, dump in enumerate(dumps_a):
        if i in matched_map_a:
            match_idx = matched_map_a[i]
            print(f"[MATCH] A:{dump['sequence']:06d}_{dump['filename']}_{dump['opname']} <-> B:{dumps_b[match_idx]['sequence']:06d}_{dumps_b[match_idx]['filename']}_{dumps_b[match_idx]['opname']}")
        else:
            print(f"[SKIP] A:{dump['sequence']:06d}_{dump['filename']}_{dump['opname']} (no match in B)")
    
    for j, dump in enumerate(dumps_b):
        if j not in matched_map_b:
            print(f"[SKIP] B:{dump['sequence']:06d}_{dump['filename']}_{dump['opname']} (no match in A)")
    
    # Phase 2: Detailed comparison
    print(f"[COMPARE] Starting detailed comparison of {lcs_len} matched pairs...")
    
    for idx_a, idx_b in matched_pairs:
        dump_a = dumps_a[idx_a]
        dump_b = dumps_b[idx_b]
        
        op_id = f"{dump_a['sequence']:06d}_{dump_a['filename']}_{dump_a['opname']}"
        print(f"[COMPARE] {op_id}:")
        
        # Compare inputs
        inputs_a = dump_a['inputs']
        inputs_b = dump_b['inputs']
        
        max_inputs = max(len(inputs_a), len(inputs_b))
        
        for i in range(max_inputs):
            if i >= len(inputs_a):
                comparator = MissingInAComparator(inputs_b[i])
            elif i >= len(inputs_b):
                comparator = MissingInBComparator(inputs_a[i])
            else:
                comparator = create_comparator(inputs_a[i], inputs_b[i])
            
            left_info, right_info = comparator.get_type_info()
            result = comparator.compare()
            print(f"  Inputs[{i}] | {left_info} | {right_info} | {result['log']}")
        
        # Compare outputs (if exists in dumps)
        if 'outputs' in dump_a and 'outputs' in dump_b:
            outputs_a = dump_a['outputs']
            outputs_b = dump_b['outputs']
            
            max_outputs = max(len(outputs_a), len(outputs_b))
            
            for i in range(max_outputs):
                if i >= len(outputs_a):
                    comparator = MissingInAComparator(outputs_b[i])
                elif i >= len(outputs_b):
                    comparator = MissingInBComparator(outputs_a[i])
                else:
                    comparator = create_comparator(outputs_a[i], outputs_b[i])
                
                left_info, right_info = comparator.get_type_info()
                result = comparator.compare()
                print(f"  Outputs[{i}] | {left_info} | {right_info} | {result['log']}")
    
    # No summary (removed per requirement)