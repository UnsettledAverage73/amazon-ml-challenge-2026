"""
Verification Suite for Phase 4: Cardinality Resolution & Macro F0.5 Optimization
"""

import sys
import os
import csv
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

from src.normalization import clean_name, clean_addr, extract_anchors
from src.blocking import MultiPassBlocker
from src.features import compute_pairwise_features
from src.model import train_reranker
from src.resolution import compute_macro_f05, resolve_matches_bipartite, optimize_threshold

def test_macro_f05_mathematics():
    print("\n--- 1. Testing Macro F0.5 Edge Cases ---")
    # Case A: True singleton
    score_a = compute_macro_f05({'S1-1': []}, {'S1-1': []})
    assert score_a == 1.0, f"Expected 1.0 for true singleton, got {score_a}"

    # Case B: False singleton (Devastating 0.0)
    score_b = compute_macro_f05({'S1-1': ['S2-A']}, {'S1-1': []})
    assert score_b == 0.0, f"Expected 0.0 for false singleton, got {score_b}"

    # Case C: False positive on singleton
    score_c = compute_macro_f05({'S1-1': []}, {'S1-1': ['S2-A']})
    assert score_c == 0.0, f"Expected 0.0 for FP on singleton, got {score_c}"

    # Case D: Exact match
    score_d = compute_macro_f05({'S1-1': ['S2-A', 'S3-B']}, {'S1-1': ['S2-A', 'S3-B']})
    assert score_d == 1.0, f"Expected 1.0 for perfect match, got {score_d}"

    # Case E: 1 TP, 1 FP (Under F0.5, TP=1, FP=1, FN=0 -> 1.25 / (1.25 + 0.25) = 1.25 / 1.50 = 0.833)
    score_e = compute_macro_f05({'S1-1': ['S2-A']}, {'S1-1': ['S2-A', 'S2-B']})
    assert abs(score_e - (1.25 / 1.50)) < 1e-4, f"Unexpected score: {score_e}"
    print(f"  -> All Macro F0.5 mathematical properties PASSED! (1 TP, 1 FP = {score_e:.3f})")

def test_target_collision_resolution():
    print("\n--- 2. Testing Target Conflict Resolution ---")
    # S1-A scored S2-T1 at 0.70, S2-T2 at 0.90
    # S1-B scored S2-T1 at 0.95
    # Result: S2-T1 must go to S1-B (0.95 > 0.70). S1-A gets S2-T2.
    scored_dict = {
        'S1-A': [('S2-T1', 0.70), ('S2-T2', 0.90)],
        'S1-B': [('S2-T1', 0.95)]
    }
    preds = resolve_matches_bipartite(scored_dict, threshold=0.50)
    print("Resolved Predictions:", preds)
    assert preds['S1-B'] == ['S2-T1'], "S1-B should have won S2-T1!"
    assert preds['S1-A'] == ['S2-T2'], "S1-A should keep S2-T2 without S2-T1!"
    print("  -> PASSED: Global target ownership correctly resolved without duplicate claims!")

def test_end_to_end_threshold_optimization():
    print("\n--- 3. Running End-To-End Threshold Optimization on Validation Set ---")
    # Load 1200 ground truth queries
    gt = {}
    with open('dataset/train/train_ground_truth.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 1200:
                break
            m = [x for x in row[1].split(',') if x.startswith('S2-')]
            gt[row[0]] = m

    all_target_ids = set()
    for t_list in gt.values():
        all_target_ids.update(t_list)

    s1_data = {}
    with open('dataset/train/train_source1.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if row[0] in gt:
                s1_data[row[0]] = (row[1], row[2], row[3])

    blocker = MultiPassBlocker(country="Mixed")
    s2_data = {}
    with open('dataset/train/train_source2.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            eid = row[0]
            if eid in all_target_ids or blocker.total_docs < 20000:
                blocker.add_record(eid, row[1], row[2])
                s2_data[eid] = (row[1], row[2])
                if blocker.total_docs >= 25000 and all_target_ids.issubset(blocker.records.keys()):
                    break

    blocker.finalize_index()

    # Pre-parse profiles
    def make_profile(raw_name, raw_addr):
        cn, comp, core, toks = clean_name(raw_name)
        ca = clean_addr(raw_addr)
        door, pin, phone = extract_anchors(raw_addr)
        ak, _ = blocker.extract_addr_keys(raw_addr)
        return (cn, comp, core, ca, door, pin, phone, ak)

    s1_profiles = {k: make_profile(v[0], v[1]) for k, v in s1_data.items()}
    s2_profiles = {k: make_profile(v[0], v[1]) for k, v in s2_data.items()}

    # Honest split: 70% train model, 30% holdout for threshold tuning
    s1_keys = list(s1_profiles.keys())
    split_idx = int(0.7 * len(s1_keys))
    train_keys = set(s1_keys[:split_idx])
    val_keys = set(s1_keys[split_idx:])

    # Generate train pairs
    X_train, y_train = [], []
    for s1_id in train_keys:
        profile1 = s1_profiles[s1_id]
        true_set = set(gt[s1_id])
        for cid in true_set:
            if cid in s2_profiles:
                X_train.append(compute_pairwise_features(profile1, s2_profiles[cid], blocker.token_idf))
                y_train.append(1)
        # Negatives
        raw_n, raw_a, _ = s1_data[s1_id]
        cands = blocker.retrieve_candidates(raw_n, raw_a, top_k=10)
        neg_c = 0
        for cid in cands:
            if cid not in true_set and cid in s2_profiles:
                X_train.append(compute_pairwise_features(profile1, s2_profiles[cid], blocker.token_idf))
                y_train.append(0)
                neg_c += 1
                if neg_c >= 5:
                    break

    print(f"Training LightGBM on {len(X_train)} training pairs...")
    model = train_reranker(np.array(X_train), np.array(y_train), n_estimators=80)

    # Score validation holdout queries
    print(f"Scoring {len(val_keys)} holdout validation queries...")
    val_scored = {}
    val_gt = {k: gt[k] for k in val_keys}

    for s1_id in val_keys:
        profile1 = s1_profiles[s1_id]
        raw_n, raw_a, _ = s1_data[s1_id]
        cands = blocker.retrieve_candidates(raw_n, raw_a, top_k=25)
        if not cands:
            val_scored[s1_id] = []
            continue

        feats = [compute_pairwise_features(profile1, s2_profiles[cid], blocker.token_idf) for cid in cands]
        probs = model.predict_proba(np.array(feats))[:, 1]
        val_scored[s1_id] = [(cands[i], float(probs[i])) for i in range(len(cands))]

    best_th, best_score, curve = optimize_threshold(val_gt, val_scored, start=0.25, end=0.65, step=0.05)
    print(f"\n--- OPTIMAL THRESHOLD DISCOVERY ---")
    print(f"Best Threshold tau*: {best_th:.2f}")
    print(f"Peak Validation Macro F0.5: {best_score * 100:.2f}%")
    print("Threshold Curve:")
    for th, sc in curve:
        bar = '#' * int(sc * 40)
        print(f"  tau = {th:.2f} | Macro F0.5 = {sc * 100:5.2f}% | {bar}")

    assert best_score >= 0.85, f"Validation score too low: {best_score}"
    print("\n[SUCCESS] Phase 4 Cardinality Resolution & Optimization PASSED!")

if __name__ == '__main__':
    test_macro_f05_mathematics()
    test_target_collision_resolution()
    test_end_to_end_threshold_optimization()
