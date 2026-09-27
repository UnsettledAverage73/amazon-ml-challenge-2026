"""
Verification Suite for Phase 3: High-Precision Feature Engineering & Pairwise Reranker
"""

import sys
import os
import csv
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

from src.normalization import clean_name, clean_addr, extract_anchors
from src.blocking import MultiPassBlocker
from src.features import compute_pairwise_features, FEATURE_NAMES
from src.model import train_reranker, get_feature_importances

def test_feature_engineering_and_model():
    print("\n--- 1. Loading Training Ground Truth & Candidates ---")
    # Load 1500 ground truth queries
    gt = {}
    with open('dataset/train/train_ground_truth.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 1500:
                break
            m = [x for x in row[1].split(',') if x.startswith('S2-')]
            if m:
                gt[row[0]] = set(m)

    all_target_ids = set()
    for t_set in gt.values():
        all_target_ids.update(t_set)

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
            if eid in all_target_ids or blocker.total_docs < 25000:
                blocker.add_record(eid, row[1], row[2])
                s2_data[eid] = (row[1], row[2])
                if blocker.total_docs >= 30000 and all_target_ids.issubset(blocker.records.keys()):
                    break

    blocker.finalize_index()
    print(f"Indexed {blocker.total_docs} target records. S1 queries: {len(s1_data)}")

    # Pre-parse profiles
    def make_profile(raw_name, raw_addr):
        cn, comp, core, toks = clean_name(raw_name)
        ca = clean_addr(raw_addr)
        door, pin, phone = extract_anchors(raw_addr)
        ak, _ = blocker.extract_addr_keys(raw_addr)
        return (cn, comp, core, ca, door, pin, phone, ak)

    s1_profiles = {k: make_profile(v[0], v[1]) for k, v in s1_data.items()}
    s2_profiles = {k: make_profile(v[0], v[1]) for k, v in s2_data.items()}

    print("\n--- 2. Constructing Honest Training Dataset (Positives + Hard Negatives) ---")
    X = []
    y = []

    for s1_id, profile1 in s1_profiles.items():
        true_set = gt[s1_id]
        # Add True Positives
        for target_id in true_set:
            if target_id in s2_profiles:
                feat = compute_pairwise_features(profile1, s2_profiles[target_id], blocker.token_idf)
                X.append(feat)
                y.append(1)

        # Retrieve candidates from Blocker for Hard Negatives
        raw_n, raw_a, _ = s1_data[s1_id]
        cands = blocker.retrieve_candidates(raw_n, raw_a, top_k=15)
        neg_count = 0
        for cid in cands:
            if cid not in true_set and cid in s2_profiles:
                feat = compute_pairwise_features(profile1, s2_profiles[cid], blocker.token_idf)
                X.append(feat)
                y.append(0)
                neg_count += 1
                if neg_count >= 5: # 5 hard negatives per query
                    break

    X = np.array(X)
    y = np.array(y)
    print(f"Dataset generated: {len(X)} pairs. Positive pairs: {sum(y)}, Negative pairs: {len(y) - sum(y)}")

    # Train / Validation Split (80% train, 20% test queries)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print("\n--- 3. Training LightGBM Pairwise Reranker ---")
    model = train_reranker(X_train, y_train, X_val, y_val, n_estimators=100)

    val_probs = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, val_probs)
    print(f"\nFinal Out-Of-Fold AUC-ROC: {auc:.4f}")

    print("\n--- 4. Feature Importance Breakdown ---")
    importances = get_feature_importances(model)
    for feat_name, imp in importances[:10]:
        print(f"  {feat_name:24s} | Importance: {imp}")

    assert auc >= 0.92, f"Model AUC {auc} below required threshold 0.92!"
    print("\n[SUCCESS] Phase 3 Feature Engineering & Reranker PASSED all validation tests!")

if __name__ == '__main__':
    test_feature_engineering_and_model()
