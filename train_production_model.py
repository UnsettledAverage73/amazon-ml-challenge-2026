"""
Production Model Training Script (Phase 5)
Trains LightGBM on diverse ground truth positive pairs and blocker-generated hard negatives.
Saves model to 'models/top_tier_lightgbm.txt' and token IDFs to 'models/top_tier_idf.json'.
"""

import os
import sys
import csv
import json
import time
import numpy as np
import lightgbm as lgb
from collections import defaultdict

from src.normalization import clean_name, clean_addr, extract_anchors
from src.blocking import MultiPassBlocker
from src.features import compute_pairwise_features, FEATURE_NAMES

def main():
    start_time = time.time()
    os.makedirs('models', exist_ok=True)
    print("=== Training Top-Tier Pairwise Model ===")

    # 1. Load Ground Truth sample (25,000 queries)
    print("Step 1: Loading ground truth...")
    gt = {}
    with open('dataset/train/train_ground_truth.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 25000:
                break
            m = [x for x in row[1].split(',') if x]
            if m:
                gt[row[0]] = m

    all_target_ids = set()
    for targets in gt.values():
        all_target_ids.update(targets)

    print(f"Loaded {len(gt)} S1 queries with {len(all_target_ids)} unique target IDs.")

    # 2. Load S1 data
    print("Step 2: Loading S1 records...")
    s1_data = {}
    with open('dataset/train/train_source1.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if row[0] in gt:
                s1_data[row[0]] = (row[1], row[2], row[3])

    # 3. Load Target Records from S2 and S3 into MultiPassBlocker
    print("Step 3: Indexing target records from S2 and S3...")
    blocker = MultiPassBlocker(country="All")
    target_data = {}

    for src_file in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        with open(src_file) as f:
            r = csv.reader(f, delimiter='\t')
            next(r)
            for row in r:
                eid = row[0]
                # Include all true positives + sample distractors
                if eid in all_target_ids or blocker.total_docs < 60000:
                    blocker.add_record(eid, row[1], row[2])
                    target_data[eid] = (row[1], row[2])

    blocker.finalize_index()
    print(f"Indexed {blocker.total_docs} target records in MultiPassBlocker.")

    # Save IDF dictionary
    with open('models/top_tier_idf.json', 'w') as f:
        json.dump(blocker.token_idf, f)
    print("Saved token IDF dictionary to models/top_tier_idf.json")

    # 4. Precompute profiles
    print("Step 4: Precomputing normalized profiles...")
    def make_profile(raw_n, raw_a):
        cn, comp, core, toks = clean_name(raw_n)
        ca = clean_addr(raw_a)
        door, pin, phone = extract_anchors(raw_a)
        ak, _ = blocker.extract_addr_keys(raw_a)
        return (cn, comp, core, ca, door, pin, phone, ak)

    s1_profiles = {k: make_profile(v[0], v[1]) for k, v in s1_data.items()}
    target_profiles = {k: make_profile(v[0], v[1]) for k, v in target_data.items()}

    # 5. Build Training Matrix
    print("Step 5: Generating honest training pairs (Positives + Hard Negatives)...")
    X = []
    y = []

    for s1_id, profile1 in s1_profiles.items():
        true_targets = set(gt[s1_id])
        # Positive pairs
        for cid in true_targets:
            if cid in target_profiles:
                feat = compute_pairwise_features(profile1, target_profiles[cid], blocker.token_idf)
                X.append(feat)
                y.append(1)

        # Blocker hard negatives
        raw_n, raw_a, _ = s1_data[s1_id]
        cands = blocker.retrieve_candidates(raw_n, raw_a, top_k=15)
        neg_added = 0
        for cid in cands:
            if cid not in true_targets and cid in target_profiles:
                feat = compute_pairwise_features(profile1, target_profiles[cid], blocker.token_idf)
                X.append(feat)
                y.append(0)
                neg_added += 1
                if neg_added >= 4:
                    break

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Generated {len(X)} training pairs. Positives: {int(np.sum(y))}, Negatives: {len(y) - int(np.sum(y))}")

    # 6. Train LightGBM model
    print("Step 6: Fitting LightGBM model...")
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 150,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 4
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    # Save model to disk
    model_path = 'models/top_tier_lightgbm.txt'
    model.booster_.save_model(model_path)
    print(f"Model successfully saved to {model_path}!")

    elapsed = time.time() - start_time
    print(f"Production model training completed in {elapsed:.1f}s.")

if __name__ == '__main__':
    main()
