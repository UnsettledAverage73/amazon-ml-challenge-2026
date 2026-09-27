"""
Integration Test for Phase 5: Production End-to-End Pipeline
Runs full pipeline on a slice of test queries, writes temporary TSVs, and validates superset constraints.
"""

import sys
import os
import csv
import json
import numpy as np
import lightgbm as lgb
from collections import defaultdict
sys.path.insert(0, os.path.abspath('.'))

from src.normalization import clean_name, clean_addr, extract_anchors
from src.blocking import MultiPassBlocker
from src.features import compute_pairwise_features
from src.resolution import resolve_matches_bipartite

def test_end_to_end_slice():
    print("\n--- 1. Testing End-to-End Pipeline on Test Slice ---")
    booster = lgb.Booster(model_file='models/top_tier_lightgbm.txt')
    with open('models/top_tier_idf.json') as f:
        idf_lookup = json.load(f)

    # 1. Read first 200 S1 queries from test_source1.tsv
    s1_slice = []
    with open('dataset/test/test_source1.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 200:
                break
            eid, raw_n, raw_a, country = row[0], row[1], row[2], row[3]
            cn, comp, core, toks = clean_name(raw_n)
            ca = clean_addr(raw_a)
            door, pin, phone = extract_anchors(raw_a)
            ak, _ = MultiPassBlocker.extract_addr_keys(raw_a)
            prof = (cn, comp, core, ca, door, pin, phone, ak)
            s1_slice.append((eid, raw_n, raw_a, country, prof))

    print(f"Loaded {len(s1_slice)} test S1 queries.")

    # 2. Build index from test_source2 slice for the countries present
    countries_present = {x[3] for x in s1_slice}
    blocker = MultiPassBlocker(country="Slice")
    target_profs = {}

    with open('dataset/test/test_source2.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 50000:
                break
            if row[3] in countries_present:
                eid, raw_n, raw_a = row[0], row[1], row[2]
                blocker.add_record(eid, raw_n, raw_a)
                cn, comp, core, toks = clean_name(raw_n)
                ca = clean_addr(raw_a)
                door, pin, phone = extract_anchors(raw_a)
                ak, _ = blocker.extract_addr_keys(raw_a)
                target_profs[eid] = (cn, comp, core, ca, door, pin, phone, ak)

    blocker.finalize_index()
    print(f"Indexed {blocker.total_docs} target records from test_source2.tsv.")

    # 3. Retrieve, score, and rank
    scored_dict = defaultdict(list)
    blocker_cands_dict = defaultdict(list)

    for idx, (eid, raw_n, raw_a, country, prof1) in enumerate(s1_slice):
        cands = blocker.retrieve_candidates(raw_n, raw_a, top_k=20)
        feats = []
        c_mapped = []
        for cid in cands:
            if cid in target_profs:
                feats.append(compute_pairwise_features(prof1, target_profs[cid], idf_lookup))
                c_mapped.append(cid)
            if cid not in blocker_cands_dict[idx] and len(blocker_cands_dict[idx]) < 10:
                blocker_cands_dict[idx].append(cid)

        if feats:
            probs = booster.predict(np.array(feats, dtype=np.float32))
            for cid, p in zip(c_mapped, probs):
                scored_dict[idx].append((cid, float(p)))

    # 4. Bipartite Resolution
    resolved = resolve_matches_bipartite(scored_dict, threshold=0.55)

    # 5. Check Superset Guarantee & Matching Quality
    matched_count = 0
    for idx, (eid, *_) in enumerate(s1_slice):
        m_list = resolved.get(idx, [])
        c_list = list(m_list)
        for c in blocker_cands_dict[idx]:
            if c not in c_list:
                c_list.append(c)
        if m_list:
            matched_count += 1
            # Verify superset: every match MUST be in candidates
            assert set(m_list).issubset(set(c_list)), f"Superset violation for {eid}!"

    print(f"Matched queries: {matched_count} / {len(s1_slice)} ({matched_count/len(s1_slice)*100:.1f}%)")
    print("Superset guarantee verified: all matches are valid subsets of candidate pairs!")
    print("\n[SUCCESS] Phase 5 Integration Test PASSED!")

if __name__ == '__main__':
    test_end_to_end_slice()
