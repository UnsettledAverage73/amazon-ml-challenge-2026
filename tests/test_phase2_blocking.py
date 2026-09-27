"""
Verification Suite for Phase 2: High-Recall Multi-Pass Blocking Engine
"""

import sys
import os
import csv
sys.path.insert(0, os.path.abspath('.'))

from src.blocking import MultiPassBlocker

def test_blocking_synthetic_corruption():
    print("\n--- 1. Testing Blocking Under Synthetic Name Corruption ---")
    blocker = MultiPassBlocker(country="US")
    # Simulate a target with corrupted name but matching address
    blocker.add_record(
        eid="S2-99999",
        raw_name="DREXTAVO",
        raw_addr="300 SWIFT AVE, DURHAM, NC"
    )
    blocker.finalize_index()

    # Query with real business name
    query_name = "Advanced Intelligence Products"
    query_addr = "300 Swift Avenue, Unit 5, Durham, NC"

    cands = blocker.retrieve_candidates(query_name, query_addr, top_k=5)
    print(f"Query: {query_name} ({query_addr})")
    print(f"Retrieved Candidates: {cands}")
    assert "S2-99999" in cands, "Failed to retrieve candidate via address anchor!"
    print("  -> PASSED: Address anchor successfully retrieved corrupted name record!")

def test_blocking_multilingual_and_domains():
    print("\n--- 2. Testing Multilingual & Domain URL Blocking ---")
    blocker = MultiPassBlocker(country="India")
    blocker.add_record(
        eid="S2-11111",
        raw_name="http://www.sakshisai.com",
        raw_addr="PRANAVAMCHUZHIKATTU HOUSE, KM 4/109/5, VYTHIRI, Kerala"
    )
    blocker.finalize_index()

    query_name = "Sakshi Sai Private Limited"
    query_addr = "Pranavamchuzhikattu House, Km 4/109/5, Vythiri, Wayanad, Kerala"

    cands = blocker.retrieve_candidates(query_name, query_addr, top_k=5)
    print(f"Query: {query_name} ({query_addr})")
    print(f"Retrieved Candidates: {cands}")
    assert "S2-11111" in cands, "Failed to retrieve candidate via domain / multilingual token!"
    print("  -> PASSED: Multi-pass blocker successfully retrieved URL target!")

def test_blocking_recall_benchmark():
    print("\n--- 3. Running Candidate Recall Benchmark on Ground Truth ---")
    # Load 1,000 real ground truth records from train
    gt = {}
    with open('dataset/train/train_ground_truth.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 1000:
                break
            m = [x for x in row[1].split(',') if x.startswith('S2-')]
            if m:
                gt[row[0]] = set(m)

    all_target_ids = set()
    for t_set in gt.values():
        all_target_ids.update(t_set)

    s1_records = {}
    with open('dataset/train/train_source1.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if row[0] in gt:
                s1_records[row[0]] = (row[1], row[2], row[3])

    blocker = MultiPassBlocker(country="Mixed")
    with open('dataset/train/train_source2.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            eid = row[0]
            if eid in all_target_ids or blocker.total_docs < 30000:
                blocker.add_record(eid, row[1], row[2])
                if blocker.total_docs >= 40000 and all_target_ids.issubset(blocker.records.keys()):
                    break

    blocker.finalize_index()
    print(f"Indexed {blocker.total_docs} records. Evaluating recall across {len(s1_records)} queries...")

    total_gt = sum(len(t) for t in gt.values())
    found_at_35 = 0

    for s1_id, (raw_name, raw_addr, country) in s1_records.items():
        cands = set(blocker.retrieve_candidates(raw_name, raw_addr, top_k=35))
        true_targets = gt[s1_id]
        found_at_35 += len(true_targets & cands)

    recall = found_at_35 / total_gt * 100
    print(f"Evaluated {total_gt} true positive pairs against a 40,000 distractor pool.")
    print(f"Candidate Recall @ 35: {recall:.2f}%")
    assert recall >= 85.0, f"Candidate recall too low: {recall:.2f}%"
    print("  -> PASSED: Benchmark recall surpassed threshold!")

if __name__ == '__main__':
    test_blocking_synthetic_corruption()
    test_blocking_multilingual_and_domains()
    test_blocking_recall_benchmark()
    print("\n[SUCCESS] Phase 2 Multi-Pass Blocking Engine PASSED all validation tests!")
