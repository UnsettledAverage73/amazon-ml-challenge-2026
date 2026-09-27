"""
Cardinality Resolution & Macro F0.5 Optimization Engine (Phase 4)

Provides:
- Source-aware bipartite resolution: allows 1-to-many S1 matches (matching 80.48% ground truth distribution)
  while ensuring each physical S2/S3 record is claimed at most once globally.
- Exact competition Macro F0.5 evaluation function.
- Empirical threshold optimization over honest holdout sets.
"""

from collections import defaultdict
import numpy as np

def compute_macro_f05(y_true_dict: dict, y_pred_dict: dict) -> float:
    """
    Computes the official competition Macro F0.5 metric:
    - If true is empty and pred is empty: score = 1.0 (True singleton)
    - If true is non-empty and pred is empty: score = 0.0 (False singleton)
    - If true is empty and pred is non-empty: score = 0.0 (False positive on singleton)
    - Otherwise: F0.5 = (1.25 * TP) / (1.25 * TP + 0.25 * FP + FN)
    """
    total_score = 0.0
    N = len(y_true_dict)
    if N == 0:
        return 0.0

    for s1_id, y_true in y_true_dict.items():
        true_set = set(y_true)
        pred_set = set(y_pred_dict.get(s1_id, []))

        if len(true_set) == 0 and len(pred_set) == 0:
            total_score += 1.0
        elif len(true_set) == 0 or len(pred_set) == 0:
            total_score += 0.0
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            if tp == 0:
                total_score += 0.0
            else:
                denom = 1.25 * tp + 0.25 * fp + fn
                total_score += (1.25 * tp) / denom

    return total_score / N

def resolve_matches_bipartite(
    scored_candidates_by_s1: dict,
    threshold: float = 0.42,
    max_matches_per_s1: int = 6
) -> dict:
    """
    Resolves final matches using global target ownership and per-S1 thresholding:
    1. Each target record (cid) can only belong to at most ONE canonical S1 entity.
       In case of competition, the S1 with the higher probability wins the target.
    2. Each S1 can claim multiple target records if prob >= threshold.
    """
    # Step 1: Assign each target to its highest-scoring S1
    target_best_s1 = {} # cid -> (prob, s1_id)
    for s1_id, cands in scored_candidates_by_s1.items():
        for cid, prob in cands:
            if prob >= threshold:
                if cid not in target_best_s1 or prob > target_best_s1[cid][0]:
                    target_best_s1[cid] = (prob, s1_id)

    # Step 2: Invert to s1_id -> [matched_target_ids]
    s1_matches = defaultdict(list)
    for cid, (prob, s1_id) in target_best_s1.items():
        s1_matches[s1_id].append((cid, prob))

    # Step 3: Sort each S1's matches by confidence descending and cap
    final_preds = {}
    for s1_id in scored_candidates_by_s1.keys():
        if s1_id in s1_matches:
            sorted_m = sorted(s1_matches[s1_id], key=lambda x: x[1], reverse=True)
            final_preds[s1_id] = [cid for cid, _ in sorted_m[:max_matches_per_s1]]
        else:
            final_preds[s1_id] = []

    return final_preds

def optimize_threshold(
    y_true_dict: dict,
    scored_candidates_by_s1: dict,
    start: float = 0.20,
    end: float = 0.70,
    step: float = 0.02
):
    """
    Sweeps the decision threshold over an honest validation set to find the global
    optimal threshold tau* that maximizes Macro F0.5.
    """
    best_th = start
    best_score = -1.0
    curve = []

    th_values = np.arange(start, end + 1e-5, step)
    for th in th_values:
        preds = resolve_matches_bipartite(scored_candidates_by_s1, threshold=th)
        score = compute_macro_f05(y_true_dict, preds)
        curve.append((round(float(th), 3), round(float(score), 4)))
        if score > best_score:
            best_score = score
            best_th = float(th)

    return best_th, best_score, curve
