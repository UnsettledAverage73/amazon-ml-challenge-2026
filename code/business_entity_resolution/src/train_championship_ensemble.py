import os
import sys
import re
import gc
import json
import time
import math
import random
import unicodedata
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from rapidfuzz import fuzz, distance

# ---------------------------------------------------------
# 1. UNICODE NORMALIZATION & ADVANCED ANCHORS
# ---------------------------------------------------------
DOMAINS = re.compile(r'(\.com|\.in|\.org|\.net|\.co|\.fr|\.io|\.biz|\.info|www\.)', re.IGNORECASE)
RE_PUNCT = re.compile(r'[^a-zA-Z0-9\s]')
RE_DOOR = re.compile(r'\b([A-Za-z]{0,3})[-/]?0*(\d{1,5})([A-Za-z]?)\b')
RE_PIN = re.compile(r'\b(\d{5,6})\b')
RE_PHONE = re.compile(r'\b\d{7,10}\b')
RE_CEDEX = re.compile(r'\bcedex\s*\d*\b', re.IGNORECASE)

LEGAL_SUFFIXES = {
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise', 'industries', 'group',
    'services', 'solutions', 'technologies', 'holdings', 'assoc', 'associates',
    'plc', 'bv', 'gmbh', 'sa', 'sarl', 'sas', 'eurl', 'sci', 'snc', 'gie', 'sca', 'scs'
}

GENERIC_TERMS = {
    'store', 'stores', 'shop', 'mart', 'market', 'supermarket', 'agency', 'agencies',
    'traders', 'trading', 'hotel', 'restaurant', 'cafe', 'bazaar', 'jewellers',
    'jewellery', 'textiles', 'pharmacy', 'chemist', 'auto', 'garage', 'consultants',
    'consultancy', 'logistics', 'transports', 'centre', 'center', 'foods', 'retail',
    'wholesale', 'supply', 'supplies', 'commercial'
}

def strip_accents(text: str) -> str:
    if not text:
        return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def clean_name(text: str):
    if not text:
        return '', '', set(), []
    text = strip_accents(text)
    text = DOMAINS.sub(' ', text)
    text = RE_PUNCT.sub(' ', text.lower())
    tokens = [w for w in text.split() if w not in LEGAL_SUFFIXES and len(w) > 1]
    cleaned = ' '.join(tokens)
    compressed = ''.join(tokens)
    core = {w for w in tokens if w not in GENERIC_TERMS}
    return cleaned, compressed, core, tokens

def clean_addr(text: str):
    if not text:
        return ''
    text = strip_accents(text)
    text = RE_CEDEX.sub(' ', text.lower()) # Separate CEDEX
    return ' '.join(RE_PUNCT.sub(' ', text).split())

def extract_anchors(raw_addr: str):
    if not raw_addr:
        return '', '', ''
    text = RE_CEDEX.sub(' ', raw_addr)
    pins = RE_PIN.findall(text)
    pin = pins[0] if pins else ''
    phones = RE_PHONE.findall(text)
    phone = phones[0] if phones else ''
    door = ''
    doors = RE_DOOR.findall(text)
    if doors:
        prefix, num, suffix = doors[0]
        door = f"{prefix.lower()}-{num}{suffix.lower()}" if prefix else f"{num}{suffix.lower()}"
    return door, pin, phone

# Character 3-Gram Jaccard
def char_3gram_jaccard(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if len(s1) < 3 or len(s2) < 3:
        return 1.0 if s1 == s2 else 0.0
    g1 = {s1[i:i+3] for i in range(len(s1)-2)}
    g2 = {s2[i:i+3] for i in range(len(s2)-2)}
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return inter / union if union > 0 else 0.0

# ---------------------------------------------------------
# 2. ADVANCED 24-FEATURE EXTRACTOR
# ---------------------------------------------------------
FEATURE_NAMES = [
    'name_jw',
    'name_tset',
    'name_tsort',
    'name_lev',
    'name_len_diff',
    'name_containment',
    'name_3gram_jaccard',
    'shared_core_count',
    'shared_token_idf_sum',
    'max_shared_token_idf',
    'comp_exact',
    'has_addr_s1',
    'has_addr_tgt',
    'addr_tset',
    'addr_lev',
    'addr_containment',
    'pin_status',
    'pin_prefix_status',
    'door_status',
    'shared_phone',
    'is_source2',
    'both_have_address',
    'interaction_name_addr',
    'interaction_door_name'
]

# Load IDF Vocabulary
IDF_LOOKUP = {}
DEFAULT_IDF = 12.0
if os.path.exists('models/token_idf.json'):
    with open('models/token_idf.json') as f:
        data = json.load(f)
        IDF_LOOKUP = data.get('idfs', {})
        DEFAULT_IDF = data.get('default_idf', 12.0)

def extract_championship_features(s1_tuple, tgt_tuple, is_s2_val):
    # s1_tuple: (cn, comp, core_set, tokens_list, ca, door, pin, phone)
    s1_cn, s1_comp, s1_core, s1_toks, s1_ca, s1_door, s1_pin, s1_phone = s1_tuple
    tgt_cn, tgt_comp, tgt_core, tgt_toks, tgt_ca, tgt_door, tgt_pin, tgt_phone = tgt_tuple

    # 1. Name Features
    if s1_cn and tgt_cn:
        name_jw = distance.JaroWinkler.similarity(s1_cn, tgt_cn)
        name_tset = fuzz.token_set_ratio(s1_cn, tgt_cn) / 100.0
        name_tsort = fuzz.token_sort_ratio(s1_cn, tgt_cn) / 100.0
        name_lev = fuzz.ratio(s1_cn, tgt_cn) / 100.0
        max_l = max(len(s1_cn), len(tgt_cn))
        name_len_diff = abs(len(s1_cn) - len(tgt_cn)) / max_l if max_l > 0 else 0.0

        # Asymmetric Containment
        set1 = set(s1_toks)
        set2 = set(tgt_toks)
        common = set1 & set2
        min_len = min(len(set1), len(set2))
        name_containment = len(common) / min_len if min_len > 0 else 0.0

        # Character 3-Gram Jaccard
        name_3gram = char_3gram_jaccard(s1_comp, tgt_comp)

        # IDF Weighted overlap
        if common:
            idf_vals = [IDF_LOOKUP.get(w, DEFAULT_IDF) for w in common]
            shared_idf_sum = sum(idf_vals)
            max_idf = max(idf_vals)
        else:
            shared_idf_sum = 0.0
            max_idf = 0.0
    else:
        name_jw = 0.0
        name_tset = 0.0
        name_tsort = 0.0
        name_lev = 0.0
        name_len_diff = 1.0
        name_containment = 0.0
        name_3gram = 0.0
        shared_idf_sum = 0.0
        max_idf = 0.0

    shared_core_count = float(len(s1_core & tgt_core))
    comp_exact = 1.0 if (s1_comp and tgt_comp and s1_comp == tgt_comp) else 0.0

    # 2. Address Features
    has_addr_s1 = 1.0 if s1_ca else 0.0
    has_addr_tgt = 1.0 if tgt_ca else 0.0
    both_have_address = 1.0 if (s1_ca and tgt_ca) else 0.0

    if both_have_address:
        addr_tset = fuzz.token_set_ratio(s1_ca, tgt_ca) / 100.0
        addr_lev = fuzz.ratio(s1_ca, tgt_ca) / 100.0
        a_tok1 = set(s1_ca.split())
        a_tok2 = set(tgt_ca.split())
        min_a = min(len(a_tok1), len(a_tok2))
        addr_containment = len(a_tok1 & a_tok2) / min_a if min_a > 0 else 0.0
    else:
        addr_tset = 0.0
        addr_lev = 0.0
        addr_containment = 0.0

    # 3. Anchor Hierarchies & Negative Conflict Signals
    if s1_pin and tgt_pin:
        pin_status = 1.0 if s1_pin == tgt_pin else -1.0
        pin_prefix_status = 1.0 if s1_pin[:3] == tgt_pin[:3] else -1.0
    else:
        pin_status = 0.0
        pin_prefix_status = 0.0

    if s1_door and tgt_door:
        door_status = 1.0 if s1_door == tgt_door else -1.0
    else:
        door_status = 0.0

    shared_phone = 1.0 if (s1_phone and tgt_phone and s1_phone == tgt_phone) else 0.0

    # 4. Feature Interactions
    interaction_name_addr = name_tset * addr_tset
    interaction_door_name = door_status * name_jw

    return [
        name_jw,
        name_tset,
        name_tsort,
        name_lev,
        name_len_diff,
        name_containment,
        name_3gram,
        shared_core_count,
        shared_idf_sum,
        max_idf,
        comp_exact,
        has_addr_s1,
        has_addr_tgt,
        addr_tset,
        addr_lev,
        addr_containment,
        pin_status,
        pin_prefix_status,
        door_status,
        shared_phone,
        is_s2_val,
        both_have_address,
        interaction_name_addr,
        interaction_door_name
    ]

# ---------------------------------------------------------
# 3. MACRO-F0.5 EVALUATOR
# ---------------------------------------------------------
def compute_macro_f05(gt_dict, pred_dict):
    total = len(gt_dict)
    if total == 0:
        return 0.0
    scores = []
    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(true_set & pred_set)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        denom = 0.25 * precision + recall
        f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
        scores.append(f05)
    return sum(scores) / total

# ---------------------------------------------------------
# 4. MAIN ENSEMBLE TRAINING & VALIDATION
# ---------------------------------------------------------
def main():
    print("=" * 65)
    print("🏆 TRAINING TOP-1 ENSEMBLE (LIGHTGBM + CATBOOST) ON 24 SIGNALS")
    print("=" * 65)
    random.seed(42)
    np.random.seed(42)
    os.makedirs('models', exist_ok=True)
    train_dir = 'dataset/train'

    # Step 1: Read Ground Truth
    print("Step 1: Reading train_ground_truth.tsv...")
    gt = {}
    with open(os.path.join(train_dir, 'train_ground_truth.tsv'), encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            matched = set(parts[1].split(',')) if len(parts) > 1 and parts[1] else set()
            gt[s1_id] = matched

    all_s1_ids = list(gt.keys())
    random.shuffle(all_s1_ids)

    # 70,000 for training, 15,000 for validation
    train_s1_ids = set(all_s1_ids[:70000])
    val_s1_ids = set(all_s1_ids[70000:85000])
    target_s1_all = train_s1_ids | val_s1_ids

    needed_targets = set()
    for s1_id in target_s1_all:
        needed_targets |= gt[s1_id]

    print(f"Selected {len(train_s1_ids)} train entities, {len(val_s1_ids)} val entities.")
    print(f"Loading {len(needed_targets)} positive target records...")

    # Step 2: Load S1 records
    s1_data = {}
    with open(os.path.join(train_dir, 'train_source1.tsv'), encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            eid = parts[0]
            if eid in target_s1_all:
                cn, comp, core, toks = clean_name(parts[1])
                ca = clean_addr(parts[2])
                door, pin, phone = extract_anchors(parts[2])
                s1_data[eid] = (cn, comp, core, toks, ca, door, pin, phone)

    # Step 3: Load Target Records & Inverted Index
    target_data = {}
    name_index = defaultdict(list)
    door_index = defaultdict(list)
    pin_index = defaultdict(list)
    target_count = 0

    for fname in ('train_source2.tsv', 'train_source3.tsv'):
        print(f"  Scanning {fname}...")
        with open(os.path.join(train_dir, fname), encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.rstrip('\r\n').split('\t')
                eid = parts[0]
                if eid in needed_targets or (target_count < 400000 and random.random() < 0.05):
                    target_count += 1
                    cn, comp, core, toks = clean_name(parts[1])
                    ca = clean_addr(parts[2])
                    door, pin, phone = extract_anchors(parts[2])
                    target_data[eid] = (cn, comp, core, toks, ca, door, pin, phone)

                    for t in set(toks):
                        if len(t) >= 3 and t not in GENERIC_TERMS:
                            name_index[t].append(eid)
                    if door:
                        door_index[door].append(eid)
                    if pin:
                        pin_index[pin].append(eid)

    print(f"Indexed {len(target_data)} targets. Building training pair dataset...")

    def get_cands(s1_t, max_c=12):
        cand_counts = defaultdict(float)
        s1_cn, s1_comp, s1_core, s1_toks, s1_ca, s1_door, s1_pin, s1_phone = s1_t
        for t in set(s1_toks):
            if len(t) >= 3 and t in name_index:
                w = IDF_LOOKUP.get(t, 2.0)
                for cid in name_index[t][:80]:
                    cand_counts[cid] += w
        if s1_door and s1_door in door_index:
            for cid in door_index[s1_door][:50]:
                cand_counts[cid] += 4.0
        if s1_pin and s1_pin in pin_index:
            for cid in pin_index[s1_pin][:50]:
                cand_counts[cid] += 2.0
        if not cand_counts:
            return []
        return [c for c, _ in sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:max_c]]

    X_train, y_train = [], []
    for s1_id in train_s1_ids:
        if s1_id not in s1_data:
            continue
        s1_t = s1_data[s1_id]
        true_matches = gt[s1_id]

        for pos_id in true_matches:
            if pos_id in target_data:
                is_s2 = 1.0 if pos_id.startswith('S2-') else 0.0
                feats = extract_championship_features(s1_t, target_data[pos_id], is_s2)
                X_train.append(feats)
                y_train.append(1)

        cands = get_cands(s1_t, max_c=8)
        neg_count = 0
        for cid in cands:
            if cid not in true_matches and cid in target_data:
                is_s2 = 1.0 if cid.startswith('S2-') else 0.0
                feats = extract_championship_features(s1_t, target_data[cid], is_s2)
                X_train.append(feats)
                y_train.append(0)
                neg_count += 1
                if neg_count >= 3:
                    break

    print(f"Total training pairs: {len(X_train)} (Pos: {sum(y_train)}, Neg: {len(y_train) - sum(y_train)})")

    # Step 4: Build Validation Candidates
    print("Building validation candidate pool...")
    val_candidates = defaultdict(list)
    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}

    for s1_id in val_s1_ids:
        if s1_id not in s1_data:
            continue
        s1_t = s1_data[s1_id]
        cands = get_cands(s1_t, max_c=12)
        cand_set = set(cands)
        for cid in gt[s1_id]:
            if cid in target_data:
                cand_set.add(cid)

        for cid in cand_set:
            if cid in target_data:
                is_s2 = 1.0 if cid.startswith('S2-') else 0.0
                feats = extract_championship_features(s1_t, target_data[cid], is_s2)
                val_candidates[s1_id].append((cid, feats))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    # Step 5: Train Model 1 (LightGBM)
    print("\nTraining Model 1: LightGBM (180 trees, num_leaves=63)...")
    lgb_dataset = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_depth': 7,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_child_samples': 25,
        'verbose': -1,
        'n_jobs': 4
    }
    lgb_model = lgb.train(lgb_params, lgb_dataset, num_boost_round=180)
    lgb_model.save_model('models/championship_lgbm.txt')
    print("LightGBM saved to models/championship_lgbm.txt")

    # Step 6: Train Model 2 (CatBoost)
    print("\nTraining Model 2: CatBoost (200 trees, depth=6)...")
    cb_model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.10,
        depth=6,
        loss_function='Logloss',
        verbose=0,
        thread_count=4,
        random_seed=42
    )
    cb_model.fit(X_train, y_train)
    cb_model.save_model('models/championship_catboost.cbm')
    print("CatBoost saved to models/championship_catboost.cbm")

    # Step 7: Evaluate Dual-Model Ensemble & Dynamic Thresholding
    print("\nEvaluating Dual-Model Ensemble on 15,000 Validation Entities...")
    val_pair_map = []
    val_feats_list = []
    for s1_id, pairs in val_candidates.items():
        for cid, feats in pairs:
            val_pair_map.append((s1_id, cid))
            val_feats_list.append(feats)

    X_val = np.array(val_feats_list, dtype=np.float32)
    p_lgb = lgb_model.predict(X_val)
    p_cb = cb_model.predict_proba(X_val)[:, 1]

    # Weighted Ensemble Blend: 65% LightGBM + 35% CatBoost
    p_ensemble = 0.65 * p_lgb + 0.35 * p_cb

    s1_scored = defaultdict(list)
    for (s1_id, cid), prob in zip(val_pair_map, p_ensemble):
        s1_scored[s1_id].append((cid, float(prob)))

    best_score = 0.0
    best_t = 0.80

    for t in np.arange(0.50, 0.95, 0.05):
        preds = {}
        for s1_id in val_s1_ids:
            s2_m = [c for c, p in s1_scored[s1_id] if p >= t and c.startswith('S2-')][:3]
            s3_m = [c for c, p in s1_scored[s1_id] if p >= t and c.startswith('S3-')][:3]
            preds[s1_id] = set(s2_m + s3_m)
        f05 = compute_macro_f05(val_gt, preds)
        print(f"  Ensemble Threshold: {t:.2f} --> Macro-F0.5: {f05*100:.2f}%")
        if f05 > best_score:
            best_score = f05
            best_t = t

    print(f"\n🏆 CHAMPIONSHIP ENSEMBLE PEAK SCORE: {best_score*100:.2f}% at Threshold: {best_t:.2f}")
    with open('models/championship_threshold.txt', 'w') as f:
        f.write(f"{best_t:.4f}\n")

if __name__ == '__main__':
    main()
