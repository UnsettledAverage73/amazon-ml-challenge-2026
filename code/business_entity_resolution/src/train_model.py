import os
import sys
import re
import gc
import time
import random
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz, distance

# ---------------------------------------------------------
# 1. NORMALIZATION & ANCHOR PARSING
# ---------------------------------------------------------
LEGAL_SUFFIXES = {
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise', 'industries', 'group',
    'services', 'solutions', 'technologies', 'holdings', 'assoc', 'associates',
    'sa', 'sarl', 'gmbh', 'bv', 'plc', 'eurl', 'sas', 'sci', 'snc'
}

GENERIC_TERMS = {
    'store', 'stores', 'shop', 'mart', 'market', 'supermarket', 'agency', 'agencies',
    'traders', 'trading', 'hotel', 'restaurant', 'cafe', 'bazaar', 'jewellers',
    'jewellery', 'textiles', 'pharmacy', 'chemist', 'auto', 'garage', 'consultants',
    'consultancy', 'logistics', 'transports', 'centre', 'center', 'foods', 'retail',
    'wholesale', 'supply', 'supplies', 'commercial'
}

DOMAINS = re.compile(r'(\.com|\.in|\.org|\.net|\.co|\.fr|\.io|\.biz|\.info|www\.)', re.IGNORECASE)
RE_PUNCT = re.compile(r'[^a-zA-Z0-9\s]')
RE_DOOR = re.compile(r'\b([A-Za-z]{0,3})[-/]?0*(\d{1,5})([A-Za-z]?)\b')
RE_PIN = re.compile(r'\b(\d{5,6})\b')
RE_PHONE = re.compile(r'\b\d{7,10}\b')

ADDR_ABBREVIATIONS = [
    (re.compile(r'\bst\b'), 'street'),
    (re.compile(r'\brd\b'), 'road'),
    (re.compile(r'\bave\b'), 'avenue'),
    (re.compile(r'\bblvd\b'), 'boulevard'),
    (re.compile(r'\bapt\b'), 'apartment'),
    (re.compile(r'\bste\b'), 'suite'),
    (re.compile(r'\bdr\b'), 'drive'),
    (re.compile(r'\bln\b'), 'lane'),
    (re.compile(r'\bct\b'), 'court'),
    (re.compile(r'\bpl\b'), 'place'),
    (re.compile(r'\bsq\b'), 'square'),
    (re.compile(r'\bpkwy\b'), 'parkway'),
    (re.compile(r'\bcir\b'), 'circle'),
    (re.compile(r'\bhwy\b'), 'highway'),
    (re.compile(r'\bopp\b'), 'opposite'),
    (re.compile(r'\bnr\b'), 'near'),
    (re.compile(r'\bsec\b|\bsect\b'), 'sector'),
    (re.compile(r'\bph\b'), 'phase'),
    (re.compile(r'\bflr\b|\bfl\b'), 'floor'),
    (re.compile(r'\bcol\b'), 'colony'),
    (re.compile(r'\bmrg\b'), 'marg'),
    (re.compile(r'\bbldg\b'), 'building'),
    (re.compile(r'\bcplx\b'), 'complex'),
    (re.compile(r'\bbd\b|\bbld\b'), 'boulevard'),
    (re.compile(r'\bav\b'), 'avenue'),
    (re.compile(r'\bimp\b'), 'impasse'),
    (re.compile(r'\ball\b'), 'allee'),
    (re.compile(r'\brte\b'), 'route'),
    (re.compile(r'\bch\b'), 'chemin'),
    (re.compile(r'\bcedex\b'), 'cedex'),
]

def clean_name(text: str):
    if not text:
        return '', '', set()
    text = DOMAINS.sub(' ', text)
    text = RE_PUNCT.sub(' ', text.lower())
    tokens = [w for w in text.split() if w not in LEGAL_SUFFIXES and len(w) > 1]
    cleaned = ' '.join(tokens)
    compressed = ''.join(tokens)
    core = {w for w in tokens if w not in GENERIC_TERMS}
    return cleaned, compressed, core

def clean_addr(text: str):
    if not text:
        return ''
    text = text.lower()
    for pattern, repl in ADDR_ABBREVIATIONS:
        text = pattern.sub(repl, text)
    return ' '.join(RE_PUNCT.sub(' ', text).split())

def extract_anchors(raw_addr: str):
    if not raw_addr:
        return set(), set(), set()
    anchors = set()
    pins = set(RE_PIN.findall(raw_addr))
    phones = set(RE_PHONE.findall(raw_addr))
    for prefix, num, suffix in RE_DOOR.findall(raw_addr):
        norm_door = f"{prefix.lower()}-{num}{suffix.lower()}" if prefix else f"{num}{suffix.lower()}"
        if len(norm_door) >= 2:
            anchors.add(norm_door)
    for p in pins:
        anchors.add(p)
    return anchors, pins, phones

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION FUNCTION
# ---------------------------------------------------------
FEATURE_NAMES = [
    'name_jw',             # Jaro-Winkler distance on clean name
    'name_tset',           # Token set ratio
    'name_tsort',          # Token sort ratio
    'name_lev',            # Levenshtein ratio
    'name_len_diff',       # Relative length difference
    'shared_core_count',   # Number of shared core name tokens
    'comp_exact',          # Exact compressed name match (0 or 1)
    'addr_tset',           # Address token set ratio
    'addr_lev',            # Address Levenshtein ratio
    'has_addr_s1',         # Does S1 have an address?
    'has_addr_tgt',        # Does Target have an address?
    'pin_status',          # +1: match, 0: missing, -1: conflict
    'door_status',         # +1: match, 0: missing, -1: conflict
    'shared_phone',        # Phone number match (0 or 1)
    'is_source2',          # 1 if S2, 0 if S3
]

def extract_pairwise_features(s1_tuple, tgt_tuple, tgt_id):
    # s1_tuple: (cn, comp, core, ca, anchors, pins, phones)
    # tgt_tuple: (cn, comp, core, ca, anchors, pins, phones)
    s1_cn, s1_comp, s1_core, s1_ca, s1_anchors, s1_pins, s1_phones = s1_tuple
    tgt_cn, tgt_comp, tgt_core, tgt_ca, tgt_anchors, tgt_pins, tgt_phones = tgt_tuple

    # 1. Name Features
    if s1_cn and tgt_cn:
        name_jw = distance.JaroWinkler.similarity(s1_cn, tgt_cn)
        name_tset = fuzz.token_set_ratio(s1_cn, tgt_cn) / 100.0
        name_tsort = fuzz.token_sort_ratio(s1_cn, tgt_cn) / 100.0
        name_lev = fuzz.ratio(s1_cn, tgt_cn) / 100.0
        max_l = max(len(s1_cn), len(tgt_cn))
        name_len_diff = abs(len(s1_cn) - len(tgt_cn)) / max_l if max_l > 0 else 0.0
    else:
        name_jw = 0.0
        name_tset = 0.0
        name_tsort = 0.0
        name_lev = 0.0
        name_len_diff = 1.0

    shared_core_count = float(len(s1_core & tgt_core))
    comp_exact = 1.0 if (s1_comp and tgt_comp and s1_comp == tgt_comp) else 0.0

    # 2. Address Features
    has_addr_s1 = 1.0 if s1_ca else 0.0
    has_addr_tgt = 1.0 if tgt_ca else 0.0
    if s1_ca and tgt_ca:
        addr_tset = fuzz.token_set_ratio(s1_ca, tgt_ca) / 100.0
        addr_lev = fuzz.ratio(s1_ca, tgt_ca) / 100.0
    else:
        addr_tset = 0.0
        addr_lev = 0.0

    # 3. Anchor & Negative Conflict Signals
    # PIN status
    if s1_pins and tgt_pins:
        pin_status = 1.0 if bool(s1_pins & tgt_pins) else -1.0
    else:
        pin_status = 0.0

    # Door status
    s1_doors = s1_anchors - s1_pins
    tgt_doors = tgt_anchors - tgt_pins
    if s1_doors and tgt_doors:
        door_status = 1.0 if bool(s1_doors & tgt_doors) else -1.0
    else:
        door_status = 0.0

    shared_phone = 1.0 if (s1_phones and tgt_phones and bool(s1_phones & tgt_phones)) else 0.0
    is_source2 = 1.0 if tgt_id.startswith('S2-') else 0.0

    return [
        name_jw,
        name_tset,
        name_tsort,
        name_lev,
        name_len_diff,
        shared_core_count,
        comp_exact,
        addr_tset,
        addr_lev,
        has_addr_s1,
        has_addr_tgt,
        pin_status,
        door_status,
        shared_phone,
        is_source2
    ]

# ---------------------------------------------------------
# 3. MACRO-F0.5 EVALUATOR
# ---------------------------------------------------------
def compute_macro_f05(gt_dict, pred_dict):
    """
    Computes per-entity macro F_0.5 score:
    - If true is empty and pred is empty: score = 1.0
    - If true is empty and pred is non-empty: score = 0.0 (false merge on singleton)
    - If true is non-empty and pred is empty: score = 0.0
    - Otherwise: standard F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    total = len(gt_dict)
    if total == 0:
        return 0.0

    scores = []
    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())

        if not true_set:
            # Singleton entity
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

        denom = (0.25 * precision + recall)
        f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
        scores.append(f05)

    return sum(scores) / total

# ---------------------------------------------------------
# 4. MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def main():
    print("=" * 65)
    print("⚡ FAST LIGHTGBM MODEL TRAINING & MACRO-F0.5 TUNING")
    print("=" * 65)
    random.seed(42)
    np.random.seed(42)

    os.makedirs('models', exist_ok=True)
    train_dir = 'dataset/train'

    # Step 1: Read Ground Truth mapping
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
    print(f"Loaded ground truth for {len(all_s1_ids)} S1 entities.")

    # Select 60,000 for training, 15,000 for validation
    random.shuffle(all_s1_ids)
    train_s1_ids = set(all_s1_ids[:60000])
    val_s1_ids = set(all_s1_ids[60000:75000])
    target_s1_all = train_s1_ids | val_s1_ids

    # Collect all needed S2 and S3 IDs to load from train_source2/3
    needed_target_ids = set()
    for s1_id in target_s1_all:
        needed_target_ids |= gt[s1_id]

    print(f"Selected {len(train_s1_ids)} train entities, {len(val_s1_ids)} val entities.")
    print(f"Positive targets to fetch: {len(needed_target_ids)}")

    # Step 2: Load S1 records
    print("Step 2: Loading S1 records...")
    s1_data = {}
    with open(os.path.join(train_dir, 'train_source1.tsv'), encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            eid = parts[0]
            if eid in target_s1_all:
                cn, comp, core = clean_name(parts[1])
                ca = clean_addr(parts[2])
                anchors, pins, phones = extract_anchors(parts[2])
                country = parts[3].strip() if len(parts) > 3 else 'US'
                s1_data[eid] = (cn, comp, core, ca, anchors, pins, phones, country)

    # Step 3: Load Target Records (S2 & S3)
    # To enable realistic blocking negatives, load positive matches + a sample of non-matches
    print("Step 3: Loading S2 & S3 records and building blocking index...")
    target_data = {}
    name_index = defaultdict(list)
    door_index = defaultdict(list)
    pin_index = defaultdict(list)

    # We will load positive matches + stream-index targets in batches
    target_count = 0
    max_targets_to_index = 350000 # Keep memory low (~1.5GB)

    for fname in ('train_source2.tsv', 'train_source3.tsv'):
        print(f"  Scanning {fname}...")
        with open(os.path.join(train_dir, fname), encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.rstrip('\r\n').split('\t')
                eid = parts[0]
                is_needed = eid in needed_target_ids
                if is_needed or (target_count < max_targets_to_index and random.random() < 0.05):
                    target_count += 1
                    cn, comp, core = clean_name(parts[1])
                    ca = clean_addr(parts[2])
                    anchors, pins, phones = extract_anchors(parts[2])
                    t_tuple = (cn, comp, core, ca, anchors, pins, phones)
                    target_data[eid] = t_tuple

                    for t in set(cn.split()):
                        if len(t) >= 3 and t not in GENERIC_TERMS:
                            name_index[t].append(eid)
                    for a in (anchors - pins):
                        door_index[a].append(eid)
                    for p in pins:
                        pin_index[p].append(eid)

    print(f"Indexed {len(target_data)} target entities. Building feature matrix...")

    # Step 4: Build Training and Validation Pair Datasets
    # Positive pairs: (s1_id, gt_target)
    # Hard negative pairs: generated via blocking
    X_train, y_train = [], []
    val_candidates = defaultdict(list) # s1_id -> list of (target_id, features)
    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}

    def get_blocking_candidates(s1_tuple, max_cands=15):
        s1_cn, s1_comp, s1_core, s1_ca, s1_anchors, s1_pins, s1_phones, _ = s1_tuple
        cand_counts = defaultdict(float)

        for t in set(s1_cn.split()):
            if len(t) >= 3 and t in name_index:
                for cid in name_index[t][:100]:
                    cand_counts[cid] += 2.0

        for a in (s1_anchors - s1_pins):
            if a in door_index:
                for cid in door_index[a][:50]:
                    cand_counts[cid] += 3.0

        for p in s1_pins:
            if p in pin_index:
                for cid in pin_index[p][:50]:
                    cand_counts[cid] += 1.5

        if not cand_counts:
            return []
        return [cid for cid, _ in sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:max_cands]]

    print("Building training samples...")
    train_count = 0
    for s1_id in train_s1_ids:
        if s1_id not in s1_data:
            continue
        s1_t = s1_data[s1_id]
        true_matches = gt[s1_id]

        # 1. Add true positives
        for pos_id in true_matches:
            if pos_id in target_data:
                feats = extract_pairwise_features(s1_t[:7], target_data[pos_id], pos_id)
                X_train.append(feats)
                y_train.append(1)

        # 2. Add hard negatives from blocking
        cands = get_blocking_candidates(s1_t, max_cands=8)
        neg_added = 0
        for cid in cands:
            if cid not in true_matches and cid in target_data:
                feats = extract_pairwise_features(s1_t[:7], target_data[cid], cid)
                X_train.append(feats)
                y_train.append(0)
                neg_added += 1
                if neg_added >= 4: # Keep balanced ratio ~ 1 pos : 2-3 neg
                    break

        train_count += 1

    print(f"Total training pairs: {len(X_train)} (Positives: {sum(y_train)}, Negatives: {len(y_train) - sum(y_train)})")

    # Step 5: Build Validation Candidate Pairs
    print("Building validation candidate pool...")
    val_processed = 0
    for s1_id in val_s1_ids:
        val_processed += 1
        if val_processed % 3000 == 0:
            print(f"  Validation pool: {val_processed}/{len(val_s1_ids)}")
        if s1_id not in s1_data:
            continue
        s1_t = s1_data[s1_id]
        cands = get_blocking_candidates(s1_t, max_cands=15)
        # Fast lookup in target_data without allocating sets
        cand_set = set(cands)
        for cid in gt[s1_id]:
            if cid in target_data:
                cand_set.add(cid)

        for cid in cand_set:
            if cid in target_data:
                feats = extract_pairwise_features(s1_t[:7], target_data[cid], cid)
                val_candidates[s1_id].append((cid, feats))

    # Step 6: Train LightGBM Classifier
    print("Step 4: Training LightGBM Classifier...")
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': 4
    }

    model = lgb.train(params, train_data, num_boost_round=150)
    model.save_model('models/lgbm_matcher.txt')
    print("Model saved to models/lgbm_matcher.txt")

    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    print("\nFeature Importances (gain):")
    for fname, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True):
        print(f"  {fname:18s}: {imp:.1f}")

    # Step 7: Sweep Threshold for Maximum Macro-F0.5
    print("\nStep 5: Sweeping Threshold on Validation Set for Maximum Macro-F0.5...")
    # Pre-predict probabilities for all validation pairs
    val_predictions = {}
    all_val_feats = []
    pair_map = [] # (s1_id, cid)

    for s1_id, pairs in val_candidates.items():
        for cid, feats in pairs:
            all_val_feats.append(feats)
            pair_map.append((s1_id, cid))

    if all_val_feats:
        val_probs = model.predict(np.array(all_val_feats, dtype=np.float32))
        s1_to_scored_cands = defaultdict(list)
        for (s1_id, cid), prob in zip(pair_map, val_probs):
            s1_to_scored_cands[s1_id].append((cid, prob))

        best_f05 = 0.0
        best_threshold = 0.70

        for thresh in np.arange(0.40, 0.96, 0.05):
            pred_dict = {}
            for s1_id in val_s1_ids:
                matches = []
                # Separate S2 and S3, keep max 3 of each
                s2_cands = [cid for cid, p in s1_to_scored_cands[s1_id] if p >= thresh and cid.startswith('S2-')]
                s3_cands = [cid for cid, p in s1_to_scored_cands[s1_id] if p >= thresh and cid.startswith('S3-')]
                matches = s2_cands[:3] + s3_cands[:3]
                pred_dict[s1_id] = set(matches)

            score = compute_macro_f05(val_gt, pred_dict)
            print(f"  Threshold: {thresh:.2f}  -->  Macro-F0.5: {score*100:.2f}%")
            if score > best_f05:
                best_f05 = score
                best_threshold = thresh

        print(f"\n🏆 Optimal Decision Threshold: {best_threshold:.2f} (Macro-F0.5: {best_f05*100:.2f}%)")
        with open('models/optimal_threshold.txt', 'w') as f:
            f.write(f"{best_threshold:.4f}\n")
    else:
        print("No validation candidates generated.")

if __name__ == '__main__':
    main()
