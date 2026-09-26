import os
import sys
import re
import gc
import json
import time
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
    'wholesale', 'supply', 'supplies', 'commercial',
    # French common generic business terms
    'agence', 'agences', 'societe', 'societes', 'boulangerie', 'coiffure', 'boucherie',
    'epicerie', 'atelier', 'ateliers', 'commerce', 'commerces', 'batiment', 'menuiserie',
    'plomberie', 'electricite', 'tabac'
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
    text = RE_CEDEX.sub(' ', text.lower())
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
# 2. FEATURE EXTRACTION
# ---------------------------------------------------------
IDF_LOOKUP = {}
DEFAULT_IDF = 12.0
if os.path.exists('models/token_idf.json'):
    with open('models/token_idf.json') as f:
        data = json.load(f)
        IDF_LOOKUP = data.get('idfs', {})
        DEFAULT_IDF = data.get('default_idf', 12.0)

def extract_championship_features(s1_tuple, tgt_tuple, is_s2_val):
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

        set1 = set(s1_toks)
        set2 = set(tgt_toks)
        common = set1 & set2
        min_len = min(len(set1), len(set2))
        name_containment = len(common) / min_len if min_len > 0 else 0.0

        name_3gram = char_3gram_jaccard(s1_comp, tgt_comp)

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

    # 3. Anchors & Negative Conflict Signals
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
# 3. MAIN CHAMPIONSHIP INFERENCE PIPELINE
# ---------------------------------------------------------
def main():
    start_time = time.time()
    test_dir = 'dataset/test'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    match_path = os.path.join(output_dir, 'matching_results.tsv')
    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    print("==================================================================")
    print("🏆 CHAMPIONSHIP PIPELINE: DUAL GBDT ENSEMBLE INFERENCE")
    print("==================================================================")

    # Load Models
    print("Loading LightGBM & CatBoost models...")
    lgb_model = lgb.Booster(model_file='models/championship_lgbm.txt')
    cb_model = CatBoostClassifier()
    cb_model.load_model('models/championship_catboost.cbm')

    base_threshold = 0.85
    if os.path.exists('models/championship_threshold.txt'):
        with open('models/championship_threshold.txt') as f:
            base_threshold = float(f.read().strip())
    print(f"Optimal Base Threshold: {base_threshold:.2f}")

    print("Step 1: Reading test_source1.tsv with Unicode normalization...")
    s1_records = []
    with open(os.path.join(test_dir, 'test_source1.tsv'), encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 4:
                eid = parts[0]
                cn, comp, core, toks = clean_name(parts[1])
                ca = clean_addr(parts[2])
                door, pin, phone = extract_anchors(parts[2])
                country = parts[3].strip()
                s1_records.append((eid, cn, comp, core, toks, ca, door, pin, phone, country))

    total_s1 = len(s1_records)
    print(f"Total S1 entities loaded: {total_s1}")

    country_to_s1 = defaultdict(list)
    for idx, item in enumerate(s1_records):
        country_to_s1[item[9]].append((idx, item))

    s1_matches = [[] for _ in range(total_s1)]
    s1_candidates = [[] for _ in range(total_s1)]

    # Track target assignments for global 1-to-1 / best-parent resolution
    target_best_s1 = {} # target_id -> (best_prob, s1_idx)

    for country in sorted(country_to_s1.keys(), key=lambda c: len(country_to_s1[c])):
        s1_country_list = country_to_s1[country]
        print(f"\n=======================================================")
        print(f"Processing Country: [{country}] ({len(s1_country_list)} S1 entities)")
        print(f"=======================================================")

        for source_name, is_s2 in [('test_source2.tsv', 1.0), ('test_source3.tsv', 0.0)]:
            s_start = time.time()
            prefix = "S2" if is_s2 == 1.0 else "S3"
            print(f"\n  --- Indexing {source_name} for [{country}] ---")

            target_records = {} # eid -> (cn, comp, core, toks, ca, door, pin, phone)
            name_index = defaultdict(list)
            door_index = defaultdict(list)
            pin_index = defaultdict(list)

            fpath = os.path.join(test_dir, source_name)
            with open(fpath, encoding='utf-8') as f:
                next(f)
                for line in f:
                    parts = line.rstrip('\r\n').split('\t')
                    if len(parts) >= 4 and parts[3].strip() == country:
                        eid = parts[0]
                        cn, comp, core, toks = clean_name(parts[1])
                        ca = clean_addr(parts[2])
                        door, pin, phone = extract_anchors(parts[2])
                        target_records[eid] = (cn, comp, core, toks, ca, door, pin, phone)

                        for t in set(toks):
                            if len(t) >= 3 and t not in GENERIC_TERMS:
                                name_index[t].append(eid)
                        if door:
                            door_index[door].append(eid)
                        if pin:
                            pin_index[pin].append(eid)

            print(f"  [{country} - {prefix}] Loaded {len(target_records)} records.")

            # Prune high-frequency keys
            for k in list(name_index.keys()):
                if len(name_index[k]) > 250:
                    del name_index[k]
            for k in list(door_index.keys()):
                if len(door_index[k]) > 100:
                    del door_index[k]
            for k in list(pin_index.keys()):
                if len(pin_index[k]) > 250:
                    del pin_index[k]

            print(f"  [{country} - {prefix}] Running Dual-GBDT Ensemble batch inference...")

            BATCH_SIZE = 10000
            matches_found = 0

            for b_start in range(0, len(s1_country_list), BATCH_SIZE):
                b_chunk = s1_country_list[b_start : b_start + BATCH_SIZE]
                batch_pairs = [] # (global_idx, cid, feats, adaptive_thresh)

                for global_idx, s1_item in b_chunk:
                    s1_id, s1_cn, s1_comp, s1_core, s1_toks, s1_ca, s1_door, s1_pin, s1_phone, _ = s1_item
                    cand_counts = defaultdict(float)

                    for t in set(s1_toks):
                        if len(t) >= 3 and t in name_index:
                            w = IDF_LOOKUP.get(t, 2.0)
                            for cid in name_index[t]:
                                cand_counts[cid] += w

                    if s1_door and s1_door in door_index:
                        for cid in door_index[s1_door]:
                            cand_counts[cid] += 4.0

                    if s1_pin and s1_pin in pin_index:
                        for cid in pin_index[s1_pin]:
                            cand_counts[cid] += 2.0

                    if not cand_counts:
                        continue

                    # Top 8 candidates per source
                    top_cands = [cid for cid, _ in sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
                    s1_t = s1_item[1:9]

                    # Context-Adaptive Threshold:
                    # If address is missing, strictly enforce 0.88 to avoid name-only false merges
                    # If door and pin match, relax to 0.75
                    for cid in top_cands:
                        tgt_t = target_records[cid]
                        feats = extract_championship_features(s1_t, tgt_t, is_s2)

                        # Context-adaptive threshold calculation
                        if not s1_ca or not tgt_t[4]:
                            thresh = max(base_threshold, 0.88)
                        elif s1_door and tgt_t[5] and s1_door == tgt_t[5] and s1_pin and tgt_t[6] and s1_pin == tgt_t[6]:
                            thresh = min(base_threshold, 0.75)
                        else:
                            thresh = base_threshold

                        batch_pairs.append((global_idx, cid, feats, thresh))

                if batch_pairs:
                    X_batch = np.array([p[2] for p in batch_pairs], dtype=np.float32)

                    # Dual-GBDT Blended Probability
                    probs_lgb = lgb_model.predict(X_batch)
                    probs_cb = cb_model.predict_proba(X_batch)[:, 1]
                    probs = 0.65 * probs_lgb + 0.35 * probs_cb

                    grouped = defaultdict(list)
                    for (g_idx, cid, _, thresh), prob in zip(batch_pairs, probs):
                        grouped[g_idx].append((cid, float(prob), thresh))

                    for g_idx, scored_list in grouped.items():
                        scored_list.sort(key=lambda x: x[1], reverse=True)
                        m_list = [cid for cid, p, th in scored_list if p >= th][:3]
                        c_list = [cid for cid, _, _ in scored_list[:4]]

                        # Enforce global 1-to-1 assignment per target
                        for cid, p, th in scored_list:
                            if p >= th:
                                if cid not in target_best_s1 or p > target_best_s1[cid][0]:
                                    target_best_s1[cid] = (p, g_idx)

                        for c in c_list:
                            if c not in s1_candidates[g_idx]:
                                s1_candidates[g_idx].append(c)

                        matches_found += len(m_list)

            elapsed = time.time() - s_start
            rate = len(s1_country_list) / elapsed if elapsed > 0 else 0
            print(f"  [{country} - {prefix}] Done in {elapsed:.1f}s ({rate:.0f} recs/s) | Raw Pairs Scored: {matches_found}")

            del target_records, name_index, door_index, pin_index
            gc.collect()

    # Step 3.5: Populate s1_matches from global 1-to-1 target_best_s1
    print("\nStep 3.5: Enforcing global 1-to-1 target resolution...")
    for cid, (prob, g_idx) in target_best_s1.items():
        s1_matches[g_idx].append(cid)

    # Step 4: Write Final Output TSVs with Superset Guarantee
    print("\nStep 4: Writing final TSVs...")
    print(f"  Writing {match_path}...")
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for idx, (eid, *_) in enumerate(s1_records):
            m_str = ','.join(s1_matches[idx]) if s1_matches[idx] else ''
            f.write(f"{eid}\t{m_str}\n")

    print(f"  Writing {cand_path}...")
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for idx, (eid, *_) in enumerate(s1_records):
            m_set = s1_matches[idx]
            c_set = list(m_set)
            for c in s1_candidates[idx]:
                if c not in c_set:
                    c_set.append(c)
                if len(c_set) >= max(len(m_set), 6):
                    break
            c_str = ','.join(c_set) if c_set else ''
            f.write(f"{eid}\t{c_str}\n")

    total_time = (time.time() - start_time) / 60
    print("\n==================================================================")
    print(f"🏆 Championship Pipeline completed successfully in {total_time:.1f} minutes!")
    print(f"  Matching TSV:  {match_path}")
    print(f"  Candidate TSV: {cand_path}")
    print("==================================================================")

if __name__ == '__main__':
    main()
