import os
import sys
import re
import gc
import time
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
        return '', '', ''
    pins = RE_PIN.findall(raw_addr)
    pin = pins[0] if pins else ''
    phones = RE_PHONE.findall(raw_addr)
    phone = phones[0] if phones else ''
    door = ''
    doors = RE_DOOR.findall(raw_addr)
    if doors:
        prefix, num, suffix = doors[0]
        door = f"{prefix.lower()}-{num}{suffix.lower()}" if prefix else f"{num}{suffix.lower()}"
    return door, pin, phone

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION FUNCTION
# ---------------------------------------------------------
def extract_pairwise_features(s1_tuple, tgt_tuple, is_source2_flag):
    # s1_tuple: (cn, comp, core_set, ca, door, pin, phone)
    # tgt_tuple: (cn, comp, core_set, ca, door, pin, phone)
    s1_cn, s1_comp, s1_core, s1_ca, s1_door, s1_pin, s1_phone = s1_tuple
    tgt_cn, tgt_comp, tgt_core, tgt_ca, tgt_door, tgt_pin, tgt_phone = tgt_tuple

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
    if s1_pin and tgt_pin:
        pin_status = 1.0 if s1_pin == tgt_pin else -1.0
    else:
        pin_status = 0.0

    if s1_door and tgt_door:
        door_status = 1.0 if s1_door == tgt_door else -1.0
    else:
        door_status = 0.0

    shared_phone = 1.0 if (s1_phone and tgt_phone and s1_phone == tgt_phone) else 0.0

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
        is_source2_flag
    ]

# ---------------------------------------------------------
# 3. MAIN MEMORY-LEAN INFERENCE PIPELINE
# ---------------------------------------------------------
def main():
    start_time = time.time()
    test_dir = 'dataset/test'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    match_path = os.path.join(output_dir, 'matching_results.tsv')
    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    model_path = 'models/lgbm_matcher.txt'

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Run train_model.py first.")
        sys.exit(1)

    print("==================================================================")
    print("🚀 PIPELINE V6: ULTRA-FAST MEMORY-LEAN LIGHTGBM INFERENCE")
    print("==================================================================")

    model = lgb.Booster(model_file=model_path)
    threshold = 0.80
    if os.path.exists('models/optimal_threshold.txt'):
        with open('models/optimal_threshold.txt') as f:
            threshold = float(f.read().strip())
    print(f"Using Optimal Decision Threshold: {threshold:.2f}")

    print("Step 1: Reading test_source1.tsv...")
    s1_records = []
    with open(os.path.join(test_dir, 'test_source1.tsv'), encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 4:
                eid = parts[0]
                cn, comp, core = clean_name(parts[1])
                ca = clean_addr(parts[2])
                door, pin, phone = extract_anchors(parts[2])
                country = parts[3].strip()
                s1_records.append((eid, cn, comp, core, ca, door, pin, phone, country))

    total_s1 = len(s1_records)
    print(f"Total S1 entities loaded: {total_s1}")

    country_to_s1 = defaultdict(list)
    for idx, item in enumerate(s1_records):
        country_to_s1[item[8]].append((idx, item))

    # Stores per S1 index: list of (target_id, prob)
    s1_matches = [[] for _ in range(total_s1)]
    s1_candidates = [[] for _ in range(total_s1)]

    # Process each country, and within each country, stream S2 then S3 separately!
    for country in sorted(country_to_s1.keys(), key=lambda c: len(country_to_s1[c])):
        s1_country_list = country_to_s1[country]
        print(f"\n=======================================================")
        print(f"Processing Country: [{country}] ({len(s1_country_list)} S1 entities)")
        print(f"=======================================================")

        for source_name, is_s2 in [('test_source2.tsv', 1.0), ('test_source3.tsv', 0.0)]:
            s_start = time.time()
            prefix = "S2" if is_s2 == 1.0 else "S3"
            print(f"\n  --- Indexing {source_name} for [{country}] ---")

            target_records = {} # eid -> (cn, comp, core, ca, door, pin, phone)
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
                        cn, comp, core = clean_name(parts[1])
                        ca = clean_addr(parts[2])
                        door, pin, phone = extract_anchors(parts[2])
                        target_records[eid] = (cn, comp, core, ca, door, pin, phone)

                        for t in set(cn.split()):
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

            print(f"  [{country} - {prefix}] Running LightGBM batch inference...")

            BATCH_SIZE = 10000
            matches_found = 0

            for b_start in range(0, len(s1_country_list), BATCH_SIZE):
                b_chunk = s1_country_list[b_start : b_start + BATCH_SIZE]
                batch_pairs = []

                for global_idx, s1_item in b_chunk:
                    s1_id, s1_cn, s1_comp, s1_core, s1_ca, s1_door, s1_pin, s1_phone, _ = s1_item
                    cand_counts = defaultdict(float)

                    for t in set(s1_cn.split()):
                        if len(t) >= 3 and t in name_index:
                            for cid in name_index[t]:
                                cand_counts[cid] += 2.0

                    if s1_door and s1_door in door_index:
                        for cid in door_index[s1_door]:
                            cand_counts[cid] += 3.0

                    if s1_pin and s1_pin in pin_index:
                        for cid in pin_index[s1_pin]:
                            cand_counts[cid] += 1.5

                    if not cand_counts:
                        continue

                    # Top 8 candidates per source
                    top_cands = [cid for cid, _ in sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
                    s1_t = s1_item[1:8]

                    for cid in top_cands:
                        tgt_t = target_records[cid]
                        feats = extract_pairwise_features(s1_t, tgt_t, is_s2)
                        batch_pairs.append((global_idx, cid, feats))

                if batch_pairs:
                    X_batch = np.array([p[2] for p in batch_pairs], dtype=np.float32)
                    probs = model.predict(X_batch)

                    grouped = defaultdict(list)
                    for (g_idx, cid, _), prob in zip(batch_pairs, probs):
                        grouped[g_idx].append((cid, float(prob)))

                    for g_idx, scored_list in grouped.items():
                        scored_list.sort(key=lambda x: x[1], reverse=True)
                        m_list = [cid for cid, p in scored_list if p >= threshold][:3]
                        c_list = [cid for cid, _ in scored_list[:4]]

                        s1_matches[g_idx].extend(m_list)
                        for c in c_list:
                            if c not in s1_candidates[g_idx]:
                                s1_candidates[g_idx].append(c)

                        matches_found += len(m_list)

            elapsed = time.time() - s_start
            rate = len(s1_country_list) / elapsed if elapsed > 0 else 0
            print(f"  [{country} - {prefix}] Done in {elapsed:.1f}s ({rate:.0f} recs/s) | Matches: {matches_found}")

            del target_records, name_index, door_index, pin_index
            gc.collect()

    # Step 4: Write Final TSV Files with Superset Guarantee
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
            c_set = list(m_set) # Guarantees all matches are in candidate_pairs
            for c in s1_candidates[idx]:
                if c not in c_set:
                    c_set.append(c)
                if len(c_set) >= max(len(m_set), 6):
                    break
            c_str = ','.join(c_set) if c_set else ''
            f.write(f"{eid}\t{c_str}\n")

    total_time = (time.time() - start_time) / 60
    print("\n==================================================================")
    print(f"✅ Pipeline v6 finished successfully in {total_time:.1f} minutes!")
    print(f"  Matching TSV:  {match_path}")
    print(f"  Candidate TSV: {cand_path}")
    print("==================================================================")

if __name__ == '__main__':
    main()
