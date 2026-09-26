from rapidfuzz import fuzz

FEATURE_NAMES = [
    'name_jaccard', 'name_ratio', 'name_token_sort',
    'addr_ratio', 'digit_overlap', 'exact_name', 'is_s2'
]

def compute_features(s1_name: str, s1_addr: str, s1_digits: set,
                     s2_name: str, s2_addr: str, s2_digits: set, s2_id: str):
    t1 = set(s1_name.split())
    t2 = set(s2_name.split())
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    name_jaccard = intersection / union if union > 0 else 0.0

    name_ratio = fuzz.ratio(s1_name, s2_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, s2_name) / 100.0
    addr_ratio = fuzz.token_set_ratio(s1_addr, s2_addr) / 100.0 if s1_addr and s2_addr else 0.0
    digit_overlap = 1.0 if (s1_digits and s2_digits and (s1_digits & s2_digits)) else 0.0
    exact_name = 1.0 if (s1_name and s1_name == s2_name) else 0.0
    is_s2 = 1.0 if s2_id.startswith('S2-') else 0.0

    return [
        name_jaccard,
        name_ratio,
        name_token_sort,
        addr_ratio,
        digit_overlap,
        exact_name,
        is_s2
    ]
