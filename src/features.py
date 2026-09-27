"""
High-Precision Pairwise Feature Extraction Engine (Phase 3)

Computes 16 discriminative pairwise features between a Query (S1) and Candidate (S2/S3):
- Name Similarity: Ratio, Partial Ratio, Token Sort, Token Set, Compressed Match
- Core Token Alignment: Jaccard, Overlap Count, IDF-Weighted Overlap Sum
- Address Alignment: Ratio, Token Set Ratio
- Exact Anchors: Door Match, Postal PIN Match, Phone Match, Door+Street Composite Match
- Cross-Channel Interaction: Synthetic Corruption Detector & Composite Product
"""

from rapidfuzz import fuzz

def compute_pairwise_features(s1_profile, s2_profile, idf_lookup=None, default_idf=12.0):
    """
    s1_profile: (clean_name, compressed, core_tokens, clean_addr, door, pin, phone, addr_keys)
    s2_profile: (clean_name, compressed, core_tokens, clean_addr, door, pin, phone, addr_keys)
    """
    (s1_cn, s1_comp, s1_core, s1_ca, s1_door, s1_pin, s1_phone, s1_ak) = s1_profile
    (s2_cn, s2_comp, s2_core, s2_ca, s2_door, s2_pin, s2_phone, s2_ak) = s2_profile

    # 1. Name Features
    name_ratio = fuzz.ratio(s1_cn, s2_cn)
    name_partial_ratio = fuzz.partial_ratio(s1_cn, s2_cn)
    name_token_sort = fuzz.token_sort_ratio(s1_cn, s2_cn)
    name_token_set = fuzz.token_set_ratio(s1_cn, s2_cn)
    name_comp_ratio = fuzz.ratio(s1_comp, s2_comp) if s1_comp and s2_comp else 0.0

    # 2. Core Token Overlap & Weighted IDF
    common_core = s1_core & s2_core
    union_core = s1_core | s2_core
    core_jaccard = (len(common_core) / len(union_core)) if union_core else 0.0
    core_overlap_count = len(common_core)
    
    core_idf_sum = 0.0
    if idf_lookup and common_core:
        for t in common_core:
            core_idf_sum += idf_lookup.get(t, default_idf)

    # 3. Address Features
    addr_ratio = fuzz.ratio(s1_ca, s2_ca) if s1_ca and s2_ca else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_ca, s2_ca) if s1_ca and s2_ca else 0.0

    # 4. Exact Anchor Comparisons
    # Door match: 1.0 (match), -1.0 (conflict), 0.0 (missing)
    if s1_door and s2_door:
        door_match = 1.0 if s1_door == s2_door else -1.0
    else:
        door_match = 0.0

    # PIN match: 1.0 (match), -1.0 (conflict), 0.0 (missing)
    if s1_pin and s2_pin:
        pin_match = 1.0 if s1_pin == s2_pin else -1.0
    else:
        pin_match = 0.0

    # Phone match
    phone_match = 1.0 if (s1_phone and s2_phone and s1_phone == s2_phone) else 0.0

    # Composite door + street anchor overlap
    ak_overlap = len(set(s1_ak) & set(s2_ak)) if s1_ak and s2_ak else 0.0

    # 5. Non-linear & Cross-Channel Interactions
    # Composite interaction score: product of normalized name & address similarities
    comp_interaction = (name_token_set / 100.0) * (addr_token_set / 100.0)

    # Synthetic corruption flag: Address is an exact anchor match, but name was scrambled
    synthetic_name_corrupt = 1.0 if (ak_overlap > 0 and name_token_set < 45.0) else 0.0

    return [
        name_ratio,
        name_partial_ratio,
        name_token_sort,
        name_token_set,
        name_comp_ratio,
        core_jaccard,
        float(core_overlap_count),
        core_idf_sum,
        addr_ratio,
        addr_token_set,
        door_match,
        pin_match,
        phone_match,
        float(ak_overlap),
        comp_interaction,
        synthetic_name_corrupt
    ]

FEATURE_NAMES = [
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort',
    'name_token_set',
    'name_comp_ratio',
    'core_jaccard',
    'core_overlap_count',
    'core_idf_sum',
    'addr_ratio',
    'addr_token_set',
    'door_match',
    'pin_match',
    'phone_match',
    'ak_overlap',
    'comp_interaction',
    'synthetic_name_corrupt'
]
