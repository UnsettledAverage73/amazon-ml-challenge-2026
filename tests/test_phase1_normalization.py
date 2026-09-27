"""
Verification Suite for Phase 1: Zero-Loss Multilingual Ingestion Engine
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.normalization import clean_name, clean_addr, extract_anchors
from rapidfuzz import fuzz

def test_indian_transliteration():
    test_cases = [
        ("Raj Investments LLP", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"),
        ("Ss Food Private Limited", "एसएस फूड प्राइवेट लिमिटेड"),
        ("Aditya Properties LLP", "आदित्य प्रॉपर्टीज एलएलपी"),
        ("Red Ventures Private Limited", "रेड वेंचर्स प्राइवेट लिमिटेड"),
    ]
    print("\n--- 1. Testing Indian Regional Scripts Transliteration ---")
    for s1, s2 in test_cases:
        c1, comp1, core1, tok1 = clean_name(s1)
        c2, comp2, core2, tok2 = clean_name(s2)
        sim = fuzz.token_set_ratio(c1, c2)
        print(f"S1: {s1}  --> Clean: '{c1}'")
        print(f"S2: {s2}  --> Clean: '{c2}'")
        print(f"Core S1: {core1} | Core S2: {core2}")
        print(f"Fuzz TokenSet: {sim:.1f}%\n")
        assert len(c2) > 0, f"Transliteration failed for {s2}: output is empty!"
        assert sim >= 50.0, f"Similarity too low ({sim}%) for {s1} vs {s2}"

def test_french_accents_and_terms():
    test_cases = [
        ("SCI Ptit Àmicale", "sci ptit amicale"),
        ("NO. 5 ALLÉE DES HÊTRES", "allee des hetres"),
        ("Thermal & Fils SASU", "thermal"),
        ("Bordeaux, Nouvelle-Aquitaine, CEDEX 02", "bordeaux nouvelle aquitaine"),
    ]
    print("--- 2. Testing French Accents and Address Terms ---")
    for raw, expected_sub in test_cases:
        cn, comp, core, tok = clean_name(raw)
        ca = clean_addr(raw)
        print(f"Raw: '{raw}'")
        print(f"  clean_name: '{cn}' | core: {core}")
        print(f"  clean_addr: '{ca}'\n")
        assert any(expected_sub.split()[0] in out for out in [cn, ca]), f"Failed on {raw}"

def test_domains_and_compressed():
    print("--- 3. Testing Domain Stems & Compressed Matching ---")
    raw_s1 = "Orelee's Barbershop"
    raw_s2 = "http://www.oreleesbarbershop.com"
    c1, comp1, core1, _ = clean_name(raw_s1)
    c2, comp2, core2, _ = clean_name(raw_s2)
    sim = fuzz.ratio(comp1, comp2)
    print(f"S1: '{raw_s1}' -> comp: '{comp1}'")
    print(f"S2: '{raw_s2}' -> comp: '{comp2}'")
    print(f"Compressed exact/fuzzy match: {sim:.1f}%\n")
    assert sim >= 90.0, f"Domain compressed match failed: {sim}%"

def test_anchors():
    print("--- 4. Testing Anchor Extraction ---")
    addrs = [
        ("KH NO. -570/13, NEW DELHI, WEST DELHI, 110041", ("570-13", "110041")),
        ("1795 Westchester Drive, High Point, NC 27265", ("1795", "27265")),
        ("20 Rue Parmentier, Dunkerque, 59140", ("20", "59140")),
    ]
    for addr, (exp_door, exp_pin) in addrs:
        door, pin, phone = extract_anchors(addr)
        print(f"Addr: '{addr}' -> Door: '{door}', PIN: '{pin}'")
        assert pin == exp_pin, f"Expected PIN {exp_pin}, got {pin}"
        assert len(door) > 0, f"Expected door anchor for {addr}"

if __name__ == '__main__':
    test_indian_transliteration()
    test_french_accents_and_terms()
    test_domains_and_compressed()
    test_anchors()
    print("\n[SUCCESS] Phase 1 Normalization Engine PASSED all validation tests!")
