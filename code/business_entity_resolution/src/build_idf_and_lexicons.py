import os
import re
import json
import math
import unicodedata
from collections import Counter

# ---------------------------------------------------------
# UNICODE NORMALIZATION & ADVANCED LEXICONS
# ---------------------------------------------------------
DOMAINS = re.compile(r'(\.com|\.in|\.org|\.net|\.co|\.fr|\.io|\.biz|\.info|www\.)', re.IGNORECASE)
RE_PUNCT = re.compile(r'[^a-zA-Z0-9\s]')
RE_CEDEX = re.compile(r'\bcedex\s*\d*\b', re.IGNORECASE)

LEGAL_SUFFIXES = {
    # US & UK
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise', 'industries', 'group',
    'services', 'solutions', 'technologies', 'holdings', 'assoc', 'associates',
    'plc', 'bv', 'gmbh',
    # France
    'sa', 'sarl', 'sas', 'eurl', 'sci', 'snc', 'gie', 'sca', 'scs', 'selarl'
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

def clean_name_tokens(text: str):
    if not text:
        return []
    text = strip_accents(text)
    text = DOMAINS.sub(' ', text)
    text = RE_PUNCT.sub(' ', text.lower())
    return [w for w in text.split() if w not in LEGAL_SUFFIXES and len(w) > 1]

def main():
    print("=" * 65)
    print("⚡ PRECOMPUTING CORPUS TOKEN IDF WEIGHTS & ADVANCED LEXICONS")
    print("=" * 65)
    os.makedirs('models', exist_ok=True)

    df_counts = Counter()
    total_docs = 0

    print("Scanning train_source1.tsv for token document frequencies...")
    with open('dataset/train/train_source1.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2:
                tokens = set(clean_name_tokens(parts[1]))
                for t in tokens:
                    df_counts[t] += 1
                total_docs += 1

    print(f"Total training documents processed: {total_docs}")
    print(f"Unique tokens observed: {len(df_counts)}")

    # Filter tokens that appear at least 2 times to avoid noise
    idf_dict = {}
    default_idf = math.log(1.0 + total_docs) # Max IDF for unseen rare words

    for token, count in df_counts.items():
        if count >= 2:
            idf = math.log(1.0 + (total_docs / (1.0 + count)))
            idf_dict[token] = round(idf, 3)

    print(f"Retained {len(idf_dict)} high-value tokens in vocabulary.")
    output_path = 'models/token_idf.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_docs': total_docs,
            'default_idf': round(default_idf, 3),
            'idfs': idf_dict
        }, f)

    print(f"IDF dictionary saved to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

if __name__ == '__main__':
    main()
