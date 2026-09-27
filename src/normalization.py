"""
Zero-Loss Multilingual Ingestion & Normalization Engine (Phase 1)
Supports:
- Indian transliterations (Hindi Devanagari, Tamil, Bengali, Telugu, Kannada)
- French accented text and postal terms (Nouvelle-Aquitaine, Dunkerque, CEDEX)
- US standard street address and corporate entity normalization
- Domain stem de-obfuscation and compressed matching
"""

import re
import unicodedata
import unidecode

# ---------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------
DOMAINS = re.compile(
    r'(https?://)?(www\.)?([a-zA-Z0-9\-]+)(\.com|\.in|\.org|\.net|\.co|\.fr|\.io|\.biz|\.info|\.edu|\.gov)(/[^\s]*)?',
    re.IGNORECASE
)
RE_PUNCT = re.compile(r'[^a-zA-Z0-9\s]')
RE_DOOR = re.compile(r'\b([A-Za-z]{0,3})[-/]?0*(\d{1,5})([A-Za-z]?)\b')
RE_PIN = re.compile(r'\b(\d{5,6})\b')
RE_PHONE = re.compile(r'\b\d{7,10}\b')
RE_CEDEX = re.compile(r'\bcedex\s*\d*\b', re.IGNORECASE)

# ---------------------------------------------------------
# Dictionaries & Gazetteers
# ---------------------------------------------------------
LEGAL_SUFFIXES = {
    # English / US / International
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise', 'industries', 'group',
    'services', 'solutions', 'technologies', 'holdings', 'assoc', 'associates',
    'plc', 'bv', 'gmbh',
    # French
    'sa', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'gie', 'sca', 'scs', 'fils'
}

GENERIC_TERMS = {
    # English / Universal
    'store', 'stores', 'shop', 'mart', 'market', 'supermarket', 'agency', 'agencies',
    'traders', 'trading', 'hotel', 'restaurant', 'cafe', 'bazaar', 'jewellers',
    'jewellery', 'textiles', 'pharmacy', 'chemist', 'auto', 'garage', 'consultants',
    'consultancy', 'logistics', 'transports', 'centre', 'center', 'foods', 'retail',
    'wholesale', 'supply', 'supplies', 'commercial',
    # French
    'agence', 'agences', 'societe', 'societes', 'boulangerie', 'coiffure', 'boucherie',
    'epicerie', 'atelier', 'ateliers', 'commerce', 'commerces', 'batiment', 'menuiserie',
    'plomberie', 'electricite', 'tabac', 'ecole', 'club', 'amicale'
}

# Phonetic normalization for Indian transliterations (from Devanagari/Tamil via unidecode)
PHONETIC_LEGAL_SUBS = [
    (re.compile(r'\b(praaivett\s+limittedd|praivet\s+limited|praaivet\s+limiteed)\b'), 'pvt ltd'),
    (re.compile(r'\b(elelpii|elelpi)\b'), 'llp'),
    (re.compile(r'\b(elelsii|elelsi)\b'), 'llc'),
    (re.compile(r'\b(limittedd|limiteed)\b'), 'ltd'),
    (re.compile(r'\b(praaivett|praivet)\b'), 'pvt'),
    (re.compile(r'\b(innnvesttmenntts|investmennts|invesstment)\b'), 'investments'),
    (re.compile(r'\b(pronprttiij|prapartiij|properrties)\b'), 'properties'),
    (re.compile(r'\b(vencrs|vennchars|venchers)\b'), 'ventures'),
    (re.compile(r'\b(ennttrrpraaizej|entrpraij)\b'), 'enterprises'),
    (re.compile(r'\b(saolyuushnns|solyushans)\b'), 'solutions'),
    (re.compile(r'\b(ttteknnolaojiij|teknolaji)\b'), 'technologies'),
]

# Standard corporate normalization
CORP_CANONICAL = [
    (re.compile(r'\bprivate\s+limited\b'), 'pvt ltd'),
    (re.compile(r'\bincorporated\b'), 'inc'),
    (re.compile(r'\bcorporation\b'), 'corp'),
    (re.compile(r'\bcompany\b'), 'co'),
    (re.compile(r'\blimited\b'), 'ltd'),
]

# Multilingual Address Normalization (US, India, France)
ADDR_ABBREVIATIONS = [
    # English / US / India
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
    # French
    (re.compile(r'\br\b|\brue\b'), 'rue'),
    (re.compile(r'\bbd\b|\bbld\b'), 'boulevard'),
    (re.compile(r'\bav\b'), 'avenue'),
    (re.compile(r'\bimp\b'), 'impasse'),
    (re.compile(r'\ball\b'), 'allee'),
    (re.compile(r'\brte\b'), 'route'),
    (re.compile(r'\bch\b'), 'chemin'),
]

def transliterate_and_unaccent(text: str) -> str:
    """
    Preserves information by mapping non-ASCII scripts (Devanagari, Tamil, etc.)
    and accented Latin (é, è, ç, ô) into phonetic Latin text.
    """
    if not text:
        return ''
    # Fast unidecode conversion handles both Indic scripts and European accents
    return unidecode.unidecode(text)

def clean_name(raw_name: str):
    """
    Cleans and canonicalizes business names.
    Returns:
        cleaned: str (normalized token string)
        compressed: str (no-space string for domain/composite matching)
        core_tokens: set (distinctive tokens excluding legal/generic terms)
        tokens: list of all normalized tokens
    """
    if not raw_name:
        return '', '', set(), []

    # 1. Transliterate and unaccent
    text = transliterate_and_unaccent(raw_name)

    # 2. Extract domain stem if URL is present
    match = DOMAINS.search(text)
    domain_stem = ''
    if match:
        domain_stem = match.group(3).lower()
        text = DOMAINS.sub(' ' + domain_stem + ' ', text)

    # 3. Apply phonetic legal substitutions for Indian transliterations
    text = text.lower()
    for pat, repl in PHONETIC_LEGAL_SUBS:
        text = pat.sub(repl, text)

    # 4. Canonicalize standard corporate terms
    for pat, repl in CORP_CANONICAL:
        text = pat.sub(repl, text)

    # 5. Remove punctuation
    text = RE_PUNCT.sub(' ', text)

    # 6. Tokenize & filter
    raw_tokens = text.split()
    tokens = [w for w in raw_tokens if len(w) > 1]
    
    cleaned = ' '.join(tokens)
    compressed = ''.join(tokens)
    core_tokens = {w for w in tokens if w not in LEGAL_SUFFIXES and w not in GENERIC_TERMS}

    # If domain stem was isolated, add its compressed form
    if domain_stem and not compressed:
        compressed = domain_stem

    return cleaned, compressed, core_tokens, tokens

def clean_addr(raw_addr: str) -> str:
    """
    Cleans and canonicalizes addresses across US, India, and France.
    """
    if not raw_addr:
        return ''

    # 1. Transliterate & unaccent
    text = transliterate_and_unaccent(raw_addr)

    # 2. Strip French CEDEX routing codes
    text = RE_CEDEX.sub(' ', text.lower())

    # 3. Standardize address abbreviations
    for pattern, repl in ADDR_ABBREVIATIONS:
        text = pattern.sub(repl, text)

    # 4. Remove punctuation
    clean = ' '.join(RE_PUNCT.sub(' ', text).split())
    return clean

def extract_anchors(raw_addr: str):
    """
    Extracts high-precision anchor features: door/unit, postal PIN/ZIP, phone number.
    """
    if not raw_addr:
        return '', '', ''

    text = transliterate_and_unaccent(raw_addr)
    text = RE_CEDEX.sub(' ', text)

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
