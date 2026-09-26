import re

LEGAL_SUFFIXES = {
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise', 'industries', 'group',
    'services', 'solutions', 'technologies', 'holdings', 'assoc', 'associates',
    'sa', 'sarl', 'gmbh', 'bv', 'plc'
}

STOPWORDS = {'and', '&', 'of', 'the', 'in', 'at', 'for', 'by', 'near', 'opp', 'opposite'}

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d{3,6}\b')

def clean_name(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = RE_PUNCT.sub(' ', text)
    tokens = [w for w in text.split() if w not in LEGAL_SUFFIXES and w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)

def extract_digits(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return set(RE_DIGITS.findall(text))

def clean_address(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\bst\b', 'street', text)
    text = re.sub(r'\brd\b', 'road', text)
    text = re.sub(r'\bave\b', 'avenue', text)
    text = re.sub(r'\bblvd\b', 'boulevard', text)
    text = re.sub(r'\bapt\b', 'apartment', text)
    text = RE_PUNCT.sub(' ', text)
    return ' '.join(text.split())
