from collections import defaultdict
from tqdm import tqdm

class FastInvertedIndex:
    def __init__(self, max_df: int = 1500):
        self.index = defaultdict(list)
        self.max_df = max_df
        # eid -> (clean_name, clean_addr, digits, country)
        self.records = {}

    def add_records(self, df):
        print(f"Indexing {len(df)} target records...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Inverted Indexing"):
            eid = row['entity_id']
            c_name = row['clean_name']
            c_addr = row['clean_addr']
            digits = row['digits']
            country = str(row['country']).strip()
            self.records[eid] = (c_name, c_addr, digits, country)

            tokens = set(c_name.split())
            for tok in tokens:
                if len(tok) >= 3:
                    self.index[(country, tok)].append(eid)

        # Prune very frequent tokens to keep candidate generation high precision & fast
        pruned = 0
        for k in list(self.index.keys()):
            if len(self.index[k]) > self.max_df:
                del self.index[k]
                pruned += 1
        print(f"Index complete: {len(self.index)} active keys, {pruned} pruned high-frequency keys.")

    def get_candidates(self, query_name: str, query_country: str, top_k: int = 20) -> list:
        counts = defaultdict(int)
        tokens = set(query_name.split())
        for tok in tokens:
            if len(tok) >= 3:
                key = (query_country, tok)
                if key in self.index:
                    for cid in self.index[key]:
                        counts[cid] += 1
        if not counts:
            return []
        sorted_cands = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_cands[:top_k]]
