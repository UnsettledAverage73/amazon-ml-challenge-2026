import os
import gc
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import clean_name, clean_address, extract_digits
from blocking import FastInvertedIndex
from features import compute_features

def run_country_inference(country: str, df_s1_country: pd.DataFrame, df_targets_country: pd.DataFrame,
                          clf, threshold: float, f_match, f_cand):
    if len(df_s1_country) == 0:
        return

    print(f"\n[{country}] Building inverted index for {len(df_targets_country)} target records...")
    df_targets_country = df_targets_country.copy()
    df_targets_country['clean_name'] = df_targets_country['business_name'].apply(clean_name)
    df_targets_country['clean_addr'] = df_targets_country['business_address'].apply(clean_address)
    df_targets_country['digits'] = df_targets_country['business_address'].apply(extract_digits)

    indexer = FastInvertedIndex(max_df=1200)
    indexer.add_records(df_targets_country)
    del df_targets_country
    gc.collect()

    print(f"[{country}] Scoring {len(df_s1_country)} Source 1 records...")
    df_s1_country = df_s1_country.copy()
    df_s1_country['clean_name'] = df_s1_country['business_name'].apply(clean_name)
    df_s1_country['clean_addr'] = df_s1_country['business_address'].apply(clean_address)
    df_s1_country['digits'] = df_s1_country['business_address'].apply(extract_digits)

    for _, row in tqdm(df_s1_country.iterrows(), total=len(df_s1_country), desc=f"Inference [{country}]"):
        s1_id = row['entity_id']
        s1_cname = row['clean_name']
        s1_caddr = row['clean_addr']
        s1_digits = row['digits']

        cands = indexer.get_candidates(s1_cname, country, top_k=25)
        f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

        matched_ids = []
        if cands:
            pair_feats = []
            for cid in cands:
                c_info = indexer.records[cid]
                pair_feats.append(compute_features(
                    s1_cname, s1_caddr, s1_digits,
                    c_info[0], c_info[1], c_info[2], cid
                ))
            probs = clf.predict_proba(np.array(pair_feats))[:, 1]
            for cid, prob in zip(cands, probs):
                if prob >= threshold:
                    matched_ids.append(cid)

        f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")

    del indexer
    gc.collect()

def main(data_dir: str = 'dataset', model_path: str = 'models/lgb_model.pkl',
         output_dir: str = 'output', threshold: float = 0.65, test_limit: int = None):
    os.makedirs(output_dir, exist_ok=True)
    test_dir = os.path.join(data_dir, 'test')

    print(f"Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    print("Loading test dataset files...")
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')

    df_s1 = pd.read_csv(s1_path, sep='\t', nrows=test_limit)
    df_s2 = pd.read_csv(s2_path, sep='\t')
    df_s3 = pd.read_csv(s3_path, sep='\t')
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    countries = list(df_s1['country'].dropna().unique())
    print(f"Target countries in test set: {countries}")

    match_path = os.path.join(output_dir, 'matching_results.tsv')
    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    with open(match_path, 'w', encoding='utf-8') as f_match, \
         open(cand_path, 'w', encoding='utf-8') as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in countries:
            s1_sub = df_s1[df_s1['country'] == country]
            tgt_sub = df_targets[df_targets['country'] == country]
            run_country_inference(country, s1_sub, tgt_sub, clf, threshold, f_match, f_cand)

    print("\nAll country batches completed! Output files written.")

if __name__ == '__main__':
    main()
