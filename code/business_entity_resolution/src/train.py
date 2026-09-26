import os
import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from utils import clean_name, clean_address, extract_digits
from blocking import FastInvertedIndex
from features import compute_features, FEATURE_NAMES

def evaluate_macro_f05(ground_truth_dict: dict, predictions_dict: dict, s1_ids: list) -> float:
    scores = []
    for s1_id in s1_ids:
        true_matches = ground_truth_dict.get(s1_id, set())
        pred_matches = predictions_dict.get(s1_id, set())
        if not true_matches and not pred_matches:
            scores.append(1.0)
        elif not true_matches and pred_matches:
            scores.append(0.0)
        elif true_matches and not pred_matches:
            scores.append(0.0)
        else:
            tp = len(true_matches & pred_matches)
            prec = tp / len(pred_matches)
            rec = tp / len(true_matches)
            denom = 0.25 * prec + rec
            f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
            scores.append(f05)
    return float(np.mean(scores))

def main(data_dir: str = 'dataset', output_dir: str = 'models', n_train: int = 50000):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, 'train')

    print(f"Loading S1 train data ({n_train} samples)...")
    df_s1 = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', nrows=n_train)
    df_gt = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t')

    df_s1['clean_name'] = df_s1['business_name'].apply(clean_name)
    df_s1['clean_addr'] = df_s1['business_address'].apply(clean_address)
    df_s1['digits'] = df_s1['business_address'].apply(extract_digits)

    gt_map = {}
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).strip()
        if matches and matches != 'nan':
            gt_map[s1_id] = set(matches.split(','))
        else:
            gt_map[s1_id] = set()

    print("Loading target records (S2 & S3)...")
    df_s2 = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', nrows=n_train * 2)
    df_s3 = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', nrows=n_train * 2)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    df_targets['clean_name'] = df_targets['business_name'].apply(clean_name)
    df_targets['clean_addr'] = df_targets['business_address'].apply(clean_address)
    df_targets['digits'] = df_targets['business_address'].apply(extract_digits)

    indexer = FastInvertedIndex(max_df=1500)
    indexer.add_records(df_targets)

    print("Generating candidate pairs & training features...")
    X, y = [], []
    for _, row in tqdm(df_s1.iterrows(), total=len(df_s1), desc="Building Training Pairs"):
        s1_id = row['entity_id']
        s1_cname = row['clean_name']
        s1_caddr = row['clean_addr']
        s1_digits = row['digits']
        country = str(row['country']).strip()
        true_matches = gt_map.get(s1_id, set())

        cands = indexer.get_candidates(s1_cname, country, top_k=15)
        for cid in cands:
            c_info = indexer.records[cid]
            feats = compute_features(
                s1_cname, s1_caddr, s1_digits,
                c_info[0], c_info[1], c_info[2], cid
            )
            label = 1 if cid in true_matches else 0
            X.append(feats)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape}, Positives: {np.sum(y)}, Positive rate: {np.mean(y):.4f}")

    # Train LightGBM model
    print("Training LightGBM model...")
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y)

    model_path = os.path.join(output_dir, 'lgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}!")

if __name__ == '__main__':
    main()
