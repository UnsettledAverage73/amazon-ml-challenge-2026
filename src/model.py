"""
LightGBM Pairwise Reranker & Probability Model (Phase 3)

Trains on honest multi-pass candidates (positives from ground truth, hard negatives from blocker).
Outputs calibrated match probabilities P(match | S1, S2).
"""

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features import compute_pairwise_features, FEATURE_NAMES

def train_reranker(X_train, y_train, X_val=None, y_val=None, n_estimators=150):
    """
    Trains a LightGBM GBDT binary classifier with calibrated probabilities.
    """
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': n_estimators,
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        ap = average_precision_score(y_val, val_probs)
        print(f"Validation AUC-ROC: {auc:.4f} | Average Precision (PR-AUC): {ap:.4f}")
    else:
        model.fit(X_train, y_train)

    return model

def get_feature_importances(model):
    """
    Returns feature importances sorted descending.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    return [(FEATURE_NAMES[i], int(importances[i])) for i in sorted_idx]
