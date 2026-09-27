"""
Model Training, Cross-Validation, and Threshold Policy for Entity Resolution
Supports LightGBMClassifier with fallback to HistGradientBoostingClassifier.
Uses GroupKFold on source1_entity_id.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.evaluate import evaluate_predictions

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

class EntityResolutionMatcher:
    def __init__(self, n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model = None

    def train(self, X, y, sample_weight=None):
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                boosting_type="gbdt",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.num_leaves,
                random_state=self.random_state
            )
            self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict_proba(X)[:, 1]

def run_group_kfold_cv(df_candidates, feature_cols, target_col="is_match", group_col="source1_entity_id", 
                       all_gt_dict=None, all_s1_ids=None, n_splits=5, thresholds=None):
    """
    Runs 5-Fold GroupKFold Cross-Validation by source1_entity_id.
    Evaluates exact per-S1 Macro F0.5 metric for various probability thresholds.
    """
    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98]

    groups = df_candidates[group_col].values
    gkf = GroupKFold(n_splits=n_splits)

    X = df_candidates[feature_cols].values
    y = df_candidates[target_col].values
    s1_ids = df_candidates[group_col].values
    target_ids = df_candidates["candidate_entity_id"].values

    print(f"Starting GroupKFold Cross-Validation ({n_splits} folds) over {len(df_candidates):,} candidate pairs using {'LightGBM' if HAS_LIGHTGBM else 'HistGradientBoosting'}...")
    
    oof_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": target_ids,
        "is_match": y,
        "prob": np.zeros(len(df_candidates))
    })

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        matcher = EntityResolutionMatcher(n_estimators=300, learning_rate=0.05)
        matcher.train(X_train, y_train)

        probs_val = matcher.predict_proba(X_val)
        oof_df.iloc[val_idx, oof_df.columns.get_loc("prob")] = probs_val
        print(f"  Fold {fold}/{n_splits} complete.")

    # Sweep thresholds on out-of-fold predictions
    print("\nEvaluating Out-of-Fold Threshold Sweep (Macro F0.5 per S1):")
    best_t = 0.50
    best_f05 = -1.0
    sweep_results = []

    for t in thresholds:
        pred_dict = {}
        # Filter pairs where probability >= t
        matched_rows = oof_df[oof_df["prob"] >= t]
        for row in matched_rows.itertuples(index=False):
            s1 = row.source1_entity_id
            target = row.candidate_entity_id
            if s1 not in pred_dict:
                pred_dict[s1] = set()
            pred_dict[s1].add(target)

        stats = evaluate_predictions(pred_dict, all_gt_dict, all_s1_ids)
        stats["threshold"] = t
        sweep_results.append(stats)

        print(f"  Threshold {t:.2f} -> Macro F0.5: {stats['macro_f05']:.4f} | Prec: {stats['macro_precision']:.4f} | Rec: {stats['macro_recall']:.4f} | Singleton FP: {stats['singleton_fp_rate']:.4f} | Mean Matches/S1: {stats['mean_pred_matches']:.2f}")

        if stats['macro_f05'] > best_f05:
            best_f05 = stats['macro_f05']
            best_t = t

    return best_t, best_f05, sweep_results, oof_df
