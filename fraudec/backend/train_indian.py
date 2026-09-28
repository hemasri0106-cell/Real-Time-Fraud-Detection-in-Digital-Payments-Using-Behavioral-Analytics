"""
train_indian.py
================
Trains per-persona XGBoost + Random Forest fraud models on the real Indian
dataset (`data/user_datasets/user_XX_transactions.csv`, produced by
`generate_user_datasets.py` - real Indian cities, INR-scale amounts, real
chronologically-derived visit-frequency features), replacing the previous
"_v2" models that were trained on `batch_pipeline.py`'s synthetic USD-scale,
`loc_user_XX_N`-coded data (now archived under models/archive_v2/,
results/archive_v2/).

Dataset used
------------
`data/user_datasets/user_{01..10}_transactions.csv` - 5,000 rows/user,
~3-4.6% fraud rate, real Indian home cities per persona, real cumulative
merchant/device/location visit counts (not random noise like v2's).

merchant_id dropped
--------------------
A pre-training leakage/quality check found `merchant_id`'s single-feature
ROC-AUC at 0.90-0.98 across all 10 users - several fraud templates in
generate_user_datasets.py mint one-off merchant IDs (`new_merch_{i}`) that
never appear in normal traffic, so the raw high-cardinality merchant_id
column risks the model memorizing IDs rather than learning a generalizable
signal. `merchant_visit_frequency` already captures "is this a
merchant this persona has seen before" as a smoother, bounded feature, so
merchant_id is dropped from the deployed feature set. `merchant_category`
is kept (also elevated, 0.87-1.00) since it's a coarse, low-cardinality,
genuinely interpretable behavioral signal - not a per-row identifier.

Variant A vs Variant B
-----------------------
Variant A (21 raw columns minus `merchant_id` = 20 columns) is the deployed
feature set: its XGBoost + RF models, ColumnTransformer, and feature-name
list are saved as the "_in" artifacts ml_service.py loads
(`models/{user_id}_in_xgboost.joblib`, `models/{user_id}_in_random_forest.joblib`,
`models/preprocessor_{user_id}_in.joblib`, `models/feature_names_{user_id}_in.json`).

Variant B (Variant A minus `merchant_category` = 19 columns) is an ablation
run for comparison only - its metrics are recorded in
`results/in_variant_comparison_summary.csv` but no model/preprocessor
artifact is saved for it, so it can never accidentally be loaded for
inference.

Split / CV settings
--------------------
- Per user: `sklearn.model_selection.train_test_split(test_size=0.2,
  stratify=y, random_state=SEED)` - stratified 80/20 train/test.
- A NEW `ColumnTransformer` (StandardScaler on numeric columns,
  OneHotEncoder(handle_unknown='ignore') on categorical columns) is fit on
  the train split only, then used to `.transform()` the test split - no
  leakage of test-set statistics into scaling/encoding.
- Hyperparameter selection: `RandomizedSearchCV(n_iter=30, scoring='f1',
  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
  random_state=SEED)` on the train split, for both XGBoost (same
  216-combination parameter space as the original train_xgboost.py:
  n_estimators/max_depth/learning_rate/subsample/colsample_bytree/
  min_child_weight - 30 of 216 combinations sampled) and Random Forest
  (same 8-combination space as batch_pipeline.py: n_estimators/max_depth/
  min_samples_split with class_weight='balanced' - since 8 < n_iter=30,
  sklearn auto-clips to 8 iterations and RF's search is effectively
  exhaustive, same as before). This replaces train_xgboost.py's old
  single-split early-stopping validation with proper 5-fold CV, trades
  XGBoost's exhaustive 216-combination grid for a 30-sample random search
  (~7x fewer fits, faster but not guaranteed to find the same optimum as
  the full grid), and this script fits its own preprocessor per
  user/variant rather than depending on a previously-saved v2
  preprocessor.joblib, so it has no dependency on any v2 artifact.
- The RandomizedSearchCV-selected best estimator is evaluated once, on the
  held-out test split, for the reported precision/recall/f1/roc_auc/pr_auc.

Seed
----
`SEED = 42` everywhere a random_state is accepted (train/test split,
StratifiedKFold, XGBClassifier, RandomForestClassifier) for reproducibility.

Run from the repo root:
    cd fraudec/backend && source venv/bin/activate && cd ../.. && python3 fraudec/backend/train_indian.py
(all paths below are relative to the repo root, matching train_xgboost.py's
and batch_pipeline.py's existing convention - this script does not use
`__file__`-relative paths, so it must be invoked with the repo root as cwd.)
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
import shap

SEED = 42
N_ITER = 30
DATA_DIR = 'data/user_datasets'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

FULL_FEATURE_COLUMNS = [
    'transaction_amount', 'merchant_category', 'merchant_id', 'payment_method',
    'device_id', 'device_type', 'city', 'hour_of_day', 'day_of_week', 'is_weekend',
    'transaction_gap_minutes', 'daily_transaction_count', 'average_amount_last_7_days',
    'std_amount_last_7_days', 'merchant_visit_frequency', 'device_usage_frequency',
    'location_visit_frequency', 'new_device', 'new_location', 'new_merchant',
    'distance_from_last_transaction_km'
]
VARIANT_A_COLUMNS = [c for c in FULL_FEATURE_COLUMNS if c != 'merchant_id']          # deployed (_in artifacts)
VARIANT_B_COLUMNS = [c for c in VARIANT_A_COLUMNS if c != 'merchant_category']       # ablation, metrics only

XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3],
}
RF_PARAM_GRID = {
    'n_estimators': [100, 300],
    'max_depth': [15, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1],
    'max_features': ['sqrt'],
    'class_weight': ['balanced'],
}


def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop')
    preprocessor.fit(X_train)
    return preprocessor


def eval_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'confusion_matrix': cm.tolist(),
        'fraud_in_test': int(y_true.sum()),
    }


def run_xgb(X_train, y_train, X_test, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    gs = RandomizedSearchCV(
        XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=1, eval_metric='logloss'),
        XGB_PARAM_GRID, n_iter=N_ITER, scoring='f1', cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = eval_metrics(y_test, y_pred, y_prob)
    metrics['best_params'] = gs.best_params_
    metrics['cv_f1'] = gs.best_score_
    return best_model, metrics


def run_rf(X_train, y_train, X_test, y_test):
    # RF_PARAM_GRID has only 8 total combinations (2*2*2*1*1*1); since
    # N_ITER=30 > 8, sklearn auto-clips to 8 iterations and this search is
    # effectively exhaustive, identical in coverage to GridSearchCV.
    gs = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        RF_PARAM_GRID, n_iter=N_ITER, scoring='f1', cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = eval_metrics(y_test, y_pred, y_prob)
    metrics['best_params'] = gs.best_params_
    metrics['cv_f1'] = gs.best_score_
    return best_model, metrics


def main():
    users = [f"user_{i:02d}" for i in range(1, 11)]
    summary_rows = []          # variant A + B, XGB + RF, all users -> for the report table
    xgb_a_final_rows = []      # final_xgboost_metrics_in.csv (variant A only, deployed)
    rf_a_final_rows = []       # final_rf_metrics_in.csv (variant A only, deployed)
    shap_top5_rows = []        # top-5 SHAP features per user (variant A XGBoost, deployed model)
    user_runtime_rows = []     # per-user wall time -> printed + saved for the runtime report

    t_start = time.time()
    for uid in users:
        t_user_start = time.time()
        print(f"\n{'='*70}\n{uid}\n{'='*70}")
        df = pd.read_csv(f'{DATA_DIR}/{uid}_transactions.csv')
        y_all = df['label'].values

        for variant_name, cols, deploy in [('A', VARIANT_A_COLUMNS, True), ('B', VARIANT_B_COLUMNS, False)]:
            X = df[cols].copy()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_all, test_size=0.2, stratify=y_all, random_state=SEED
            )
            preprocessor = build_preprocessor(X_train)
            Xtr = preprocessor.transform(X_train)
            Xte = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out().tolist()

            t0 = time.time()
            xgb_model, xgb_metrics = run_xgb(Xtr, y_train, Xte, y_test)
            t1 = time.time()
            rf_model, rf_metrics = run_rf(Xtr, y_train, Xte, y_test)
            t2 = time.time()
            print(f"  variant {variant_name}: XGB {t1-t0:.1f}s (f1={xgb_metrics['f1']:.4f} roc_auc={xgb_metrics['roc_auc']:.4f}) | "
                  f"RF {t2-t1:.1f}s (f1={rf_metrics['f1']:.4f} roc_auc={rf_metrics['roc_auc']:.4f}) | "
                  f"test_fraud={xgb_metrics['fraud_in_test']}")

            for model_name, m in [('XGBoost', xgb_metrics), ('RandomForest', rf_metrics)]:
                summary_rows.append({
                    'user_id': uid, 'variant': variant_name, 'model': model_name,
                    'n_features': len(cols), 'test_fraud_count': m['fraud_in_test'],
                    'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'],
                    'roc_auc': m['roc_auc'], 'pr_auc': m['pr_auc'],
                })

            if deploy:
                # Save the "_in" deployed artifacts (variant A only)
                joblib.dump(xgb_model, f'{MODELS_DIR}/{uid}_in_xgboost.joblib')
                joblib.dump(rf_model, f'{MODELS_DIR}/{uid}_in_random_forest.joblib')
                joblib.dump(preprocessor, f'{MODELS_DIR}/preprocessor_{uid}_in.joblib')
                with open(f'{MODELS_DIR}/feature_names_{uid}_in.json', 'w') as f:
                    json.dump(feature_names, f)

                xgb_a_final_rows.append({
                    'user_id': uid, 'test_precision': xgb_metrics['precision'], 'test_recall': xgb_metrics['recall'],
                    'test_f1': xgb_metrics['f1'], 'test_roc_auc': xgb_metrics['roc_auc'], 'test_pr_auc': xgb_metrics['pr_auc'],
                })
                rf_a_final_rows.append({
                    'user_id': uid, 'test_precision': rf_metrics['precision'], 'test_recall': rf_metrics['recall'],
                    'test_f1': rf_metrics['f1'], 'test_roc_auc': rf_metrics['roc_auc'], 'test_pr_auc': rf_metrics['pr_auc'],
                })

                # Per-user results/feature-importance CSVs (mirrors train_xgboost.py's outputs)
                pd.DataFrame([{**xgb_metrics['best_params'], 'cv_f1': xgb_metrics['cv_f1'],
                               'test_precision': xgb_metrics['precision'], 'test_recall': xgb_metrics['recall'],
                               'test_f1': xgb_metrics['f1'], 'test_roc_auc': xgb_metrics['roc_auc'],
                               'test_pr_auc': xgb_metrics['pr_auc']}]).to_csv(
                    f'{RESULTS_DIR}/{uid}_in_xgboost_results.csv', index=False)
                pd.DataFrame([{**rf_metrics['best_params'], 'cv_f1': rf_metrics['cv_f1'],
                               'test_precision': rf_metrics['precision'], 'test_recall': rf_metrics['recall'],
                               'test_f1': rf_metrics['f1'], 'test_roc_auc': rf_metrics['roc_auc'],
                               'test_pr_auc': rf_metrics['pr_auc']}]).to_csv(
                    f'{RESULTS_DIR}/{uid}_in_random_forest_results.csv', index=False)

                importances = xgb_model.feature_importances_
                fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(
                    'importance', ascending=False)
                fi_df.head(20).to_csv(f'{RESULTS_DIR}/{uid}_in_xgboost_feature_importance.csv', index=False)

                # SHAP top-5 for the deployed XGBoost model, on a sample of the real test set
                explainer = shap.TreeExplainer(xgb_model)
                sample = Xte[:200] if Xte.shape[0] > 200 else Xte
                sv = explainer.shap_values(sample)
                if sv.ndim == 3:
                    sv = sv[:, :, list(xgb_model.classes_).index(1)]
                mean_abs = np.abs(sv).mean(axis=0)
                top5_idx = np.argsort(mean_abs)[::-1][:5]
                clean_names = [feature_names[i].replace('num__', '').replace('cat__', '') for i in top5_idx]
                shap_top5_rows.append({'user_id': uid, 'top5_shap_features': ', '.join(clean_names)})

        user_runtime = time.time() - t_user_start
        user_runtime_rows.append({'user_id': uid, 'runtime_seconds': round(user_runtime, 1)})
        print(f"  {uid} TOTAL RUNTIME (variants A+B, XGB+RF): {user_runtime:.1f}s")

    pd.DataFrame(summary_rows).to_csv(f'{RESULTS_DIR}/in_variant_comparison_summary.csv', index=False)
    pd.DataFrame(xgb_a_final_rows).to_csv(f'{RESULTS_DIR}/final_xgboost_metrics_in.csv', index=False)
    pd.DataFrame(rf_a_final_rows).to_csv(f'{RESULTS_DIR}/final_rf_metrics_in.csv', index=False)
    pd.DataFrame(shap_top5_rows).to_csv(f'{RESULTS_DIR}/in_shap_top5_per_user.csv', index=False)
    pd.DataFrame(user_runtime_rows).to_csv(f'{RESULTS_DIR}/in_runtime_per_user.csv', index=False)

    print("\n\nPER-USER RUNTIME SUMMARY:")
    for r in user_runtime_rows:
        print(f"  {r['user_id']}: {r['runtime_seconds']}s")
    print(f"\nTOTAL WALL TIME: {time.time()-t_start:.1f}s")
    print("DONE")


if __name__ == '__main__':
    main()
