import os
import re
import glob
import joblib
import pandas as pd
from typing import Dict, Any, List
from collections import OrderedDict
from datetime import datetime, timezone
import numpy as np
import shap

# Cache configuration
MAX_CACHED_MODELS = 3
_model_cache = OrderedDict()
_preprocessor_cache = OrderedDict()
_explainer_cache = OrderedDict()
_historical_data_cache = OrderedDict()
_replay_cursor: Dict[str, int] = {}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "user_datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# The 20 raw columns the preprocessor was fit on (excludes label, transaction_id,
# user_id, timestamp - and, as of the Indian-dataset retrain, merchant_id: dropped
# as a high-cardinality memorization risk, see train_indian.py's docstring).
# Shared by predict_transaction and the historical replay path so both build the
# exact same schema.
EXPECTED_FEATURE_COLUMNS = [
    'transaction_amount', 'merchant_category', 'payment_method',
    'device_id', 'device_type', 'city', 'hour_of_day', 'day_of_week', 'is_weekend',
    'transaction_gap_minutes', 'daily_transaction_count', 'average_amount_last_7_days',
    'std_amount_last_7_days', 'merchant_visit_frequency', 'device_usage_frequency',
    'location_visit_frequency', 'new_device', 'new_location', 'new_merchant',
    'distance_from_last_transaction_km'
]

_VALID_USER_ID_PATTERN = re.compile(r"^(.+)_in_xgboost\.joblib$")

def get_valid_user_ids() -> List[str]:
    """The real whitelist of personas this app can actually score - derived
    from which trained model artifacts exist on disk, not a hand-maintained
    list that can drift out of sync with models/."""
    paths = glob.glob(os.path.join(MODELS_DIR, "*_in_xgboost.joblib"))
    ids = []
    for p in paths:
        m = _VALID_USER_ID_PATTERN.match(os.path.basename(p))
        if m:
            ids.append(m.group(1))
    return sorted(ids)

def get_cached_model(user_id: str):
    if user_id in _model_cache:
        _model_cache.move_to_end(user_id)
        return _model_cache[user_id]

    model_path = os.path.join(MODELS_DIR, f"{user_id}_in_xgboost.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {user_id} not found.")

    model = joblib.load(model_path)
    _model_cache[user_id] = model

    if len(_model_cache) > MAX_CACHED_MODELS:
        _model_cache.popitem(last=False)

    return model

def get_cached_preprocessor(user_id: str):
    if user_id in _preprocessor_cache:
        _preprocessor_cache.move_to_end(user_id)
        return _preprocessor_cache[user_id]

    prep_path = os.path.join(MODELS_DIR, f"preprocessor_{user_id}_in.joblib")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor for {user_id} not found.")
        
    preprocessor = joblib.load(prep_path)
    _preprocessor_cache[user_id] = preprocessor
    
    if len(_preprocessor_cache) > MAX_CACHED_MODELS:
        _preprocessor_cache.popitem(last=False)

    return preprocessor

def get_cached_explainer(user_id: str):
    if user_id in _explainer_cache:
        _explainer_cache.move_to_end(user_id)
        return _explainer_cache[user_id]

    model = get_cached_model(user_id)
    explainer = shap.TreeExplainer(model)
    _explainer_cache[user_id] = explainer

    if len(_explainer_cache) > MAX_CACHED_MODELS:
        _explainer_cache.popitem(last=False)

    return explainer

def get_global_feature_importances(user_id: str, top_n: int = 6) -> List[Dict[str, Any]]:
    """The XGBoost model's real global feature_importances_ (fit-time, not
    per-transaction) - the correct real source for an aggregate "what does this
    persona's model weigh most" panel, as opposed to the per-instance SHAP values
    used on the sandbox and live monitor pages."""
    model = get_cached_model(user_id)
    preprocessor = get_cached_preprocessor(user_id)
    names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1][:top_n]
    total = float(importances[indices].sum()) or 1.0
    return [
        {
            "feature": names[i].replace("num__", "").replace("cat__", ""),
            "importance": float(importances[i]),
            "share": float(importances[i]) / total,
        }
        for i in indices
    ]

def get_model_last_modified(user_id: str) -> str:
    """Real last-modified timestamp of the model artifact on disk - i.e. when
    it was actually last retrained/saved, not a fabricated 'Today, 04:00 UTC'."""
    model_path = os.path.join(MODELS_DIR, f"{user_id}_in_xgboost.joblib")
    if not os.path.exists(model_path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc).isoformat()

def get_user_metrics(user_id: str) -> Dict[str, Any]:
    metrics_path = os.path.join(RESULTS_DIR, "final_xgboost_metrics_in.csv")
    if not os.path.exists(metrics_path):
        return {}

    df = pd.read_csv(metrics_path)
    user_row = df[df["user_id"] == user_id]
    if user_row.empty:
        return {}

    result = user_row.iloc[0].to_dict()

    # Real trained-sample count: the final model is retrained on train+val
    # (the first 90% of the chronological split, test held out) - see
    # train_xgboost.py's load_and_preprocess / final_training_and_eval.
    try:
        historical_df = _get_historical_df(user_id)
        result["trained_samples"] = int(len(historical_df) * 0.9)
    except FileNotFoundError:
        pass

    return result

def get_average_roc_auc() -> float:
    """Real mean test ROC-AUC across all trained personas - used for the
    pre-login telemetry bar, where no single user_id is selected yet."""
    metrics_path = os.path.join(RESULTS_DIR, "final_xgboost_metrics_in.csv")
    if not os.path.exists(metrics_path):
        return None
    df = pd.read_csv(metrics_path)
    if df.empty or "test_roc_auc" not in df.columns:
        return None
    return float(df["test_roc_auc"].mean())

def get_transactions(user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    data_path = os.path.join(DATA_DIR, f"{user_id}_transactions.csv")
    if not os.path.exists(data_path):
        return []

    # In a real app, read from DB. Here we read chunks from CSV.
    df = pd.read_csv(data_path, skiprows=range(1, offset + 1), nrows=limit)
    return df.to_dict(orient="records")

def _get_historical_df(user_id: str) -> pd.DataFrame:
    if user_id in _historical_data_cache:
        _historical_data_cache.move_to_end(user_id)
        return _historical_data_cache[user_id]

    data_path = os.path.join(DATA_DIR, f"{user_id}_transactions.csv")
    df = pd.read_csv(data_path)
    _historical_data_cache[user_id] = df

    if len(_historical_data_cache) > MAX_CACHED_MODELS:
        _historical_data_cache.popitem(last=False)

    return df

def get_next_historical_transaction(user_id: str) -> dict:
    """Pull the next row (in original order) from this persona's real historical
    dataset, for the Live Monitor's replay feed. Wraps around at the end so the
    demo can run indefinitely. These are real recorded transactions/features -
    only the pacing of "arrival" is simulated, not the data itself."""
    df = _get_historical_df(user_id)
    idx = _replay_cursor.get(user_id, 0) % len(df)
    row = df.iloc[idx].to_dict()
    _replay_cursor[user_id] = idx + 1
    return row

def predict_transaction(user_id: str, transaction_data: dict) -> dict:
    preprocessor = get_cached_preprocessor(user_id)
    model = get_cached_model(user_id)
    
    # We must match the expected schema for the preprocessor.
    # The preprocessor expects a DataFrame with the exact columns.
    # Convert dict to DataFrame.
    df = pd.DataFrame([transaction_data])
    
    # Preprocessor requires these exact columns (excluding label, transaction_id, user_id, timestamp)
    # Ensure they exist or are filled with defaults if missing
    for col in EXPECTED_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if 'freq' in col or 'amount' in col or 'gap' in col else 'unknown'

    df = df[EXPECTED_FEATURE_COLUMNS]
    
    # Apply boolean conversion to int as done during training
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
        
    X_trans = preprocessor.transform(df)

    pred = model.predict(X_trans)[0]
    prob = model.predict_proba(X_trans)[0][1]

    # Real per-instance SHAP values for this specific transaction (not global
    # feature_importances_, which is the same for every request).
    #
    # shap.TreeExplainer's output shape depends on the underlying model:
    # sklearn's RandomForestClassifier returns a 3D array (1, n_features, n_classes)
    # - one set of contributions per class - so the fraud class (1) must be
    # selected explicitly. XGBoost's XGBClassifier (binary) instead returns a
    # single 2D array (1, n_features): there's only one output (log-odds
    # margin for the positive class), so the whole array already IS the
    # fraud-class contribution - no class dimension to index into. Verified:
    # sigmoid(shap_values.sum() + expected_value) == predict_proba[:, 1].
    explainer = get_cached_explainer(user_id)
    shap_values = explainer.shap_values(X_trans)
    if shap_values.ndim == 3:
        fraud_class_index = list(model.classes_).index(1)
        instance_shap = shap_values[0, :, fraud_class_index]
    else:
        instance_shap = shap_values[0, :]

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Sort and get top 5 by absolute contribution to this transaction
    indices = np.argsort(np.abs(instance_shap))[::-1][:5]
    top_features = [
        {
            "feature": feature_names[i].replace("num__", "").replace("cat__", ""),
            "shap_value": float(instance_shap[i]),
        }
        for i in indices
    ]

    return {
        "prediction": int(pred),
        "fraud_probability": float(prob),
        "top_features": top_features,
        "feature_vector_dim": int(X_trans.shape[1]),
    }
