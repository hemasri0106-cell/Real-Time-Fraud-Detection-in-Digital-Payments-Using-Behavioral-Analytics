import os
import joblib
import pandas as pd
from typing import Dict, Any, List
from collections import OrderedDict
import numpy as np
from xgboost import XGBClassifier

from dashboard_data import ALLOWED_USER_IDS, MODEL_FEATURES

# Cache configuration
MAX_CACHED_MODELS = 3
_model_cache = OrderedDict()
_preprocessor_cache = OrderedDict()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models", "archive_v2")
DATA_DIR = os.path.join(BASE_DIR, "data", "user_datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def get_cached_model(user_id: str):
    if user_id not in ALLOWED_USER_IDS:
        raise ValueError("Unsupported demo profile")
    if user_id in _model_cache:
        _model_cache.move_to_end(user_id)
        return _model_cache[user_id]
        
    model_path = os.path.join(MODELS_DIR, f"{user_id}_v2_xgboost.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {user_id} not found.")
        
    model = joblib.load(model_path)
    if not isinstance(model, XGBClassifier):
        raise ValueError("Unexpected model artifact type")
    _model_cache[user_id] = model
    
    if len(_model_cache) > MAX_CACHED_MODELS:
        _model_cache.popitem(last=False)
        
    return model

def get_cached_preprocessor(user_id: str):
    if user_id not in ALLOWED_USER_IDS:
        raise ValueError("Unsupported demo profile")
    if user_id in _preprocessor_cache:
        _preprocessor_cache.move_to_end(user_id)
        return _preprocessor_cache[user_id]
        
    prep_path = os.path.join(MODELS_DIR, f"preprocessor_{user_id}_v2.joblib")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor for {user_id} not found.")
        
    preprocessor = joblib.load(prep_path)
    if list(preprocessor.feature_names_in_) != list(MODEL_FEATURES):
        raise ValueError("Model preprocessor feature schema mismatch")
    _preprocessor_cache[user_id] = preprocessor
    
    if len(_preprocessor_cache) > MAX_CACHED_MODELS:
        _preprocessor_cache.popitem(last=False)
        
    return preprocessor

def get_user_metrics(user_id: str) -> Dict[str, Any]:
    metrics_path = os.path.join(RESULTS_DIR, "final_rf_metrics.csv")
    if not os.path.exists(metrics_path):
        return {}
        
    df = pd.read_csv(metrics_path)
    user_row = df[df["user_id"] == user_id]
    if user_row.empty:
        return {}
        
    return user_row.iloc[0].to_dict()

def get_transactions(user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    data_path = os.path.join(DATA_DIR, f"{user_id}_transactions_v2.csv")
    if not os.path.exists(data_path):
        return []
        
    # In a real app, read from DB. Here we read chunks from CSV.
    df = pd.read_csv(data_path, skiprows=range(1, offset + 1), nrows=limit)
    return df.to_dict(orient="records")

def predict_transaction(user_id: str, transaction_data: dict) -> dict:
    preprocessor = get_cached_preprocessor(user_id)
    model = get_cached_model(user_id)
    
    # We must match the expected schema for the preprocessor.
    # The preprocessor expects a DataFrame with the exact columns.
    # Convert dict to DataFrame.
    df = pd.DataFrame([transaction_data])
    
    # The fitted v2 preprocessor requires all model features in its training order.
    missing_cols = [col for col in MODEL_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError("Transaction is missing required model features")
    df = df[list(MODEL_FEATURES)]
    
    # Apply boolean conversion to int as done during training
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
        
    X_trans = preprocessor.transform(df)
    if model.n_features_in_ != X_trans.shape[1]:
        raise ValueError("Model and preprocessor feature dimensions do not match")
    
    pred = model.predict(X_trans)[0]
    prob = model.predict_proba(X_trans)[0][1]
    
    # Get feature importances for "Why Flagged" / "Key Model Features"
    feature_names = preprocessor.get_feature_names_out().tolist()
    importances = model.feature_importances_
    
    # Sort and get top 5
    indices = np.argsort(importances)[::-1][:5]
    top_features = [{"feature": feature_names[i], "importance": float(importances[i])} for i in indices]
    
    return {
        "prediction": int(pred),
        "fraud_probability": float(prob),
        "top_features": top_features
    }
