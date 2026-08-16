import os
import joblib
import pandas as pd
from typing import Dict, Any, List
from collections import OrderedDict
import numpy as np

# Cache configuration
MAX_CACHED_MODELS = 3
_model_cache = OrderedDict()
_preprocessor_cache = OrderedDict()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "user_datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def get_cached_model(user_id: str):
    if user_id in _model_cache:
        _model_cache.move_to_end(user_id)
        return _model_cache[user_id]
        
    model_path = os.path.join(MODELS_DIR, f"{user_id}_v2_random_forest.joblib")
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
        
    prep_path = os.path.join(MODELS_DIR, f"preprocessor_{user_id}_v2.joblib")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor for {user_id} not found.")
        
    preprocessor = joblib.load(prep_path)
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
    
    # Preprocessor requires these exact columns (excluding label, transaction_id, user_id, timestamp)
    # Ensure they exist or are filled with defaults if missing
    expected_cols = [
        'transaction_amount', 'merchant_category', 'merchant_id', 'payment_method',
        'device_id', 'device_type', 'city', 'hour_of_day', 'day_of_week', 'is_weekend',
        'transaction_gap_minutes', 'daily_transaction_count', 'average_amount_last_7_days',
        'std_amount_last_7_days', 'merchant_visit_frequency', 'device_usage_frequency',
        'location_visit_frequency', 'new_device', 'new_location', 'new_merchant',
        'distance_from_last_transaction_km'
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0 if 'freq' in col or 'amount' in col or 'gap' in col else 'unknown'
            
    df = df[expected_cols]
    
    # Apply boolean conversion to int as done during training
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
        
    X_trans = preprocessor.transform(df)
    
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
