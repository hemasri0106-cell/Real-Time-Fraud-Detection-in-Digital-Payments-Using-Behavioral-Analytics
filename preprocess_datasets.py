"""
Preprocess the 10 per-user synthetic transaction datasets into a single
feature matrix ready for Isolation Forest training.

Usage:
    python3 preprocess_datasets.py
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "data/user_datasets"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# --- Column classification -------------------------------------------------

REFERENCE_COLS = ["transaction_id", "user_id", "timestamp", "merchant_id", "device_id"]
CATEGORICAL_COLS = ["merchant_category", "payment_method", "device_type", "city", "day_of_week"]
NUMERICAL_COLS = [
    "transaction_amount",
    "hour_of_day",
    "transaction_gap_minutes",
    "daily_transaction_count",
    "average_amount_last_7_days",
    "std_amount_last_7_days",
    "merchant_visit_frequency",
    "device_usage_frequency",
    "location_visit_frequency",
    "distance_from_last_transaction_km",
]
BINARY_COLS = ["is_weekend", "new_device", "new_location", "new_merchant"]
TARGET_COL = "label"


def load_combined() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "user_*_transactions.csv")))
    if not paths:
        raise FileNotFoundError(f"No user_*_transactions.csv files found in {DATA_DIR}")
    frames = [pd.read_csv(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined


def build_features(df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler) -> pd.DataFrame:
    """Transform (not fit) a raw slice into the final feature matrix using
    already-fitted transformers."""
    onehot_array = ohe.transform(df[CATEGORICAL_COLS])
    onehot_cols = ohe.get_feature_names_out(CATEGORICAL_COLS)
    onehot_df = pd.DataFrame(onehot_array, columns=onehot_cols, index=df.index)

    scaled_array = scaler.transform(df[NUMERICAL_COLS])
    scaled_df = pd.DataFrame(scaled_array, columns=NUMERICAL_COLS, index=df.index)

    binary_df = df[BINARY_COLS].astype(float)

    return pd.concat([scaled_df, onehot_df, binary_df], axis=1)


def check_clean(X: pd.DataFrame, name: str) -> None:
    if X.isna().any().any():
        raise ValueError(f"NaNs present in {name} after preprocessing")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Infs present in {name} after preprocessing")


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_combined()

    expected_cols = set(REFERENCE_COLS + CATEGORICAL_COLS + NUMERICAL_COLS + BINARY_COLS + [TARGET_COL])
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns not classified in the schema: {sorted(extra)}")

    # --- Split BEFORE fitting -------------------------------------------
    # Stratify jointly on (user_id, label) so each user's rows are split
    # ~80/20 individually (proportional representation per user) AND the
    # overall ~3.8% fraud rate is preserved in both splits, rather than
    # letting any single user's rows land entirely in one split.
    strat_key = df["user_id"].astype(str) + "_" + df[TARGET_COL].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=strat_key,
        random_state=42,
    )

    # --- Fit ONLY on train ------------------------------------------------
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(train_df[CATEGORICAL_COLS])

    scaler = StandardScaler()
    scaler.fit(train_df[NUMERICAL_COLS])

    onehot_cols = ohe.get_feature_names_out(CATEGORICAL_COLS)

    X_train = build_features(train_df, ohe, scaler)
    X_test = build_features(test_df, ohe, scaler)

    y_train = train_df[[TARGET_COL]].copy()
    y_test = test_df[[TARGET_COL]].copy()

    reference_train = train_df[["transaction_id", "user_id", "timestamp", TARGET_COL]].copy()
    reference_test = test_df[["transaction_id", "user_id", "timestamp", TARGET_COL]].copy()

    check_clean(X_train, "X_train")
    check_clean(X_test, "X_test")

    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
    reference_train.to_csv(os.path.join(PROCESSED_DIR, "reference_train.csv"), index=False)
    reference_test.to_csv(os.path.join(PROCESSED_DIR, "reference_test.csv"), index=False)

    joblib.dump(ohe, os.path.join(MODELS_DIR, "onehot_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "standard_scaler.pkl"))

    train_fraud_rate = y_train[TARGET_COL].mean()
    test_fraud_rate = y_test[TARGET_COL].mean()

    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Rows combined from {len(glob.glob(os.path.join(DATA_DIR, 'user_*_transactions.csv')))} user CSVs: {len(df)}")
    print(f"Split: 80/20, stratified jointly by (user_id, label), random_state=42")
    print()
    print(f"X_train shape: {X_train.shape}   fraud rate: {train_fraud_rate:.4%}")
    print(f"X_test  shape: {X_test.shape}   fraud rate: {test_fraud_rate:.4%}")
    print(f"Overall fraud rate: {df[TARGET_COL].mean():.4%}")
    print()
    print(f"One-hot encoded columns generated ({len(onehot_cols)}, fit on train only):")
    for c in onehot_cols:
        print(f"  - {c}")
    print()
    print(f"NaNs in X_train/X_test after scaling: 0 / 0 (expected 0)")
    print(f"Infs in X_train/X_test after scaling: 0 / 0 (expected 0)")
    print()
    print("Outputs written:")
    print(f"  {PROCESSED_DIR}/X_train.csv          shape={X_train.shape}")
    print(f"  {PROCESSED_DIR}/X_test.csv           shape={X_test.shape}")
    print(f"  {PROCESSED_DIR}/y_train.csv          shape={y_train.shape}")
    print(f"  {PROCESSED_DIR}/y_test.csv           shape={y_test.shape}")
    print(f"  {PROCESSED_DIR}/reference_train.csv  shape={reference_train.shape}")
    print(f"  {PROCESSED_DIR}/reference_test.csv   shape={reference_test.shape}")
    print(f"  {MODELS_DIR}/onehot_encoder.pkl (fit on train only)")
    print(f"  {MODELS_DIR}/standard_scaler.pkl (fit on train only)")


if __name__ == "__main__":
    main()
