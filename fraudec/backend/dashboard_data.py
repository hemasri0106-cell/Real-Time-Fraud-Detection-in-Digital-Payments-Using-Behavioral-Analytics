"""Bounded, profile-scoped reads and real aggregates from v2 transaction CSVs."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "user_datasets"
ALLOWED_USER_IDS = frozenset(f"user_{number:02d}" for number in range(1, 11))
SAFE_TRANSACTION_FIELDS = (
    "transaction_id",
    "timestamp",
    "transaction_amount",
    "merchant_category",
    "payment_method",
    "device_type",
    "city",
    "hour_of_day",
    "day_of_week",
    "label",
)
MODEL_FEATURES = (
    "transaction_amount", "merchant_category", "merchant_id", "payment_method",
    "device_id", "device_type", "city", "hour_of_day", "day_of_week",
    "is_weekend", "transaction_gap_minutes", "daily_transaction_count",
    "average_amount_last_7_days", "std_amount_last_7_days",
    "merchant_visit_frequency", "device_usage_frequency",
    "location_visit_frequency", "new_device", "new_location", "new_merchant",
    "distance_from_last_transaction_km",
)


def is_allowed_user_id(user_id: str) -> bool:
    return user_id in ALLOWED_USER_IDS


@lru_cache(maxsize=10)
def get_prediction_options(user_id: str) -> Dict[str, List[str]]:
    """Actual categorical choices in the selected demo dataset."""
    frame = load_dataset(user_id)
    fields = ("merchant_category", "merchant_id", "payment_method", "device_id", "device_type", "city")
    return {field: sorted(frame[field].dropna().astype(str).unique().tolist()) for field in fields}


@lru_cache(maxsize=10)
def load_dataset(user_id: str) -> pd.DataFrame:
    """Load one allowlisted CSV; never interpolate an unchecked path."""
    if not is_allowed_user_id(user_id):
        raise ValueError("Unsupported demo profile")

    data_path = DATA_DIR / f"{user_id}_transactions_v2.csv"
    if not data_path.is_file():
        raise FileNotFoundError("Transaction dataset is unavailable")

    frame = pd.read_csv(data_path)
    required = set(SAFE_TRANSACTION_FIELDS) | {
        "is_weekend", "transaction_gap_minutes", "daily_transaction_count",
        "average_amount_last_7_days", "new_device", "new_location", "new_merchant",
        "distance_from_last_transaction_km",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Transaction dataset schema is incomplete")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame


def _counts(series: pd.Series) -> List[Dict[str, object]]:
    counts = series.value_counts(dropna=False).sort_index()
    return [{"key": str(key), "count": int(count)} for key, count in counts.items()]


def _numeric_bins(series: pd.Series, bin_count: int = 10) -> List[Dict[str, object]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return []
    if values.nunique() <= 1:
        value = float(values.iloc[0])
        return [{"range": f"{value:g}", "count": int(len(values))}]

    categories = pd.cut(values, bins=bin_count, duplicates="drop", include_lowest=True)
    counts = categories.value_counts(sort=False)
    return [
        {"range": f"{interval.left:.2f}–{interval.right:.2f}", "count": int(count)}
        for interval, count in counts.items()
    ]


def get_overview(user_id: str) -> Dict[str, object]:
    frame = load_dataset(user_id)
    grouped = frame.assign(date=frame["timestamp"].dt.strftime("%Y-%m-%d")).groupby("date", sort=True)
    volume_timeline = [
        {
            "date": str(date),
            "transaction_count": int(group["transaction_id"].count()),
            "fraud_labelled_count": int(group["label"].eq(1).sum()),
        }
        for date, group in grouped
    ]
    amount_timeline = [
        {"date": str(date), "transaction_amount": round(float(group["transaction_amount"].sum()), 2)}
        for date, group in grouped
    ]
    total = int(len(frame))
    fraud_count = int(frame["label"].eq(1).sum())
    amount_total = float(frame["transaction_amount"].sum())
    return {
        "user_id": user_id,
        "label_semantics": "synthetic_ground_truth",
        "total_transaction_count": total,
        "fraud_labelled_count": fraud_count,
        "normal_count": int(frame["label"].eq(0).sum()),
        "fraud_labelled_percentage": round(fraud_count * 100 / total, 2) if total else 0.0,
        "total_transaction_amount": round(amount_total, 2),
        "average_transaction_amount": round(amount_total / total, 2) if total else 0.0,
        "transaction_volume_over_time": volume_timeline,
        "transaction_amount_over_time": amount_timeline,
    }


def get_behavior(user_id: str) -> Dict[str, object]:
    frame = load_dataset(user_id)
    weekday_names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    weekday_counts = frame["day_of_week"].value_counts().to_dict()
    hour_counts = frame["hour_of_day"].value_counts().to_dict()
    daily = frame.assign(date=frame["timestamp"].dt.strftime("%Y-%m-%d")).groupby("date", sort=True)
    return {
        "user_id": user_id,
        "transactions_by_hour": [
            {"key": f"{hour:02d}:00", "count": int(hour_counts.get(hour, 0))}
            for hour in range(24)
        ],
        "transactions_by_day_of_week": [
            {"key": name, "count": int(weekday_counts.get(day, 0))}
            for day, name in enumerate(weekday_names)
        ],
        "merchant_category_distribution": _counts(frame["merchant_category"]),
        "payment_method_distribution": _counts(frame["payment_method"]),
        "device_type_distribution": _counts(frame["device_type"]),
        "new_device_count": int(frame["new_device"].eq(1).sum()),
        "new_location_count": int(frame["new_location"].eq(1).sum()),
        "new_merchant_count": int(frame["new_merchant"].eq(1).sum()),
        "transaction_gap_distribution": _numeric_bins(frame["transaction_gap_minutes"]),
        "distance_from_previous_transaction": _numeric_bins(frame["distance_from_last_transaction_km"]),
        "daily_transaction_count_over_time": [
            {"date": str(date), "count": int(group["transaction_id"].count())}
            for date, group in daily
        ],
        "average_amount_last_7_days_over_time": [
            {"date": str(date), "average_amount": round(float(group["average_amount_last_7_days"].mean()), 2)}
            for date, group in daily
        ],
    }


def get_transaction_page(
    user_id: str,
    page: int,
    page_size: int,
    label: int = None,
    search: str = None,
    sort_by: str = "timestamp",
    sort_order: str = "desc",
) -> Dict[str, object]:
    frame = load_dataset(user_id)
    if label is not None:
        frame = frame[frame["label"] == label]
    if search:
        needle = search.strip().casefold()
        if needle:
            searchable = ("transaction_id", "merchant_category", "payment_method", "device_type", "city")
            mask = pd.Series(False, index=frame.index)
            for field in searchable:
                mask |= frame[field].astype(str).str.casefold().str.contains(needle, regex=False)
            frame = frame[mask]

    sort_field = {
        "timestamp": "timestamp",
        "transaction_amount": "transaction_amount",
        "merchant_category": "merchant_category",
        "label": "label",
    }[sort_by]
    frame = frame.sort_values(sort_field, ascending=sort_order == "asc", kind="stable")
    total = int(len(frame))
    start = (page - 1) * page_size
    selected = frame.iloc[start:start + page_size].loc[:, SAFE_TRANSACTION_FIELDS]
    records = selected.copy()
    records["timestamp"] = records["timestamp"].map(lambda value: value.to_pydatetime())
    records = records.to_dict(orient="records")
    return {
        "user_id": user_id,
        "label_semantics": "synthetic_ground_truth",
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
        "transactions": records,
    }
