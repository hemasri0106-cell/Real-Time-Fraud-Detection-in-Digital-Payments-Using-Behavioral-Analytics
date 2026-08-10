#!/usr/bin/env python3
"""
generate_user_datasets.py
==========================
Synthetic transaction-dataset generator for the Fraudec project.

Produces one CSV per user profile (10 total), each with EXACTLY 5,000
transactions, for training an Isolation Forest on customer-relative
behavioral deviations. Every derived/rolling feature (columns 14-24) is
computed by walking each user's own transaction history in chronological
order, so it reflects genuine running state rather than random sampling.

Run standalone:
    python generate_user_datasets.py

Output:
    data/user_datasets/user_01_transactions.csv ... user_10_transactions.csv
    data/user_datasets/manifest.csv

--------------------------------------------------------------------------
NOTE ON A SPEC TRADE-OFF (read this before trusting daily_transaction_count)
--------------------------------------------------------------------------
The brief asks for EXACTLY 5,000 rows per user spread across ~90 days, but
also gives each profile a literal "transactions/day" range (e.g. Retired
Senior: 1-4/day, Business Owner: 10-25/day). 5,000 rows / 90 days is a
forced average of ~55-56 transactions/day for every user -- that average
is incompatible with a literal 1-4/day range for the Senior Citizen (which
would total only ~90-360 rows over 90 days).

Resolution used here: the exact-row-count and ~90-day-window constraints
are treated as hard (they're stated as "exactly" / are needed for a fixed
training-set size). Each profile's stated txns/day range is instead used
to shape the *relative* day-to-day burstiness/variance of that user's
activity pattern (via a per-day weight sampled from that range), so a
"low volume" profile still looks visibly less spiky than a "high volume"
one -- but every user's daily_transaction_count will average out to
~50-60/day rather than the small literal numbers in the brief. This is
flagged explicitly here and in the console summary.
"""

import math
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Global config
# --------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "user_datasets")
ROWS_PER_USER = 5000
NUM_DAYS = 90
END_DATE = datetime(2026, 8, 10, 23, 59, 0)
START_DATE = END_DATE - timedelta(days=NUM_DAYS)
FRAUD_RATE_RANGE = (0.03, 0.05)

# --------------------------------------------------------------------------
# City -> (lat, lon) lookup, used for haversine distance_from_last_transaction_km
# --------------------------------------------------------------------------

CITY_COORDS = {
    # usual / home cities
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Pune": (18.5204, 73.8567),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Goa": (15.2993, 74.1240),
    "Lucknow": (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794),
    "Kochi": (9.9312, 76.2673),
    "Surat": (21.1702, 72.8311),
    "Indore": (22.7196, 75.8577),
    # foreign / unusual cities used only for fraud injection
    "Dubai": (25.2048, 55.2708),
    "Singapore": (1.3521, 103.8198),
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Bangkok": (13.7563, 100.5018),
    "Unknown Country": (0.0, 0.0),
}

FOREIGN_CITIES = ["Dubai", "Singapore", "London", "New York", "Bangkok", "Unknown Country"]


def haversine_km(city_a, city_b):
    if city_a is None or city_b is None or city_a == city_b:
        return 0.0
    lat1, lon1 = CITY_COORDS[city_a]
    lat2, lon2 = CITY_COORDS[city_b]
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return round(r * 2 * math.asin(math.sqrt(a)), 2)


# --------------------------------------------------------------------------
# User profiles
# --------------------------------------------------------------------------
# home_cities: list of (city, weight) the user legitimately transacts from
# day_count_range: used only to SHAPE relative daily burstiness (see note above)
# fraud_templates: cycled through when injecting fraud rows

PROFILES = [
    dict(
        user_id=1, name="College Student", age=20,
        amount_range=(50, 700),
        categories={"Food": 0.35, "Cafes": 0.25, "Grocery": 0.2, "Movies": 0.2},
        payment_methods={"UPI": 1.0},
        active_hours=(11, 22),
        day_count_range=(3, 8),
        home_cities=[("Pune", 0.97)],
        num_devices=1,
        num_merchants=12,
        fraud_templates=["big_electronics", "odd_hour_3am", "new_city", "unknown_device"],
    ),
    dict(
        user_id=2, name="Software Engineer", age=27,
        amount_range=(200, 3000),
        categories={"Grocery": 0.25, "Fuel": 0.15, "Amazon": 0.3, "Swiggy": 0.3},
        payment_methods={"UPI": 0.6, "Credit Card": 0.4},
        active_hours=(7, 23),
        day_count_range=(5, 10),
        home_cities=[("Bangalore", 0.97)],
        num_devices=2,
        num_merchants=20,
        fraud_templates=["luxury_purchase", "foreign_city", "rapid_burst_30s"],
    ),
    dict(
        user_id=3, name="Homemaker", age=45,
        amount_range=(100, 1500),
        categories={"Grocery": 0.45, "Pharmacy": 0.25, "Household": 0.3},
        payment_methods={"UPI": 0.5, "Card": 0.5},
        active_hours=(8, 19),
        day_count_range=(2, 5),
        home_cities=[("Chennai", 0.98)],
        num_devices=1,
        num_merchants=10,
        fraud_templates=["midnight_fuel", "gaming_subscription", "airport_txn"],
    ),
    dict(
        user_id=4, name="Business Owner", age=42,
        amount_range=(2000, 50000),
        categories={"Suppliers": 0.3, "Office": 0.2, "Fuel": 0.2, "Hotels": 0.3},
        payment_methods={"Card": 0.5, "UPI": 0.3, "Bank Transfer": 0.2},
        active_hours=(8, 20),
        day_count_range=(10, 25),
        home_cities=[("Mumbai", 0.85), ("Delhi", 0.08), ("Pune", 0.07)],
        num_devices=2,
        num_merchants=30,
        fraud_templates=["foreign_merchant", "unknown_device", "rapid_repeated_transfers"],
    ),
    dict(
        user_id=5, name="Sales Executive", age=33,
        amount_range=(200, 4000),
        categories={"Fuel": 0.25, "Hotels": 0.25, "Restaurants": 0.35, "Taxi": 0.15},
        payment_methods={"UPI": 0.5, "Card": 0.5},
        active_hours=(6, 23),
        day_count_range=(6, 12),
        home_cities=[("Delhi", 0.3), ("Mumbai", 0.25), ("Pune", 0.2), ("Bangalore", 0.15), ("Jaipur", 0.1)],
        num_devices=2,
        num_merchants=25,
        fraud_templates=["huge_transfer", "unknown_country_purchase", "late_night_spree"],
    ),
    dict(
        user_id=6, name="Doctor", age=39,
        amount_range=(500, 6000),
        categories={"Medical": 0.4, "Grocery": 0.3, "Dining": 0.3},
        payment_methods={"Card": 0.55, "UPI": 0.45},
        active_hours=(6, 22),
        day_count_range=(3, 8),
        home_cities=[("Delhi", 0.97)],
        num_devices=1,
        num_merchants=10,
        fraud_templates=["casino_payment", "luxury_abroad", "multi_atm_withdrawals"],
    ),
    dict(
        user_id=7, name="Retired Senior Citizen", age=68,
        amount_range=(100, 2000),
        categories={"Pharmacy": 0.35, "Grocery": 0.35, "Utility Bills": 0.3},
        payment_methods={"UPI": 0.8, "Card": 0.2},
        active_hours=(8, 20),
        day_count_range=(1, 4),
        home_cities=[("Kolkata", 0.98)],
        num_devices=1,
        num_merchants=8,
        fraud_templates=["crypto_exchange", "gaming_website", "big_transfer_45000"],
    ),
    dict(
        user_id=8, name="Online Shopper", age=29,
        amount_range=(300, 6000),
        categories={"Amazon": 0.3, "Myntra": 0.25, "Flipkart": 0.25, "Electronics": 0.2},
        payment_methods={"Card": 0.55, "UPI": 0.45},
        active_hours=(9, 23),
        day_count_range=(5, 15),
        home_cities=[("Hyderabad", 0.97)],
        num_devices=2,
        num_merchants=20,
        fraud_templates=["luxury_jewellery", "unknown_merchant", "midnight_purchase"],
    ),
    dict(
        user_id=9, name="Freelancer", age=31,
        amount_range=(200, 8000),
        categories={"Software": 0.25, "Coffee": 0.25, "Restaurants": 0.25, "Cloud Services": 0.25},
        payment_methods={"UPI": 0.5, "Card": 0.5},
        active_hours=(8, 24),
        day_count_range=(3, 10),
        home_cities=[("Pune", 0.7), ("Goa", 0.18), ("Bangalore", 0.12)],
        num_devices=2,
        num_merchants=15,
        fraud_templates=["intl_transfer", "new_device_fraud", "hourly_location_hopping"],
    ),
    dict(
        user_id=10, name="Frequent Traveler", age=36,
        amount_range=(500, 10000),
        categories={"Airlines": 0.2, "Hotels": 0.25, "Food": 0.3, "Taxi": 0.25},
        payment_methods={"Card": 0.6, "UPI": 0.4},
        active_hours=(5, 23),
        day_count_range=(8, 20),
        home_cities=[("Mumbai", 0.2), ("Delhi", 0.2), ("Bangalore", 0.15), ("Chennai", 0.15),
                     ("Hyderabad", 0.15), ("Goa", 0.15)],
        num_devices=2,
        num_merchants=25,
        fraud_templates=["distant_800km_15min", "unknown_device", "large_luxury_purchase"],
    ),
]

DEVICE_TYPES = ["Android", "iPhone", "Web"]


# --------------------------------------------------------------------------
# Per-user static pools (merchants, devices)
# --------------------------------------------------------------------------

def build_merchant_pool(profile, rng):
    """Stable merchant pool for this user: each merchant belongs to one of the
    user's usual categories. merchant_id encodes user_id for global uniqueness."""
    cats = list(profile["categories"].keys())
    weights = np.array(list(profile["categories"].values()))
    weights = weights / weights.sum()
    n = profile["num_merchants"]
    pool = []
    for i in range(n):
        cat = rng.choice(cats, p=weights)
        merchant_id = profile["user_id"] * 10000 + i + 1
        pool.append({"merchant_id": merchant_id, "category": cat})
    return pool


def build_device_pool(profile, rng):
    devices = []
    for i in range(profile["num_devices"]):
        dtype = rng.choice(DEVICE_TYPES, p=[0.55, 0.35, 0.10] if len(DEVICE_TYPES) == 3 else None)
        devices.append({"device_id": f"DEV-{profile['user_id']:02d}-{i+1}", "device_type": dtype})
    return devices


# --------------------------------------------------------------------------
# Daily count shaping (see module docstring trade-off note)
# --------------------------------------------------------------------------

def build_daily_counts(profile, rng, total_rows):
    lo, hi = profile["day_count_range"]
    raw_weights = rng.uniform(lo, hi, size=NUM_DAYS)
    # mild weekly pattern: slightly busier on weekends for consumer profiles
    day_dates = [START_DATE + timedelta(days=d) for d in range(NUM_DAYS)]
    weekend_boost = np.array([1.15 if d.weekday() >= 5 else 1.0 for d in day_dates])
    raw_weights = raw_weights * weekend_boost
    scaled = raw_weights / raw_weights.sum() * total_rows
    floors = np.floor(scaled).astype(int)
    remainder = total_rows - floors.sum()
    frac_order = np.argsort(-(scaled - floors))
    for i in range(remainder):
        floors[frac_order[i % NUM_DAYS]] += 1
    floors = np.maximum(floors, 0)
    # fix any rounding drift
    drift = total_rows - floors.sum()
    if drift != 0:
        floors[np.argmax(floors)] += drift
    return day_dates, floors


def random_time_in_hours(rng, day_date, active_hours):
    lo, hi = active_hours
    hi = min(hi, 23)
    hour = int(rng.integers(lo, hi + 1)) if hi >= lo else int(rng.integers(0, 24))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return day_date.replace(hour=min(hour, 23), minute=minute, second=second, microsecond=0)


# --------------------------------------------------------------------------
# Normal transaction generation
# --------------------------------------------------------------------------

def pick_city(profile, rng, novel_prob=0.01):
    cities = [c for c, _ in profile["home_cities"]]
    weights = np.array([w for _, w in profile["home_cities"]])
    weights = weights / weights.sum()
    if rng.random() < novel_prob:
        candidates = [c for c in CITY_COORDS if c not in cities and c not in FOREIGN_CITIES]
        return rng.choice(candidates)
    return rng.choice(cities, p=weights)


def pick_merchant(profile, merchant_pool, category, rng, new_merchant_prob=0.03):
    same_cat = [m for m in merchant_pool if m["category"] == category]
    if rng.random() < new_merchant_prob or not same_cat:
        new_id = max(m["merchant_id"] for m in merchant_pool) + 1
        m = {"merchant_id": new_id, "category": category}
        merchant_pool.append(m)
        return m
    # skew toward a few favorite merchants (zipf-like)
    n = len(same_cat)
    ranks = np.arange(1, n + 1)
    probs = 1.0 / ranks
    probs = probs / probs.sum()
    idx = rng.choice(n, p=probs)
    return same_cat[idx]


def pick_device(profile, device_pool, rng, new_device_prob=0.01):
    if rng.random() < new_device_prob:
        new_id = f"DEV-{profile['user_id']:02d}-{len(device_pool)+1}"
        dtype = rng.choice(DEVICE_TYPES)
        d = {"device_id": new_id, "device_type": dtype}
        device_pool.append(d)
        return d
    n = len(device_pool)
    if n > 1:
        # favor earlier (original/usual) devices, decaying weight for later additions
        weights = np.array([0.7 ** i for i in range(n)])
        weights = weights / weights.sum()
    else:
        weights = np.array([1.0])
    idx = rng.choice(n, p=weights)
    return device_pool[idx]


def pick_payment(profile, rng):
    methods = list(profile["payment_methods"].keys())
    weights = np.array(list(profile["payment_methods"].values()))
    weights = weights / weights.sum()
    return rng.choice(methods, p=weights)


def gen_amount(lo, hi, rng, skew=True):
    if skew:
        # lognormal-ish skew toward lower end of range, clipped to [lo, hi]
        mid = (lo + hi) / 2
        val = rng.gamma(shape=2.0, scale=mid / 2.5) + lo
        return float(np.clip(val, lo, hi))
    return float(rng.uniform(lo, hi))


def generate_normal_rows(profile, rng, merchant_pool, device_pool, count):
    day_dates, daily_counts = build_daily_counts(profile, rng, count)
    rows = []
    for day_date, n in zip(day_dates, daily_counts):
        if n <= 0:
            continue
        times = sorted(random_time_in_hours(rng, day_date, profile["active_hours"]) for _ in range(n))
        for ts in times:
            cats = list(profile["categories"].keys())
            cat_weights = np.array(list(profile["categories"].values()))
            cat_weights = cat_weights / cat_weights.sum()
            category = rng.choice(cats, p=cat_weights)
            merchant = pick_merchant(profile, merchant_pool, category, rng)
            device = pick_device(profile, device_pool, rng)
            city = pick_city(profile, rng)
            payment = pick_payment(profile, rng)
            amount = round(gen_amount(*profile["amount_range"], rng), 2)
            rows.append(dict(
                user_id=profile["user_id"], timestamp=ts, transaction_amount=amount,
                merchant_category=category, merchant_id=merchant["merchant_id"],
                payment_method=payment, device_id=device["device_id"],
                device_type=device["device_type"], city=city, label=0, fraud_type=None,
            ))
    return rows


# --------------------------------------------------------------------------
# Fraud transaction generation (one template per row-slot, some templates
# produce a linked pair to satisfy "rapid succession" / "N-km-apart" cases)
# --------------------------------------------------------------------------

def random_fraud_time(rng):
    day_offset = int(rng.integers(0, NUM_DAYS))
    return (START_DATE + timedelta(days=day_offset)).replace(
        hour=int(rng.integers(0, 24)), minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60))
    )


def base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type):
    """Sensible legitimate-looking defaults a specific template then overrides."""
    device = rng_choice_dict(device_pool)
    city = profile["home_cities"][0][0]
    return dict(
        user_id=profile["user_id"], timestamp=ts,
        transaction_amount=round((profile["amount_range"][0] + profile["amount_range"][1]) / 2, 2),
        merchant_category=list(profile["categories"].keys())[0],
        merchant_id=merchant_pool[0]["merchant_id"],
        payment_method=list(profile["payment_methods"].keys())[0],
        device_id=device["device_id"], device_type=device["device_type"],
        city=city, label=1, fraud_type=fraud_type,
    )


def rng_choice_dict(pool):
    return pool[0]


def make_fraud_rows(profile, rng, merchant_pool, device_pool, fraud_type, budget_left):
    """Returns a list of 1-2 raw fraud rows for the given template, never
    exceeding budget_left rows."""
    ts = random_fraud_time(rng)
    lo, hi = profile["amount_range"]
    rows = []

    if fraud_type == "big_electronics":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(transaction_amount=18000.0, merchant_category="Electronics", payment_method="UPI")
        rows = [r]
    elif fraud_type == "odd_hour_3am":
        r = base_fraud_row(profile, ts.replace(hour=3, minute=int(rng.integers(0, 60))), merchant_pool, device_pool, fraud_type)
        r.update(transaction_amount=round(gen_amount(lo, hi, rng), 2))
        rows = [r]
    elif fraud_type == "new_city":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        candidates = [c for c in CITY_COORDS if c not in [h for h, _ in profile["home_cities"]]]
        r.update(city=rng.choice(candidates), transaction_amount=round(gen_amount(lo, hi, rng) * 1.5, 2))
        rows = [r]
    elif fraud_type == "unknown_device":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(device_id=f"DEV-UNK-{int(rng.integers(1000, 9999))}", device_type=rng.choice(DEVICE_TYPES))
        rows = [r]
    elif fraud_type == "luxury_purchase":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(transaction_amount=60000.0, merchant_category="Luxury Goods", payment_method="Credit Card")
        rows = [r]
    elif fraud_type == "foreign_city":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(city=rng.choice(FOREIGN_CITIES), transaction_amount=round(gen_amount(lo, hi, rng) * 3, 2))
        rows = [r]
    elif fraud_type == "rapid_burst_30s":
        n = min(3, max(2, budget_left))
        city = profile["home_cities"][0][0]
        for i in range(n):
            r = base_fraud_row(profile, ts + timedelta(seconds=i * int(rng.integers(5, 12))), merchant_pool, device_pool, fraud_type)
            r.update(city=city, transaction_amount=round(gen_amount(lo, hi, rng), 2),
                      merchant_id=rng.choice(merchant_pool)["merchant_id"])
            rows.append(r)
    elif fraud_type == "midnight_fuel":
        r = base_fraud_row(profile, ts.replace(hour=0, minute=int(rng.integers(0, 60))), merchant_pool, device_pool, fraud_type)
        r.update(merchant_category="Fuel", transaction_amount=round(gen_amount(lo, hi, rng), 2))
        rows = [r]
    elif fraud_type == "gaming_subscription":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(merchant_category="Gaming Subscription", transaction_amount=round(gen_amount(lo, hi, rng) * 0.5, 2))
        rows = [r]
    elif fraud_type == "airport_txn":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        candidates = [c for c in CITY_COORDS if c not in [h for h, _ in profile["home_cities"]] and c not in FOREIGN_CITIES]
        r.update(merchant_category="Airport", city=rng.choice(candidates), transaction_amount=round(gen_amount(lo, hi, rng) * 2, 2))
        rows = [r]
    elif fraud_type == "foreign_merchant":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(city=rng.choice(FOREIGN_CITIES), merchant_id=rng.integers(900000, 999999),
                  merchant_category="Overseas Merchant", transaction_amount=round(gen_amount(lo, hi, rng) * 1.5, 2))
        rows = [r]
    elif fraud_type == "rapid_repeated_transfers":
        n = min(3, max(2, budget_left))
        for i in range(n):
            r = base_fraud_row(profile, ts + timedelta(minutes=i * int(rng.integers(1, 4))), merchant_pool, device_pool, fraud_type)
            r.update(merchant_category="Bank Transfer", payment_method="Bank Transfer",
                      transaction_amount=round(gen_amount(hi * 0.5, hi, rng), 2))
            rows.append(r)
    elif fraud_type == "huge_transfer":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(transaction_amount=100000.0, merchant_category="Bank Transfer", payment_method="Card")
        rows = [r]
    elif fraud_type == "unknown_country_purchase":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(city="Unknown Country", transaction_amount=round(gen_amount(lo, hi, rng) * 4, 2))
        rows = [r]
    elif fraud_type == "late_night_spree":
        n = min(2, max(2, budget_left))
        for i in range(n):
            r = base_fraud_row(profile, ts.replace(hour=1 + i, minute=int(rng.integers(0, 60))), merchant_pool, device_pool, fraud_type)
            r.update(merchant_category="Shopping", transaction_amount=round(gen_amount(lo, hi, rng) * 2, 2))
            rows.append(r)
    elif fraud_type == "casino_payment":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(city=rng.choice(FOREIGN_CITIES), merchant_category="Casino", transaction_amount=round(gen_amount(lo, hi, rng) * 2, 2))
        rows = [r]
    elif fraud_type == "luxury_abroad":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(city=rng.choice(FOREIGN_CITIES), merchant_category="Luxury Goods", transaction_amount=round(gen_amount(lo, hi, rng) * 2.5, 2))
        rows = [r]
    elif fraud_type == "multi_atm_withdrawals":
        n = min(3, max(2, budget_left))
        for i in range(n):
            r = base_fraud_row(profile, ts + timedelta(minutes=i * int(rng.integers(2, 6))), merchant_pool, device_pool, fraud_type)
            r.update(merchant_category="ATM Withdrawal", payment_method="Card", transaction_amount=round(gen_amount(lo, hi, rng), 2))
            rows.append(r)
    elif fraud_type == "crypto_exchange":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(merchant_category="Crypto Exchange", transaction_amount=round(gen_amount(lo, hi, rng) * 3, 2))
        rows = [r]
    elif fraud_type == "gaming_website":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(merchant_category="Gaming Website", transaction_amount=round(gen_amount(lo, hi, rng), 2))
        rows = [r]
    elif fraud_type == "big_transfer_45000":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(transaction_amount=45000.0, merchant_category="Bank Transfer")
        rows = [r]
    elif fraud_type == "luxury_jewellery":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(merchant_category="Jewellery", transaction_amount=round(max(hi * 3, 40000), 2))
        rows = [r]
    elif fraud_type == "unknown_merchant":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(merchant_id=int(rng.integers(900000, 999999)), merchant_category="Unknown Merchant",
                  transaction_amount=round(gen_amount(lo, hi, rng) * 1.5, 2))
        rows = [r]
    elif fraud_type == "midnight_purchase":
        r = base_fraud_row(profile, ts.replace(hour=0, minute=int(rng.integers(0, 60))), merchant_pool, device_pool, fraud_type)
        r.update(transaction_amount=round(gen_amount(lo, hi, rng) * 1.3, 2))
        rows = [r]
    elif fraud_type == "intl_transfer":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(city=rng.choice(FOREIGN_CITIES), merchant_category="International Transfer",
                  transaction_amount=round(gen_amount(lo, hi, rng) * 4, 2))
        rows = [r]
    elif fraud_type == "new_device_fraud":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(device_id=f"DEV-UNK-{int(rng.integers(1000, 9999))}", device_type=rng.choice(DEVICE_TYPES))
        rows = [r]
    elif fraud_type == "hourly_location_hopping":
        n = min(3, max(2, budget_left))
        candidates = [c for c in CITY_COORDS if c not in [h for h, _ in profile["home_cities"]]]
        for i in range(n):
            r = base_fraud_row(profile, ts + timedelta(hours=i), merchant_pool, device_pool, fraud_type)
            r.update(city=rng.choice(candidates), transaction_amount=round(gen_amount(lo, hi, rng), 2))
            rows.append(r)
    elif fraud_type == "distant_800km_15min":
        city_a = profile["home_cities"][0][0]
        pair_candidates = [c for c in CITY_COORDS if c not in FOREIGN_CITIES and c != city_a
                            and 650 <= haversine_km(city_a, c) <= 1000]
        city_b = rng.choice(pair_candidates) if pair_candidates else rng.choice(
            [c for c in CITY_COORDS if c not in FOREIGN_CITIES and c != city_a])
        r1 = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r1.update(city=city_a, transaction_amount=round(gen_amount(lo, hi, rng), 2))
        r2 = base_fraud_row(profile, ts + timedelta(minutes=int(rng.integers(5, 15))), merchant_pool, device_pool, fraud_type)
        r2.update(city=city_b, transaction_amount=round(gen_amount(lo, hi, rng), 2))
        rows = [r1, r2]
    elif fraud_type == "unknown_device":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(device_id=f"DEV-UNK-{int(rng.integers(1000, 9999))}", device_type=rng.choice(DEVICE_TYPES))
        rows = [r]
    elif fraud_type == "large_luxury_purchase":
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        r.update(merchant_category="Luxury Goods", transaction_amount=round(max(hi * 2.5, 50000), 2))
        rows = [r]
    else:
        r = base_fraud_row(profile, ts, merchant_pool, device_pool, fraud_type)
        rows = [r]

    return rows[:budget_left] if budget_left < len(rows) else rows


def generate_fraud_rows(profile, rng, merchant_pool, device_pool, fraud_count):
    templates = profile["fraud_templates"]
    rows = []
    t_idx = 0
    guard = 0
    while len(rows) < fraud_count and guard < fraud_count * 5 + 20:
        guard += 1
        fraud_type = templates[t_idx % len(templates)]
        t_idx += 1
        budget_left = fraud_count - len(rows)
        new_rows = make_fraud_rows(profile, rng, merchant_pool, device_pool, fraud_type, budget_left)
        rows.extend(new_rows)
    return rows[:fraud_count]


# --------------------------------------------------------------------------
# Derived / rolling feature computation (walked in chronological order)
# --------------------------------------------------------------------------

def compute_derived_features(rows):
    rows = sorted(rows, key=lambda r: r["timestamp"])

    prev_ts = None
    prev_city = None
    daily_count = {}
    merchant_seen = {}
    device_seen = {}
    city_seen = {}
    amount_history = []  # list of (timestamp, amount) for trailing-7-day window

    out = []
    for r in rows:
        ts = r["timestamp"]
        day_key = ts.date()

        # transaction_gap_minutes
        gap = 0.0 if prev_ts is None else round((ts - prev_ts).total_seconds() / 60.0, 2)

        # daily_transaction_count
        daily_count[day_key] = daily_count.get(day_key, 0) + 1

        # trailing 7-day mean/std (based on history strictly before this txn)
        window_start = ts - timedelta(days=7)
        window_vals = [amt for (t, amt) in amount_history if t >= window_start]
        if window_vals:
            avg_7d = round(float(np.mean(window_vals)), 2)
            std_7d = round(float(np.std(window_vals)), 2) if len(window_vals) > 1 else 0.0
        else:
            avg_7d = round(r["transaction_amount"], 2)
            std_7d = 0.0

        # cumulative frequencies (including current txn)
        merchant_seen[r["merchant_id"]] = merchant_seen.get(r["merchant_id"], 0) + 1
        device_seen[r["device_id"]] = device_seen.get(r["device_id"], 0) + 1
        city_seen[r["city"]] = city_seen.get(r["city"], 0) + 1

        new_merchant = 1 if merchant_seen[r["merchant_id"]] == 1 else 0
        new_device = 1 if device_seen[r["device_id"]] == 1 else 0
        new_location = 1 if city_seen[r["city"]] == 1 else 0

        distance = haversine_km(prev_city, r["city"]) if prev_ts is not None else 0.0

        out.append(dict(
            user_id=r["user_id"], timestamp=ts, transaction_amount=r["transaction_amount"],
            merchant_category=r["merchant_category"], merchant_id=r["merchant_id"],
            payment_method=r["payment_method"], device_id=r["device_id"], device_type=r["device_type"],
            city=r["city"], hour_of_day=ts.hour, day_of_week=ts.weekday(),
            is_weekend=1 if ts.weekday() >= 5 else 0,
            transaction_gap_minutes=gap, daily_transaction_count=daily_count[day_key],
            average_amount_last_7_days=avg_7d, std_amount_last_7_days=std_7d,
            merchant_visit_frequency=merchant_seen[r["merchant_id"]],
            device_usage_frequency=device_seen[r["device_id"]],
            location_visit_frequency=city_seen[r["city"]],
            new_device=new_device, new_location=new_location, new_merchant=new_merchant,
            distance_from_last_transaction_km=distance, label=r["label"],
        ))

        amount_history.append((ts, r["transaction_amount"]))
        prev_ts = ts
        prev_city = r["city"]

    return out


# --------------------------------------------------------------------------
# Per-user pipeline
# --------------------------------------------------------------------------

COLUMNS = [
    "transaction_id", "user_id", "timestamp", "transaction_amount", "merchant_category",
    "merchant_id", "payment_method", "device_id", "device_type", "city", "hour_of_day",
    "day_of_week", "is_weekend", "transaction_gap_minutes", "daily_transaction_count",
    "average_amount_last_7_days", "std_amount_last_7_days", "merchant_visit_frequency",
    "device_usage_frequency", "location_visit_frequency", "new_device", "new_location",
    "new_merchant", "distance_from_last_transaction_km", "label",
]


def generate_user_dataset(profile):
    rng = np.random.default_rng(100 + profile["user_id"])

    merchant_pool = build_merchant_pool(profile, rng)
    device_pool = build_device_pool(profile, rng)

    fraud_rate = rng.uniform(*FRAUD_RATE_RANGE)
    fraud_count = int(round(ROWS_PER_USER * fraud_rate))
    normal_count = ROWS_PER_USER - fraud_count

    normal_rows = generate_normal_rows(profile, rng, merchant_pool, device_pool, normal_count)
    fraud_rows = generate_fraud_rows(profile, rng, merchant_pool, device_pool, fraud_count)

    all_rows = normal_rows + fraud_rows
    assert len(all_rows) == ROWS_PER_USER, f"user {profile['user_id']}: got {len(all_rows)} rows, expected {ROWS_PER_USER}"

    enriched = compute_derived_features(all_rows)

    df = pd.DataFrame(enriched)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{profile['user_id']:02d}-{i+1:05d}" for i in range(len(df))])
    df = df[COLUMNS]

    return df, fraud_count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_rows = []
    summary_rows = []

    for profile in PROFILES:
        df, fraud_count = generate_user_dataset(profile)
        out_path = os.path.join(OUTPUT_DIR, f"user_{profile['user_id']:02d}_transactions.csv")
        df.to_csv(out_path, index=False)

        n_rows = len(df)
        actual_fraud = int(df["label"].sum())
        fraud_pct = round(100 * actual_fraud / n_rows, 2)
        amt_min, amt_max = df["transaction_amount"].min(), df["transaction_amount"].max()

        manifest_rows.append(dict(
            user_id=profile["user_id"], profile_name=profile["name"], row_count=n_rows,
            fraud_count=actual_fraud, fraud_rate_pct=fraud_pct,
            amount_min=round(amt_min, 2), amount_max=round(amt_max, 2),
            file=os.path.basename(out_path),
        ))
        summary_rows.append((profile["user_id"], profile["name"], n_rows, actual_fraud, fraud_pct))
        print(f"  wrote {out_path}  ({n_rows} rows, {actual_fraud} fraud = {fraud_pct}%)")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print("\n" + "=" * 78)
    print("FRAUDEC SYNTHETIC DATASET GENERATION — SUMMARY")
    print("=" * 78)
    header = f"{'ID':<4}{'Profile':<26}{'Rows':<8}{'Fraud':<8}{'Fraud %':<10}"
    print(header)
    print("-" * 78)
    for uid, name, n_rows, fraud, pct in summary_rows:
        print(f"{uid:<4}{name:<26}{n_rows:<8}{fraud:<8}{pct:<10}")
    print("-" * 78)
    total_rows = sum(r[2] for r in summary_rows)
    total_fraud = sum(r[3] for r in summary_rows)
    print(f"{'':<4}{'TOTAL':<26}{total_rows:<8}{total_fraud:<8}{round(100*total_fraud/total_rows, 2):<10}")
    print("=" * 78)
    print(f"\nManifest written to: {manifest_path}")
    print(f"Per-user CSVs written to: {OUTPUT_DIR}")
    print("\nNOTE: daily_transaction_count averages ~50-60/day for every profile,")
    print("since 5,000 rows/user is spread across a fixed ~90-day window regardless")
    print("of each profile's stated txns/day range. See module docstring for detail.")


if __name__ == "__main__":
    main()
