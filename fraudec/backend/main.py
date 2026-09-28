import os
import json
import logging
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

# Must run before `import auth` - auth.py reads JWT_SECRET_KEY from the
# environment at import time, so .env has to be loaded first regardless of
# whether the caller's shell has it pre-exported.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text

import models
import schemas
import auth
import ml_service
from database import engine, get_db

logger = logging.getLogger("fraudec")

models.Base.metadata.create_all(bind=engine)

# create_all() only creates missing tables - it won't add a column to a
# predictions table that already existed before true_label was introduced.
# SQLite supports ADD COLUMN directly, so patch it in if it's missing.
with engine.connect() as _conn:
    try:
        _conn.execute(text("ALTER TABLE predictions ADD COLUMN true_label INTEGER"))
        _conn.commit()
    except Exception:
        pass  # column already exists

app = FastAPI(title="Fraud Detection Dashboard - Auth Service")

# Design-system foundation (Stitch export) - templates/base.html holds the
# shared <head>/nav/header; static/ serves the Tailwind theme config + logo.
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Loosened for local dev with a React dev server. Tighten to your real
# frontend origin before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/auth/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(
        (models.User.username == payload.username) | (models.User.email == payload.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered")

    user = models.User(
        username=payload.username,
        email=payload.email,
        hashed_password=auth.hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/auth/login", response_model=schemas.Token)
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Capture the same behavioral signals (IP, timestamp, device) the fraud
    # model profiles elsewhere in the system - the auth layer eats its own dogfood.
    user.last_login_at = datetime.now(timezone.utc)
    user.last_login_ip = request.client.host if request.client else None
    user.last_login_device = request.headers.get("x-device-id", "unknown")
    db.commit()
    db.refresh(user)

    access_token = auth.create_access_token(data={"sub": user.username, "role": user.role.value})
    return {"access_token": access_token, "token_type": "bearer", "user": user}


@app.get("/api/auth/me", response_model=schemas.UserOut)
def read_current_user(current_user: models.User = Depends(auth.get_current_user)):
    return current_user


from pydantic import BaseModel

class DemoLoginRequest(BaseModel):
    persona_name: str
    user_id: str

@app.post("/api/auth/demo-login")
def demo_login(request: DemoLoginRequest):
    if request.user_id not in ml_service.get_valid_user_ids():
        raise HTTPException(status_code=404, detail="Unknown persona")

    access_token = auth.create_access_token(
        data={"sub": request.persona_name, "role": models.UserRole.persona, "user_id": request.user_id}
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user": {"username": request.persona_name, "role": "persona", "user_id": request.user_id}
    }

@app.get("/api/metrics")
def get_metrics(current_user: models.User = Depends(auth.get_current_user)):
    user_id = getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Not a persona user")
    metrics = ml_service.get_user_metrics(user_id)
    return metrics

@app.get("/api/transactions")
def get_transactions(limit: int = 50, offset: int = 0, current_user: models.User = Depends(auth.get_current_user)):
    user_id = getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Not a persona user")
    txs = ml_service.get_transactions(user_id, limit, offset)
    return {"transactions": txs}

def _log_prediction(db: Session, user_id: str, source: str, feature_payload: dict, result: dict, true_label: int = None) -> models.Prediction:
    row = models.Prediction(
        user_id=user_id,
        source=source,
        transaction_amount=float(feature_payload.get("transaction_amount", 0) or 0),
        merchant_category=str(feature_payload.get("merchant_category", "unknown")),
        merchant_id=str(feature_payload.get("merchant_id", "unknown")),
        payment_method=str(feature_payload.get("payment_method", "unknown")),
        device_id=str(feature_payload.get("device_id", "unknown")),
        device_type=str(feature_payload.get("device_type", "unknown")),
        city=str(feature_payload.get("city", "unknown")),
        hour_of_day=int(feature_payload.get("hour_of_day", 0) or 0),
        distance_from_last_transaction_km=float(feature_payload.get("distance_from_last_transaction_km", 0) or 0),
        new_device=int(bool(feature_payload.get("new_device", 0))),
        new_location=int(bool(feature_payload.get("new_location", 0))),
        new_merchant=int(bool(feature_payload.get("new_merchant", 0))),
        prediction=result["prediction"],
        fraud_probability=result["fraud_probability"],
        top_features=json.dumps(result["top_features"]),
        true_label=true_label,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _serialize_prediction(row: models.Prediction) -> dict:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "source": row.source,
        "transaction_amount": row.transaction_amount,
        "merchant_category": row.merchant_category,
        "merchant_id": row.merchant_id,
        "payment_method": row.payment_method,
        "device_id": row.device_id,
        "device_type": row.device_type,
        "city": row.city,
        "hour_of_day": row.hour_of_day,
        "distance_from_last_transaction_km": row.distance_from_last_transaction_km,
        "new_device": row.new_device,
        "new_location": row.new_location,
        "new_merchant": row.new_merchant,
        "prediction": row.prediction,
        "fraud_probability": row.fraud_probability,
        "top_features": json.loads(row.top_features) if row.top_features else [],
        "true_label": row.true_label,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@app.post("/api/predict")
def predict(payload: dict, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    user_id = getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Not a persona user")
    try:
        result = ml_service.predict_transaction(user_id, payload)
        _log_prediction(db, user_id, "manual", payload, result)
        return result
    except Exception:
        logger.exception("predict_transaction failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs for details.")


@app.get("/api/predictions")
def list_predictions(limit: int = 20, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    user_id = getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Not a persona user")
    limit = max(1, min(limit, 100))
    rows = (
        db.query(models.Prediction)
        .filter(models.Prediction.user_id == user_id)
        .order_by(models.Prediction.created_at.desc(), models.Prediction.id.desc())
        .limit(limit)
        .all()
    )
    return {"predictions": [_serialize_prediction(r) for r in rows]}


@app.post("/api/predictions/replay")
def replay_predictions(count: int = 1, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Advance this persona's real historical transaction log by `count` rows,
    run each one through the real model, log it, and return it - the "Live
    Monitor" feed source when nothing has been submitted from the sandbox yet."""
    user_id = getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Not a persona user")

    count = max(1, min(count, 10))
    created = []
    try:
        for _ in range(count):
            historical_row = ml_service.get_next_historical_transaction(user_id)
            feature_payload = {k: historical_row[k] for k in ml_service.EXPECTED_FEATURE_COLUMNS}
            result = ml_service.predict_transaction(user_id, feature_payload)
            true_label = int(historical_row["label"]) if "label" in historical_row else None
            row = _log_prediction(db, user_id, "replay", feature_payload, result, true_label=true_label)
            created.append(_serialize_prediction(row))
    except Exception:
        logger.exception("replay_predictions failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="Replay failed. Check server logs for details.")

    return {"predictions": created}

@app.get("/api/analytics")
def get_analytics(days: int = None, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Everything the Analytics screen needs, computed from this persona's real
    logged predictions (traffic-derived) plus the real trained model artifact
    (fit-derived) - no fabricated aggregates."""
    user_id = getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Not a persona user")

    query = db.query(models.Prediction).filter(models.Prediction.user_id == user_id)
    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        query = query.filter(models.Prediction.created_at >= cutoff)
    rows = query.order_by(models.Prediction.created_at.asc(), models.Prediction.id.asc()).all()

    total = len(rows)
    blocked_rows = [r for r in rows if r.prediction == 1]
    blocked_count = len(blocked_rows)
    blocked_amount_sum = sum(r.transaction_amount for r in blocked_rows)
    avg_fraud_probability = (sum(r.fraud_probability for r in rows) / total) if total else 0.0

    # False positive rate is only meaningful where we know the ground truth -
    # that's real historical label, only available on replayed rows.
    fp_eligible = [r for r in rows if r.source == "replay" and r.true_label == 0]
    fp_rows = [r for r in fp_eligible if r.prediction == 1]
    false_positive_rate = (len(fp_rows) / len(fp_eligible)) if fp_eligible else None

    source_mix = {
        "manual": sum(1 for r in rows if r.source == "manual"),
        "replay": sum(1 for r in rows if r.source == "replay"),
    }

    hourly_counts = {}
    for r in rows:
        if r.created_at:
            bucket = r.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_counts[bucket] = hourly_counts.get(bucket, 0) + 1
    peak_hourly_count = max(hourly_counts.values()) if hourly_counts else 0

    cat_stats = {}
    for r in rows:
        stats = cat_stats.setdefault(r.merchant_category, {"count": 0, "blocked": 0})
        stats["count"] += 1
        if r.prediction == 1:
            stats["blocked"] += 1
    category_breakdown = [
        {
            "category": cat,
            "count": stats["count"],
            "share": (stats["count"] / total) if total else 0,
            "block_rate": (stats["blocked"] / stats["count"]) if stats["count"] else 0,
        }
        for cat, stats in sorted(cat_stats.items(), key=lambda kv: kv[1]["count"], reverse=True)
    ]

    timeline = [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "fraud_probability": r.fraud_probability,
            "prediction": r.prediction,
            "transaction_amount": r.transaction_amount,
        }
        for r in rows[-50:]
    ]

    # 10 real bins (0.0-0.1 ... 0.9-1.0) over every predicted probability for
    # this persona - not just the last-50 slice used for the timeline chart.
    probability_histogram = [0] * 10
    for r in rows:
        bin_index = min(int(r.fraud_probability * 10), 9)
        probability_histogram[bin_index] += 1

    recent_blocked = [
        _serialize_prediction(r)
        for r in sorted(blocked_rows, key=lambda r: r.created_at or datetime.min, reverse=True)[:8]
    ]

    return {
        "range_days": days,
        "totals": {
            "total_transactions": total,
            "blocked_count": blocked_count,
            "blocked_amount_sum": blocked_amount_sum,
            "avg_fraud_probability": avg_fraud_probability,
            "false_positive_rate": false_positive_rate,
            "false_positive_eligible_count": len(fp_eligible),
            "false_positive_count": len(fp_rows),
            "peak_hourly_count": peak_hourly_count,
            "source_mix": source_mix,
        },
        "category_breakdown": category_breakdown,
        "timeline": timeline,
        "probability_histogram": probability_histogram,
        "recent_blocked": recent_blocked,
        "feature_importances": ml_service.get_global_feature_importances(user_id, top_n=6),
        "model_last_modified": ml_service.get_model_last_modified(user_id),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Throwaway preview routes - confirm each migrated Stitch screen renders
# correctly against the shared layout(s). Not auth-gated, not wired to real
# data. Remove once real routes take over.
@app.get("/preview/manual-predict")
def preview_manual_predict(request: Request):
    return templates.TemplateResponse(
        "manual_predict_sandbox.html", {"request": request, "active_page": "manual-predict"}
    )


@app.get("/preview/live-monitor")
def preview_live_monitor(request: Request):
    return templates.TemplateResponse(
        "live_monitor_anomaly_stream.html", {"request": request, "active_page": "live-monitor"}
    )


@app.get("/preview/analytics")
def preview_analytics(request: Request):
    return templates.TemplateResponse(
        "persona_analytics_model_telemetry.html", {"request": request, "active_page": "analytics"}
    )


@app.get("/preview/login")
def preview_login(request: Request, db: Session = Depends(get_db)):
    persona_count = len(ml_service.get_valid_user_ids())
    avg_roc_auc = ml_service.get_average_roc_auc()

    # Real aggregate false-positive rate across every persona's replayed
    # history with known ground truth (true_label only exists on replay rows).
    fp_eligible = db.query(models.Prediction).filter(
        models.Prediction.source == "replay", models.Prediction.true_label == 0
    ).all()
    fp_blocked = [r for r in fp_eligible if r.prediction == 1]
    aggregate_fpr = (len(fp_blocked) / len(fp_eligible)) if fp_eligible else None

    return templates.TemplateResponse("demo_persona_login_gateway.html", {
        "request": request,
        "persona_count": persona_count,
        "avg_roc_auc": avg_roc_auc,
        "aggregate_fpr": aggregate_fpr,
    })
