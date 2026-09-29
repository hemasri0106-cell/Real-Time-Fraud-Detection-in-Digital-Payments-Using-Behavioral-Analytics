from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import logging
import os
import re
import secrets
import smtplib
import ssl
from email.message import EmailMessage

from fastapi import FastAPI, Depends, HTTPException, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field

import models
import schemas
import auth
from database import engine, get_db
import dashboard_data

models.Base.metadata.create_all(bind=engine)
OTP_TTL_MINUTES = 5
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_SECONDS = 60
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection Dashboard - Auth Service")

# Loosened for local dev with a React dev server. Tighten to your real
# frontend origin before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmailRequest(BaseModel):
    email: EmailStr


class OTPVerifyRequest(EmailRequest):
    otp: str = Field(pattern=r"^\d{6}$")


class DemoProfileRequest(BaseModel):
    user_id: str


def _otp_hash(email: str, code: str) -> str:
    key = os.environ.get("OTP_HASH_SECRET") or auth.SECRET_KEY
    return hmac.new(key.encode(), f"{email}:{code}".encode(), hashlib.sha256).hexdigest()


def _safe_smtp_error(exc: Exception) -> str:
    """Keep SMTP diagnostics useful while removing configured credentials and addresses."""
    message = str(exc)
    for name in ("SMTP_PASSWORD", "SMTP_USERNAME", "SMTP_FROM", "JWT_SECRET_KEY", "OTP_HASH_SECRET", "RESEND_API_KEY"):
        secret = os.environ.get(name)
        if not secret:
            continue
        candidates = {secret, secret.strip(), secret.strip().strip("\"'")}
        candidates.update(candidate.replace(" ", "") for candidate in tuple(candidates))
        for candidate in candidates:
            if candidate:
                message = message.replace(candidate, "[REDACTED]")
    message = re.sub(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+", "[EMAIL]", message)
    return " ".join(message.split())[:500] or "No SMTP error detail was provided"


def _send_otp_email(email: str, code: str) -> None:
    """Send OTP using Gmail SMTP with STARTTLS and a Google App Password."""
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT")
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM")
    if not all((host, port, username, password, sender)):
        raise HTTPException(status_code=503, detail="Gmail SMTP settings are not configured")
    try:
        port_number = int(port)
    except ValueError:
        raise HTTPException(status_code=503, detail="Gmail SMTP_PORT must be 587") from None
    if host != "smtp.gmail.com" or port_number != 587:
        raise HTTPException(status_code=503, detail="Gmail SMTP requires SMTP_HOST=smtp.gmail.com and SMTP_PORT=587")

    message = EmailMessage()
    message["Subject"] = "Your FraudEC sign-in code"
    message["From"] = sender
    message["To"] = email
    message.set_content(f"Your FraudEC verification code is {code}. It expires in {OTP_TTL_MINUTES} minutes.")
    try:
        with smtplib.SMTP(host, port_number, timeout=15) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(username, password)
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        logger.error("Gmail SMTP send failed (%s): %s", type(exc).__name__, _safe_smtp_error(exc))
        raise HTTPException(
            status_code=503,
            detail="Gmail rejected SMTP authentication. Check that SMTP_USERNAME is your Gmail address and SMTP_PASSWORD is a current Google App Password, not your normal Gmail password.",
        ) from None
    except Exception as exc:
        logger.error("Gmail SMTP send failed (%s): %s", type(exc).__name__, _safe_smtp_error(exc))
        raise HTTPException(status_code=503, detail="Gmail SMTP delivery failed; see the sanitized backend diagnostic.") from None


@app.post("/api/auth/send-otp")
def send_otp(payload: EmailRequest, db: Session = Depends(get_db)):
    email = str(payload.email).strip().lower()
    now = datetime.now(timezone.utc)
    recent = db.query(models.OTPChallenge).filter(
        models.OTPChallenge.email == email,
        models.OTPChallenge.created_at >= now - timedelta(seconds=OTP_RESEND_SECONDS),
        models.OTPChallenge.consumed_at.is_(None),
    ).first()
    if recent:
        raise HTTPException(status_code=429, detail="Please wait before requesting another code")
    code = f"{secrets.randbelow(1_000_000):06d}"
    challenge = models.OTPChallenge(email=email, otp_hash=_otp_hash(email, code), expires_at=now + timedelta(minutes=OTP_TTL_MINUTES), attempts=0)
    db.add(challenge)
    db.commit()
    try:
        _send_otp_email(email, code)
    except HTTPException:
        db.delete(challenge)
        db.commit()
        raise
    except Exception:
        db.delete(challenge)
        db.commit()
        raise HTTPException(status_code=503, detail="Email delivery failed")
    return {"message": "If this address can receive mail, a verification code has been sent.", "expires_in_seconds": OTP_TTL_MINUTES * 60, "resend_after_seconds": OTP_RESEND_SECONDS}


@app.post("/api/auth/verify-otp", response_model=schemas.Token)
def verify_otp(payload: OTPVerifyRequest, request: Request, db: Session = Depends(get_db)):
    email = str(payload.email).strip().lower()
    challenge = db.query(models.OTPChallenge).filter(models.OTPChallenge.email == email, models.OTPChallenge.consumed_at.is_(None)).order_by(models.OTPChallenge.created_at.desc()).first()
    now = datetime.now(timezone.utc)
    if not challenge or challenge.expires_at.replace(tzinfo=timezone.utc) <= now:
        raise HTTPException(status_code=400, detail="The code is invalid or expired")
    attempts_updated = db.query(models.OTPChallenge).filter(
        models.OTPChallenge.id == challenge.id,
        models.OTPChallenge.attempts < OTP_MAX_ATTEMPTS,
        models.OTPChallenge.consumed_at.is_(None),
    ).update({models.OTPChallenge.attempts: models.OTPChallenge.attempts + 1}, synchronize_session=False)
    if attempts_updated != 1:
        db.rollback()
        raise HTTPException(status_code=400, detail="The code is invalid or expired")
    db.refresh(challenge)
    if not hmac.compare_digest(challenge.otp_hash, _otp_hash(email, payload.otp)):
        db.commit()
        raise HTTPException(status_code=400, detail="The code is invalid or expired")
    challenge.consumed_at = now
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        user = models.User(username=f"email_{secrets.token_hex(5)}", email=email, hashed_password=auth.hash_password(secrets.token_urlsafe(32)), role=models.UserRole.analyst)
        db.add(user)
        db.flush()
    user.last_login_at = now
    user.last_login_ip = request.client.host if request.client else None
    user.last_login_device = request.headers.get("x-device-id", "unknown")
    db.commit()
    db.refresh(user)
    token = auth.create_access_token({"sub": user.username, "role": user.role.value, "email": user.email})
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.post("/api/auth/logout", status_code=204)
def logout(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
    except auth.JWTError:
        raise HTTPException(status_code=401, detail="Invalid session")
    jti, exp = payload.get("jti"), payload.get("exp")
    if jti and exp:
        db.merge(models.RevokedToken(jti=jti, expires_at=datetime.fromtimestamp(exp, timezone.utc)))
        db.commit()
    return None


@app.post("/api/auth/select-demo-profile")
def select_demo_profile(payload: DemoProfileRequest, current_user: models.User = Depends(auth.get_current_user)):
    if not dashboard_data.is_allowed_user_id(payload.user_id):
        raise HTTPException(status_code=400, detail="Unsupported dataset profile")
    token = auth.create_access_token({"sub": current_user.username, "role": current_user.role.value, "email": current_user.email, "user_id": payload.user_id, "dataset_access": "demo"})
    return {"access_token": token, "token_type": "bearer", "user": {"username": current_user.username, "email": current_user.email, "role": current_user.role.value, "user_id": payload.user_id}}


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
import ml_service

class DemoLoginRequest(BaseModel):
    persona_name: str
    user_id: str


def get_demo_profile_id(current_user: models.User) -> str:
    user_id = getattr(current_user, "user_id", None)
    if getattr(current_user, "dataset_access", None) != "demo" or not dashboard_data.is_allowed_user_id(user_id):
        raise HTTPException(status_code=403, detail="Select an allowlisted dataset profile to access demo data")
    return user_id

@app.post("/api/auth/demo-login")
def demo_login(request: DemoLoginRequest, current_user: models.User = Depends(auth.get_current_user)):
    if not dashboard_data.is_allowed_user_id(request.user_id):
        raise HTTPException(status_code=400, detail="Unsupported demo profile")
    access_token = auth.create_access_token(
        data={"sub": current_user.username, "role": current_user.role.value, "email": current_user.email, "user_id": request.user_id, "dataset_access": "demo"}
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user": {"username": current_user.username, "email": current_user.email, "role": current_user.role.value, "user_id": request.user_id}
    }

@app.get("/api/metrics")
def get_metrics(current_user: models.User = Depends(auth.get_current_user)):
    user_id = get_demo_profile_id(current_user)
    metrics = ml_service.get_user_metrics(user_id)
    return metrics

@app.get("/api/dashboard/overview", response_model=schemas.TransactionOverview)
def transaction_overview(current_user: models.User = Depends(auth.get_current_user)):
    user_id = get_demo_profile_id(current_user)
    try:
        return dashboard_data.get_overview(user_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=503, detail="Transaction dataset is unavailable")


@app.get("/api/dashboard/behavior", response_model=schemas.BehavioralAnalytics)
def behavioral_analytics(current_user: models.User = Depends(auth.get_current_user)):
    user_id = get_demo_profile_id(current_user)
    try:
        return dashboard_data.get_behavior(user_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=503, detail="Transaction dataset is unavailable")


@app.get("/api/prediction/options")
def prediction_options(current_user: models.User = Depends(auth.get_current_user)):
    user_id = get_demo_profile_id(current_user)
    try:
        return dashboard_data.get_prediction_options(user_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=503, detail="Transaction dataset is unavailable")


@app.get("/api/transactions", response_model=schemas.PaginatedTransactions)
def get_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    label: Optional[int] = Query(None, ge=0, le=1),
    search: Optional[str] = Query(None, min_length=1, max_length=100),
    sort_by: Literal["timestamp", "transaction_amount", "merchant_category", "label"] = "timestamp",
    sort_order: Literal["asc", "desc"] = "desc",
    current_user: models.User = Depends(auth.get_current_user),
):
    user_id = get_demo_profile_id(current_user)
    try:
        return dashboard_data.get_transaction_page(
            user_id, page, page_size, label, search, sort_by, sort_order
        )
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=503, detail="Transaction dataset is unavailable")

@app.post("/api/predict")
def predict(payload: dict, current_user: models.User = Depends(auth.get_current_user)):
    user_id = get_demo_profile_id(current_user)
    try:
        result = ml_service.predict_transaction(user_id, payload)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=503, detail="Prediction model is unavailable")
    except Exception:
        # Keep request bodies, authorization headers, and tokens out of logs.
        # The traceback identifies the failing artifact/runtime operation.
        logger.exception("Unexpected prediction failure for profile %s", user_id)
        raise HTTPException(status_code=500, detail="Prediction could not be completed")

@app.get("/api/health")
def health():
    return {"status": "ok"}
