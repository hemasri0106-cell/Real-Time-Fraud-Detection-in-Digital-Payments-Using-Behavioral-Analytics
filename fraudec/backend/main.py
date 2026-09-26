from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import models
import schemas
import auth
from database import engine, get_db

models.Base.metadata.create_all(bind=engine)

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

@app.post("/api/auth/demo-login")
def demo_login(request: DemoLoginRequest):
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

@app.post("/api/predict")
def predict(payload: dict, current_user: models.User = Depends(auth.get_current_user)):
    user_id = getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Not a persona user")
    try:
        result = ml_service.predict_transaction(user_id, payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ok"}
