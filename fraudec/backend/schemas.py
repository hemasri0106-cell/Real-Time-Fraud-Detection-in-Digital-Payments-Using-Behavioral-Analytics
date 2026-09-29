from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, field_validator
from models import UserRole


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.analyst

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login_at: Optional[datetime] = None
    last_login_ip: Optional[str] = None
    last_login_device: Optional[str] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class LoginRequest(BaseModel):
    username: str
    password: str
    device_id: Optional[str] = None


class VolumeTimeSeriesPoint(BaseModel):
    date: str
    transaction_count: int
    fraud_labelled_count: int


class AmountTimeSeriesPoint(BaseModel):
    date: str
    transaction_amount: float


class TransactionOverview(BaseModel):
    user_id: str
    label_semantics: str
    total_transaction_count: int
    fraud_labelled_count: int
    normal_count: int
    fraud_labelled_percentage: float
    total_transaction_amount: float
    average_transaction_amount: float
    transaction_volume_over_time: List[VolumeTimeSeriesPoint]
    transaction_amount_over_time: List[AmountTimeSeriesPoint]


class CountPoint(BaseModel):
    key: str
    count: int


class NumericBin(BaseModel):
    range: str
    count: int


class DailyCountPoint(BaseModel):
    date: str
    count: int


class DailyAverageAmountPoint(BaseModel):
    date: str
    average_amount: float


class BehavioralAnalytics(BaseModel):
    user_id: str
    transactions_by_hour: List[CountPoint]
    transactions_by_day_of_week: List[CountPoint]
    merchant_category_distribution: List[CountPoint]
    payment_method_distribution: List[CountPoint]
    device_type_distribution: List[CountPoint]
    new_device_count: int
    new_location_count: int
    new_merchant_count: int
    transaction_gap_distribution: List[NumericBin]
    distance_from_previous_transaction: List[NumericBin]
    daily_transaction_count_over_time: List[DailyCountPoint]
    average_amount_last_7_days_over_time: List[DailyAverageAmountPoint]


class SafeTransaction(BaseModel):
    transaction_id: str
    timestamp: datetime
    transaction_amount: float
    merchant_category: str
    payment_method: str
    device_type: str
    city: str
    hour_of_day: int
    day_of_week: int
    label: int


class PaginatedTransactions(BaseModel):
    user_id: str
    label_semantics: str
    page: int
    page_size: int
    total: int
    total_pages: int
    transactions: List[SafeTransaction]
