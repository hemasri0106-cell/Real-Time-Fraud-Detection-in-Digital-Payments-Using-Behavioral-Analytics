import enum
from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func
from database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    analyst = "analyst"
    persona = "persona"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.analyst, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Behavioral metadata captured at login - mirrors the same signal set
    # (device, IP, timestamp) the fraud model itself profiles. Useful later
    # if you want to demo "we practice what we detect."
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(String, nullable=True)
    last_login_device = Column(String, nullable=True)
