from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite for the college-project demo. Swap the URL for Postgres/MySQL
# later without touching anything else - SQLAlchemy abstracts it.
SQLALCHEMY_DATABASE_URL = "sqlite:///./fraud_dashboard.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
