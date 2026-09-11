from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.config import settings

Base = declarative_base()

_engine = None
_SessionLocal = None


def init_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine, _SessionLocal


def get_engine():
    global _engine
    if _engine is None:
        init_engine()
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        init_engine()
    return _SessionLocal


def get_db():
    if _SessionLocal is None:
        init_engine()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_postgis():
    """Enable PostGIS extension - call this before creating tables."""
    if _engine is None:
        init_engine()
    with _engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        conn.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
        conn.commit()