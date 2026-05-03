"""SQLAlchemy engine + session factory used by every service."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

_engine: Engine | None = None
_factory: sessionmaker | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        kwargs: dict = {"future": True, "pool_pre_ping": True}
        if settings.database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs["pool_size"] = settings.database_pool_size
            kwargs["max_overflow"] = settings.database_max_overflow
            kwargs["pool_recycle"] = 1800
        _engine = create_engine(settings.database_url, **kwargs)
    return _engine


def session_factory() -> sessionmaker:
    global _factory
    if _factory is None:
        _factory = sessionmaker(bind=engine(), expire_on_commit=False, future=True)
    return _factory


@contextmanager
def session_scope() -> Iterator[Session]:
    s = session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
