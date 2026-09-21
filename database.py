"""
SQLite database models and helpers for IQ Option accounts and signals.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# Default DB file next to this module
DB_PATH = os.getenv("DATABASE_URL", "sqlite:///./qt_trading.db")
if DB_PATH.startswith("sqlite:///./"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = f"sqlite:///{os.path.join(base_dir, 'qt_trading.db')}"

engine = create_engine(
    DB_PATH,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Account(Base):
    """Stored IQ Option / broker account credentials & last known state."""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hint = Column(String(64), nullable=True)
    broker = Column(String(50), default="iq")  # iq | pocket
    mode = Column(String(20), default="demo")  # demo | real
    session_id = Column(String(128), unique=True, index=True, nullable=True)
    balance = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")
    connected = Column(Boolean, default=False)
    last_connect_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    signals = relationship("Signal", back_populates="account", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="account", cascade="all, delete-orphan")


class Signal(Base):
    """Trading signals (from Telegram or internal)."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(64), unique=True, index=True, nullable=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    pair = Column(String(32), nullable=False)  # e.g. EUR/USD
    direction = Column(String(10), nullable=False)  # BUY | SELL
    minutes = Column(Integer, default=5)
    confidence = Column(String(16), default="—")
    raw_text = Column(Text, nullable=True)
    source = Column(String(50), default="telegram")
    status = Column(String(20), default="pending")  # pending | confirmed | expired | deleted
    created_at = Column(DateTime, default=datetime.utcnow)
    confirmed_at = Column(DateTime, nullable=True)

    account = relationship("Account", back_populates="signals")


class Trade(Base):
    """Recorded trades (demo simulation or real IQ orders)."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(64), unique=True, index=True, nullable=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    pair = Column(String(32), nullable=False)
    direction = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    payout_pct = Column(Float, default=85.0)
    minutes = Column(Integer, default=1)
    status = Column(String(20), default="open")  # open | closed
    won = Column(Boolean, nullable=True)
    profit = Column(Float, nullable=True)
    iq_order_id = Column(String(64), nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)

    account = relationship("Account", back_populates="trades")


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yield a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Account helpers ----------

def upsert_account(
    db: Session,
    *,
    email: str,
    broker: str = "iq",
    mode: str = "demo",
    session_id: Optional[str] = None,
    balance: float = 0.0,
    currency: str = "USD",
    connected: bool = True,
) -> Account:
    acc = db.query(Account).filter(Account.email == email).first()
    now = datetime.utcnow()
    if acc is None:
        acc = Account(
            email=email,
            broker=broker,
            mode=mode,
            session_id=session_id,
            balance=balance,
            currency=currency,
            connected=connected,
            last_connect_at=now if connected else None,
        )
        db.add(acc)
    else:
        acc.broker = broker
        acc.mode = mode
        acc.session_id = session_id
        acc.balance = balance
        acc.currency = currency
        acc.connected = connected
        acc.updated_at = now
        if connected:
            acc.last_connect_at = now
    db.commit()
    db.refresh(acc)
    return acc


def get_account_by_session(db: Session, session_id: str) -> Optional[Account]:
    return db.query(Account).filter(Account.session_id == session_id).first()


def get_account_by_email(db: Session, email: str) -> Optional[Account]:
    return db.query(Account).filter(Account.email == email).first()


def mark_disconnected(db: Session, session_id: str) -> None:
    acc = get_account_by_session(db, session_id)
    if acc:
        acc.connected = False
        acc.session_id = None
        acc.updated_at = datetime.utcnow()
        db.commit()


# ---------- Signal helpers ----------

def save_signal(
    db: Session,
    *,
    pair: str,
    direction: str,
    minutes: int = 5,
    confidence: str = "—",
    raw_text: Optional[str] = None,
    source: str = "telegram",
    account_id: Optional[int] = None,
    external_id: Optional[str] = None,
) -> Signal:
    sig = Signal(
        external_id=external_id,
        account_id=account_id,
        pair=pair.upper().replace(" ", ""),
        direction=direction.upper(),
        minutes=minutes,
        confidence=confidence,
        raw_text=raw_text,
        source=source,
        status="pending",
    )
    db.add(sig)
    db.commit()
    db.refresh(sig)
    return sig


def list_signals(
    db: Session,
    status: Optional[str] = "pending",
    limit: int = 100,
) -> List[Signal]:
    q = db.query(Signal).order_by(Signal.created_at.desc())
    if status:
        q = q.filter(Signal.status == status)
    return q.limit(limit).all()


def delete_signal(db: Session, signal_id: int) -> bool:
    sig = db.query(Signal).filter(Signal.id == signal_id).first()
    if not sig:
        return False
    db.delete(sig)
    db.commit()
    return True


def clear_signals(db: Session, status: Optional[str] = None) -> int:
    q = db.query(Signal)
    if status:
        q = q.filter(Signal.status == status)
    count = q.delete()
    db.commit()
    return count


# ---------- Trade helpers ----------

def save_trade(
    db: Session,
    *,
    pair: str,
    direction: str,
    amount: float,
    payout_pct: float = 85.0,
    minutes: int = 1,
    account_id: Optional[int] = None,
    external_id: Optional[str] = None,
    iq_order_id: Optional[str] = None,
    expires_at: Optional[datetime] = None,
) -> Trade:
    trade = Trade(
        external_id=external_id,
        account_id=account_id,
        pair=pair,
        direction=direction,
        amount=amount,
        payout_pct=payout_pct,
        minutes=minutes,
        status="open",
        iq_order_id=iq_order_id,
        expires_at=expires_at,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade
