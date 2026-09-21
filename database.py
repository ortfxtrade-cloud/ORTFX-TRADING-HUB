"""SQLite database for accounts, signals, trades."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    Session as DBSession,
    declarative_base,
    relationship,
    sessionmaker,
)

# ── Engine ───────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    base_dir = Path(__file__).resolve().parent
    DATABASE_URL = f"sqlite:///{base_dir / 'qt_trading.db'}"

# Render Postgres URLs sometimes start with postgres:// — SQLAlchemy 2 wants
# postgresql://. Normalize.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
else:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Models ───────────────────────────────────────────────────────────────────


class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    broker = Column(String(50), default="iq")
    mode = Column(String(20), default="demo")
    session_id = Column(String(128), unique=True, index=True, nullable=True)
    balance = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")
    connected = Column(Boolean, default=False)
    last_connect_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    signals = relationship("Signal", back_populates="account", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="account", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Account id={self.id} email={self.email!r} mode={self.mode}>"


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(64), unique=True, index=True, nullable=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=True)
    pair = Column(String(32), nullable=False)
    direction = Column(String(10), nullable=False)
    minutes = Column(Integer, default=5)
    confidence = Column(String(16), default="—")
    raw_text = Column(Text, nullable=True)
    source = Column(String(50), default="auto")
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=utcnow)
    confirmed_at = Column(DateTime, nullable=True)

    account = relationship("Account", back_populates="signals")

    __table_args__ = (
        Index("ix_signals_status_created", "status", "created_at"),
    )


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(64), unique=True, index=True, nullable=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=True)
    pair = Column(String(32), nullable=False)
    direction = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    payout_pct = Column(Float, default=85.0)
    minutes = Column(Integer, default=1)
    status = Column(String(20), default="open")
    won = Column(Boolean, nullable=True)
    profit = Column(Float, nullable=True)
    iq_order_id = Column(String(64), nullable=True)
    opened_at = Column(DateTime, default=utcnow)
    expires_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)

    account = relationship("Account", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_account_status", "account_id", "status"),
        Index("ix_trades_opened_at", "opened_at"),
    )

    def __repr__(self) -> str:
        return f"<Trade id={self.id} pair={self.pair} dir={self.direction}>"


# ── Setup ────────────────────────────────────────────────────────────────────


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Accounts ─────────────────────────────────────────────────────────────────


def upsert_account(
    db: DBSession,
    *,
    email: str,
    broker: str = "iq",
    mode: str = "demo",
    session_id: Optional[str] = None,
    balance: float = 0.0,
    currency: str = "USD",
    connected: bool = True,
) -> Account:
    email = (email or "").strip().lower()
    acc = db.query(Account).filter(Account.email == email).first()
    now = utcnow()
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

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        acc = db.query(Account).filter(Account.email == email).first()
        if acc is None:
            raise

    db.refresh(acc)
    return acc


def get_account_by_session(db: DBSession, session_id: str) -> Optional[Account]:
    return db.query(Account).filter(Account.session_id == session_id).first()


def get_account_by_email(db: DBSession, email: str) -> Optional[Account]:
    return db.query(Account).filter(Account.email == (email or "").strip().lower()).first()


def mark_disconnected(db: DBSession, session_id: str) -> None:
    acc = get_account_by_session(db, session_id)
    if acc and acc.session_id == session_id:
        acc.connected = False
        acc.session_id = None
        acc.updated_at = utcnow()
        db.commit()


# ── Signals ──────────────────────────────────────────────────────────────────


def save_signal(
    db: DBSession,
    *,
    pair: str,
    direction: str,
    minutes: int = 5,
    confidence: str = "—",
    raw_text: Optional[str] = None,
    source: str = "auto",
    account_id: Optional[int] = None,
    external_id: Optional[str] = None,
) -> Signal:
    sig = Signal(
        external_id=external_id,
        account_id=account_id,
        pair=(pair or "").upper().replace(" ", ""),
        direction=(direction or "").strip().upper(),
        minutes=minutes,
        confidence=confidence,
        raw_text=raw_text,
        source=source,
        status="pending",
    )
    db.add(sig)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if external_id:
            existing = db.query(Signal).filter(Signal.external_id == external_id).first()
            if existing:
                return existing
        raise
    db.refresh(sig)
    return sig


def list_signals(
    db: DBSession, status: Optional[str] = "pending", limit: int = 100
) -> List[Signal]:
    limit = max(1, min(int(limit or 100), 500))
    q = db.query(Signal).order_by(Signal.created_at.desc())
    if status:
        q = q.filter(Signal.status == status)
    return q.limit(limit).all()


def delete_signal(db: DBSession, signal_id: int) -> bool:
    sig = db.query(Signal).filter(Signal.id == signal_id).first()
    if not sig:
        return False
    db.delete(sig)
    db.commit()
    return True


def clear_signals(db: DBSession, status: Optional[str] = None) -> int:
    q = db.query(Signal)
    if status:
        q = q.filter(Signal.status == status)
    count = q.delete(synchronize_session=False)
    db.commit()
    return count or 0


# ── Trades ───────────────────────────────────────────────────────────────────


def save_trade(
    db: DBSession,
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
        pair=(pair or "").strip(),
        direction=(direction or "").strip().upper(),
        amount=amount,
        payout_pct=payout_pct,
        minutes=minutes,
        status="open",
        iq_order_id=iq_order_id,
        expires_at=expires_at,
    )
    db.add(trade)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if iq_order_id:
            existing = db.query(Trade).filter(Trade.iq_order_id == iq_order_id).first()
            if existing:
                return existing
        if external_id:
            existing = db.query(Trade).filter(Trade.external_id == external_id).first()
            if existing:
                return existing
        raise
    db.refresh(trade)
    return trade


def list_trades(
    db: DBSession, status: Optional[str] = None, limit: int = 100
) -> List[Trade]:
    limit = max(1, min(int(limit or 100), 500))
    q = db.query(Trade).order_by(Trade.opened_at.desc())
    if status:
        q = q.filter(Trade.status == status)
    return q.limit(limit).all()


def close_expired_trades(db: DBSession) -> int:
    """Move open trades whose expires_at has passed → closed."""
    now = utcnow()
    expired = (
        db.query(Trade)
        .filter(Trade.status == "open")
        .filter(Trade.expires_at.isnot(None))
        .filter(Trade.expires_at <= now)
        .all()
    )
    for t in expired:
        t.status = "closed"
        t.closed_at = now
    if expired:
        db.commit()
    return len(expired)
