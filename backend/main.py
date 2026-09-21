"""
QT Trading backend — FastAPI

Endpoints expected by the frontend (index.html):
  POST /api/iq/connect      { email, password, mode }
  POST /api/iq/disconnect  { session_id }

Extra helpers for signals, balance, and health.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import connect as iq
from database import (
    init_db,
    get_db,
    upsert_account,
    get_account_by_session,
    mark_disconnected,
    save_signal,
    list_signals,
    delete_signal,
    clear_signals,
    save_trade,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qt.main")

API_KEY = os.getenv("BACKEND_API_KEY", "").strip()

app = FastAPI(
    title="QT Trading API",
    description="Backend for IQ Option connect + signals storage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Database ready")


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ---------- Schemas ----------

class ConnectBody(BaseModel):
    email: str
    password: str
    mode: str = Field(default="demo", description="demo | real")


class DisconnectBody(BaseModel):
    session_id: str


class SignalBody(BaseModel):
    pair: str
    direction: str  # BUY | SELL
    minutes: int = 5
    confidence: str = "—"
    raw_text: Optional[str] = None
    source: str = "telegram"
    account_id: Optional[int] = None
    external_id: Optional[str] = None


class TradeBody(BaseModel):
    session_id: Optional[str] = None
    pair: str
    direction: str
    amount: float
    minutes: int = 1
    payout_pct: float = 85.0
    place_on_iq: bool = False


# ---------- Health ----------

@app.get("/")
def root():
    return {"service": "QT Trading API", "status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "sessions": len(iq.list_active_sessions()),
    }


# ---------- IQ Option ----------

@app.post("/api/iq/connect")
def api_iq_connect(
    body: ConnectBody,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    ok, payload = iq.connect_iq(body.email, body.password, body.mode)
    if not ok:
        return {
            "status": "error",
            "message": payload.get("message", "Connection failed"),
        }

    upsert_account(
        db,
        email=body.email.strip().lower(),
        broker="iq",
        mode=body.mode.lower(),
        session_id=payload["session_id"],
        balance=float(payload.get("balance") or 0),
        currency=payload.get("currency") or "USD",
        connected=True,
    )

    return payload


@app.post("/api/iq/disconnect")
def api_iq_disconnect(
    body: DisconnectBody,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    ok, msg = iq.disconnect_iq(body.session_id)
    mark_disconnected(db, body.session_id)
    return {"status": "success" if ok else "error", "message": msg}


@app.get("/api/iq/balance")
def api_iq_balance(
    session_id: str,
    _: None = Depends(require_api_key),
):
    bal = iq.refresh_balance(session_id)
    if bal is None:
        raise HTTPException(status_code=404, detail="Session not found or disconnected")
    sess = iq.get_session(session_id)
    return {
        "status": "success",
        "balance": bal,
        "currency": sess.currency if sess else "USD",
        "mode": sess.mode if sess else None,
    }


@app.get("/api/iq/sessions")
def api_iq_sessions(_: None = Depends(require_api_key)):
    return {"sessions": iq.list_active_sessions()}


# ---------- Signals ----------

@app.post("/api/signals")
def api_create_signal(
    body: SignalBody,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    sig = save_signal(
        db,
        pair=body.pair,
        direction=body.direction,
        minutes=body.minutes,
        confidence=body.confidence,
        raw_text=body.raw_text,
        source=body.source,
        account_id=body.account_id,
        external_id=body.external_id,
    )
    return {
        "status": "success",
        "id": sig.id,
        "pair": sig.pair,
        "direction": sig.direction,
        "minutes": sig.minutes,
        "confidence": sig.confidence,
        "created_at": sig.created_at.isoformat() + "Z",
    }


@app.get("/api/signals")
def api_list_signals(
    status: Optional[str] = "pending",
    limit: int = 100,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    rows = list_signals(db, status=status, limit=limit)
    return {
        "status": "success",
        "count": len(rows),
        "signals": [
            {
                "id": s.id,
                "pair": s.pair,
                "direction": s.direction,
                "minutes": s.minutes,
                "confidence": s.confidence,
                "raw_text": s.raw_text,
                "source": s.source,
                "status": s.status,
                "created_at": s.created_at.isoformat() + "Z" if s.created_at else None,
            }
            for s in rows
        ],
    }


@app.delete("/api/signals/{signal_id}")
def api_delete_signal(
    signal_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    if not delete_signal(db, signal_id):
        raise HTTPException(status_code=404, detail="Signal not found")
    return {"status": "success"}


@app.delete("/api/signals")
def api_clear_signals(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    n = clear_signals(db, status=status)
    return {"status": "success", "deleted": n}


# ---------- Trade ----------

@app.post("/api/trade")
def api_place_trade(
    body: TradeBody,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    iq_order_id = None
    if body.place_on_iq and body.session_id:
        ok, result = iq.place_binary_order(
            body.session_id,
            active=body.pair,
            direction=body.direction,
            amount=body.amount,
            duration_min=body.minutes,
        )
        if not ok:
            return {"status": "error", "message": result.get("message")}
        iq_order_id = result.get("order_id")

    account_id = None
    if body.session_id:
        acc = get_account_by_session(db, body.session_id)
        if acc:
            account_id = acc.id

    trade = save_trade(
        db,
        pair=body.pair,
        direction=body.direction.upper(),
        amount=body.amount,
        payout_pct=body.payout_pct,
        minutes=body.minutes,
        account_id=account_id,
        iq_order_id=iq_order_id,
    )
    return {
        "status": "success",
        "trade_id": trade.id,
        "iq_order_id": iq_order_id,
        "pair": trade.pair,
        "direction": trade.direction,
        "amount": trade.amount,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
