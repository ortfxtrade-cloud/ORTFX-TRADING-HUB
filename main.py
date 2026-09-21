"""
QT Trading — FastAPI (UI + API).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import yfinance as yf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

import connect as iq
from database import (
    SessionLocal,
    Signal,
    Trade,
    advance_expired_trades,
    clear_signals,
    delete_signal,
    get_account_by_session,
    get_db,
    init_db,
    list_signals,
    list_trades,
    list_unresolved_trades,
    mark_disconnected,
    resolve_trade,
    save_signal,
    save_trade,
    upsert_account,
    void_trade,
)
from signal_worker import signal_loop

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qt.main")

API_KEY = os.getenv("BACKEND_API_KEY", "").strip()
ENV = os.getenv("ENV", "dev").lower()

IQ_EMAIL = os.getenv("IQ_EMAIL", "").strip()
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "")
IQ_MODE = os.getenv("IQ_MODE", "demo").strip().lower()
IQ_AUTO_CONNECT = os.getenv("IQ_AUTO_CONNECT", "true").lower() in ("1", "true", "yes")

RESULT_POLL_INTERVAL = int(os.getenv("RESULT_POLL_INTERVAL", "10"))
RESULT_VOID_AFTER = int(os.getenv("RESULT_VOID_AFTER", "600"))

if not API_KEY and ENV == "production":
    raise RuntimeError("BACKEND_API_KEY must be set in production")


# ── Background: trade result resolver ────────────────────────────────────────


async def _trade_result_resolver(stop: asyncio.Event) -> None:
    """
    Every RESULT_POLL_INTERVAL seconds:
      1. Move expired 'open' trades → 'pending_result'
      2. Ask IQ for the result of every 'pending_result' trade
      3. Resolve as won/lost, or void if too much time has passed
    """
    logger.info(
        "Trade resolver started (poll=%ds, void_after=%ds)",
        RESULT_POLL_INTERVAL, RESULT_VOID_AFTER,
    )

    while not stop.is_set():
        try:
            # Stage 1: advance expired open trades
            db = SessionLocal()
            try:
                n_advanced = advance_expired_trades(db)
                if n_advanced:
                    logger.info("Advanced %d trade(s) to pending_result", n_advanced)
            finally:
                db.close()

            # Stage 2: query IQ for results
            sessions = iq.list_active_sessions()
            live = next((s for s in sessions if s["alive"]), None)
            session_id = live["session_id"] if live else None

            db = SessionLocal()
            try:
                unresolved = list_unresolved_trades(db)
                for trade in unresolved:
                    if trade.status == "open":
                        continue
                    if not session_id:
                        continue

                    result = iq.get_order_result(session_id, trade.iq_order_id)

                    if result and result.get("status") == "closed":
                        won = result.get("won")
                        profit = result.get("profit")
                        if profit is None and won is not None:
                            if won:
                                profit = round(trade.amount * (trade.payout_pct / 100.0), 2)
                            else:
                                profit = -trade.amount
                        resolve_trade(db, trade.id, won=won, profit=profit)
                        logger.info(
                            "Resolved trade #%d %s %s → won=%s profit=%s",
                            trade.id, trade.pair, trade.direction, won, profit,
                        )
                    elif result and result.get("status") == "open":
                        continue
                    else:
                        # No usable answer. Void after grace period.
                        if trade.expires_at:
                            age = (datetime.utcnow() - trade.expires_at).total_seconds()
                            if age > RESULT_VOID_AFTER:
                                void_trade(db, trade.id)
                                logger.warning(
                                    "Voided trade #%d (no IQ result after %ds)",
                                    trade.id, int(age),
                                )
            finally:
                db.close()

        except Exception as e:
            logger.exception("trade resolver error: %s", e)

        try:
            await asyncio.wait_for(stop.wait(), timeout=RESULT_POLL_INTERVAL)
        except asyncio.TimeoutError:
            pass

    logger.info("Trade resolver stopped")


# ── Auto connect ─────────────────────────────────────────────────────────────


async def _auto_connect_iq() -> None:
    if not (IQ_AUTO_CONNECT and IQ_EMAIL and IQ_PASSWORD):
        logger.info("IQ auto-connect skipped (no creds or disabled)")
        return

    for attempt in range(1, 4):
        ok, payload = iq.connect_iq(IQ_EMAIL, IQ_PASSWORD, IQ_MODE)
        if ok:
            logger.info(
                "IQ auto-connect OK: %s (%s) balance=%.2f",
                IQ_EMAIL, IQ_MODE, payload.get("balance", 0),
            )
            db = SessionLocal()
            try:
                upsert_account(
                    db,
                    email=IQ_EMAIL,
                    broker="iq",
                    mode=IQ_MODE,
                    session_id=payload["session_id"],
                    balance=float(payload.get("balance") or 0),
                    currency=payload.get("currency") or "USD",
                    connected=True,
                )
            finally:
                db.close()
            return
        logger.warning(
            "IQ auto-connect attempt %d failed: %s", attempt, payload.get("message")
        )
        await asyncio.sleep(5)

    logger.error("IQ auto-connect gave up after 3 attempts")


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database ready")

    stop = asyncio.Event()
    worker = asyncio.create_task(signal_loop(stop))
    resolver = asyncio.create_task(_trade_result_resolver(stop))
    await _auto_connect_iq()

    logger.info("Background tasks started")
    yield

    stop.set()
    for task in (worker, resolver):
        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.TimeoutError:
            task.cancel()
    for s in iq.list_active_sessions():
        iq.disconnect_iq(s["session_id"])


app = FastAPI(title="QT Trading API", version="1.0.0", lifespan=lifespan)

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ── Schemas ──────────────────────────────────────────────────────────────────


class ConnectBody(BaseModel):
    email: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = Field(default=None, max_length=256, repr=False)
    mode: Optional[Literal["demo", "real"]] = None


class DisconnectBody(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class SignalBody(BaseModel):
    pair: str = Field(min_length=3, max_length=32)
    direction: str = Field(pattern=r"^(BUY|SELL|buy|sell|CALL|PUT|call|put)$")
    minutes: int = Field(default=5, ge=1, le=1440)
    confidence: str = Field(default="—", max_length=16)
    raw_text: Optional[str] = None
    source: str = Field(default="auto", max_length=50)
    account_id: Optional[int] = None
    external_id: Optional[str] = Field(default=None, max_length=64)


class ConfirmSignalBody(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=128)
    amount: float = Field(default=1.0, gt=0, le=100_000)
    minutes_override: Optional[int] = Field(default=None, ge=1, le=60)


class TradeBody(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=128)
    pair: str = Field(min_length=3, max_length=32)
    direction: str = Field(pattern=r"^(BUY|SELL|buy|sell|CALL|PUT|call|put)$")
    amount: float = Field(gt=0, le=100_000)
    minutes: int = Field(default=1, ge=1, le=60)
    payout_pct: float = Field(default=85.0, ge=0, le=1000)
    place_on_iq: bool = False


# ── UI ───────────────────────────────────────────────────────────────────────


@app.get("/")
def serve_ui():
    path = Path(__file__).parent / "index.html"
    if path.exists():
        return FileResponse(str(path), media_type="text/html")
    return {"service": "QT Trading API", "status": "ok", "note": "index.html not found"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "sessions": len(iq.list_active_sessions()),
    }


# ── IQ ───────────────────────────────────────────────────────────────────────


@app.post("/api/iq/connect")
def api_iq_connect(
    body: ConnectBody,
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    email = (body.email or IQ_EMAIL).strip().lower()
    password = body.password or IQ_PASSWORD
    mode = (body.mode or IQ_MODE or "demo").lower()

    if not email or not password:
        raise HTTPException(400, "Email and password required (or set IQ_EMAIL/IQ_PASSWORD)")

    ok, payload = iq.connect_iq(email, password, mode)
    if not ok:
        raise HTTPException(401, payload.get("message", "Connection failed"))

    upsert_account(
        db,
        email=email,
        broker="iq",
        mode=mode,
        session_id=payload["session_id"],
        balance=float(payload.get("balance") or 0),
        currency=payload.get("currency") or "USD",
        connected=True,
    )
    return payload


@app.post("/api/iq/disconnect")
def api_iq_disconnect(
    body: DisconnectBody,
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    ok, msg = iq.disconnect_iq(body.session_id)
    if not ok:
        raise HTTPException(404, msg)
    mark_disconnected(db, body.session_id)
    return {"status": "success", "message": msg}


@app.get("/api/iq/balance")
def api_iq_balance(session_id: str, _: None = Depends(require_api_key)):
    sess = iq.get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if not sess.is_alive():
        raise HTTPException(410, "Session disconnected")
    bal = iq.refresh_balance(session_id)
    if bal is None:
        raise HTTPException(502, "Could not fetch balance from IQ")
    return {
        "status": "success",
        "balance": bal,
        "currency": sess.currency,
        "mode": sess.mode,
    }


@app.get("/api/iq/sessions")
def api_iq_sessions(_: None = Depends(require_api_key)):
    return {"sessions": iq.list_active_sessions()}


@app.get("/api/iq/current")
def api_iq_current(_: None = Depends(require_api_key)):
    sess = iq.get_first_session()
    if not sess:
        return {"status": "none"}
    return {
        "status": "connected",
        "session_id": sess.session_id,
        "email": sess.email,
        "mode": sess.mode,
        "balance": sess.balance,
        "currency": sess.currency,
        "account_type": sess.account_type,
    }


# ── Signals ──────────────────────────────────────────────────────────────────


@app.post("/api/signals")
def api_create_signal(
    body: SignalBody,
    db: DBSession = Depends(get_db),
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
        "created_at": sig.created_at.isoformat() + "Z" if sig.created_at else None,
    }


@app.get("/api/signals")
def api_list_signals(
    status: Optional[str] = "pending",
    limit: int = 100,
    db: DBSession = Depends(get_db),
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
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    if not delete_signal(db, signal_id):
        raise HTTPException(404, "Signal not found")
    return {"status": "success"}


@app.delete("/api/signals")
def api_clear_signals(
    status: Optional[str] = None,
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    n = clear_signals(db, status=status)
    return {"status": "success", "deleted": n}


@app.post("/api/signals/{signal_id}/confirm")
def api_confirm_signal(
    signal_id: int,
    body: ConfirmSignalBody,
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    sig = db.query(Signal).filter(Signal.id == signal_id).first()
    if not sig:
        raise HTTPException(404, "Signal not found")
    if sig.status != "pending":
        raise HTTPException(409, f"Signal already {sig.status}")

    minutes = body.minutes_override or sig.minutes or 5

    session_id = body.session_id
    if not session_id:
        sess = iq.get_first_session()
        if sess:
            session_id = sess.session_id

    if not session_id:
        raise HTTPException(400, "No IQ session connected")

    ok, result = iq.place_binary_order(
        session_id,
        active=sig.pair,
        direction=sig.direction,
        amount=body.amount,
        duration_min=minutes,
    )
    if not ok:
        raise HTTPException(502, result.get("message", "Order rejected"))
    iq_order_id = result.get("order_id")

    account_id = None
    acc = get_account_by_session(db, session_id)
    if acc:
        account_id = acc.id

    expires = datetime.utcnow() + timedelta(minutes=minutes)
    trade = save_trade(
        db,
        pair=sig.pair,
        direction=sig.direction.upper(),
        amount=body.amount,
        payout_pct=85.0,
        minutes=minutes,
        account_id=account_id,
        iq_order_id=iq_order_id,
        expires_at=expires,
    )

    sig.status = "executed"
    sig.confirmed_at = datetime.utcnow()
    db.commit()

    return {
        "status": "success",
        "signal_id": sig.id,
        "trade_id": trade.id,
        "iq_order_id": iq_order_id,
        "pair": sig.pair,
        "direction": sig.direction,
        "amount": body.amount,
        "minutes": minutes,
    }


@app.post("/api/signals/{signal_id}/reject")
def api_reject_signal(
    signal_id: int,
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    sig = db.query(Signal).filter(Signal.id == signal_id).first()
    if not sig:
        raise HTTPException(404, "Signal not found")
    if sig.status != "pending":
        raise HTTPException(409, f"Signal already {sig.status}")
    sig.status = "rejected"
    db.commit()
    return {"status": "success", "signal_id": sig.id}


@app.post("/api/signals/generate")
async def api_generate_now(_: None = Depends(require_api_key)):
    from signal_worker import scan_once
    asyncio.create_task(scan_once())
    return {"status": "scan_started"}


# ── Trades ───────────────────────────────────────────────────────────────────


@app.get("/api/trades")
def api_list_trades(
    status: Optional[str] = None,
    limit: int = 100,
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    rows = list_trades(db, status=status, limit=limit)

    closed = [t for t in rows if t.status == "closed" and t.won is not None]
    wins = sum(1 for t in closed if t.won)
    losses = sum(1 for t in closed if t.won is False)
    total_pnl = sum((t.profit or 0) for t in closed)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    return {
        "status": "success",
        "count": len(rows),
        "summary": {
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
        },
        "trades": [
            {
                "id": t.id,
                "pair": t.pair,
                "direction": t.direction,
                "amount": t.amount,
                "payout_pct": t.payout_pct,
                "minutes": t.minutes,
                "status": t.status,
                "won": t.won,
                "profit": t.profit,
                "iq_order_id": t.iq_order_id,
                "opened_at": t.opened_at.isoformat() + "Z" if t.opened_at else None,
                "expires_at": t.expires_at.isoformat() + "Z" if t.expires_at else None,
                "closed_at": t.closed_at.isoformat() + "Z" if t.closed_at else None,
            }
            for t in rows
        ],
    }


@app.post("/api/trades/refresh")
async def api_refresh_trades(_: None = Depends(require_api_key)):
    """Force one pass of the trade resolver right now."""
    sessions = iq.list_active_sessions()
    live = next((s for s in sessions if s["alive"]), None)
    session_id = live["session_id"] if live else None

    db = SessionLocal()
    updated = 0
    try:
        advance_expired_trades(db)
        unresolved = list_unresolved_trades(db)
        for trade in unresolved:
            if trade.status == "open":
                continue
            if not session_id:
                break
            result = iq.get_order_result(session_id, trade.iq_order_id)
            if result and result.get("status") == "closed":
                won = result.get("won")
                profit = result.get("profit")
                if profit is None and won is not None:
                    profit = (
                        round(trade.amount * (trade.payout_pct / 100.0), 2)
                        if won else -trade.amount
                    )
                resolve_trade(db, trade.id, won=won, profit=profit)
                updated += 1
    finally:
        db.close()
    return {"status": "success", "updated": updated}


@app.post("/api/trade")
def api_place_trade(
    body: TradeBody,
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    session_id = body.session_id
    if body.place_on_iq and not session_id:
        sess = iq.get_first_session()
        if sess:
            session_id = sess.session_id
    if body.place_on_iq and not session_id:
        raise HTTPException(400, "session_id is required when place_on_iq=true")

    iq_order_id = None
    if body.place_on_iq and session_id:
        ok, result = iq.place_binary_order(
            session_id,
            active=body.pair,
            direction=body.direction,
            amount=body.amount,
            duration_min=body.minutes,
        )
        if not ok:
            raise HTTPException(502, result.get("message", "Order rejected"))
        iq_order_id = result.get("order_id")

    account_id = None
    if session_id:
        acc = get_account_by_session(db, session_id)
        if acc:
            account_id = acc.id

    expires = datetime.utcnow() + timedelta(minutes=body.minutes)
    trade = save_trade(
        db,
        pair=body.pair,
        direction=body.direction.upper(),
        amount=body.amount,
        payout_pct=body.payout_pct,
        minutes=body.minutes,
        account_id=account_id,
        iq_order_id=iq_order_id,
        expires_at=expires,
    )
    return {
        "status": "success",
        "trade_id": trade.id,
        "iq_order_id": iq_order_id,
        "pair": trade.pair,
        "direction": trade.direction,
        "amount": trade.amount,
    }


# ── Candles ──────────────────────────────────────────────────────────────────


TF_TO_YF = {
    "M1":  ("1d",  "1m"),
    "M5":  ("5d",  "5m"),
    "M15": ("1mo", "15m"),
    "M30": ("1mo", "30m"),
    "H1":  ("3mo", "1h"),
}


def _to_yf_symbol(pair: str) -> str:
    p = pair.upper().replace("/", "").replace(" ", "")
    if p.endswith("=X"):
        return p
    if p in ("BTCUSD", "ETHUSD"):
        return p[:3] + "-" + p[3:]
    return p + "=X"


@app.get("/api/candles")
def api_candles(
    pair: str,
    tf: str = "M1",
    limit: int = 200,
    _: None = Depends(require_api_key),
):
    yf_sym = _to_yf_symbol(pair)
    period, interval = TF_TO_YF.get(tf.upper(), ("1d", "1m"))
    limit = max(10, min(int(limit or 200), 500))
    try:
        df = yf.Ticker(yf_sym).history(period=period, interval=interval)
    except Exception as e:
        raise HTTPException(502, f"YF error: {e}")
    if df is None or df.empty:
        return {"status": "success", "candles": []}
    df = df.tail(limit)
    candles = []
    for idx, row in df.iterrows():
        candles.append({
            "time": int(idx.timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        })
    return {"status": "success", "pair": pair, "tf": tf.upper(), "candles": candles}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = ENV != "production"
    uvicorn.run("main:app", host=host, port=port, reload=reload_flag)
