"""
QT Trading — FastAPI (UI + API).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
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


# ── Biquote chart provider ──────────────────────────────────────────────────

_BIQUOTE_TF = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D1": "1d",
}


def _biquote_symbol(pair: str) -> str:
    """EURUSD → EUR/USD, EUR/USD → EUR/USD"""
    p = pair.upper().replace("OTC", "").replace(" ", "")
    if "/" not in p and len(p) == 6 and p.isalpha():
        p = p[:3] + "/" + p[3:]
    return p


def _fetch_biquote_candles(pair: str, tf: str, limit: int):
    """Fetch candles from Biquote. No IQ websocket involved."""
    from biquote import get_candles

    symbol = _biquote_symbol(pair)
    interval = _BIQUOTE_TF.get(tf.upper(), "1m")
    limit = max(10, min(int(limit or 500), 5000))

    raw = get_candles(symbol=symbol, interval=interval, limit=limit)

    candles = []
    for c in raw or []:
        try:
            t = c.get("time") or c.get("timestamp") or c.get("datetime")
            if isinstance(t, str):
                from datetime import datetime as _dt
                try:
                    t = int(_dt.fromisoformat(t.replace("Z", "+00:00")).timestamp())
                except Exception:
                    continue
            candles.append({
                "time": int(t),
                "open": float(c.get("open") or 0),
                "high": float(c.get("high") or 0),
                "low": float(c.get("low") or 0),
                "close": float(c.get("close") or 0),
            })
        except Exception:
            continue

    candles.sort(key=lambda x: x["time"])
    return candles


# ── Background tasks ────────────────────────────────────────────────────────


async def _trade_result_resolver(stop: asyncio.Event) -> None:
    logger.info(
        "Trade resolver started (poll=%ds, void_after=%ds)",
        RESULT_POLL_INTERVAL, RESULT_VOID_AFTER,
    )
    while not stop.is_set():
        try:
            db = SessionLocal()
            try:
                n = advance_expired_trades(db)
                if n:
                    logger.info("Advanced %d trade(s) to pending_result", n)
            finally:
                db.close()

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
                            profit = (
                                round(trade.amount * (trade.payout_pct / 100.0), 2)
                                if won else -trade.amount
                            )
                        resolve_trade(db, trade.id, won=won, profit=profit)
                        logger.info(
                            "Resolved trade #%d %s %s → won=%s profit=%s",
                            trade.id, trade.pair, trade.direction, won, profit,
                        )
                    elif result and result.get("status") == "open":
                        continue
                    else:
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
        logger.warning("IQ auto-connect attempt %d failed: %s", attempt, payload.get("message"))
        await asyncio.sleep(5)
    logger.error("IQ auto-connect gave up after 3 attempts")


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
    bal = iq.refresh_balance(session_id)
    if bal is None:
        return {
            "status": "success",
            "balance": sess.balance,
            "currency": sess.currency,
            "mode": sess.mode,
        }
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


# ── Instruments cache (60s TTL) ─────────────────────────────────────────────

_INSTRUMENTS_CACHE = {"data": None, "timestamp": 0, "session_id": None}
_INSTRUMENTS_CACHE_TTL = 60


def _build_instruments(sess) -> dict:
    try:
        actives = sess.api.get_all_ACTIVES_OPCODE() or {}
    except Exception as e:
        logger.warning("get_all_ACTIVES_OPCODE failed: %s", e)
        actives = {}

    try:
        profits = sess.api.get_all_profit() or {}
    except Exception as e:
        logger.warning("get_all_profit failed: %s", e)
        profits = {}

    open_time = {}
    try:
        if hasattr(sess.api, "get_all_open_time"):
            open_time = sess.api.get_all_open_time() or {}
    except Exception:
        pass

    def classify(symbol: str) -> str:
        s = symbol.upper()
        if s in ("BTCUSD","ETHUSD","LTCUSD","XRPUSD","BCHUSD","BTCUSDT","ETHUSDT"): return "crypto"
        if s.startswith(("XAU","XAG","XPT","XPD")): return "commodity"
        if s in ("US30","NAS100","SPX500","GER40","UK100","JP225","US500","F40","E35"): return "index"
        if s in ("AAPL","TSLA","AMZN","FB","MSFT","NFLX","GOOGL","INTC","JPM","VISA"): return "stock"
        if len(s) == 6 and s.isalpha(): return "forex"
        return "other"

    account_type = sess.account_type

    def payout_of(symbol: str):
        data = profits.get(symbol)
        if isinstance(data, dict):
            pct = data.get(account_type) or data.get("turbo") or data.get("binary")
            if pct is not None:
                try: return round(float(pct) * 100, 1)
                except Exception: return None
        elif isinstance(data, (int, float)) and data:
            return round(float(data) * 100, 1)
        return None

    def durations_of(symbol: str):
        for kind in ("turbo", "binary"):
            node = (open_time.get(kind) or {}).get(symbol)
            if isinstance(node, dict):
                d = node.get("durations")
                if isinstance(d, list) and d:
                    return sorted({int(x) for x in d if str(x).isdigit()})
        return [1, 2, 3, 5, 10, 15, 30, 60]

    def is_open(symbol: str):
        for kind in ("turbo", "binary"):
            node = (open_time.get(kind) or {}).get(symbol)
            if isinstance(node, dict):
                return bool(node.get("open"))
        return (payout_of(symbol) or 0) > 0

    out = []
    seen = set()
    for symbol in actives.keys():
        if symbol in seen:
            continue
        seen.add(symbol)
        # Skip OTC pairs entirely
        if symbol.upper().endswith("OTC"):
            continue
        out.append({
            "symbol": symbol,
            "kind": classify(symbol),
            "is_otc": False,
            "payout": payout_of(symbol),
            "open": is_open(symbol),
            "durations": durations_of(symbol),
        })

    kind_order = {"forex": 1, "crypto": 2, "commodity": 3, "index": 4, "stock": 5, "other": 6}
    out.sort(key=lambda x: (not x["open"], kind_order.get(x["kind"], 99), x["symbol"]))

    open_count = sum(1 for x in out if x["open"])
    logger.info("instruments: %d total, %d open (OTC excluded)", len(out), open_count)

    return {"status": "success", "count": len(out), "open_count": open_count, "instruments": out}


@app.get("/api/iq/instruments")
def api_iq_instruments(refresh: bool = False, _: None = Depends(require_api_key)):
    import time as _time

    sess = iq.get_first_session()
    if not sess:
        raise HTTPException(410, "No IQ session — connect your account")

    now = _time.time()
    cache = _INSTRUMENTS_CACHE

    if (not refresh and cache["data"] is not None
        and cache["session_id"] == sess.session_id
        and (now - cache["timestamp"]) < _INSTRUMENTS_CACHE_TTL):
        logger.info("instruments: served from cache (age=%.1fs)", now - cache["timestamp"])
        return cache["data"]

    try:
        payload = _build_instruments(sess)
        cache["data"] = payload
        cache["timestamp"] = now
        cache["session_id"] = sess.session_id
        return payload
    except Exception as e:
        logger.exception("instruments build failed: %s", e)
        if cache["data"] is not None:
            logger.warning("serving stale instruments cache after error")
            return cache["data"]
        raise HTTPException(502, f"Could not load instruments: {e}")


# ── Candles (Biquote) ───────────────────────────────────────────────────────


@app.get("/api/iq/candles")
def api_candles(
    pair: str,
    tf: str = "M1",
    limit: int = 10,
    _: None = Depends(require_api_key),
):
    """Recent candles from Biquote — no IQ websocket involved."""
    try:
        candles = _fetch_biquote_candles(pair, tf, limit)
    except Exception as e:
        logger.error("biquote candles failed: %s", e)
        raise HTTPException(502, f"Chart provider error: {e}")
    return {
        "status": "success", "pair": pair, "tf": tf.upper(),
        "candles": candles, "source": "biquote",
    }


@app.get("/api/iq/candles/history")
def api_candles_history(
    pair: str,
    tf: str = "M1",
    total: int = 500,
    _: None = Depends(require_api_key),
):
    """Historical candles from Biquote — no IQ websocket involved."""
    try:
        candles = _fetch_biquote_candles(pair, tf, total)
    except Exception as e:
        logger.error("biquote history failed: %s", e)
        raise HTTPException(502, f"Chart provider error: {e}")
    return {
        "status": "success", "pair": pair, "tf": tf.upper(),
        "count": len(candles), "candles": candles, "source": "biquote",
    }


@app.get("/api/iq/indicators")
def api_iq_indicators(
    pair: str,
    tf: str = "M1",
    total: int = 500,
    rsi_period: int = 14,
    rsi_upper: float = 70,
    rsi_middle: float = 50,
    rsi_lower: float = 30,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    _: None = Depends(require_api_key),
):
    total = max(50, min(int(total or 500), 1000))
    rsi_period = max(2, min(int(rsi_period), 100))
    macd_fast = max(2, min(int(macd_fast), 200))
    macd_slow = max(macd_fast + 1, min(int(macd_slow), 400))
    macd_signal = max(2, min(int(macd_signal), 100))

    try:
        rows = _fetch_biquote_candles(pair, tf, total)
    except Exception as e:
        raise HTTPException(502, f"Chart provider error: {e}")

    logger.info(
        "indicators: %s %s rsi=%d/%s/%s/%s macd=%d/%d/%d candles=%d",
        pair, tf, rsi_period, rsi_lower, rsi_middle, rsi_upper,
        macd_fast, macd_slow, macd_signal, len(rows or []),
    )

    params = {
        "rsi_period": rsi_period,
        "rsi_lower": rsi_lower, "rsi_middle": rsi_middle, "rsi_upper": rsi_upper,
        "macd_fast": macd_fast, "macd_slow": macd_slow, "macd_signal": macd_signal,
    }

    if not rows:
        return {"status": "success", "candles": 0, "rsi": [], "macd": [], "signal": [], "hist": [], "params": params}

    rows.sort(key=lambda r: r["time"])
    df = pd.DataFrame(rows)
    closes = df
