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


@app.get("/api/iq/instruments")
def api_iq_instruments(_: None = Depends(require_api_key)):
    """
    Every asset IQ currently offers, with payout, open state, durations.
    """
    sess = iq.get_first_session()
    if not sess or not sess.is_alive():
        raise HTTPException(410, "No live IQ session")

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
    except Exception as e:
        logger.debug("get_all_open_time failed: %s", e)

    def classify(symbol: str) -> str:
        s = symbol.upper()
        if s.endswith("OTC"):
            return "forex_otc"
        if s in ("BTCUSD","ETHUSD","LTCUSD","XRPUSD","BCHUSD","BTCUSDT","ETHUSDT"):
            return "crypto"
        if s.startswith("XAU") or s.startswith("XAG") or s.startswith("XPT") or s.startswith("XPD"):
            return "commodity"
        if s in ("US30","NAS100","SPX500","GER40","UK100","JP225","US500","F40","E35"):
            return "index"
        if s in ("AAPL","TSLA","AMZN","FB","MSFT","NFLX","GOOGL","INTC","JPM","VISA"):
            return "stock"
        if len(s) == 6 and s.isalpha():
            return "forex"
        return "other"

    account_type = sess.account_type

    def payout_of(symbol: str):
        data = profits.get(symbol)
        if isinstance(data, dict):
            pct = data.get(account_type) or data.get("turbo") or data.get("binary")
            if pct is not None:
                try:
                    return round(float(pct) * 100, 1)
                except Exception:
                    return None
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
        out.append({
            "symbol": symbol,
            "kind": classify(symbol),
            "is_otc": symbol.upper().endswith("OTC"),
            "payout": payout_of(symbol),
            "open": is_open(symbol),
            "durations": durations_of(symbol),
        })

    kind_order = {"forex": 1, "forex_otc": 2, "crypto": 3, "commodity": 4, "index": 5, "stock": 6, "other": 7}
    out.sort(key=lambda x: (not x["open"], kind_order.get(x["kind"], 99), x["symbol"]))

    open_count = sum(1 for x in out if x["open"])
    logger.info("instruments: %d total, %d open", len(out), open_count)

    return {
        "status": "success",
        "count": len(out),
        "open_count": open_count,
        "instruments": out,
    }


@app.get("/api/iq/available")
def api_iq_available(
    pair: str,
    minutes: int = 1,
    _: None = Depends(require_api_key),
):
    """Pre-flight check: is this pair tradeable right now?"""
    sess = iq.get_first_session()
    if not sess or not sess.is_alive():
        raise HTTPException(410, "No live IQ session")

    active = pair.upper().replace("/", "").replace(" ", "")
    account_type = sess.account_type

    in_list = False
    try:
        actives = sess.api.get_all_ACTIVES_OPCODE() or {}
        in_list = active in actives
    except Exception:
        pass

    payout = None
    try:
        profits = sess.api.get_all_profit() or {}
        data = profits.get(active)
        if isinstance(data, dict):
            pct = data.get(account_type) or data.get("turbo") or data.get("binary")
            if pct:
                payout = round(float(pct) * 100, 1)
        elif isinstance(data, (int, float)) and data:
            payout = round(float(data) * 100, 1)
    except Exception:
        pass

    durations = []
    try:
        if hasattr(sess.api, "get_all_open_time"):
            otime = sess.api.get_all_open_time() or {}
            for kind in ("turbo", "binary"):
                node = otime.get(kind, {}).get(active)
                if isinstance(node, dict):
                    d = node.get("durations") or node.get("min_duration")
                    if isinstance(d, list) and d:
                        durations = [int(x) for x in d if str(x).isdigit()]
                    elif isinstance(d, int):
                        durations = [d]
                    if durations:
                        break
    except Exception:
        pass

    if not durations:
        durations = [1, 2, 3, 5, 10, 15, 30, 60]

    tradeable = bool(in_list or payout)
    duration_ok = (minutes in durations) if durations else True

    reason = None
    if not tradeable:
        reason = f"{active} not in IQ's tradeable list (market may be closed)"
    elif payout is not None and payout <= 0:
        reason = f"{active} has 0% payout right now"
    elif not duration_ok:
        reason = f"{active} doesn't support {minutes}m — try {sorted(set(durations))[:5]}"

    return {
        "status": "success",
        "pair": active,
        "tradeable": tradeable and duration_ok and (payout is None or payout > 0),
        "in_asset_list": in_list,
        "payout": payout,
        "durations": sorted(set(durations)),
        "minutes_requested": minutes,
        "duration_ok": duration_ok,
        "account_type": account_type,
        "mode": sess.mode,
        "reason": reason,
    }


@app.get("/api/iq/candles")
def api_iq_candles(
    pair: str,
    tf: str = "M1",
    limit: int = 1000,
    _: None = Depends(require_api_key),
):
    sess = iq.get_first_session()
    if not sess or not sess.is_alive():
        raise HTTPException(410, "No live IQ session")

    active = pair.upper().replace("/", "").replace(" ", "")
    tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}.get(tf.upper(), 60)
    limit = max(10, min(int(limit or 1000), 1000))

    try:
        raw = sess.api.get_candles(active, tf_seconds, limit, time.time())
    except Exception as e:
        raise HTTPException(502, f"IQ candle error: {e}")

    if not raw:
        return {"status": "success", "candles": []}

    candles = []
    for c in raw:
        candles.append({
            "time": int(c.get("at") or c.get("from") or 0),
            "open": float(c.get("open") or 0),
            "high": float(c.get("max") or c.get("high") or 0),
            "low": float(c.get("min") or c.get("low") or 0),
            "close": float(c.get("close") or 0),
        })
    candles.sort(key=lambda c: c["time"])
    return {"status": "success", "pair": pair, "tf": tf.upper(), "candles": candles}


@app.get("/api/iq/candles/history")
def api_iq_candles_history(
    pair: str,
    tf: str = "M1",
    total: int = 3000,
    _: None = Depends(require_api_key),
):
    sess = iq.get_first_session()
    if not sess or not sess.is_alive():
        raise HTTPException(410, "No live IQ session")

    active = pair.upper().replace("/", "").replace(" ", "")
    tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}.get(tf.upper(), 60)

    total = max(100, min(int(total or 3000), 20000))
    PAGE = 1000

    all_candles = []
    seen_times = set()
    end_time = time.time()
    pages = min((total + PAGE - 1) // PAGE, 20)

    for _ in range(pages):
        try:
            raw = sess.api.get_candles(active, tf_seconds, PAGE, end_time)
        except Exception as e:
            logger.warning("history page failed: %s", e)
            break

        if not raw:
            break

        page_candles = []
        for c in raw:
            t = int(c.get("at") or c.get("from") or 0)
            if t <= 0 or t in seen_times:
                continue
            seen_times.add(t)
            page_candles.append({
                "time": t,
                "open": float(c.get("open") or 0),
                "high": float(c.get("max") or c.get("high") or 0),
                "low": float(c.get("min") or c.get("low") or 0),
                "close": float(c.get("close") or 0),
            })

        if not page_candles:
            break

        all_candles.extend(page_candles)
        oldest = min(page_candles, key=lambda x: x["time"])["time"]
        end_time = oldest - 1

        if len(all_candles) >= total:
            break

    all_candles.sort(key=lambda c: c["time"])
    if len(all_candles) > total:
        all_candles = all_candles[-total:]

    return {
        "status": "success",
        "pair": pair,
        "tf": tf.upper(),
        "count": len(all_candles),
        "candles": all_candles,
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
    sess = iq.get_first_session()
    if not sess or not sess.is_alive():
        raise HTTPException(410, "No live IQ session")

    active = pair.upper().replace("/", "").replace(" ", "")
    tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}.get(tf.upper(), 60)
    total = max(50, min(int(total or 500), 1000))

    rsi_period = max(2, min(int(rsi_period), 100))
    macd_fast = max(2, min(int(macd_fast), 200))
    macd_slow = max(macd_fast + 1, min(int(macd_slow), 400))
    macd_signal = max(2, min(int(macd_signal), 100))

    raw = None
    for attempt in range(2):
        try:
            raw = sess.api.get_candles(active, tf_seconds, total, time.time())
        except Exception as e:
            logger.warning("indicators attempt %d failed: %s", attempt + 1, e)
            raw = None
        if raw:
            break
        time.sleep(0.4)

    logger.info(
        "indicators: %s %s rsi=%d/%s/%s/%s macd=%d/%d/%d candles=%d",
        pair, tf, rsi_period, rsi_lower, rsi_middle, rsi_upper,
        macd_fast, macd_slow, macd_signal, len(raw or []),
    )

    params = {
        "rsi_period": rsi_period,
        "rsi_lower": rsi_lower, "rsi_middle": rsi_middle, "rsi_upper": rsi_upper,
        "macd_fast": macd_fast, "macd_slow": macd_slow, "macd_signal": macd_signal,
    }

    if not raw:
        return {
            "status": "success", "candles": 0,
            "rsi": [], "macd": [], "signal": [], "hist": [],
            "params": params,
        }

    rows = []
    for c in raw:
        t = int(c.get("at") or c.get("from") or 0)
        if t <= 0:
            continue
        rows.append({
            "time": t,
            "open": float(c.get("open") or 0),
            "high": float(c.get("max") or c.get("high") or 0),
            "low": float(c.get("min") or c.get("low") or 0),
            "close": float(c.get("close") or 0),
        })

    if not rows:
        return {"status": "success", "candles": 0, "rsi": [], "macd": [], "signal": [], "hist": [], "params": params}

    rows.sort(key=lambda r: r["time"])
    df = pd.DataFrame(rows)
    closes = df["close"].astype(float)

    ema_fast = closes.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = closes.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    histogram = macd_line - signal_line

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_line = 100 - (100 / (1 + rs))

    rsi_out, macd_out, signal_out, hist_out = [], [], [], []
    for i, t in enumerate(df["time"].tolist()):
        rv = rsi_line.iloc[i]
        mv = macd_line.iloc[i]
        sv = signal_line.iloc[i]
        hv = histogram.iloc[i]
        if pd.isna(rv) or pd.isna(mv) or pd.isna(sv) or pd.isna(hv):
            continue
        rsi_out.append({"time": int(t), "value": round(float(rv), 2)})
        macd_out.append({"time": int(t), "value": round(float(mv), 6)})
        signal_out.append({"time": int(t), "value": round(float(sv), 6)})
        hist_out.append({"time": int(t), "value": round(float(hv), 6)})

    logger.info("indicators: computed rsi=%d macd=%d", len(rsi_out), len(macd_out))

    return {
        "status": "success",
        "candles": len(rows),
        "rsi": rsi_out, "macd": macd_out, "signal": signal_out, "hist": hist_out,
        "params": params,
    }


@app.get("/api/iq/payouts")
def api_iq_payouts(_: None = Depends(require_api_key)):
    sess = iq.get_first_session()
    if not sess or not sess.is_alive():
        raise HTTPException(410, "No live IQ session")
    try:
        profits = sess.api.get_all_profit() or {}
    except Exception as e:
        raise HTTPException(502, f"payout error: {e}")

    out = {}
    for pair, data in profits.items():
        try:
            if isinstance(data, dict):
                pct = data.get(sess.account_type) or data.get("turbo") or data.get("binary")
                out[pair] = round(float(pct) * 100, 1) if pct else None
            elif isinstance(data, (int, float)):
                out[pair] = round(float(data) * 100, 1)
        except Exception:
            out[pair] = None
    return {"status": "success", "payouts": out}


# ── Profile stats ───────────────────────────────────────────────────────────


@app.get("/api/profile/stats")
def api_profile_stats(
    db: DBSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    import collections

    trades = db.query(Trade).order_by(Trade.opened_at.asc()).all()
    now = datetime.utcnow()

    def within(days):
        cutoff = now - timedelta(days=days)
        return [t for t in trades if t.closed_at and t.closed_at >= cutoff]

    def pnl(items):
        return round(sum((t.profit or 0) for t in items if t.won is not None), 2)

    today_items = [t for t in trades if t.closed_at and t.closed_at.date() == now.date()]
    week_items = within(7)
    month_items = within(30)
    year_items = within(365)

    closed = [t for t in trades if t.status == "closed" and t.won is not None]
    best_streak = worst_streak = cur_win = cur_loss = 0
    for t in closed:
        if t.won:
            cur_win += 1; cur_loss = 0
        else:
            cur_loss += 1; cur_win = 0
        best_streak = max(best_streak, cur_win)
        worst_streak = max(worst_streak, cur_loss)

    wins = [t for t in closed if t.won]
    losses = [t for t in closed if t.won is False]
    avg_size = round(sum(t.amount for t in closed) / len(closed), 2) if closed else 0
    avg_profit = round(sum((t.profit or 0) for t in closed) / len(closed), 2) if closed else 0

    pair_counts = collections.Counter(t.pair for t in closed)
    favorite = pair_counts.most_common(1)[0][0] if pair_counts else "—"

    hours = collections.Counter(t.opened_at.hour for t in trades if t.opened_at)
    top_hour = hours.most_common(1)[0][0] if hours else None
    hour_range = f"{top_hour:02d}:00 – {(top_hour+2)%24:02d}:00" if top_hour is not None else "—"

    spark = []
    cumulative = 0.0
    for t in closed[-30:]:
        cumulative += (t.profit or 0)
        spark.append(round(cumulative, 2))

    first_trade = trades[0].opened_at if trades else None
    created = first_trade.strftime("%b %Y") if first_trade else "—"

    total = len(closed)
    win_rate = round(len(wins) / total * 100, 1) if total else 0

    return {
        "status": "success",
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit": {
            "today": pnl(today_items),
            "week": pnl(week_items),
            "month": pnl(month_items),
            "year": pnl(year_items),
        },
        "streaks": {"best": best_streak, "worst": worst_streak},
        "averages": {"size": avg_size, "profit_per_trade": avg_profit},
        "favorite_asset": favorite,
        "most_active_hour": hour_range,
        "account_created": created,
        "sparkline": spark,
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

    DURATION_OK = {1, 2, 3, 5, 10, 15, 30, 60}
    if minutes not in DURATION_OK:
        minutes = 5

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
            "total": len(rows),
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
    if not session_id:
        sess = iq.get_first_session()
        if sess:
            session_id = sess.session_id
    if not session_id:
        raise HTTPException(400, "No IQ session connected")

    active = body.pair.upper().replace("/", "").replace(" ", "")

    DURATION_OK = {1, 2, 3, 5, 10, 15, 30, 60}
    if body.minutes not in DURATION_OK:
        raise HTTPException(400, f"Duration {body.minutes}m not supported")

    logger.info(
        "TRADE REQUEST: pair=%s → active=%s dir=%s amt=%.2f dur=%dm",
        body.pair, active, body.direction, body.amount, body.minutes,
    )

    payout = body.payout_pct
    try:
        sess = iq.get_session(session_id)
        if sess and sess.is_alive():
            profits = sess.api.get_all_profit() or {}
            data = profits.get(active)
            if isinstance(data, dict):
                pct = data.get(sess.account_type) or data.get("turbo") or data.get("binary")
                if pct:
                    payout = round(float(pct) * 100, 1)
            elif isinstance(data, (int, float)) and data:
                payout = round(float(data) * 100, 1)
    except Exception as e:
        logger.warning("payout fetch failed: %s", e)

    ok, result = iq.place_binary_order(
        session_id,
        active=active,
        direction=body.direction,
        amount=body.amount,
        duration_min=body.minutes,
    )

    logger.info("TRADE RESULT: ok=%s result=%s", ok, result)

    if not ok:
        detail = result.get("message", "Order rejected")
        low = detail.lower()
        if "not available" in low:
            detail = (
                f"{active} is not tradeable right now on IQ. "
                f"Try a different pair, a different duration, or wait for the market to open."
            )
        elif "amount" in low:
            detail = f"Invalid amount: {detail}"
        elif "duration" in low or "expiration" in low:
            detail = f"Invalid duration: {detail}"
        raise HTTPException(502, detail)

    iq_order_id = result.get("order_id")

    account_id = None
    acc = get_account_by_session(db, session_id)
    if acc:
        account_id = acc.id

    expires = datetime.utcnow() + timedelta(minutes=body.minutes)
    trade = save_trade(
        db,
        pair=body.pair,
        direction=body.direction.upper(),
        amount=body.amount,
        payout_pct=payout,
        minutes=body.minutes,
        account_id=account_id,
        iq_order_id=iq_order_id,
        expires_at=expires,
    )

    try:
        new_balance = iq.refresh_balance(session_id)
    except Exception:
        new_balance = None

    return {
        "status": "success",
        "trade_id": trade.id,
        "iq_order_id": iq_order_id,
        "pair": trade.pair,
        "direction": trade.direction,
        "amount": trade.amount,
        "payout_pct": payout,
        "balance": new_balance,
    }


# ── Fallback candles (yfinance) ─────────────────────────────────────────────

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
