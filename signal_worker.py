"""Background worker: scan the watchlist, save pending signals to the DB."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from database import SessionLocal, Signal, save_signal
from signal_engine import scan_symbols

logger = logging.getLogger("qt.signal_worker")

WATCHLIST: List[str] = [
    p.strip()
    for p in os.getenv(
        "SIGNAL_WATCHLIST",
        "EURUSD=X,GBPUSD=X,USDJPY=X,USDCHF=X,AUDUSD=X,"
        "NZDUSD=X,USDCAD=X,EURGBP=X,EURJPY=X,EURCHF=X,"
        "EURAUD=X,EURNZD=X,EURCAD=X,GBPJPY=X,GBPCHF=X,"
        "GBPAUD=X,GBPNZD=X,GBPCAD=X,CHFJPY=X,CADJPY=X,"
        "AUDJPY=X,NZDJPY=X,AUDNZD=X,AUDCAD=X,AUDCHF=X,"
        "NZDCAD=X,NZDCHF=X,CADCHF=X",
    ).split(",")
    if p.strip()
]

SCAN_INTERVAL = int(os.getenv("SIGNAL_SCAN_INTERVAL", "30"))
COOLDOWN = int(os.getenv("SIGNAL_COOLDOWN", "300"))

_last_emit: Dict[str, float] = {}


def _recent_pending_exists(symbol: str, within_seconds: int) -> bool:
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=within_seconds)
        row = (
            db.query(Signal)
            .filter(Signal.pair == symbol.replace("=X", ""))
            .filter(Signal.status == "pending")
            .filter(Signal.created_at >= cutoff)
            .first()
        )
        return row is not None
    finally:
        db.close()


async def scan_once() -> int:
    saved = 0
    loop = asyncio.get_running_loop()

    def _run() -> List:
        return scan_symbols(WATCHLIST)

    try:
        signals = await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.exception("scan_symbols failed: %s", e)
        return 0

    for sig in signals:
        if time.time() - _last_emit.get(sig.symbol, 0) < COOLDOWN:
            continue
        if _recent_pending_exists(sig.symbol, COOLDOWN):
            continue

        db = SessionLocal()
        try:
            save_signal(
                db,
                pair=sig.pair,
                direction=sig.direction,
                minutes=sig.minutes,
                confidence=f"{sig.confidence}%",
                raw_text=sig.reason,
                source="auto",
            )
            _last_emit[sig.symbol] = time.time()
            saved += 1
            logger.info(
                "Signal saved: %s %s (%d%%) — %s",
                sig.pair, sig.direction, sig.confidence, sig.reason,
            )
        except Exception as e:
            logger.exception("save_signal failed: %s", e)
        finally:
            db.close()

    return saved


async def signal_loop(stop: asyncio.Event) -> None:
    logger.info(
        "Signal worker started · %d symbols · interval=%ds · cooldown=%ds",
        len(WATCHLIST), SCAN_INTERVAL, COOLDOWN,
    )
    await asyncio.sleep(5)
    while not stop.is_set():
        try:
            n = await scan_once()
            if n:
                logger.info("Scan complete: %d new signal(s)", n)
        except Exception as e:
            logger.exception("scan loop error: %s", e)
        try:
            await asyncio.wait_for(stop.wait(), timeout=SCAN_INTERVAL)
        except asyncio.TimeoutError:
            pass
    logger.info("Signal worker stopped")
