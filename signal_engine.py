"""
QT Trading - Signal engine (from the Telegram bot).

Same rules as the bot:
  - yfinance 5m + 1m candles
  - MACD (12/26/9) on 5m and 1m
  - Latest 1m cross confirmation
  - RSI(14) filter with configurable bounds
  - MACD compression filter
  - yfinance 5m flat-market spread filter
  - Optional OANDA live spread check
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger("qt.signal_engine")


OANDA_API_KEY = os.environ.get("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
MAX_SPREAD_PIPS = float(os.environ.get("MAX_SPREAD_PIPS", "3.0"))
UNBLOCK_CONSECUTIVE_CHECKS = int(os.environ.get("UNBLOCK_CONSECUTIVE_CHECKS", "2"))

DEFAULT_RSI_BUY_MIN = int(os.environ.get("RSI_BUY_MIN", "30"))
DEFAULT_RSI_BUY_MAX = int(os.environ.get("RSI_BUY_MAX", "40"))
DEFAULT_RSI_SELL_MIN = int(os.environ.get("RSI_SELL_MIN", "60"))
DEFAULT_RSI_SELL_MAX = int(os.environ.get("RSI_SELL_MAX", "70"))

MIN_DIFF = float(os.environ.get("MIN_DIFF", "0.00001"))


@dataclass
class GeneratedSignal:
    symbol: str
    pair: str
    direction: str
    minutes: int
    confidence: int
    reason: str
    macd_5m: float
    signal_5m: float
    diff_5m: float
    macd_1m: float
    signal_1m: float
    diff_1m: float
    rsi_5m: float
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


pair_settings: Dict[str, Dict[str, int]] = {}


def get_effective_settings(symbol: str) -> Dict[str, int]:
    if symbol in pair_settings:
        return {
            "rsi_buy_min": pair_settings[symbol].get("rsi_buy_min", DEFAULT_RSI_BUY_MIN),
            "rsi_buy_max": pair_settings[symbol].get("rsi_buy_max", DEFAULT_RSI_BUY_MAX),
            "rsi_sell_min": pair_settings[symbol].get("rsi_sell_min", DEFAULT_RSI_SELL_MIN),
            "rsi_sell_max": pair_settings[symbol].get("rsi_sell_max", DEFAULT_RSI_SELL_MAX),
        }
    return {
        "rsi_buy_min": DEFAULT_RSI_BUY_MIN,
        "rsi_buy_max": DEFAULT_RSI_BUY_MAX,
        "rsi_sell_min": DEFAULT_RSI_SELL_MIN,
        "rsi_sell_max": DEFAULT_RSI_SELL_MAX,
    }


def calculate_strategy(df: pd.DataFrame):
    fast_ema = df["Close"].ewm(span=12, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=26, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return macd, signal, hist, rsi


def latest_1m_cross(diff_series: pd.Series) -> Optional[str]:
    diffs = list(diff_series)
    for i in range(len(diffs) - 1, 0, -1):
        if diffs[i - 1] < 0 and diffs[i] > 0:
            return "bull"
        elif diffs[i - 1] > 0 and diffs[i] < 0:
            return "bear"
    return None


def is_spread_present_5m(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="5m")
        if len(df) < 10:
            return False
        last_candles = df.tail(5)
        high_low_range = last_candles["High"] - last_candles["Low"]
        avg_range = high_low_range.mean()
        avg_price = last_candles["Close"].mean()
        body_size = abs(last_candles["Close"] - last_candles["Open"])
        avg_body = body_size.mean()
        c1 = avg_price > 0 and avg_range < avg_price * 0.0002
        c2 = avg_price > 0 and avg_body < avg_price * 0.00005
        c3 = df.iloc[-1]["High"] == df.iloc[-1]["Low"]
        return sum([c1, c2, c3]) >= 2
    except Exception as e:
        logger.error("5m spread check error %s: %s", symbol, e)
        return False


def get_oanda_spread_pips(symbol: str) -> Optional[float]:
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        return None
    instrument = symbol.replace("=X", "")
    base = instrument[:3]
    quote = instrument[3:]
    oanda_symbol = f"{base}_{quote}"
    try:
        url = "https://api-fxtrade.oanda.com/v3/instruments/" + oanda_symbol + "/pricing"
        headers = {"Authorization": "Bearer " + OANDA_API_KEY}
        params = {"instruments": oanda_symbol}
        r = requests.get(url, headers=headers, params=params, timeout=5)
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return None
        bid = float(prices[0]["bids"][0]["price"])
        ask = float(prices[0]["asks"][0]["price"])
        spread = ask - bid
        pip_size = 0.01 if "JPY" in quote else 0.0001
        return round(spread / pip_size, 2)
    except Exception as e:
        logger.error("OANDA spread check error %s: %s", symbol, e)
        return None


compression_counter: Dict[str, int] = {}
compression_blocked: Dict[str, bool] = {}


def check_compression(symbol: str, h: pd.Series) -> bool:
    hist_abs = h.abs()
    avg_hist_abs = hist_abs.tail(100).mean() if len(hist_abs) >= 100 else hist_abs.mean()
    current_abs = abs(h.iloc[-1])
    low_threshold = avg_hist_abs * 0.20
    high_threshold = avg_hist_abs * 0.50

    compression_counter.setdefault(symbol, 0)
    compression_blocked.setdefault(symbol, False)

    if compression_blocked[symbol]:
        if current_abs > high_threshold:
            compression_blocked[symbol] = False
            compression_counter[symbol] = 0
            return False
        return True
    else:
        if current_abs < low_threshold:
            compression_counter[symbol] += 1
            if compression_counter[symbol] >= 10:
                compression_blocked[symbol] = True
                return True
        else:
            compression_counter[symbol] = 0
        return False


def compute_confidence(direction, rsi_val, diff_5m):
    if direction == "BUY":
        band_center = (DEFAULT_RSI_BUY_MIN + DEFAULT_RSI_BUY_MAX) / 2
    else:
        band_center = (DEFAULT_RSI_SELL_MIN + DEFAULT_RSI_SELL_MAX) / 2
    rsi_score = max(0.0, 1.0 - abs(rsi_val - band_center) / 20.0)
    if MIN_DIFF > 0:
        diff_score = min(1.0, abs(diff_5m) / (5 * MIN_DIFF))
    else:
        diff_score = 0.0
    score = 55 + 25 * rsi_score + 20 * diff_score
    return int(max(0, min(99, score)))


def _build_reason(rsi_val, diff_5m, diff_1m):
    parts = []
    parts.append("MACD cross + 1m confirm")
    parts.append("RSI " + str(round(rsi_val, 1)))
    parts.append("diff5m " + str(round(diff_5m, 5)))
    parts.append("diff1m " + str(round(diff_1m, 5)))
    return " | ".join(parts)


def generate_signal(symbol: str) -> Optional[GeneratedSignal]:
    try:
        if is_spread_present_5m(symbol):
            logger.debug("%s blocked: 5m flat market", symbol)
            return None

        df = yf.Ticker(symbol).history(period="5d", interval="5m")
        if len(df) < 50:
            return None
        m, s, h, rsi = calculate_strategy(df)
        prev_diff = m.iloc[-1] - s.iloc[-1]
        prev_diff_before = m.iloc[-2] - s.iloc[-2]

        if check_compression(symbol, h):
            logger.debug("%s blocked: MACD compression", symbol)
            return None

        df_1m = yf.Ticker(symbol).history(period="1d", interval="1m")
        if len(df_1m) < 30:
            return None
        m_1m, s_1m, h_1m, rsi_1m = calculate_strategy(df_1m)
        diff_1m_series = m_1m - s_1m
        cross_1m = latest_1m_cross(diff_1m_series)

        is_1m_bull = cross_1m == "bull"
        is_1m_bear = cross_1m == "bear"

        settings = get_effective_settings(symbol)
        rsi_val = float(rsi.iloc[-1])

        confirm_bull = (
            prev_diff_before < 0
            and prev_diff > 0
            and abs(prev_diff) >= MIN_DIFF
            and is_1m_bull
            and settings["rsi_buy_min"] <= rsi_val <= settings["rsi_buy_max"]
        )
        confirm_bear = (
            prev_diff_before > 0
            and prev_diff < 0
            and abs(prev_diff) >= MIN_DIFF
            and is_1m_bear
            and settings["rsi_sell_min"] <= rsi_val <= settings["rsi_sell_max"]
        )

        if not (confirm_bull or confirm_bear):
            return None

        if confirm_bull:
            direction = "BUY"
        else:
            direction = "SELL"

        pair_display = symbol.replace("=X", "")

        confidence = compute_confidence(direction, rsi_val, float(prev_diff))
        reason = _build_reason(rsi_val, float(prev_diff), float(diff_1m_series.iloc[-1]))

        return GeneratedSignal(
            symbol=symbol,
            pair=pair_display,
            direction=direction,
            minutes=5,
            confidence=confidence,
            reason=reason,
            macd_5m=float(m.iloc[-1]),
            signal_5m=float(s.iloc[-1]),
            diff_5m=float(prev_diff),
            macd_1m=float(m_1m.iloc[-1]),
            signal_1m=float(s_1m.iloc[-1]),
            diff_1m=float(diff_1m_series.iloc[-1]),
            rsi_5m=rsi_val,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error("generate_signal error %s: %s", symbol, e)
        return None


def scan_symbols(symbols: List[str]) -> List[GeneratedSignal]:
    out = []
    for sym in symbols:
        try:
            sig = generate_signal(sym)
            if sig:
                out.append(sig)
        except Exception as e:
            logger.error("scan error %s: %s", sym, e)
    return out
