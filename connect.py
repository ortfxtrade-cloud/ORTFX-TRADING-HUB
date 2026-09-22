"""IQ Option connection helpers."""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("qt.connect")

_sessions: Dict[str, "IQSession"] = {}
_lock = threading.Lock()

DURATION_WHITELIST = {1, 2, 3, 5, 10, 15, 30, 60}


@dataclass
class IQSession:
    session_id: str
    email: str
    mode: str
    api: Any = None
    balance: float = 0.0
    currency: str = "USD"
    account_type: str = "PRACTICE"
    connected_at: float = field(default_factory=time.time)
    last_error: Optional[str] = None

    def is_alive(self) -> bool:
        """Note: check_connect() is unreliable on cloud hosts. Use only for diagnostics."""
        if self.api is None:
            return False
        try:
            return bool(self.api.check_connect())
        except Exception as e:
            logger.debug("is_alive check failed for %s: %s", self.session_id, e)
            return False


def _import_iq_option():
    from iqoptionapi.stable_api import IQ_Option
    return IQ_Option


def _safe_logout(sess: "IQSession") -> None:
    try:
        if sess.api is not None and hasattr(sess.api, "logout"):
            sess.api.logout()
    except Exception as e:
        logger.debug("logout cleanup failed for %s: %s", sess.session_id, e)
    finally:
        sess.api = None


def connect_iq(email: str, password: str, mode: str = "demo") -> Tuple[bool, Dict[str, Any]]:
    email = (email or "").strip().lower()
    password = password or ""
    mode = (mode or "demo").lower()
    if mode not in ("demo", "real"):
        mode = "demo"

    if not email or not password:
        return False, {"status": "error", "message": "Email and password are required"}

    try:
        IQ_Option = _import_iq_option()
    except ImportError as e:
        logger.exception("iqoptionapi import failed")
        return False, {"status": "error", "message": f"iqoptionapi not installed: {e}"}

    try:
        api = IQ_Option(email, password)
        check, reason = api.connect()
    except Exception as e:
        logger.exception("IQ connect exception")
        return False, {"status": "error", "message": f"Connection error: {e}"}

    if not check:
        msg = reason
        if isinstance(reason, dict):
            msg = reason.get("message") or reason.get("code") or str(reason)
        if reason == "2FA" or (isinstance(msg, str) and "2FA" in str(msg).upper()):
            return False, {
                "status": "error",
                "message": "Two-factor authentication required. Disable 2FA or use connect_2fa.",
            }
        return False, {"status": "error", "message": str(msg) or "Invalid credentials"}

    balance_type = "PRACTICE" if mode == "demo" else "REAL"
    try:
        api.change_balance(balance_type)
    except Exception as e:
        logger.warning("change_balance failed: %s", e)

    balance = 0.0
    currency = "USD"
    try:
        balance = float(api.get_balance() or 0)
    except Exception:
        pass

    try:
        if hasattr(api, "get_currency"):
            currency = api.get_currency() or "USD"
        elif hasattr(api, "get_balance_v2"):
            info = api.get_balance_v2()
            if isinstance(info, dict):
                currency = info.get("currency", currency)
    except Exception:
        pass

    session_id = f"iq_{secrets.token_urlsafe(18)}"

    sess = IQSession(
        session_id=session_id,
        email=email,
        mode=mode,
        api=api,
        balance=balance,
        currency=currency,
        account_type=balance_type,
    )

    with _lock:
        old = [s for s in _sessions.values() if s.email == email]
        for s in old:
            _sessions.pop(s.session_id, None)
        _sessions[session_id] = sess

    for s in old:
        _safe_logout(s)

    logger.info("IQ connected: %s mode=%s balance=%.2f", email, mode, balance)

    return True, {
        "status": "success",
        "session_id": session_id,
        "balance": balance,
        "currency": currency,
        "account_type": balance_type,
        "message": f"Connected ({balance_type})",
        "email": email,
        "mode": mode,
    }


def disconnect_iq(session_id: str) -> Tuple[bool, str]:
    with _lock:
        sess = _sessions.pop(session_id, None)
    if not sess:
        return False, "Session not found"
    _safe_logout(sess)
    return True, "Disconnected"


def get_session(session_id: str) -> Optional[IQSession]:
    with _lock:
        return _sessions.get(session_id)


def get_first_session() -> Optional[IQSession]:
    with _lock:
        if not _sessions:
            return None
        return max(_sessions.values(), key=lambda s: s.connected_at)


def refresh_balance(session_id: str) -> Optional[float]:
    # NOTE: do NOT gate on is_alive() — check_connect() is unreliable on Render.
    sess = get_session(session_id)
    if not sess:
        return None
    try:
        bal = float(sess.api.get_balance() or 0)
        sess.balance = bal
        return bal
    except Exception as e:
        logger.warning("refresh_balance failed for %s: %s", session_id, e)
        return None


def place_binary_order(
    session_id: str,
    active: str,
    direction: str,
    amount: float,
    duration_min: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    # NOTE: do NOT gate on is_alive() — check_connect() is unreliable on Render.
    sess = get_session(session_id)
    if not sess:
        return False, {"message": "Session not found — reconnect"}

    active = re.sub(r"[^A-Z0-9]", "", (active or "").upper())
    if not active:
        return False, {"message": "active is required"}

    d = (direction or "").lower()
    if d in ("buy", "up", "call"):
        d = "call"
    elif d in ("sell", "down", "put"):
        d = "put"
    else:
        return False, {"message": "direction must be BUY/SELL or call/put"}

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return False, {"message": "amount must be numeric"}
    if amount <= 0:
        return False, {"message": "amount must be positive"}

    try:
        duration_min = int(duration_min)
    except (TypeError, ValueError):
        return False, {"message": "duration must be an integer"}

    if duration_min not in DURATION_WHITELIST:
        return False, {
            "message": f"Unsupported duration: {duration_min} "
            f"(allowed: {sorted(DURATION_WHITELIST)})"
        }

    import concurrent.futures

    def _do_buy():
        return sess.api.buy(amount, active, d, duration_min)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_buy)
            ok, order_id = fut.result(timeout=20)
    except concurrent.futures.TimeoutError:
        logger.error("place order TIMEOUT after 20s: %s %s %.2f %dm",
                     active, d, amount, duration_min)
        return False, {
            "message": (
                "IQ did not respond within 20 seconds. "
                "The order may or may not have been placed — check IQ history."
            )
        }
    except Exception as e:
        logger.exception("place order failed")
        return False, {"message": str(e)}

    if not ok:
        return False, {"message": f"Order rejected: {order_id}"}
    return True, {
        "order_id": str(order_id),
        "active": active,
        "direction": d,
        "amount": amount,
        "duration": duration_min,
    }


def list_active_sessions() -> List[Dict[str, Any]]:
    with _lock:
        sessions = list(_sessions.values())
    return [
        {
            "session_id": s.session_id,
            "email": s.email,
            "mode": s.mode,
            "balance": s.balance,
            "currency": s.currency,
            "alive": s.is_alive(),
        }
        for s in sessions
    ]


# ── Trade result polling ─────────────────────────────────────────────────────


def _extract_profit(msg: Dict[str, Any], won: bool) -> Optional[float]:
    for key in ("profit_amount", "profit", "pnl", "win_amount", "close_profit"):
        if key in msg and msg[key] is not None:
            try:
                return float(msg[key])
            except (TypeError, ValueError):
                pass
    return None


def get_order_result(session_id: str, order_id: str) -> Optional[Dict[str, Any]]:
    # NOTE: no is_alive() gate.
    sess = get_session(session_id)
    if not sess:
        return None

    api = sess.api

    try:
        if hasattr(api, "get_optioninfo"):
            info = api.get_optioninfo(10, order_id)
            if isinstance(info, list) and info:
                info = info[0]
            if isinstance(info, dict) and info:
                msg = info.get("msg", info)
                status = str(msg.get("status") or "").lower()
                if status in ("closed", "win", "loss"):
                    won = status == "win" or bool(msg.get("win"))
                    profit = _extract_profit(msg, won)
                    return {"status": "closed", "won": won, "profit": profit}
                if status in ("open", "pending"):
                    return {"status": "open"}
    except Exception as e:
        logger.debug("get_optioninfo failed for %s: %s", order_id, e)

    try:
        if hasattr(api, "get_async_order"):
            info = api.get_async_order(order_id)
            if isinstance(info, dict):
                msg = info.get("msg", info)
                status = str(msg.get("status") or info.get("status") or "").lower()
                if status in ("closed", "win", "loss"):
                    won = status == "win" or bool(msg.get("win"))
                    profit = _extract_profit(msg, won)
                    return {"status": "closed", "won": won, "profit": profit}
                if status in ("open", "pending"):
                    return {"status": "open"}
    except Exception as e:
        logger.debug("get_async_order failed for %s: %s", order_id, e)

    return None
