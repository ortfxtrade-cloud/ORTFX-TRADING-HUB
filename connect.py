"""
IQ Option connection helpers.

Uses the community `iqoptionapi` library (unofficial).
Install with:
  pip install git+https://github.com/williansandi/iqoptionapi-2025-Atualizada-.git
  # or older:
  # pip install git+https://github.com/iqoptionapi/iqoptionapi.git

IMPORTANT
---------
- There is NO official public IQ Option API.
- Always test on PRACTICE (demo) first.
- Automating real accounts can violate ToS and risk bans.
- Credentials should never be logged or committed.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("qt.connect")

# ---------------------------------------------------------------------------
# In-memory session store (process lifetime)
# session_id -> IQSession
# ---------------------------------------------------------------------------
_sessions: Dict[str, "IQSession"] = {}
_lock = threading.Lock()


@dataclass
class IQSession:
    session_id: str
    email: str
    mode: str  # demo | real
    api: Any = None  # IQ_Option instance
    balance: float = 0.0
    currency: str = "USD"
    account_type: str = "PRACTICE"
    connected_at: float = field(default_factory=time.time)
    last_error: Optional[str] = None

    def is_alive(self) -> bool:
        if self.api is None:
            return False
        try:
            return bool(self.api.check_connect())
        except Exception:
            return False


def _import_iq_option():
    """Lazy import so the rest of the app can start even if lib is missing."""
    try:
        from iqoptionapi.stable_api import IQ_Option  # type: ignore
        return IQ_Option
    except ImportError as e:
        raise ImportError(
            "iqoptionapi is not installed. Run:\n"
            "  pip install git+https://github.com/williansandi/iqoptionapi-2025-Atualizada-.git\n"
            f"Original error: {e}"
        ) from e


def connect_iq(
    email: str,
    password: str,
    mode: str = "demo",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Connect to IQ Option and return a session payload matching the frontend.

    Returns:
        (success, payload)
        payload keys on success:
          status, session_id, balance, currency, account_type, message
        on failure:
          status, message
    """
    email = (email or "").strip().lower()
    password = password or ""
    mode = (mode or "demo").lower()
    if mode not in ("demo", "real"):
        mode = "demo"

    if not email or not password:
        return False, {"status": "error", "message": "Email and password are required"}

    IQ_Option = _import_iq_option()

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
        if reason == "2FA" or (isinstance(msg, str) and "2FA" in msg.upper()):
            return False, {
                "status": "error",
                "message": "Two-factor authentication required. Disable 2FA or use connect_2fa flow.",
            }
        return False, {"status": "error", "message": str(msg) or "Invalid credentials"}

    # Switch balance type
    balance_type = "PRACTICE" if mode == "demo" else "REAL"
    try:
        api.change_balance(balance_type)
    except Exception as e:
        logger.warning("change_balance failed: %s", e)

    # Read balance
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

    session_id = f"iq_{uuid.uuid4().hex[:16]}_{secrets.token_hex(4)}"

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
        # Drop any previous session for same email
        for sid, old in list(_sessions.items()):
            if old.email == email:
                try:
                    if old.api and old.is_alive():
                        pass
                except Exception:
                    pass
                del _sessions[sid]
        _sessions[session_id] = sess

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
    """Remove session and try to clean up."""
    with _lock:
        sess = _sessions.pop(session_id, None)
    if not sess:
        return False, "Session not found"
    try:
        if sess.api is not None:
            try:
                if hasattr(sess.api, "logout"):
                    sess.api.logout()
            except Exception:
                pass
            sess.api = None
    except Exception as e:
        logger.warning("disconnect cleanup: %s", e)
    return True, "Disconnected"


def get_session(session_id: str) -> Optional[IQSession]:
    with _lock:
        return _sessions.get(session_id)


def refresh_balance(session_id: str) -> Optional[float]:
    sess = get_session(session_id)
    if not sess or not sess.is_alive():
        return None
    try:
        bal = float(sess.api.get_balance() or 0)
        sess.balance = bal
        return bal
    except Exception:
        return None


def place_binary_order(
    session_id: str,
    active: str,
    direction: str,
    amount: float,
    duration_min: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Place a binary options order on IQ Option.

    active: e.g. "EURUSD" (no slash)
    direction: "call" / "put"  (BUY→call, SELL→put)
    amount: stake in account currency
    duration_min: expiry in minutes (1, 5, ...)
    """
    sess = get_session(session_id)
    if not sess or not sess.is_alive():
        return False, {"message": "Session not connected"}

    active = active.replace("/", "").replace(" ", "").upper()
    direction = direction.lower()
    if direction in ("buy", "up", "call"):
        direction = "call"
    elif direction in ("sell", "down", "put"):
        direction = "put"
    else:
        return False, {"message": "direction must be BUY/SELL or call/put"}

    try:
        ok, order_id = sess.api.buy(amount, active, direction, duration_min)
        if not ok:
            return False, {"message": f"Order rejected: {order_id}"}
        return True, {
            "order_id": str(order_id),
            "active": active,
            "direction": direction,
            "amount": amount,
            "duration": duration_min,
        }
    except Exception as e:
        logger.exception("place order failed")
        return False, {"message": str(e)}


def list_active_sessions() -> list:
    with _lock:
        return [
            {
                "session_id": s.session_id,
                "email": s.email,
                "mode": s.mode,
                "balance": s.balance,
                "currency": s.currency,
                "alive": s.is_alive(),
            }
            for s in _sessions.values()
        ]
