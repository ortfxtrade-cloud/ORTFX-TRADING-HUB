"""
connect.py — IQ Option connection + session management
"""
import uuid
import time
from typing import Optional, Dict
from iqoptionapi.stable_api import IQ_Option


# In-memory session store: { session_id: IQSession }
_sessions: Dict[str, "IQSession"] = {}


class IQSession:
    def __init__(self, email: str, mode: str = "demo"):
        self.id = str(uuid.uuid4())
        self.email = email
        self.mode = mode
        self.api: Optional[IQ_Option] = None
        self.balance: float = 0.0
        self.currency: str = "USD"
        self.account_type = "PRACTICE" if mode == "demo" else "REAL"
        self.created_at = time.time()
        self.last_used = time.time()

    def connect(self, password: str) -> bool:
        self.api = IQ_Option(self.email, password)
        ok, reason = self.api.connect()
        if not ok:
            self.api = None
            raise RuntimeError(reason or "IQ Option login failed")

        self.api.change_balance(self.account_type)

        try:
            self.balance = float(self.api.get_balance())
        except Exception:
            self.balance = 0.0

        try:
            self.currency = self.api.get_currency() or "USD"
        except Exception:
            self.currency = "USD"

        self.last_used = time.time()
        return True

    def disconnect(self):
        try:
            if self.api:
                self.api.logout()
        except Exception:
            pass
        self.api = None

    def expired(self, ttl_seconds: int = 60 * 60 * 4) -> bool:
        return (time.time() - self.last_used) > ttl_seconds

    def to_dict(self):
        return {
            "session_id": self.id,
            "email": self.email,
            "mode": self.mode,
            "account_type": self.account_type,
            "balance": self.balance,
            "currency": self.currency,
        }


def create_session(email: str, password: str, mode: str = "demo") -> IQSession:
    """Log into IQ Option and register a new session."""
    session = IQSession(email=email, mode=mode)
    session.connect(password)
    _sessions[session.id] = session
    return session


def get_session(session_id: str) -> Optional[IQSession]:
    session = _sessions.get(session_id)
    if not session:
        return None
    if session.expired():
        session.disconnect()
        _sessions.pop(session_id, None)
        return None
    session.last_used = time.time()
    return session


def remove_session(session_id: str) -> bool:
    session = _sessions.pop(session_id, None)
    if not session:
        return False
    session.disconnect()
    return True


def cleanup_expired():
    dead = [sid for sid, s in _sessions.items() if s.expired()]
    for sid in dead:
        s = _sessions.pop(sid, None)
        if s:
            s.disconnect()
    return len(dead)


def session_count() -> int:
    return len(_sessions)
