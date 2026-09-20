"""
main.py — FastAPI server that serves the HTML and handles IQ Option requests.
Run with:  python main.py
"""
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import connect  # our connect.py file


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(title="QT Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent


# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────
class ConnectRequest(BaseModel):
    email: str
    password: str
    mode: str = "demo"


class DisconnectRequest(BaseModel):
    session_id: str


class TradeRequest(BaseModel):
    session_id: str
    pair: str
    direction: str      # "call" or "put"
    amount: float
    duration: int       # minutes
    mode: str = "demo"


# ─────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "sessions": connect.session_count(),
        "time": time.time(),
    }


@app.post("/api/iq/connect")
def iq_connect(req: ConnectRequest):
    email = (req.email or "").strip()
    if not email or not req.password:
        raise HTTPException(400, "Email and password are required")
    if req.mode not in ("demo", "real"):
        raise HTTPException(400, "mode must be 'demo' or 'real'")

    try:
        session = connect.create_session(email, req.password, req.mode)
    except RuntimeError as e:
        raise HTTPException(401, f"Login failed: {e}")
    except Exception as e:
        raise HTTPException(500, f"Connection error: {e}")

    return {
        "status": "success",
        "session_id": session.id,
        "balance": session.balance,
        "currency": session.currency,
        "account_type": session.account_type,
        "mode": session.mode,
        "message": f"Connected to IQ Option ({session.account_type})",
    }


@app.post("/api/iq/disconnect")
def iq_disconnect(req: DisconnectRequest):
    ok = connect.remove_session(req.session_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"status": "success", "message": "Disconnected"}


@app.get("/api/iq/balance/{session_id}")
def iq_balance(session_id: str):
    session = connect.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired")
    try:
        session.balance = float(session.api.get_balance())
        return {
            "status": "success",
            "balance": session.balance,
            "currency": session.currency,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/iq/trade")
def iq_trade(req: TradeRequest):
    session = connect.get_session(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired")

    direction = req.direction.lower()
    if direction not in ("call", "put"):
        raise HTTPException(400, "direction must be 'call' or 'put'")

    try:
        session.api.change_balance("PRACTICE" if req.mode == "demo" else "REAL")
        check, order_id = session.api.buy(
            req.amount, req.pair, direction, req.duration
        )
        if not check:
            raise HTTPException(400, f"Trade rejected: {order_id}")

        result = session.api.check_win_v4(order_id)
        win_amount = 0.0
        try:
            win_amount = float(result[1] if isinstance(result, (list, tuple)) else result)
        except Exception:
            pass

        won = win_amount > 0
        return {
            "status": "success",
            "order_id": order_id,
            "won": won,
            "profit": win_amount if won else -req.amount,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Trade error: {e}")


# ─────────────────────────────────────────────
# Routes — serve your index.html
# ─────────────────────────────────────────────
@app.get("/")
def serve_index():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        return {"error": "index.html not found — put it next to main.py"}
    return FileResponse(index_file)


# Serve any static file (favicon, images, etc.)
try:
    app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")
except Exception:
    pass


# ─────────────────────────────────────────────
# Background cleanup (every 5 minutes)
# ─────────────────────────────────────────────
def _cleanup_loop():
    while True:
        time.sleep(300)
        removed = connect.cleanup_expired()
        if removed:
            print(f"[cleanup] removed {removed} expired session(s)")


threading.Thread(target=_cleanup_loop, daemon=True).start()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  ➜  Open http://localhost:{port} in your browser\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
