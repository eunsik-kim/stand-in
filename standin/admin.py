"""Accounts, daily limits and the admin view, on top of sign-in (standin/auth.py) and the store (standin/store.py).

- Every paid request is counted per account per day, by kind; past a kind's daily cap it is refused.
  Caps come from STANDIN_CAP_<KIND> (e.g. STANDIN_CAP_TURN=200).
- Admins (STANDIN_ADMIN_EMAILS, comma-separated) have no caps and get /admin: today's spend against
  the daily budget, the last 7 days, and every account's usage, with block and unblock.
- A test account signs in with STANDIN_TEST_TOKEN (POST /auth/test), for automated tests and the demo
  recorder. It has the ordinary caps and is off while the token is unset.
"""
import contextvars
import hmac
import os
import time
from collections.abc import Callable

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from standin import auth
from standin.store import DailySpend, Store, today

# which daily allowance each paid request draws on; anything else under /api/ is free
KINDS = {
    "/api/turn": "turn", "/api/auto": "turn", "/api/replies": "turn", "/api/coach": "turn",
    "/api/speak": "voice",
    "/api/episode": "create", "/api/build": "chat",
    "/api/build/finish": "scene", "/api/ending": "scene", "/api/end": "scene",  # about two per scene played
}
CAPS = {"turn": 400, "voice": 1500, "create": 5, "chat": 80, "scene": 60}
TEST = {"sub": "test-account", "email": "test@stand-in.test", "name": "Test account"}

current = contextvars.ContextVar("account", default="")  # whose request the paid calls are for


def cap(kind: str) -> int:
    return int(os.getenv(f"STANDIN_CAP_{kind.upper()}", CAPS[kind]))


def admins() -> set[str]:
    return {e.strip().lower() for e in os.getenv("STANDIN_ADMIN_EMAILS", "").split(",") if e.strip()}


def is_admin(u: dict | None) -> bool:
    return bool(u and u.get("sub") != TEST["sub"] and u.get("email", "").lower() in admins())


class Gate:
    """Decides, before a request runs, whether this account may spend on it today."""

    def __init__(self, store: Store):
        self.store = store
        self.noted: dict[str, float] = {}  # sub -> when last_seen was written, so it isn't written on every request

    def check(self, request: Request) -> JSONResponse | None:
        """None lets the request through; otherwise the response that stops it.

        What it takes is noted on `request.state.taken`, so `settle()` can give it back if the request fails anyway."""
        post = request.method == "POST" and request.url.path.startswith("/api/")
        kind = KINDS.get(request.url.path) if post else None
        u = auth.user(request)
        if u:
            current.set(u["sub"])
            if time.time() - self.noted.get(u["sub"], 0) > 300:
                self.store.seen(u["sub"], u.get("email", ""), u.get("name", ""))
                self.noted[u["sub"]] = time.time()
            if post and self.store.blocked(u["sub"]):  # free routes too: a blocked account starts nothing
                return JSONResponse({"detail": "this account is blocked"}, status_code=403)
        if not kind:
            return None
        if auth.on() and not u:
            return JSONResponse({"detail": "sign in first"}, status_code=401)
        if not u:
            return None  # sign-in is off (local runs): only the per-address limit applies
        if not self.store.take(u["sub"], kind, 10**9 if is_admin(u) else cap(kind)):  # admins are counted, never capped
            return JSONResponse({"detail": f"today's limit for {kind} is used up", "kind": kind, "limit": cap(kind)},
                                status_code=429)
        request.state.taken = (u["sub"], kind)
        return None

    def settle(self, request: Request, status: int) -> None:
        """A request that ended in an error doesn't use up the day's allowance."""
        taken = getattr(request.state, "taken", None)
        if taken and status >= 400:
            self.store.refund(*taken)


class TestLogin(BaseModel):
    token: str = Field(max_length=200)


class Block(BaseModel):
    sub: str = Field(max_length=200)
    blocked: bool = True


def router(store: Store, spend: DailySpend, live: Callable[[], dict]) -> APIRouter:
    """The admin and test-login routes. `live()` reports what only the server knows right now (open scenes and so on)."""
    r = APIRouter()

    def admin_or_none(request: Request) -> dict | None:
        u = auth.user(request)
        return u if is_admin(u) else None

    @r.post("/auth/test")
    def test_login(req: TestLogin, request: Request):
        token = os.getenv("STANDIN_TEST_TOKEN", "")
        if not auth.on() or not token or not hmac.compare_digest(req.token.encode(), token.encode()):
            return JSONResponse({"detail": "not available"}, status_code=404)
        res = JSONResponse({"ok": True, "email": TEST["email"]})
        res.set_cookie(auth.SESSION, auth.sign({**TEST, "exp": int(time.time()) + 86400}), max_age=86400, path="/",
                       **auth._cookie_args(request))
        return res

    @r.get("/api/usage")
    def my_usage(request: Request):
        """What the signed-in account has used today, against its caps."""
        u = auth.user(request)
        if not u:
            return {"signed_in": False}
        used = store.used(u["sub"])
        return {"signed_in": True, "admin": is_admin(u), "day": today(),
                "kinds": {k: {"used": used.get(k, 0), "limit": None if is_admin(u) else cap(k)} for k in CAPS}}

    @r.get("/admin")
    def admin_page(request: Request):
        if not admin_or_none(request):
            return RedirectResponse("/", status_code=302)
        return FileResponse(os.path.join(os.path.dirname(__file__), "..", "static", "admin.html"))

    @r.get("/api/admin/usage")
    def usage(request: Request):
        if not admin_or_none(request):
            return JSONResponse({"detail": "admins only"}, status_code=403)
        spend.flush()
        return {"day": today(), "budget": spend.budget, "spent": store.spent_today(), "days": store.spend_days(7),
                "caps": {k: cap(k) for k in CAPS}, "admins": sorted(admins()), "live": live(),
                "accounts": store.accounts_on()}

    @r.post("/api/admin/block")
    def block(req: Block, request: Request):
        me = admin_or_none(request)
        if not me:
            return JSONResponse({"detail": "admins only"}, status_code=403)
        if req.sub == me["sub"]:
            return JSONResponse({"detail": "you can't block yourself"}, status_code=400)
        store.block(req.sub, req.blocked)
        return {"ok": True}

    return r
