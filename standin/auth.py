"""Google sign-in, on only when GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are set.

Nothing is stored on the server: the session is a cookie the server signs itself (HMAC with
STANDIN_SESSION_SECRET) holding the Google account id, email, name and when it runs out.
Standard library only, so the image needs no new dependency.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
import urllib.request
from html import escape

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

AUTHORIZE = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN = "https://oauth2.googleapis.com/token"
TOKENINFO = "https://oauth2.googleapis.com/tokeninfo"
ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
SESSION, STATE = "standin_session", "standin_oauth_state"
DAYS = 30
_spare_key = secrets.token_bytes(32)  # without STANDIN_SESSION_SECRET, sign-ins last until the next restart

router = APIRouter()


def on() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


def _key() -> bytes:
    return os.getenv("STANDIN_SESSION_SECRET", "").encode() or _spare_key


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _mac(body: str) -> str:
    return _b64(hmac.new(_key(), body.encode(), hashlib.sha256).digest())


def sign(data: dict) -> str:
    body = _b64(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode())
    return f"{body}.{_mac(body)}"


def user(request: Request) -> dict | None:
    """The signed-in Google account, or None: no cookie, a cookie this server didn't sign, or one past its time."""
    if not on():
        return None
    body, _, mac = request.cookies.get(SESSION, "").partition(".")
    if not body or not hmac.compare_digest(mac.encode(), _mac(body).encode()):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except ValueError:
        return None
    return data if isinstance(data, dict) and data.get("sub") and data.get("exp", 0) > time.time() else None


def origin(request: Request) -> str:
    """Where Google sends people back to: STANDIN_PUBLIC_URL, else this request as the browser saw it (https behind Fly)."""
    public = os.getenv("STANDIN_PUBLIC_URL", "").rstrip("/")
    if public:
        return public
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    return f"{proto}://{request.headers.get('host') or request.url.netloc}"


def _cookie_args(request: Request) -> dict:
    return {"httponly": True, "secure": origin(request).startswith("https://"), "samesite": "lax"}


def _fetch(url: str, form: dict | None = None) -> dict:
    data = urllib.parse.urlencode(form).encode() if form else None
    with urllib.request.urlopen(urllib.request.Request(url, data), timeout=10) as r:
        return json.loads(r.read())


def _account(code: str, redirect_uri: str) -> dict:
    """Trade the code for an ID token and have Google check it; only a verified email for this app gets through."""
    token = _fetch(TOKEN, {"code": code, "client_id": os.environ["GOOGLE_CLIENT_ID"], "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                           "redirect_uri": redirect_uri, "grant_type": "authorization_code"})["id_token"]
    c = _fetch(f"{TOKENINFO}?{urllib.parse.urlencode({'id_token': token})}")
    if (c.get("aud") != os.environ["GOOGLE_CLIENT_ID"] or c.get("iss") not in ISSUERS
            or int(c.get("exp", 0)) < time.time() or str(c.get("email_verified")).lower() != "true" or not c.get("sub")):
        raise ValueError("ID token rejected")
    return c


def _failed(message: str) -> HTMLResponse:
    page = (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<body style="background:#08090c;color:#eef0f4;font:16px system-ui;padding:24px">'
            f'<p>{escape(message)}</p><p><a href="/" style="color:#f2c14e">Back to Stand-In</a></p></body>')
    r = HTMLResponse(page, status_code=400)
    r.delete_cookie(STATE, path="/auth")
    return r


@router.get("/auth/login")
def login(request: Request):
    if not on():
        return RedirectResponse("/", status_code=302)
    state = secrets.token_urlsafe(24)
    query = urllib.parse.urlencode({"client_id": os.environ["GOOGLE_CLIENT_ID"], "redirect_uri": f"{origin(request)}/auth/callback",
                                    "response_type": "code", "scope": "openid email profile", "state": state, "prompt": "select_account"})
    r = RedirectResponse(f"{AUTHORIZE}?{query}", status_code=302)
    r.set_cookie(STATE, state, max_age=600, path="/auth", **_cookie_args(request))
    return r


@router.get("/auth/callback")
def callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if not on():
        return RedirectResponse("/", status_code=302)
    if error:  # they backed out at Google: the sign-in card is waiting at home
        r = RedirectResponse("/", status_code=302)
        r.delete_cookie(STATE, path="/auth")
        return r
    expected = request.cookies.get(STATE, "")
    if not code or not expected or not hmac.compare_digest(state.encode(), expected.encode()):
        return _failed("That sign-in link has expired or didn't start here. Please sign in again.")
    try:
        c = _account(code, f"{origin(request)}/auth/callback")
    except Exception as e:
        print(f"google sign-in failed: {e!r:.200}", flush=True)
        return _failed("Google sign-in didn't go through. Please try again.")
    r = RedirectResponse("/", status_code=302)
    r.delete_cookie(STATE, path="/auth")
    r.set_cookie(SESSION, sign({"sub": c["sub"], "email": c.get("email", ""), "name": c.get("name", ""), "exp": int(time.time()) + DAYS * 86400}),
                 max_age=DAYS * 86400, path="/", **_cookie_args(request))
    return r


@router.get("/auth/logout")
def logout():
    r = RedirectResponse("/", status_code=302)
    r.delete_cookie(SESSION, path="/")
    return r


@router.get("/api/me")
def me(request: Request):
    u = user(request) or {}
    return {"required": on(), "signed_in": bool(u), "email": u.get("email", ""), "name": u.get("name", "")}
