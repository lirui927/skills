"""A local HTTP fake standing in for the Azure AD and Teams chat services.

The tests drive ``teams_client`` over real ``urllib`` calls against this
server, so they exercise the request building, header shapes, token refresh and
``Location``-header parsing exactly as the live services would see them -- not
stubbed-out copies of the module's own methods.
"""

import base64
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "teams-channel"
sys.path.insert(0, str(SKILL_DIR))

# The account the fake's tokens describe. me() decodes this out of the JWT.
IDENTITY = {
    "oid": "oid-abc-123",
    "name": "Li Rui",
    "preferred_username": "lirui927@contoso.com",
}


def make_jwt(claims: dict) -> str:
    """A syntactically valid unsigned JWT carrying ``claims`` in its payload."""

    def seg(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


class FakeState:
    """Shared, inspectable state for one fake server."""

    def __init__(self) -> None:
        self.events: list[str] = []          # one label per request, in order
        self.requests: list[dict] = []       # full records for detailed asserts
        self.messages: dict[str, list] = {}  # chat id -> stored messages
        self.device_expires_in = 3600        # device-grant token lifetime
        self.pending_polls = 0               # device grant pends this many times
        self.refresh_count = 0
        self.next_chat_id = "19:newchat@thread.v2"


def _make_handler(state: FakeState, root: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the test output quiet
            pass

        def _read(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def _send(self, code: int, payload: dict | None = None, headers: dict | None = None):
            body = json.dumps(payload or {}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _record(self, label: str, parsed) -> None:
            state.events.append(label)
            state.requests.append(
                {"method": self.command, "path": self.path, "label": label,
                 "headers": dict(self.headers), "body": parsed}
            )

        def do_POST(self):  # noqa: N802
            path = urllib.parse.urlparse(self.path).path
            raw = self._read()
            if path.endswith("/oauth2/v2.0/devicecode"):
                self._record("devicecode", urllib.parse.parse_qs(raw.decode()))
                return self._send(200, {
                    "device_code": "DEV-CODE", "user_code": "ABCD-EFGH",
                    "verification_uri": "https://microsoft.com/devicelogin",
                    "interval": 1, "expires_in": 900,
                    "message": "Go to the URL and enter ABCD-EFGH",
                })
            if path.endswith("/oauth2/v2.0/token"):
                form = {k: v[0] for k, v in urllib.parse.parse_qs(raw.decode()).items()}
                grant = form.get("grant_type", "")
                if grant.endswith("device_code"):
                    if state.pending_polls > 0:
                        state.pending_polls -= 1
                        self._record("token:pending", form)
                        return self._send(400, {"error": "authorization_pending"})
                    self._record("token:device", form)
                    return self._send(200, {
                        "access_token": make_jwt(IDENTITY), "refresh_token": "refresh-1",
                        "expires_in": state.device_expires_in, "scope": "scope",
                    })
                if grant == "refresh_token":
                    state.refresh_count += 1
                    self._record("token:refresh", form)
                    return self._send(200, {
                        "access_token": make_jwt(IDENTITY), "refresh_token": "refresh-2",
                        "expires_in": 3600, "scope": "scope",
                    })
                self._record("token:unknown", form)
                return self._send(400, {"error": "unsupported_grant_type"})
            if path.endswith("/authsvc/v1.0/authz"):
                self._record("authz", {})
                return self._send(200, {
                    "tokens": {"skypeToken": "SKYPE-TOKEN", "expiresIn": 86400},
                    "regionGtms": {"chatService": root},
                })
            if path.endswith("/v1/threads"):
                self._record("create", json.loads(raw or b"{}"))
                location = f"{root}/v1/threads/{urllib.parse.quote(state.next_chat_id, safe='')}"
                return self._send(201, {}, {"Location": location})
            if path.endswith("/messages"):
                cid = urllib.parse.unquote(path.split("/conversations/")[1].split("/messages")[0])
                body = json.loads(raw or b"{}")
                self._record("send", body)
                msg = {"id": str(len(state.messages.get(cid, [])) + 1),
                       "content": body.get("content"), "messagetype": body.get("messagetype")}
                state.messages.setdefault(cid, []).append(msg)
                return self._send(201, msg)
            self._record("post:unknown", None)
            return self._send(404, {"error": "not found"})

        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.endswith("/messages"):
                cid = urllib.parse.unquote(parsed.path.split("/conversations/")[1].split("/messages")[0])
                self._record("list", urllib.parse.parse_qs(parsed.query))
                return self._send(200, {"messages": state.messages.get(cid, [])})
            self._record("get:unknown", None)
            return self._send(404, {"error": "not found"})

    return Handler


@pytest.fixture
def fake(tmp_path):
    """Yield ``(client, state, root)`` wired to a running fake server."""
    import teams_client

    state = FakeState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), None)
    root = f"http://127.0.0.1:{server.server_address[1]}"
    server.RequestHandlerClass = _make_handler(state, root)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = teams_client.TeamsClient(
        state_dir=tmp_path / "state",
        login_base=root,
        authz_url=f"{root}/api/authsvc/v1.0/authz",
    )
    try:
        yield client, state, root
    finally:
        server.shutdown()
        server.server_close()
