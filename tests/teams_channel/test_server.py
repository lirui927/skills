"""End-to-end tests for the Teams channel MCP server.

These run ``server.py`` as a real subprocess talking JSON-RPC over stdio, with
a local HTTP fake standing in for the Teams chat service. Nothing stubs the
server's own functions or the ``teams_client`` module it consumes: the state
directory is seeded with a token and a skype token so the unmodified client
reaches the fake, the subprocess frames every message on stdin/stdout itself,
and the assertions are about what crosses that boundary.
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "teams-channel"
SERVER = SKILL_DIR / "server.py"

OID = "oid-server-42"
USER_MRI = f"8:orgid:{OID}"
DISPLAY_NAME = "Li Rui"
USERNAME = "lirui927@contoso.com"
CHAT_ID = "19:thechat@thread.v2"


def make_jwt(claims: dict) -> str:
    """A syntactically valid unsigned JWT carrying ``claims`` in its payload."""

    def seg(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


class FakeChat:
    """In-process store of one Teams chat, served over HTTP to the client."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.messages: list[dict] = []
        self.counter = 0
        self.fail_lists = 0  # list requests to answer 500 before recovering

    def _add(self, content: str, frm: str, messagetype: str = "Text") -> dict:
        with self.lock:
            self.counter += 1
            msg = {
                "id": str(self.counter),
                "content": content,
                "messagetype": messagetype,
                "from": frm,
                "imdisplayname": DISPLAY_NAME,
                "originalarrivaltime": "2026-01-01T00:00:00Z",
            }
            self.messages.append(msg)
            return msg

    def add_user_message(self, content: str, messagetype: str = "Text") -> dict:
        """A message the signed-in person typed into the chat."""
        return self._add(content, USER_MRI, messagetype)

    def list_newest_first(self) -> list[dict]:
        with self.lock:
            return list(reversed(self.messages))

    def sent_by_server(self) -> list[dict]:
        """Messages the server posted: everything carrying the [Claude] mark."""
        with self.lock:
            return [m for m in self.messages if str(m["content"]).startswith("[Claude]")]


def _make_handler(fake: FakeChat):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep test output quiet
            pass

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def _send(self, code: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0].endswith("/messages"):
                with fake.lock:
                    fail = fake.fail_lists > 0
                    if fail:
                        fake.fail_lists -= 1
                if fail:
                    return self._send(500, {"error": "transient"})
                return self._send(200, {"messages": fake.list_newest_first()})
            return self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            raw = self._body()
            if self.path.endswith("/messages"):
                body = json.loads(raw or b"{}")
                msg = fake._add(body.get("content", ""), USER_MRI, body.get("messagetype", "Text"))
                return self._send(201, msg)
            return self._send(404, {"error": "not found"})

    return Handler


class Harness:
    """A running server subprocess wired to a fake chat, with stdio helpers."""

    def __init__(self, tmp_path: Path, fake: FakeChat) -> None:
        self.fake = fake
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(fake))
        self.root = f"http://127.0.0.1:{self.server.server_address[1]}"
        Thread(target=self.server.serve_forever, daemon=True).start()

        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "chat_id").write_text(CHAT_ID)
        far = time.time() + 100000
        claims = {"oid": OID, "name": DISPLAY_NAME, "preferred_username": USERNAME}
        (state / "token.json").write_text(json.dumps({
            "access_token": make_jwt(claims),
            "refresh_token": "refresh-x",
            "expires_at": far,
            "scope": "scope",
        }))
        (state / "skypetoken.json").write_text(json.dumps({
            "skypetoken": "SKYPE-TOKEN",
            "chat_service": self.root,
            "expires_at": far,
        }))

        env = dict(os.environ, TEAMS_STATE_DIR=str(state))
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        self._in_lock = Lock()
        self._msgs: list[dict] = []
        self._msgs_lock = Lock()
        self._watching = Event()
        Thread(target=self._pump_stdout, daemon=True).start()
        Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            with self._msgs_lock:
                self._msgs.append(msg)

    def _pump_stderr(self):
        for line in self.proc.stderr:
            if "watching" in line:
                self._watching.set()

    def send(self, obj: dict):
        with self._in_lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def recv(self, predicate, timeout: float = 2.0):
        """Return the first received message matching ``predicate`` in time."""
        deadline = time.time() + timeout
        seen = 0
        while time.time() < deadline:
            with self._msgs_lock:
                while seen < len(self._msgs):
                    msg = self._msgs[seen]
                    seen += 1
                    if predicate(msg):
                        return msg
            time.sleep(0.02)
        return None

    def none_matches(self, predicate, window: float = 1.0) -> bool:
        """True when no received message matches ``predicate`` within window."""
        return self.recv(predicate, timeout=window) is None

    def handshake(self):
        self.send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "0"}},
        })
        result = self.recv(lambda m: m.get("id") == 1, timeout=5.0)
        assert result is not None, "no initialize response"
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result["result"]

    def wait_watching(self, timeout: float = 5.0):
        assert self._watching.wait(timeout), "server never reported it was watching"

    def request(self, request_id, method, params=None):
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method,
                   "params": params or {}})
        resp = self.recv(lambda m: m.get("id") == request_id, timeout=5.0)
        assert resp is not None, f"no response to {method}"
        return resp

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        finally:
            self.server.shutdown()
            self.server.server_close()


@pytest.fixture
def harness(tmp_path):
    fake = FakeChat()
    h = Harness(tmp_path, fake)
    try:
        yield h
    finally:
        h.close()


# -- the named result ------------------------------------------------------


def test_inbound_message_and_permission_round_trip(harness):
    harness.handshake()
    harness.wait_watching()

    # An inbound message arrives as a channel notification within 2 seconds.
    harness.fake.add_user_message("hello from teams")
    note = harness.recv(
        lambda m: m.get("method") == "notifications/claude/channel", timeout=2.0
    )
    assert note is not None, "inbound message was not relayed within 2s"
    assert note["params"]["content"] == "hello from teams"
    meta = note["params"]["meta"]
    assert meta["chat_id"] == CHAT_ID
    assert meta["user_id"] == OID
    assert meta["message_id"]

    # One permission round trip: request reaches the chat, a reply answers it.
    harness.send({
        "jsonrpc": "2.0",
        "method": "notifications/claude/channel/permission_request",
        "params": {"request_id": "abcde", "tool_name": "Bash",
                   "description": "run ls", "input_preview": "ls"},
    })

    deadline = time.time() + 2.0
    prompt = None
    while time.time() < deadline:
        for m in harness.fake.sent_by_server():
            if "abcde" in m["content"] and "Bash" in m["content"]:
                prompt = m
                break
        if prompt:
            break
        time.sleep(0.02)
    assert prompt is not None, "permission prompt never reached the chat"

    harness.fake.add_user_message("yes abcde")
    perm = harness.recv(
        lambda m: m.get("method") == "notifications/claude/channel/permission",
        timeout=2.0,
    )
    assert perm is not None, "permission reply was not relayed within 2s"
    assert perm["params"] == {"request_id": "abcde", "behavior": "allow"}


def test_permission_reply_is_not_relayed_as_a_message(harness):
    harness.handshake()
    harness.wait_watching()
    harness.fake.add_user_message("no fghij")
    assert harness.none_matches(
        lambda m: m.get("method") == "notifications/claude/channel"
        and "fghij" in m["params"]["content"]
    )


# -- the handshake ---------------------------------------------------------


def test_initialize_declares_channel_capabilities(harness):
    result = harness.handshake()
    exp = result["capabilities"]["experimental"]
    assert "claude/channel" in exp
    assert "claude/channel/permission" in exp
    assert result["serverInfo"]["name"] == "teams-channel"
    assert result["instructions"].strip()


def test_tools_list_exposes_reply_and_fetch(harness):
    harness.handshake()
    resp = harness.request(2, "tools/list")
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {"reply", "fetch_messages"}
    reply = next(t for t in resp["result"]["tools"] if t["name"] == "reply")
    assert reply["inputSchema"]["required"] == ["chat_id", "text"]


def test_ping_answers_empty(harness):
    harness.handshake()
    resp = harness.request(3, "ping")
    assert resp["result"] == {}


def test_unknown_method_is_a_jsonrpc_error(harness):
    harness.handshake()
    resp = harness.request(4, "no/such/method")
    assert resp["error"]["code"] == -32601


# -- the tools -------------------------------------------------------------


def test_reply_sends_with_the_claude_prefix(harness):
    harness.handshake()
    resp = harness.request(5, "tools/call", {
        "name": "reply", "arguments": {"chat_id": CHAT_ID, "text": "on my way"},
    })
    assert resp["result"]["content"][0]["text"] == "sent"
    sent = harness.fake.sent_by_server()
    assert len(sent) == 1
    assert sent[0]["content"] == "[Claude] on my way"


def test_reply_rejects_a_foreign_chat_id(harness):
    harness.handshake()
    resp = harness.request(6, "tools/call", {
        "name": "reply", "arguments": {"chat_id": "19:other@thread.v2", "text": "hi"},
    })
    assert resp["result"]["isError"] is True
    assert harness.fake.sent_by_server() == []


def test_fetch_messages_returns_history_oldest_first(harness):
    harness.fake.add_user_message("first")
    harness.fake.add_user_message("second")
    harness.handshake()
    resp = harness.request(7, "tools/call", {
        "name": "fetch_messages", "arguments": {"chat_id": CHAT_ID},
    })
    text = resp["result"]["content"][0]["text"]
    assert text.index("first") < text.index("second")
    assert "(id: 1)" in text


def test_fetch_messages_rejects_a_foreign_chat_id(harness):
    harness.handshake()
    resp = harness.request(8, "tools/call", {
        "name": "fetch_messages", "arguments": {"chat_id": "19:other@thread.v2"},
    })
    assert resp["result"]["isError"] is True


# -- access and robustness -------------------------------------------------


def test_claude_prefixed_messages_are_not_relayed(harness):
    harness.handshake()
    harness.wait_watching()
    # A message the server itself posted appears in the poll; it must not be
    # relayed back as if the person had typed it.
    harness.fake._add("[Claude] my own words", USER_MRI)
    assert harness.none_matches(
        lambda m: m.get("method") == "notifications/claude/channel"
    )


def test_html_content_is_reduced_to_text(harness):
    harness.handshake()
    harness.wait_watching()
    harness.fake.add_user_message(
        "<p>line one<br>line two &amp; more</p>", messagetype="RichText/Html"
    )
    note = harness.recv(
        lambda m: m.get("method") == "notifications/claude/channel", timeout=2.0
    )
    assert note is not None
    assert note["params"]["content"] == "line one\nline two & more"


def test_a_failing_poll_is_retried_and_never_ends_the_server(harness):
    harness.handshake()
    harness.wait_watching()
    # Make the next few polls fail, then recover; the server must log and keep
    # serving, delivering a message posted after the failures.
    with harness.fake.lock:
        harness.fake.fail_lists = 3
    harness.fake.add_user_message("after the outage")
    note = harness.recv(
        lambda m: m.get("method") == "notifications/claude/channel", timeout=10.0
    )
    assert note is not None, "server did not recover from failing polls"
    assert note["params"]["content"] == "after the outage"
    assert harness.proc.poll() is None  # still running
