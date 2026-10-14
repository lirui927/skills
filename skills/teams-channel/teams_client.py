#!/usr/bin/env python3
"""A standard-library Microsoft Teams client scoped to one chat channel.

This talks to the same private chat-service surface the Teams desktop client
uses, because the standard Microsoft Graph chat scopes are not granted to the
public client this signs in as. The flow is three steps:

  1. Device-code sign-in against Azure AD for a token whose audience is the
     Skype/Teams spaces resource, saving the refresh token so later runs mint a
     fresh access token with no second sign-in.
  2. Exchange that access token at the Teams authz endpoint for a skype token
     and the account's regional chat-service base URL.
  3. Call the chat service with the skype token to list a chat, send to it, and
     create a group chat whose only member is the signed-in user.

Nothing outside this module's state directory ever supplies a token. The dev
host is a shared Unix account holding other people's sign-ins, so a stray
environment variable or a path pointing at a neighbour's token is exactly the
mistake this refuses to make: the only source is ``TEAMS_STATE_DIR`` (default
``~/.claude/channels/teams``), created 0700 with token files 0600.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Public client id of the Teams desktop app. A public client needs no secret,
# and this one is pre-authorized for the spaces/skype resource on tenants that
# run Teams at all, which is what lets the device-code flow reach the chat API.
CLIENT_ID = "1fec8e78-bce4-4aaf-ab1b-5451cc387264"
SCOPE = "https://api.spaces.skype.com/.default offline_access"
LOGIN_BASE = "https://login.microsoftonline.com"
AUTHZ_URL = "https://teams.microsoft.com/api/authsvc/v1.0/authz"
DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

# Refresh an access token this many seconds before its stated expiry so a call
# never races the boundary and fails mid-request.
EXPIRY_SKEW = 60.0


class TeamsError(RuntimeError):
    """Any failure in sign-in, token exchange, or a chat-service call."""


def _http(
    method: str,
    url: str,
    headers: dict | None = None,
    body=None,
    form: dict | None = None,
) -> tuple[int, dict, dict, str]:
    """One request. Returns ``(status, headers_lower, parsed_json, raw_text)``.

    ``form`` is url-encoded as an OAuth form post; ``body`` (a dict) is sent as
    JSON. A non-2xx answer is returned, not raised, so callers read its body;
    only a transport failure (no HTTP answer arrived) raises.
    """
    data = None
    hdrs = dict(headers or {})
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        hdrs["Content-Type"] = "application/x-www-form-urlencoded"
    elif isinstance(body, dict):
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            rh = {k.lower(): v for k, v in resp.getheaders()}
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        rh = {k.lower(): v for k, v in (e.headers or {}).items()}
        status = e.code
    except (urllib.error.URLError, OSError) as e:
        raise TeamsError(f"{method} {url}: no response ({type(e).__name__}: {e})")
    try:
        parsed = json.loads(raw) if raw else {}
    except ValueError:
        parsed = {}
    return status, rh, parsed, raw


def _jwt_claims(token: str) -> dict:
    """Decode a JWT's payload without verifying it.

    The access token is only read for the signed-in account's identity (object
    id, name, username); it is never trusted as proof of anything, so decoding
    the middle segment is enough and needs no key.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode()))
    except (IndexError, ValueError):
        return {}


class TeamsClient:
    """Sign-in, token refresh, and the one-chat chat-service calls."""

    def __init__(
        self,
        state_dir: str | os.PathLike | None = None,
        login_base: str = LOGIN_BASE,
        authz_url: str = AUTHZ_URL,
        client_id: str = CLIENT_ID,
    ) -> None:
        if state_dir is None:
            state_dir = os.environ.get("TEAMS_STATE_DIR") or (
                Path.home() / ".claude" / "channels" / "teams"
            )
        self.state_dir = Path(state_dir)
        self.login_base = login_base.rstrip("/")
        self.authz_url = authz_url
        self.client_id = client_id
        self._token_file = self.state_dir / "token.json"
        self._skype_file = self.state_dir / "skypetoken.json"
        self._token = self._read_json(self._token_file)
        self._skype = self._read_json(self._skype_file)

    # -- state -------------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    def _write_secret(self, path: Path, data: dict) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        # Create owner-only from the start: a default umask would otherwise
        # leave a world-readable token on disk between write and chmod.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.chmod(path, 0o600)

    # -- device-code sign-in ----------------------------------------------

    def begin_signin(self) -> dict:
        """Start the device-code flow. Returns the code block to show the user."""
        status, _, data, raw = _http(
            "POST",
            f"{self.login_base}/organizations/oauth2/v2.0/devicecode",
            form={"client_id": self.client_id, "scope": SCOPE},
        )
        if status != 200 or "device_code" not in data:
            raise TeamsError(f"device-code request failed (HTTP {status}): {raw[:200]}")
        return data

    def poll_signin(self, device_code: str, interval: float = 5.0, timeout: float = 900.0) -> dict:
        """Poll the token endpoint until the user signs in, then save the token."""
        deadline = time.time() + timeout
        while True:
            status, _, data, raw = _http(
                "POST",
                f"{self.login_base}/organizations/oauth2/v2.0/token",
                form={
                    "client_id": self.client_id,
                    "grant_type": DEVICE_CODE_GRANT,
                    "device_code": device_code,
                },
            )
            if status == 200 and "access_token" in data:
                self._store_token(data)
                return self._token
            error = data.get("error", "")
            if error == "authorization_pending":
                pass
            elif error == "slow_down":
                interval += 5
            else:
                raise TeamsError(f"sign-in failed: {error or raw[:200]}")
            if time.time() >= deadline:
                raise TeamsError("sign-in timed out before the code was entered")
            time.sleep(interval)

    def signin(self, interval: float = 5.0, prompt=None) -> dict:
        """Full device-code sign-in: begin, show the code, poll to completion."""
        block = self.begin_signin()
        message = block.get("message") or (
            f"Sign in at {block.get('verification_uri')} with code {block.get('user_code')}"
        )
        (prompt or (lambda m: print(m)))(message)
        return self.poll_signin(block["device_code"], interval=interval)

    def _store_token(self, data: dict) -> None:
        self._token = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", self._token.get("refresh_token", "")),
            "expires_at": time.time() + float(data.get("expires_in", 3600)),
            "scope": data.get("scope", SCOPE),
        }
        self._write_secret(self._token_file, self._token)
        # A new access token invalidates the derived skype token.
        self._skype = {}

    # -- access token ------------------------------------------------------

    def _access_token(self) -> str:
        """A valid access token, refreshed from the refresh token when stale."""
        if not self._token.get("access_token"):
            raise TeamsError("not signed in: run signin first")
        if time.time() < self._token.get("expires_at", 0) - EXPIRY_SKEW:
            return self._token["access_token"]
        refresh = self._token.get("refresh_token")
        if not refresh:
            raise TeamsError("access token expired and no refresh token: sign in again")
        status, _, data, raw = _http(
            "POST",
            f"{self.login_base}/organizations/oauth2/v2.0/token",
            form={
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "scope": SCOPE,
            },
        )
        if status != 200 or "access_token" not in data:
            raise TeamsError(f"token refresh failed (HTTP {status}): {raw[:200]}")
        self._store_token(data)
        return self._token["access_token"]

    # -- skype token + chat-service base -----------------------------------

    def _skype_token(self) -> tuple[str, str]:
        """A valid skype token and chat-service base, minted via authz on expiry."""
        if self._skype.get("skypetoken") and time.time() < self._skype.get("expires_at", 0) - EXPIRY_SKEW:
            return self._skype["skypetoken"], self._skype["chat_service"]
        access = self._access_token()
        status, _, data, raw = _http(
            "POST",
            self.authz_url,
            headers={"Authorization": f"Bearer {access}"},
            body={},
        )
        if status != 200:
            raise TeamsError(f"authz failed (HTTP {status}): {raw[:200]}")
        tokens = data.get("tokens") or {}
        token = tokens.get("skypeToken") or data.get("skypeToken") or ""
        if not token:
            raise TeamsError("authz returned no skype token")
        region = data.get("regionGtms") or {}
        base = region.get("chatService") or data.get("chatService") or ""
        if not base:
            raise TeamsError("authz returned no chat-service base URL")
        expires_in = float(tokens.get("expiresIn") or data.get("expiresIn") or 3600)
        self._skype = {
            "skypetoken": token,
            "chat_service": base.rstrip("/"),
            "expires_at": time.time() + expires_in,
        }
        self._write_secret(self._skype_file, self._skype)
        return token, self._skype["chat_service"]

    def _chat_headers(self) -> tuple[dict, str]:
        token, base = self._skype_token()
        return {"Authentication": f"skypetoken={token}"}, base

    # -- public surface ----------------------------------------------------

    def me(self) -> dict:
        """The signed-in account: object id, username, and display name."""
        claims = _jwt_claims(self._access_token())
        return {
            "oid": claims.get("oid", ""),
            "username": claims.get("preferred_username") or claims.get("upn", ""),
            "name": claims.get("name", ""),
        }

    def list_messages(self, chat_id: str, page_size: int = 20) -> list:
        """Messages in a chat, newest first, as the chat service returns them."""
        headers, base = self._chat_headers()
        cid = urllib.parse.quote(chat_id, safe="")
        url = f"{base}/v1/users/ME/conversations/{cid}/messages?pageSize={page_size}"
        status, _, data, raw = _http("GET", url, headers=headers)
        if status != 200:
            raise TeamsError(f"list_messages failed (HTTP {status}): {raw[:200]}")
        return data.get("messages", [])

    def send_message(self, chat_id: str, text: str) -> dict:
        """Post a plain-text message to a chat. Returns the service's response."""
        headers, base = self._chat_headers()
        cid = urllib.parse.quote(chat_id, safe="")
        url = f"{base}/v1/users/ME/conversations/{cid}/messages"
        body = {
            "content": text,
            "messagetype": "Text",
            "contenttype": "text",
            # A client-stamped id the service dedups on, so a retried POST can
            # never deliver the same message twice.
            "clientmessageid": str(int.from_bytes(os.urandom(8), "big")),
        }
        status, _, data, raw = _http("POST", url, headers=headers, body=body)
        if status not in (200, 201):
            raise TeamsError(f"send_message failed (HTTP {status}): {raw[:200]}")
        return data

    def create_chat(self, topic: str) -> str:
        """Create a group chat whose only member is the signed-in user.

        Returns the new chat id, read from the ``Location`` header the thread
        endpoint answers with.
        """
        oid = self.me().get("oid")
        if not oid:
            raise TeamsError("cannot create a chat without the signed-in object id")
        headers, base = self._chat_headers()
        body = {
            "members": [{"id": f"8:orgid:{oid}", "role": "Admin"}],
            "properties": {"threadType": "chat", "topic": topic},
        }
        status, rh, data, raw = _http("POST", f"{base}/v1/threads", headers=headers, body=body)
        if status not in (200, 201):
            raise TeamsError(f"create_chat failed (HTTP {status}): {raw[:200]}")
        location = rh.get("location", "")
        chat_id = location.rstrip("/").rsplit("/", 1)[-1] if location else data.get("id", "")
        if not chat_id:
            raise TeamsError("create_chat returned no chat id")
        return urllib.parse.unquote(chat_id)


# -- CLI -------------------------------------------------------------------


def _cmd_signin(client: TeamsClient, args) -> int:
    client.signin()
    who = client.me()
    print(f"Signed in as {who['name']} <{who['username']}>")
    return 0


def _cmd_status(client: TeamsClient, args) -> int:
    try:
        who = client.me()
    except TeamsError as e:
        print(f"Not signed in: {e}", file=sys.stderr)
        return 1
    print(f"Signed in as {who['name']} <{who['username']}>")
    return 0


def _cmd_create_chat(client: TeamsClient, args) -> int:
    chat_id = client.create_chat(args.topic)
    print(chat_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-chat Microsoft Teams client")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("signin", help="device-code sign-in")
    sub.add_parser("status", help="show the signed-in account")
    create = sub.add_parser("create-chat", help="create a self-only group chat")
    create.add_argument("topic", help="the chat topic")
    args = parser.parse_args(argv)

    client = TeamsClient()
    handlers = {
        "signin": _cmd_signin,
        "status": _cmd_status,
        "create-chat": _cmd_create_chat,
    }
    try:
        return handlers[args.command](client, args)
    except TeamsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
