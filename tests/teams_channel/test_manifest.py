"""The real manifest's inline MCP server answers ``initialize`` as a channel.

This drives the repository's own ``.claude-plugin/marketplace.json`` -- not a
copy -- by starting the ``teams-channel`` server with exactly the command and
arguments the manifest declares, and asserting its ``initialize`` reply offers
both the ``claude/channel`` and ``claude/channel/permission`` capabilities. A
manifest whose entry drops the ``mcpServers`` object, or names a command that
cannot start the server, fails here.
"""

import base64
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / ".claude-plugin" / "marketplace.json"


def _manifest_server() -> dict:
    """The ``teams-channel`` bundle's single inline MCP server definition."""
    plugins = json.loads(MANIFEST.read_text())["plugins"]
    entry = next(p for p in plugins if p["name"] == "teams-channel")
    servers = entry["mcpServers"]
    (name,) = servers  # exactly one inline server on this entry
    return servers[name]


def _unsigned_jwt(claims: dict) -> str:
    def seg(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


def _signed_in_state(state_dir: Path) -> None:
    """Write the chat id and a fresh token the server reads on startup."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "chat_id").write_text("19:fake-chat@thread.v2")
    (state_dir / "token.json").write_text(
        json.dumps(
            {
                "access_token": _unsigned_jwt({"oid": "oid-1", "name": "Tester"}),
                "refresh_token": "refresh",
                "expires_at": time.time() + 3600,
                "scope": "scope",
            }
        )
    )


def test_manifest_server_initializes_as_a_channel(tmp_path):
    server = _manifest_server()
    argv = [server["command"]] + [
        arg.replace("${CLAUDE_PLUGIN_ROOT}", str(REPO_ROOT)) for arg in server["args"]
    ]

    state_dir = tmp_path / "state"
    _signed_in_state(state_dir)
    env = {**os.environ, "TEAMS_STATE_DIR": str(state_dir)}

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
    }
    proc = subprocess.run(
        argv,
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    reply = json.loads(proc.stdout.splitlines()[0])
    assert reply["id"] == 1, proc.stderr
    experimental = reply["result"]["capabilities"]["experimental"]
    assert "claude/channel" in experimental
    assert "claude/channel/permission" in experimental
