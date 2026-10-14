"""以子进程真实运行 ``launch.py``，``PATH`` 最前面放假的 ``tmux`` 和 ``claude``。

假 ``tmux`` 记下每次调用，``new-session`` 时真的执行收到的命令（也就是真的
``env -i ... claude ...``），``capture-pane`` 交出假 ``claude`` 写的屏幕，
``send-keys`` 把按键记到文件里，``list-panes`` 报出它启动的进程号。假 ``claude``
记下自己的参数和环境，先显示开发通道警告，收到 Enter 后回到输入框，再按模式
起一个命令行里带 ``skills/teams-channel/server.py`` 的子进程：``up`` 立刻起，
``reconnect`` 要等 ``/mcp`` 和三次 Enter，``never`` 不起。
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "teams-channel"
LAUNCH = SKILL_DIR / "launch.py"
RECONNECT = ["Enter", "/mcp", "Enter", "Enter", "Enter", "Escape", "Escape", "Escape"]
# 会话只该收到这些变量；调用方环境里的其余变量都不能传进去。
ALLOWED_ENV = set(
    "HOME USER LOGNAME SHELL TERM LANG LC_ALL PATH http_proxy https_proxy no_proxy "
    "HTTP_PROXY HTTPS_PROXY NO_PROXY SSL_CERT_FILE REQUESTS_CA_BUNDLE TEAMS_STATE_DIR".split()
)

FAKE_TMUX = """#!{python}
import json, os, signal, subprocess, sys
from pathlib import Path
D = Path({root!r})
args = sys.argv[1:]
with open(D / "tmux.log", "a") as f:
    f.write(json.dumps(args) + "\\n")
pidfile = D / "pane.pid"
cmd = args[0]
if cmd == "has-session":
    sys.exit(0 if pidfile.exists() else 1)
if cmd == "new-session":
    at = args.index("-c")
    proc = subprocess.Popen(args[at + 2:], cwd=args[at + 1], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=open(D / "pane.err", "w"))
    pidfile.write_text(str(proc.pid))
elif cmd == "list-panes":
    if not pidfile.exists():
        sys.exit(1)
    print(pidfile.read_text())
elif cmd == "capture-pane":
    print((D / "screen").read_text() if (D / "screen").exists() else "")
elif cmd == "send-keys":
    with open(D / "keys", "a") as f:
        f.write(args[-1] + "\\n")
elif cmd == "kill-session":
    os.killpg(int(pidfile.read_text()), signal.SIGKILL)
    pidfile.unlink()
"""

FAKE_CLAUDE = """#!{python}
import json, os, subprocess, sys, time
from pathlib import Path
D = Path({root!r})
# 只记变量名：断言失败时不会把调用方环境里的密钥打印出来。
(D / "claude.json").write_text(json.dumps({{
    "argv": sys.argv[1:], "env": sorted(os.environ),
    "state": os.environ.get("TEAMS_STATE_DIR"),
}}))
def keys():
    return (D / "keys").read_text().split() if (D / "keys").exists() else []
def until(pred):
    end = time.time() + 60
    while not pred() and time.time() < end:
        time.sleep(0.02)
def serve():
    subprocess.Popen([sys.executable, str(D / "skills" / "teams-channel" / "server.py")])
(D / "screen").write_text("WARNING\\n1. I am using this for local development\\n")
until(lambda: keys()[:1] == ["Enter"])
(D / "screen").write_text("> \\n  ? for shortcuts\\n")
if {mode!r} == "up":
    serve()
elif {mode!r} == "reconnect":
    until(lambda: keys()[1:5] == ["/mcp", "Enter", "Enter", "Enter"])
    serve()
time.sleep(60)
"""


def _script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


class Rig:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.home = root / "home"
        self.home.mkdir()
        _script(root / "bin" / "tmux", FAKE_TMUX.format(python=sys.executable, root=str(root)))
        _script(root / "skills" / "teams-channel" / "server.py", "import time; time.sleep(60)\n")
        self.decoys = []

    def claude(self, mode: str) -> None:
        text = FAKE_CLAUDE.format(python=sys.executable, root=str(self.root), mode=mode)
        _script(self.root / "bin" / "claude", text)

    def state(self, agent: str, *names: str) -> Path:
        state = self.home / ".claude" / "channels" / agent
        state.mkdir(parents=True)
        for name in names:
            (state / name).write_text("x")
        return state

    def decoy(self) -> None:
        """另一个会话的 server.py：进程名对得上，但不在本会话的进程树下。"""
        path = self.root / "other" / "skills" / "teams-channel" / "server.py"
        _script(path, "import time; time.sleep(60)\n")
        self.decoys.append(subprocess.Popen([sys.executable, str(path)]))

    def run(self, *args: str, exe=(sys.executable, str(LAUNCH))):
        env = {k: v for k, v in os.environ.items() if k != "TEAMS_STATE_DIR"}
        env.update(
            PATH=f"{self.root / 'bin'}{os.pathsep}{env['PATH']}",
            HOME=str(self.home),
            CLAUDE_CODE_SESSION_ID="caller-session-id",
            CLAUDE_TEAMS_PACE="0.1",
        )
        return subprocess.run([*exe, *args], env=env, cwd=self.root,
                              capture_output=True, text=True, timeout=60)

    def keys(self) -> list[str]:
        path = self.root / "keys"
        return path.read_text().split() if path.exists() else []

    def tmux_calls(self) -> list[list[str]]:
        path = self.root / "tmux.log"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def seen_by_claude(self) -> dict:
        return json.loads((self.root / "claude.json").read_text())

    def close(self) -> None:
        pidfile = self.root / "pane.pid"
        if pidfile.exists():
            try:
                os.killpg(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        for proc in self.decoys:
            proc.kill()
            proc.wait()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def test_reconnects_when_server_appears_only_after_mcp(rig):
    rig.claude("reconnect")
    rig.state("box-agent", "token.json", "chat_id")
    rig.decoy()

    proc = rig.run("start", "--agent", "box-agent")

    assert proc.returncode == 0, proc.stderr
    assert rig.keys() == RECONNECT


def test_no_mcp_when_server_is_already_up(rig):
    rig.claude("up")
    rig.state("box-agent", "token.json", "chat_id")

    proc = rig.run("start", "--agent", "box-agent")

    assert proc.returncode == 0, proc.stderr
    assert rig.keys() == ["Enter"]
    status = rig.run("status", "--agent", "box-agent")
    assert status.returncode == 0
    assert "已连上" in status.stdout


def test_caller_session_id_does_not_reach_claude(rig):
    rig.claude("up")
    state = rig.state("box-agent", "token.json", "chat_id")

    assert rig.run("start", "--agent", "box-agent").returncode == 0

    seen = rig.seen_by_claude()
    assert "CLAUDE_CODE_SESSION_ID" not in seen["env"]
    assert set(seen["env"]) <= ALLOWED_ENV
    assert seen["state"] == str(state)
    assert seen["argv"][:2] == [
        "--dangerously-load-development-channels", "plugin:teams-channel@jgong5"
    ]


def test_exits_nonzero_when_server_never_comes_up(rig):
    rig.claude("never")
    rig.state("box-agent", "token.json", "chat_id")

    proc = rig.run("start", "--agent", "box-agent")

    assert proc.returncode != 0
    assert rig.keys() == RECONNECT


def test_missing_chat_id_exits_nonzero_and_starts_nothing(rig):
    rig.claude("up")
    rig.state("box-agent", "token.json")

    proc = rig.run("start", "--agent", "box-agent")

    assert proc.returncode != 0
    assert "/teams-channel" in proc.stderr
    assert not any(call[0] == "new-session" for call in rig.tmux_calls())


def test_agent_defaults_to_short_hostname(rig):
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()

    rig.run("status")

    assert ["has-session", "-t", f"claude-{host}-agent"] in rig.tmux_calls()


def test_installed_wrapper_runs_with_the_options_given_to_install(rig):
    rig.claude("up")
    rig.state("box-agent", "token.json", "chat_id")
    workdir = rig.root / "work"
    workdir.mkdir()

    made = rig.run("install", "--agent", "box-agent", "--workdir", str(workdir),
                   "--", "--append-system-prompt", "be brief")
    assert made.returncode == 0, made.stderr
    proc = rig.run("start", exe=(str(rig.home / ".local" / "bin" / "claude-teams"),))

    assert proc.returncode == 0, proc.stderr
    new = next(call for call in rig.tmux_calls() if call[0] == "new-session")
    assert new[new.index("-s") + 1] == "claude-box-agent"
    assert new[new.index("-c") + 1] == str(workdir)
    assert rig.seen_by_claude()["argv"][2:] == ["--append-system-prompt", "be brief"]


def test_install_leaves_a_wrapper_it_did_not_write(rig):
    wrapper = rig.home / ".local" / "bin" / "claude-teams"
    _script(wrapper, "#!/bin/sh\necho hand-written\n")

    proc = rig.run("install")

    assert proc.returncode != 0
    assert wrapper.read_text() == "#!/bin/sh\necho hand-written\n"
