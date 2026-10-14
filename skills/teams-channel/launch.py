#!/usr/bin/env python3
"""在 tmux 里常驻一个挂着 Teams 通道的 Claude Code 会话。

    launch.py start|attach|status|stop [--agent NAME] [--workdir DIR] [-- CLAUDE_ARGS...]
    launch.py install [--agent NAME] [--workdir DIR] [-- CLAUDE_ARGS...]

按机器区分：agent 名默认 ``<hostname -s>-agent``；它的状态（登录和 chat id）在
``TEAMS_STATE_DIR``，没设置时是 ``~/.claude/channels/<agent>``；会话是 tmux 会话
``claude-<agent>``。``--`` 之后的参数原样交给 ``claude``。``install`` 写
``~/.local/bin/claude-teams``，把 install 时给的选项固定进去。

为什么不直接运行启动命令：

* 干净环境。tmux 服务器把启动它的进程的环境交给每个新会话。那个进程若是另一个
  Claude 会话，新会话就继承它的 ``CLAUDE_CODE_SESSION_ID``、父进程和消息通道，
  把自己当成那个会话的子会话。所以用 ``env -i`` 只传 ``KEEP_ENV``；模型的 API
  配置来自 ``~/.claude/settings.json``，不需要传。
* 自动确认。本插件不在 Claude Code 的通道白名单里，只有组织的托管配置
  （``allowedChannelPlugins``）能加，所以每次启动都会弹开发通道警告。
* 自动重连。警告确认得晚时，本会话的服务会停在失败状态，要到 ``/mcp`` 里
  Reconnect 才起来；确认得快时一开始就是好的。``/mcp`` 里服务失败时第一项是
  Reconnect，已连上时第一项是查看工具，所以先查本会话有没有 ``server.py``，
  没有才发那串按键。
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

PLUGIN = "plugin:teams-channel@jgong5"
SERVER_MARK = "skills/teams-channel/server.py"
KEEP_ENV = (
    "HOME USER LOGNAME SHELL TERM LANG LC_ALL PATH "
    "http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY "
    "SSL_CERT_FILE REQUESTS_CA_BUNDLE TEAMS_STATE_DIR"
).split()
WARNING = "I am using this for local development"
# 回到输入框的标志：底栏出现两个 U+23F5（"⏵⏵"）或 "? for shortcuts"。
READY = ("⏵⏵", "? for shortcuts")
# 在 /mcp 里重连：进入服务列表、进入本服务页面、选第一项 Reconnect，再退出来。
# 每个按键后等待的秒数照着开发机上实测可用的节奏。
RECONNECT_KEYS = (
    ("/mcp", 1), ("Enter", 3), ("Enter", 2), ("Enter", 6),
    ("Escape", 0.5), ("Escape", 0.5), ("Escape", 0.5),
)
# 所有等待时长都乘以这个倍数：慢机器调大，测试调小。
PACE = float(os.environ.get("CLAUDE_TEAMS_PACE") or 1)
WRAPPER = Path.home() / ".local" / "bin" / "claude-teams"
WRAPPER_MARK = "# 由 teams-channel 的 launch.py install 写入"


def tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def running(session: str) -> bool:
    return tmux("has-session", "-t", session).returncode == 0


def screen(session: str) -> str:
    return tmux("capture-pane", "-p", "-t", session).stdout


def wait_until(check, seconds: float) -> bool:
    deadline = time.monotonic() + seconds * PACE
    while not check():
        if time.monotonic() >= deadline:
            return False
        time.sleep(PACE)
    return True


def shows(session: str, marks) -> bool:
    text = screen(session)
    return any(mark in text for mark in marks)


def server_owned(session: str) -> bool:
    """本会话下有没有 server.py：从它沿父进程链往上能找到本窗格的进程才算。

    同一台机器上别的会话也会跑 server.py，只看进程名会把它们当成本会话的。
    """
    panes = tmux("list-panes", "-t", session, "-F", "#{pane_pid}")
    pane_pids = {int(p) for p in panes.stdout.split()} if panes.returncode == 0 else set()
    ps = subprocess.run(["ps", "-ww", "-eo", "pid=,ppid=,args="], capture_output=True, text=True)
    parent, servers = {}, []
    for line in ps.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 2:
            continue
        pid, ppid = int(fields[0]), int(fields[1])
        parent[pid] = ppid
        if len(fields) == 3 and SERVER_MARK in fields[2]:
            servers.append(pid)
    for pid in servers:
        seen = set()
        while pid > 1 and pid not in seen:
            if pid in pane_pids:
                return True
            seen.add(pid)
            pid = parent.get(pid, 0)
    return False


def start(opts) -> int:
    session = opts.session
    if running(session):
        print(f"{session} 已在运行；查看用 attach")
        return 0
    for name in ("token.json", "chat_id"):
        path = opts.state_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            print(f"缺少 {path}：先运行 /teams-channel 登录并建群聊", file=sys.stderr)
            return 1
    env = {k: os.environ[k] for k in KEEP_ENV if os.environ.get(k)}
    env["TEAMS_STATE_DIR"] = str(opts.state_dir)
    command = [
        "env", "-i", *(f"{k}={v}" for k, v in env.items()),
        "claude", "--dangerously-load-development-channels", PLUGIN, *opts.claude_args,
    ]
    made = tmux("new-session", "-d", "-s", session, "-c", str(opts.workdir), *command)
    if made.returncode != 0:
        print(f"tmux new-session 失败：{made.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"已在 {opts.workdir} 启动 tmux 会话 {session}")

    if wait_until(lambda: shows(session, (WARNING,)), 60):
        tmux("send-keys", "-t", session, "Enter")
    wait_until(lambda: shows(session, READY), 60)
    time.sleep(5 * PACE)
    if not server_owned(session):
        for key, pause in RECONNECT_KEYS:
            tmux("send-keys", "-t", session, key)
            time.sleep(pause * PACE)
    if wait_until(lambda: server_owned(session), 10):
        print("teams-channel 服务已连上；到群聊里发条消息试试")
        return 0
    print("teams-channel 服务没有起来：attach 进去，在 /mcp 里 Reconnect", file=sys.stderr)
    return 1


def status(opts) -> int:
    if not running(opts.session):
        print(f"{opts.session} 未运行")
        return 1
    up = server_owned(opts.session)
    print(f"{opts.session} 正在运行；teams-channel 服务{'已连上' if up else '未连上'}")
    lines = [line for line in screen(opts.session).splitlines() if line.strip()]
    print("\n".join(lines[-15:]))
    return 0 if up else 1


def stop(opts) -> int:
    done = tmux("kill-session", "-t", opts.session).returncode == 0
    print(f"已停止 {opts.session}" if done else f"{opts.session} 未运行")
    return 0


def attach(opts) -> int:
    os.execvp("tmux", ["tmux", "attach", "-t", opts.session])


def install(opts) -> int:
    # 只覆盖自己写过的文件：同名的手写脚本可能是别人正在用的。
    if WRAPPER.exists() and WRAPPER_MARK not in WRAPPER.read_text(errors="replace"):
        print(f"{WRAPPER} 不是 install 写的，先把它移走", file=sys.stderr)
        return 1
    me = str(Path(__file__).resolve())
    run = [sys.executable, me]
    fixed = ["--agent", opts.agent, "--workdir", str(opts.workdir), "--", *opts.claude_args]
    usage = "用法：claude-teams [start|attach|status|stop]"
    gone = f"找不到 {me}（插件更新过？）：重新运行 launch.py install"
    WRAPPER.parent.mkdir(parents=True, exist_ok=True)
    WRAPPER.write_text(
        "#!/bin/sh\n"
        f"{WRAPPER_MARK}；要改选项就重新运行 install。\n"
        f"[ $# -le 1 ] || {{ echo {shlex.quote(usage)} >&2; exit 2; }}\n"
        f"[ -f {shlex.quote(me)} ] || {{ echo {shlex.quote(gone)} >&2; exit 1; }}\n"
        f"export TEAMS_STATE_DIR={shlex.quote(str(opts.state_dir))}\n"
        f'exec {shlex.join(run)} "${{1:-start}}" {shlex.join(fixed)}\n'
    )
    WRAPPER.chmod(0o755)
    print(f"已写入 {WRAPPER}：claude-teams start|attach|status|stop")
    return 0


COMMANDS = {"start": start, "attach": attach, "status": status, "stop": stop, "install": install}


def parse(argv: list[str]):
    claude_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, claude_args = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(
        prog="launch.py", description="在 tmux 里常驻一个挂着 Teams 通道的 Claude Code 会话。"
    )
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--agent", default=f"{socket.gethostname().split('.')[0]}-agent",
                        help="agent 名，默认 <hostname -s>-agent")
    parser.add_argument("--workdir", default=os.getcwd(), help="会话的工作目录，默认当前目录")
    opts = parser.parse_args(argv)
    opts.claude_args = claude_args
    opts.workdir = Path(opts.workdir).expanduser().resolve()
    opts.session = f"claude-{opts.agent}"
    opts.state_dir = Path(
        os.environ.get("TEAMS_STATE_DIR") or Path.home() / ".claude" / "channels" / opts.agent
    ).expanduser()
    return opts


def main(argv: list[str] | None = None) -> int:
    opts = parse(sys.argv[1:] if argv is None else argv)
    try:
        return COMMANDS[opts.command](opts)
    except FileNotFoundError as exc:
        print(f"找不到 {exc.filename}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
