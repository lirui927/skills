# Claude Code 配置指南：skills 插件、agent-team、Web UI、Teams 桥接

按顺序做即可。这份指南最初是在一台 MI355 节点的**共享 Unix 账号**上配出来的：同一个账号下还有其他人，各自有自己的 GitHub 和 Teams 登录。所以下面很多步骤都是为了做隔离。

占位符：

| 占位符 | 含义 |
|---|---|
| `<你的 AMD 邮箱>` | 你登录 Teams 用的公司邮箱（小写） |
| `<新机器>` | 新机器的 ssh 地址 |

`~` 指你的 home 目录；需要绝对路径的地方都用 `$HOME` 自动展开。

---

## 0. 先确认两件事

**新机器和原机器是否共用同一个 home？**

| 情况 | 影响 |
|---|---|
| 共用同一个 home（同一个共享文件系统） | `~/.claude`、`~/.config`、`~/lirui` 下的东西已经都在了。主要是**不要两台机器同时跑同一个东西**（见第 3、5 节） |
| 独立的 home | 按下面每一步完整做一遍；第 5 节的 Teams 文件需要从原机器拷过去 |

**基础工具**（原机器上的版本，供参考）：

```bash
claude --version        # 2.1.285
gh --version
git --version           # 2.56，agent-team 要求 >= 2.40
python3 --version       # 3.10，agent-team 要求 >= 3.10
python3 -c "import pytest"
node --version          # 24.18（nvm），Web UI 要求 >= 22
tmux -V                 # 3.2a
```

如果新机器是你独占的账号，第 1 节可以简化成直接 `gh auth login`，第 5 节的隔离要求也可以放松。

---

## 1. GitHub 账号隔离（agent-team 和 Web UI 都依赖）

共享账号上默认的 `gh` 登录（`~/.config/gh/`）可能是别人的。给 lirui927 单独一个配置目录，只在需要的项目里启用。

### 1.1 创建 fine-grained token

用 lirui927 登录 GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token：

- Repository access：**Only select repositories** → `lirui927/skills`（Web UI 也要用的话加上 `lirui927/claudecodeui`）
- Repository permissions：**Contents**、**Issues**、**Pull requests** 设为 Read and write，其余不动
- Expiration：30 或 90 天，到期后重做 1.2 即可

### 1.2 登录到单独目录

```bash
mkdir -p ~/.config/gh-lirui927 && chmod 700 ~/.config/gh-lirui927
GH_CONFIG_DIR=~/.config/gh-lirui927 gh auth login --with-token   # 粘贴 token，回车，Ctrl-D
```

验证（两个账号互不影响）：

```bash
GH_CONFIG_DIR=~/.config/gh-lirui927 gh api user --jq .login   # lirui927
gh api user --jq .login                                        # 仍是原来的默认账号
```

> **不要**运行不带 `GH_CONFIG_DIR` 的 `gh auth login`，那会改掉共享账号的默认身份。

---

## 2. 安装 skills 插件（amd-gpu、agent-team）

插件来源是 fork `lirui927/skills`（上游 `jgong5/skills`）。fork 里 marketplace 的名字仍叫 `jgong5`，所以命令里写 `@jgong5`。

```bash
claude plugin marketplace list                 # 如果已有指向上游的 jgong5，先 remove
# claude plugin marketplace remove jgong5

claude plugin marketplace add lirui927/skills
# 如果报 SSH 克隆失败（新机器没配 GitHub SSH key），改用：
# claude plugin marketplace add https://github.com/lirui927/skills.git

claude plugin install amd-gpu@jgong5
claude plugin install agent-team@jgong5
claude plugin list                             # 两个都应为 enabled
```

装完要重载：VS Code 里 "Developer: Reload Window"，或新开会话。

| 插件 | 包含 | 怎么用 |
|---|---|---|
| `amd-gpu` | `kernel-perf`、`asm-tutorial` | `kernel-perf` 自动触发，比如"这个 kernel 为什么慢、到瓶颈了吗"；也可以 `/amd-gpu:kernel-perf`。`asm-tutorial` 生成 PDF，需要 pandoc、Chrome、poppler、`pip install websockets` |
| `agent-team` | `agent-team`、`gh-prose` | `agent-team` 只能手动调用，见第 3 节 |

`kernel-perf` 的脚本在 `~/.claude/plugins/cache/jgong5/amd-gpu/<版本>/skills/kernel-perf/`，在容器里跑：

```bash
python <上面的目录>/probe_roofline.py                  # 实测算力/带宽/LLC 天花板
python <上面的目录>/probe_roofline.py --what compute   # 只测一项
```

注意：脚本里 gfx950 的 FLOP/clk 常数作者没核实过，输出超过 100% 就说明常数不对。MI455 是新架构，常数表里可能根本没有，需要用 `--flop-per-clk` 手动指定。

**更新插件**（fork 有新提交后）：

```bash
claude plugin marketplace update jgong5
claude plugin update agent-team@jgong5
claude plugin update amd-gpu@jgong5
```

改了 fork 里的插件却没生效时，先检查 `marketplace.json` 里的 `version` 有没有升。同步上游：`GH_CONFIG_DIR=~/.config/gh-lirui927 gh repo sync lirui927/skills`。

> 插件装在用户级（`~/.claude`），共享账号下别人的 Claude 也能看到。`kernel-perf` 可能在别人问 kernel 性能时自动触发。

---

## 3. agent-team

多个 agent 围绕 GitHub issue 池干活：developer 写代码开 PR → 另一个 reviewer 审到 `APPROVE @ <sha>` → orchestrator squash 合并。状态全在 GitHub 上，随时中断随时续。

### 3.1 准备仓库（一次性）

```bash
cd ~/lirui
git clone https://github.com/lirui927/skills.git skills
cd skills
git remote add upstream https://github.com/jgong5/skills.git

# 本仓库内的提交身份
git config user.name  lirui927
git config user.email 49193342+lirui927@users.noreply.github.com

# 在这个项目里开的 Claude 会话自动用 lirui927 的 gh（EOF 不加引号，$HOME 会展开）
mkdir -p .claude
cat > .claude/settings.local.json <<EOF
{
  "env": {
    "GH_CONFIG_DIR": "$HOME/.config/gh-lirui927"
  }
}
EOF
echo ".claude/settings.local.json" >> .git/info/exclude
```

如果 `~/lirui` 本身是 git 仓库，让它忽略这两个目录（**要带开头的 `/`**，否则会把任何层级叫 `skills/` 的目录都忽略掉）：

```bash
printf '/skills/\n/skills-worktrees/\n' >> ~/lirui/.git/info/exclude
```

再建 `CLAUDE.local.md`（已被 `.gitignore` 忽略，agent 会读）：

```bash
cat > ~/lirui/skills/CLAUDE.local.md <<EOF
# CLAUDE.local.md -- this checkout only (git-ignored)

- This host is one Unix account shared by several people. The default \`gh\`
  sign-in (\`~/.config/gh/\`) is someone else's and can only read this fork. Run
  every \`gh\` command, and every \`git push\` or \`git fetch\` against \`origin\`,
  with \`GH_CONFIG_DIR=$HOME/.config/gh-lirui927\` (lirui927). A session opened
  in this directory already has it set through \`.claude/settings.local.json\`;
  check \`gh api user --jq .login\` prints \`lirui927\` before writing anything
  to GitHub.
- Commands run as written: \`python3\` is 3.10 with pytest, no container.
- Never run Python with \`/tmp\` as the working directory: \`/tmp\` holds other
  people's \`.py\` files (one is \`struct.py\`) that shadow standard modules.
EOF
```

### 3.2 检查配置和测试基线

项目配置 `.claude/agent-team.md` 已经提交在 fork 里（`repo: lirui927/skills`，`never_touch: jgong5/skills`），新机器不用改。

```bash
cd ~/lirui/skills
A=$(ls -d ~/.claude/plugins/cache/jgong5/agent-team/*/skills/agent-team | tail -1)
python3 $A/overlay.py            # 退出码 0 并打印 JSON 即为正常
python3 -m pytest tests -q --ignore-glob='tests/*/test_make_pdf.py'   # 必须全绿
```

GitHub 侧一次性设置（已做过，fork 上一直有效）：fork 的 Settings → General → Features 勾选 **Issues**；`need human` 标签第一次 run 时会自动创建。

### 3.3 使用

**必须在 `~/lirui/skills` 目录下开 Claude 会话**（VS Code Open Folder，或 `cd ~/lirui/skills && claude`），否则读不到配置，gh 也不是 lirui927。

| 命令 | 作用 |
|---|---|
| `/agent-team:agent-team plan "<目标>"` | 拆成 issue（brief：文件范围、验收标准、行数估计、`Depends on #N`），**你批准后**才发布 |
| `/agent-team:agent-team run` | 循环推进：合并已批准的 → 派 reviewer → 派 developer，直到没事可做 |
| `/agent-team:agent-team status` | 只读，看每个 issue / PR 卡在哪 |

想让 run 在你离开时继续跑，放进 tmux：

```bash
tmux new -s agent-team -c ~/lirui/skills claude
# 进去后输入 /agent-team:agent-team run，然后 Ctrl-b d 退出查看
```

控制手段：

- 给 issue 或 PR 贴 **`need human`**：agent 立刻停手，只有你能摘。agent 遇到需要你裁决的事也会自己贴，并写明要你决定什么。
- 贴 **`hold landing`**：只阻止合并，开发和 review 照常。
- 合并**不需要你批准**：reviewer 给出 APPROVE 后直接 squash 进 `main`。

**一个仓库同一时间只跑一个 `run`。** 之前同时开了三个 run 会话，互相重复认领、争着合并，最后靠会话之间互发消息才收敛成一个。两台机器共用 home 时更要注意：只在一台机器上跑。

---

## 4. Web UI（claudecodeui）

在浏览器里用 Claude Code。代码是 fork `lirui927/claudecodeui`（上游 `jgong5/claudecodeui`），装在 `~/lirui/claudecodeui`。

### 4.1 安装

```bash
cd ~/lirui
git clone https://github.com/lirui927/claudecodeui.git
cd claudecodeui
nvm use 22 || nvm install 22     # .nvmrc 写的是 v22；原机器用 24.18 也能跑
npm install
cp .env.example .env
```

`.env` 里原机器用的值：

```
SERVER_PORT=3001
VITE_PORT=5173
HOST=0.0.0.0
VITE_CONTEXT_WINDOW=160000
CONTEXT_WINDOW=160000
```

新机器上如果 3001 或 5173 已被占用，换成别的端口。`HOST=0.0.0.0` 会让集群网络里的其他机器也能访问到登录页。如果你只通过 VS Code 端口转发访问，可以试着改成 `127.0.0.1` 来减少暴露（原机器没试过这样改）。

### 4.2 启动

```bash
cd ~/lirui/claudecodeui
GITHUB_TOKEN=$(GH_CONFIG_DIR=~/.config/gh-lirui927 gh auth token) \
  nohup npm run dev > /tmp/ccui-dev.log 2>&1 &
```

`GITHUB_TOKEN` 给 UI 里的 GitHub 相关功能用。显式写 `GH_CONFIG_DIR`，保证传进去的是 lirui927 的 token，不是共享账号默认的那个。

### 4.3 访问

1. VS Code 的 **Ports** 面板里转发 `5173`。
2. 浏览器打开 **`http://localhost:5173`**。注意是 **http**；用 https 会报"不安全的连接"。
3. 第一次打开会让你创建用户名和密码，之后用它登录。
4. 工具默认全部禁用，在右上角齿轮里按需打开。

### 4.4 日常操作

| 操作 | 命令 |
|---|---|
| 看日志 | `tail -f /tmp/ccui-dev.log` |
| 停止 | `pkill -f "concurrently --kill-others"` |
| 重启 | 先停止，再执行 4.2 |
| 忘记密码 | 删掉 `~/.cloudcli/auth.db`，刷新页面，重新创建账号 |
| 更新 | `git pull`，再 `npm install`，然后重启 |

> `~/.cloudcli/` 在 home 里。两台机器共用 home 时，两边的 UI 共用同一个账号库；**不要两台同时跑**，或者给第二台换端口并确认没有冲突。

---

## 5. Teams 桥接（手机控制 Claude Code）

**为什么用 Teams**：Discord、Telegram、Claude App 在大陆都要 VPN；Teams 不用。Discord 方案已放弃。

**工作方式**：Teams 里有一个只有你一个成员的群聊「Claude Agent」。tmux 里常驻一个 Claude 会话，每 2 秒查一次这个群。你在手机上直接发消息，不用任何前缀，它执行完后回复到群里，回复以 `[Claude]` 开头。消息显示的是你自己的账号，因为用的是你的登录，不是独立的机器人。

### 5.1 涉及的文件

这些文件不在任何仓库里，只在原机器上：

| 路径 | 内容 |
|---|---|
| `~/lirui/.claude/skills/teams-bridge/` | skill 本体：`SKILL.md`、`bridge_poll.py`、`scripts/`（一套 Teams 脚本的私有副本，已改成只用你自己的凭证） |
| `~/.local/bin/claude-teams` | tmux 启动脚本：`start` / `attach` / `status` / `stop` |
| `~/.config/microsoft-graph-lirui/` | 你的 Teams 登录凭证、群聊 ID（`bridge_chat_id`）、读到哪条（`bridge_lastid`），权限 700 |

**隔离**：共享账号上其他人的 Teams 和 M365 skill（例如 `~/.claude/skills/claude-home`、`~/.claude/skills/m365-*`）和它们的凭证目录 `~/.config/microsoft-graph/` 都是别人的。副本已经改成绝不读这些路径，也不读 `MS_ACCESS_TOKEN` 环境变量。启动脚本还会提示会话不要用那些 skill。

### 5.2 两台机器怎么分工

**推荐：只在一台机器上跑桥接。** 要操作另一台机器，直接在群里说"去 455 上执行 xxx"，Claude 会 ssh 过去。

原因：如果两台共用 home，凭证、`bridge_chat_id`、`bridge_lastid` 也是共用的，两台同时跑会抢同一个群的消息、重复回复。要在两台上各跑一个独立的桥接，需要先改脚本，按主机名区分群聊 ID 和状态文件。

### 5.3 在新机器上部署（独立 home 的情况）

**第 1 步，拷文件**（在原机器上执行）：

```bash
rsync -a --exclude __pycache__ ~/lirui/.claude/skills/teams-bridge/ <新机器>:~/lirui/.claude/skills/teams-bridge/
scp ~/.local/bin/claude-teams <新机器>:~/.local/bin/
```

凭证不要拷，在新机器上重新登录。启动脚本里写死了 `WORKDIR=$HOME/lirui`，新机器的目录结构不同的话要改。

**第 2 步，登录**（在新机器终端里）：

```bash
python3 ~/lirui/.claude/skills/teams-bridge/scripts/auth.py
# 打开打印的 microsoft.com/devicelogin，输入验证码，用 <你的 AMD 邮箱> 登录
# 然后运行它最后打印的 --complete 命令
python3 ~/lirui/.claude/skills/teams-bridge/scripts/auth.py --status   # Authenticated: YES
python3 ~/lirui/.claude/skills/teams-bridge/scripts/whoami.py          # 必须是 <你的 AMD 邮箱>
```

**第 3 步，建这台机器专用的群聊**。在 `~/lirui` 下运行，**不要在 `/tmp` 下运行**；先把 `MY_MAIL` 改成你的邮箱：

```bash
cd ~/lirui && python3 - <<'EOF'
MY_MAIL = '<你的 AMD 邮箱>'
import sys, os
sys.path.insert(0, os.path.expanduser('~/lirui/.claude/skills/teams-bridge/scripts'))
from graph_client import GraphClient
from skype_client import SkypeClient
me = GraphClient().get('/me', params={'$select': 'id,mail'})
assert me['mail'].lower() == MY_MAIL.lower(), me['mail']
code, tid, detail = SkypeClient().create_thread([me['id']], one_on_one=False, topic='Claude Agent 455')
print(code, tid, detail)
open(os.path.expanduser('~/.config/microsoft-graph-lirui/bridge_chat_id'), 'w').write(tid)
EOF
```

**第 4 步，启动**：

```bash
claude-teams start
claude-teams attach   # 第一次会弹"是否信任 ~/lirui"，选 Yes；然后 Ctrl-b d 退出查看
claude-teams status   # 应看到"已上线"，并且有 1 monitor 在运行
```

**第 5 步，用手机试**：Teams → 聊天 →「Claude Agent 455」→ 发"你好"，几秒内会收到 `[Claude]` 开头的回复。

### 5.4 日常操作

| 情况 | 做法 |
|---|---|
| 看它在干什么 | `claude-teams attach`（`Ctrl-b d` 退出查看） |
| 收不到回复 | `claude-teams status`；看到 `[poll-error]` 说明登录过期或网络有问题 → 重做 5.3 第 2 步 → `claude-teams stop && claude-teams start` |
| 改了 skill 或脚本 | `claude-teams stop && claude-teams start` |
| 机器重启后 | `claude-teams start` |

### 5.5 已知限制和风险

- 桥接靠会话里每 30 分钟重新挂一次的 Monitor 维持，偶尔会断，断了就重启。
- 如果 `~/.claude/settings.json` 里 `defaultMode` 是 `bypassPermissions`，**手机发来的指令会直接执行，不会弹确认**。skill 里要求删除、杀进程、推代码之前先在群里问你，但这只是对 Claude 的要求，不是硬性限制。
- 部分接口是 Teams 网页版用的内部接口，没有公开文档，微软改了就可能失效。

### 5.6 后续：完整的 Teams 通道插件

正在用 agent-team 开发正式的通道插件，见本仓库的 issue #1 到 #3。它会把"会话自己轮询"换成插件推送：不再需要每 30 分钟重新挂一次，手机上可以回复 `yes xxxxx` 批准权限，并且作为一个 bundle 用第 2 节的方式安装。完成后这一节会换成插件的安装方式。

---

## 6. 常见坑

| 坑 | 说明 |
|---|---|
| 在 `/tmp` 下跑 Python | `/tmp` 里可能有别人的 `struct.py` 之类的文件，会被当成标准库导入执行。脚本要在自己的目录下跑，或者用 `python3 -I` |
| 直接 `gh auth login` | 会改掉共享账号的默认 gh 身份，必须带 `GH_CONFIG_DIR` |
| `.git/info/exclude` 不加 `/` | `skills/` 会匹配任何层级的 `skills/` 目录 |
| 多个 `/agent-team run` | 同一仓库只跑一个，两台机器也只在一台上跑 |
| 两台机器跑同一个 Teams 群的桥接 | 会抢消息、重复回复，每台机器一个群或只跑一台 |
| 插件命令找不到 | 用带前缀的写法：`/agent-team:agent-team`、`/amd-gpu:kernel-perf`；装完记得重载窗口 |
| fork 插件改了不生效 | `marketplace.json` 里没升 `version` |
| Web UI 打不开 | 用 `http://` 而不是 `https://`；端口转发要转 5173 |
| fine-grained token 过期 | 重做 1.2 |
| tmux 里 Claude 不动 | 第一次在某个目录启动会弹信任提示，要 `attach` 进去确认 |
