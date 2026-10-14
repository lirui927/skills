# jgong5's Claude Code skills

Independent plugin bundles, installed separately. See each bundle's own
section for what it does and how to install it.

| Bundle | Skills | What it's for |
| --- | --- | --- |
| [`pr-review-kit`](#pr-review-kit) | `pr-explain`, `pr-review-draft`, `pr-review-dossier` | Context-first pull-request review |
| [`amd-gpu`](#amd-gpu) | `kernel-perf`, `asm-tutorial` | Find a CDNA kernel's bottleneck by ablation; turn its assembly into a verified PDF tutorial |
| [`code-study`](#code-study) | `implementation-study`, `architecture-study` | Turn code you did not write into a diagram-first, evidence-grounded, verified PDF study -- one algorithm, or a whole system's architecture |
| [`agent-team`](#agent-team) | `agent-team`, `gh-prose` | Run developer and reviewer agents over a GitHub issue pool, from brief to squash landing, with the owner ruling only on escalations |
| [`teams-channel`](#teams-channel) | `teams-channel` | Talk to a running session from Microsoft Teams: an MCP channel server relays one self-chat into the session, and a setup skill signs in and wires it up |

## pr-review-kit

Claude Code skills for reviewing pull requests **context-first**: understand the
change's blast radius before reading a line of it, keep the review private until
you have judged every comment, and publish only on an explicit say-so.

The skills are complementary, and each works alone.

| Skill | Invocation | What it does |
| --- | --- | --- |
| `pr-explain` | automatic, or ask to "explain this PR" | Turns a PR into a reviewer briefing: why it exists, what it touches beyond the diff, and two plain-ASCII diagrams (architecture and data/control flow) that render in any terminal. Read-only. |
| `pr-review-draft` | `/pr-review-draft` | Runs your installed `pr-review` skill, then submits its findings as a GitHub **pending** review that stays private until you click Submit. Publishes only when you explicitly ask. |
| `pr-review-dossier` | `/pr-review-dossier` | Builds a printable PDF case file for the drafted comments: each one with the code it points at, the precedent behind it, the failure scenario, and a Confirmed/Speculative/Refuted verdict, plus tick-boxes for Publish / Reword / Drop and a typable reword note. Hand the marked-up PDF back and it applies your decisions. |

Why the split: a diff can be read locally, but its interactions cannot, and a
review comment that looks sharp in chat often rests on nothing once you go
looking for the precedent. `pr-explain` supplies the interactions; the dossier
forces every comment to show its evidence before it reaches the author.

### Install

In Claude Code:

```
/plugin marketplace add jgong5/skills
/plugin install pr-review-kit@jgong5
```

Or from your shell:

```bash
claude plugin marketplace add jgong5/skills
claude plugin install pr-review-kit@jgong5
```

Restart Claude Code (or start a new session) for the skills to load. Confirm
with `claude plugin list`, or by typing `/pr-review-draft` and seeing it resolve.

### Requirements

- **[`gh`](https://cli.github.com/), authenticated** (`gh auth login`) -- every
  skill in this bundle reads PRs through it, and `pr-review-draft` writes through it.
- **A `pr-review` skill**, for `pr-review-draft` only. This kit deliberately does
  not ship one. `pr-review-draft` is a wrapper that adds draft submission and
  publication safety; the review standards themselves belong to your
  project, and many repositories ship their own `pr-review` under
  `.claude/skills/pr-review/`. Without one available, `pr-review-draft` stops
  rather than substituting a generic review.
- **Chrome or Edge, and Python 3**, for `pr-review-dossier` only. The PDF route
  is deliberately zero-install: a self-contained HTML file printed by headless
  Chrome, then post-processed by `pdf_forms.py`, which is pure standard library.
  No pandoc, reportlab, weasyprint, or pypdf needed.

Nothing here is tied to a particular language or project. `pr-review-dossier`'s
`example.html` is rendered from a real PyTorch PR, included only as a layout and
density reference for the generated pages.

### Safety model

These skills are built so that nothing reaches a PR author by accident.

- `pr-explain` and `pr-review-dossier` never write to GitHub at all.
- `pr-review-draft` posts only when asked, and posts a **draft** by default.
  Approving a comment's *wording* is treated as a separate question from
  approving its *publication* -- a bare "looks good" never escalates a draft
  into a published review.
- Before any write to GitHub, the exact review body and every inline comment are
  shown for explicit approval, and every posted comment is marked as
  AI-generated. If your repository has its own AI-contribution policy
  (`AI_POLICY.md`, `CONTRIBUTING.md`), the skill reads and complies with it too.

## amd-gpu

### kernel-perf

Optimises an AMD CDNA compute kernel by **ablation**: delete one path from the
kernel, time what is left, and read the delta. Each deletion yields a *floor*
-- the time with that path free -- which bounds what optimising it can ever
win, so a floor above the target retires an idea before it is built.

The discipline the skill exists to enforce is about *claiming* a ceiling. A
floor measured with an axis **pinned** -- one value shared by every
configuration measured -- is not a ceiling. In the work this came from, three
"this is structural" conclusions were each overturned by moving one pinned
axis, and the skill makes naming those axes a completion criterion.

It also carries what is expensive to rediscover: an executable LDS
bank-conflict model (`bank_model.py`, pinned by tests against profiler-measured
conflict rates on a real gfx942 GEMM), a roofline probe (`probe_roofline.py`)
that measures compute peak at the clock the part actually sustains rather than
its boost figure and finds the last-level cache that `rocminfo` does not
report, the occupancy and intensity arithmetic to run before writing a tile,
and a table of failure signatures -- spills that
read as architectural limits, tuning constants that are global when they should
be per-config, measurements corrupted by another job on the GPU.

Architectures: CDNA (gfx90a, gfx942, gfx950). The method is
architecture-independent; the constants in `skills/kernel-perf/modelling.md`
are gfx942's. No install requirements beyond a working build and a GPU;
`probe_roofline.py` additionally uses torch, triton and amdsmi, all of which
come with a ROCm PyTorch install.

### asm-tutorial

Turns an AMD CDNA GPU assembly listing (`.s`) into a verified PDF tutorial:
analyze the listing against its source, write a tutorial with a fixed spine
(what the kernel computes, tiling, register/LDS budget, occupancy, the
compiler's own resource report), render it to PDF, then verify the render --
mechanically over every page (embedded fonts, no wrapped code lines, every
cross-reference resolves) plus a sampled visual look at six representative
pages.

Architectures: the CDNA family -- gfx90a, gfx942, gfx950 -- which share a
wave64 execution model and an MFMA lineage. Only gfx942 currently has sourced
MFMA-cost and occupancy constants; gfx90a and gfx950 take the same degraded
path as any unrecognized target -- instruction commentary only, no MFMA cost
or occupancy claims -- until their own constants are sourced. Per-architecture
constants and their sources are in `skills/asm-tutorial/cdna-facts.md`.

### Install

```
/plugin marketplace add jgong5/skills
/plugin install amd-gpu@jgong5
```

Or from your shell:

```bash
claude plugin marketplace add jgong5/skills
claude plugin install amd-gpu@jgong5
```

### Requirements

- Python 3 (standard library, plus `websockets` for the PDF-render step)
- `pandoc`
- `google-chrome-stable` (or another Chromium-family binary)
- `poppler-utils` (`pdftotext`, `pdffonts`, `pdftoppm`, `pdfinfo`)

Unlike `pr-review-dossier` above, this skill's PDF route is not zero-install:
a tutorial this dense in 100-column assembly needs pandoc's markdown handling
and a real print stylesheet, which is worth the extra dependencies for this
one skill. That is a property of `asm-tutorial`, not a rule for this repo.

## code-study

Skills for reading code you did not write and turning what you learned into
a verified PDF. `implementation-study` zooms in on one algorithm;
`architecture-study` zooms out to a whole system. Both are user-invoked only,
both are language-agnostic, and both enforce the same rule: every substantive
claim is cited to a line of the code, derived from something already cited,
measured by an approved experiment, or omitted -- nothing is estimated to fill
a gap.

| Skill | Invocation | What it produces |
| --- | --- | --- |
| `implementation-study` | `/implementation-study path/to/file.py:symbol` | A study of one algorithm: what it computes, where it sits, how it departs from the canonical form, and why each choice beat its alternatives. |
| `architecture-study` | `/architecture-study path/to/system` | A review of a system's design: the paradigms its structure witnesses, its contracts and seams, one traced execution, its trade-offs, where it blocks, and what a different design would buy. |

### implementation-study

Turns one algorithm implementation into a diagram-first study document and a
verified PDF: what it computes, where it sits, why each choice was made rather
than the alternatives it beat, and what would have to be true for a different
choice to win. Required inline-SVG views form the visual backbone -- implementation
structure, execution/data/state flow, and the decision landscape -- with
additional state, layout, lifecycle, concurrency, or algorithm-stage figures
when the implementation needs them. Prose interprets those visuals instead of
reading every box and arrow aloud, and each key algorithm gets pseudocode in
the domain's own vocabulary -- a top level at one altitude, with complex
steps refined into their own named blocks rather than piled into one.

```
/implementation-study path/to/file.py:symbol
```

The argument is one entry point -- `path/to/file:symbol`, or `path/to/file`
when the file is the unit. Anything vaguer (a bare symbol name, a directory,
"the queue implementation") is not a resolved entry point, and the skill asks
rather than guessing.

Its phases run in order, each reading only its own reference doc:

| Phase | What it does |
| --- | --- |
| Analyze | Resolves the boundary, the contract, and the caller map; opens the comprehension ledger; records the integrity baseline the last phase checks against. |
| Investigate | Turns the ledger into a decision inventory -- the implementation's choices, their realistic alternatives, and the trade-off between them. Proposes an experiment only when a trade-off cannot be settled by reasoning. |
| Write | Turns the ledger and the inventories into a diagram-first study: a fixed section spine, the required inline-SVG views, supplemental visuals where relationships beat enumeration, refined pseudocode for each key algorithm, a middle shaped by the code, decision blocks, and back matter. Discovers nothing new. |
| Render | Turns the markdown and vector figures into a PDF through pandoc and headless Chrome, classifying crowded diagrams and overlong code before touching the stylesheet. |
| Verify | The gate: mechanical PDF and SVG-survival checks, a mechanical evidence check, rasterized inspection of every figure page, and a manual read-through, with every finding routed back to the phase that caused it. |

Outputs land next to the code, in the nearest ancestor `docs/` directory found
without crossing the repository boundary -- an ancestor `docs/` outside the
repository is never a candidate, no matter how close it sits -- or, with the
user's explicit approval, the entry point's own directory if no `docs/`
exists inside the repository, provided that directory is
not the repository root itself -- an entry point at the root with no `docs/`
above it makes the skill stop and ask for an output subdirectory rather than
write to the root): the study document
`<stem>_study.md`, the comprehension ledger `<stem>_study.notes.md`, any
approved experiment scripts and their captured output under
`<stem>_study_experiments/`, and the rendered `<stem>_study.pdf`. The notes
file is the authoritative, machine-checked ledger; before rendering, the skill
projects it into a generated terminal Evidence ledger in the study, so every
`[C1]`-style mark is defined inside the PDF and the PDF can be read on its
own. Those marks are hyperlinks in both directions -- click one to jump to its
evidence, click the entry's back-reference to return to the sentence.
Every one of those paths must be new -- the skill creates files and never
overwrites one.

The rule the whole pipeline exists to enforce: every substantive claim is
**cited** to a specific line of the code under study, **derived** from
something already cited, **measured** by an approved experiment, or
**omitted**. Nothing is estimated to fill a gap. That shows up in the
document as ledger IDs behind the prose, as decision blocks that must state
the alternatives and the trade-off rather than only the choice, and as an
Improvements chapter whose every entry is falsifiable -- a reader can go
check the stated condition and find it false.

### architecture-study

Turns a directory into an architecture review a peer would recognise: the
paradigm each layer's structure actually witnesses and the forcing function
that produced it, the contract at every seam and whether that seam could be
cut, one execution traced from entry to exit with its state mutations and
allocations, the trade-offs priced with the condition that makes each cost
bite, the blocking and coupling mechanisms, one or two alternative paradigms
with the condition under which each wins, and two to four questions the
report cannot answer and you can.

```
/architecture-study path/to/system
```

The argument is a directory -- the working directory when omitted. A single
file is not an architecture, and the skill says so and points at
`/implementation-study` instead.

Fixed sections answer the same questions every time, checked verbatim, so a
reader who has read one report can navigate any other. Required inline-SVG
views form the backbone -- system decomposition, contract map,
execution lifecycle, and trade-off landscape -- with blocking-path,
alternative-paradigm, state, layout, memory-lifetime, and concurrency figures
where they beat a paragraph. Every abbreviation is spelled out and footnoted
at its first use, once, and the checker enforces that too.

Its phases run in order, each reading only its own reference doc:

| Phase | What it does |
| --- | --- |
| Survey | Records the integrity baseline, declares the boundary, and builds the component and contract inventories. The boundary is exactly the files that were opened and read -- a component named but never read is reported as a boundary edge, not described. |
| Trace | Follows one concrete execution end to end, recording state mutations, ownership transfers, allocations, seam crossings, and every point where it blocks. |
| Weigh | Turns the inventories into decisions, realistic alternatives, priced trade-offs, alternative paradigms, and the open questions the report closes on. |
| Write | Turns the inventories into the fixed-section report: the boundary block, the required views, decision blocks, footnoted abbreviations, and the generated evidence ledger. Discovers nothing new. |
| Render | Turns markdown and vector figures into a PDF through pandoc and headless Chrome, classifying crowded diagrams and overlong code before touching the stylesheet. |
| Verify | The gate: mechanical PDF and evidence checks, rasterized inspection of every sample page, and the read-through sweeps, with every finding routed back to the phase that caused it. |

Outputs land beside the code under the same rules as `implementation-study`:
`<stem>_architecture.md`, `<stem>_architecture.notes.md`, the non-git
`<stem>_architecture.integrity.json`, and `<stem>_architecture.pdf`.

What keeps the report honest is the **witness rule**: every paradigm,
contract, constraint, and coupling it names carries the code structure that
instantiates it, cited to `path:line`. A pattern is not witnessed by a class
name, a docstring, or a folder called `pipeline/`. The rule bites hardest on
the constraints section, where "Python holds a Global Interpreter Lock" and
"model calls are latency-bound" are facts about the world that will write
themselves into any review that does not stop them. A mechanism with no
witness becomes one of the closing questions instead of a finding.

Unlike `implementation-study`, this skill runs nothing at all -- no
experiments, no test suite, no service. A claim that needs a benchmark before
it is credible is reported as a question, not implied to have been measured.

### Safety model

- **The repository under study is read-only.** Each skill writes only inside
  its own output paths above, and its final phase checks that mechanically: a
  clean `git status` against the baseline the first phase recorded, or a file
  snapshot comparison when the repository is not a git work tree. A phase
  that wants to reformat or "just quickly fix" something in the code under
  study has left the skill's scope. The git check covers what `git status`
  covers -- no tracked file changed, and nothing untracked appeared outside
  the output directory -- which does not extend to paths `.gitignore`
  excludes. That gap is closed by prevention rather than by a claim: an
  experiment must run with its language's cache and artifact writes
  suppressed (`PYTHONDONTWRITEBYTECODE=1` and the per-language equivalents),
  as part of the wrapper command, so those writes never happen.
- **Experiments run only with explicit approval**, and only in
  `implementation-study`; `architecture-study` executes nothing. Each one is proposed as a
  single plan line naming the claim it supports, the script, what it
  measures, and its bounds; nothing is written or run until you approve that
  exact line. A declined experiment is recorded as declined, and the claim it
  would have supported is derived from what is already known or dropped.
- **An experiment executes code from the repository under study, and this
  skill adds no sandbox** beyond not writing outside the experiment
  directory: no container, no restricted interpreter, no network isolation.
  Approve one with the same care you would give to running someone else's
  script on your machine, because that is what it is.

### Install

```
/plugin marketplace add jgong5/skills
/plugin install code-study@jgong5
```

Or from your shell:

```bash
claude plugin marketplace add jgong5/skills
claude plugin install code-study@jgong5
```

### Requirements

- Python 3 (standard library, plus `websockets` for the PDF-render step)
- `pandoc`
- `google-chrome-stable` (or another Chromium-family binary)
- `poppler-utils` (`pdftotext`, `pdffonts`, `pdftoppm`, `pdfinfo`)

Same dependency set as `asm-tutorial`, and for the same reason: a dense
technical document needs pandoc's markdown handling, a real print stylesheet,
and poppler to check what actually came out.

Both skills are **user-invoked only** -- neither fires on its own, because a
study is a deliberate, expensive act, not something to start by mentioning a
file. Both are language-agnostic: nothing in either pipeline assumes Python.
Both stop rather than studying generated, vendored, or minified code
(`vendor/`, `third_party/`, `node_modules/`, `*.min.*`, or a "do not edit"
header) -- code nobody hand-wrote has no design decisions to recover.

## agent-team

A standing workflow for autonomous multi-agent development on any GitHub
repository. You write (or approve) the briefs; developer agents build each
task in its own worktree and PR; reviewer agents review every head until
APPROVE; the orchestrating agent lands approved PRs squashed onto your
integration branch. You are asked only when an agent escalates, through a
`need human` label that stops all automation on that item until you remove
it.

| Skill | Invocation | What it does |
| --- | --- | --- |
| `agent-team` | `/agent-team plan`, `/agent-team run`, `/agent-team status` | `plan` turns a design document, a goal, or existing issues into briefed issues linked by `Depends on #N`, and publishes them after you approve. `run` drives the pool: claim, develop, review, land, until nothing moves. `status` reports PR turns, holds, and what waits on you. |
| `gh-prose` | automatic when writing GitHub text, or ask for it | Style rules and a lint for issues, PR bodies, comments and review records: state and ask first, each fact once, evidence folded. |

Why this shape: every rule in it came from a run that went wrong. The task
record lives only on GitHub, so any session can resume where another
stopped. A PR's state is read from its thread, never from a summary, so a
stale APPROVE cannot land. An approval names the sha it covers, and landing
checks that the tree that lands is the tree that was approved, because two
green PRs can merge red. A test counts only once someone has seen it fail,
because a test that cannot fail passes every gate.

### The project overlay

Everything project-specific lives in one file in your repository,
`.claude/agent-team.md`: the repository agents write to, the integration
branch, the per-task gate command, the design document agents read first,
and optional keys such as a repository agents must never touch. Its body
holds free-text project rules. The first `plan` or `run` drafts it with
you: it infers values from your remotes, docs and CI, runs the proposed gate
once on the integration branch to show it is green, and writes the file only
after you approve. The full key table is in
[`skills/agent-team/overlay.md`](skills/agent-team/overlay.md).

### Safety model

- **Automation is on by default.** Under `run`, agents create branches,
  worktrees, issues, PRs, comments and labels in the overlay's `repo`, and
  land approved PRs onto its integration branch **without asking you**.
  Point `repo` at a fork or a repository where that is acceptable.
- **`need human` is the brake.** An agent applies it the moment it
  escalates, and it stops all agent action on that issue or PR; only you
  remove it. A repository named in `never_touch` is never written to.
- **Nothing is rewritten.** PR branches only gain commits -- no force-push,
  rebase or amend -- and nothing is pushed to the integration branch except
  by a squash landing of an approved head.
- **`plan` publishes nothing until you approve** the task graph and its
  briefs.

### Install

```
/plugin marketplace add jgong5/skills
/plugin install agent-team@jgong5
```

Or from your shell:

```bash
claude plugin marketplace add jgong5/skills
claude plugin install agent-team@jgong5
```

### Requirements

- `gh`, authenticated with write access to the overlay's `repo`
- git 2.40 or newer (`merge-tree --write-tree --merge-base`)
- Python 3.10 or newer, standard library only
- Recommended: the [`ponytail`](https://github.com/DietrichGebert/ponytail)
  plugin, whose `ponytail` and `ponytail-review` skills developers and
  reviewers use when installed; and the `github/gh-stack` extension for
  stacking dependent PRs. Both are optional, and the skill says once when
  it runs without them.

`agent-team` is **user-invoked only**: an autonomous team that pushes and
lands is not something to start by mentioning a PR.

## teams-channel

Talk to a Claude Code session from Microsoft Teams. This bundle ships an **MCP
channel server** that relays one Teams chat into a running session: a message
you send in the chat reaches the session, and the session's replies and its
permission prompts come back to the same chat, so you can drive and approve work
from your phone. It is the Teams counterpart of the official Discord channel.

Unlike every other bundle here, `teams-channel` declares its server **inline in
the manifest**, so installing the bundle is all it takes to wire the server into
Claude Code. `claude plugin details teams-channel` shows one skill and one MCP
server; every other bundle still shows zero MCP servers.

| Skill | Invocation | What it does |
| --- | --- | --- |
| `teams-channel` | `/teams-channel` | Signs in to Teams with a device code, creates the one self-chat the session relays, records its id, and prints the command that launches a session with the channel attached. |

The server serves exactly **one chat per agent**, the one recorded at
`TEAMS_STATE_DIR/chat_id`. Give each agent that shares a home its own
`TEAMS_STATE_DIR` and its own chat, or two agents relay the same chat and answer
over each other. The setup skill names each chat `<short hostname>-agent` unless
you name it.

### Install

```
/plugin marketplace add jgong5/skills
/plugin install teams-channel@jgong5
```

Or from your shell:

```bash
claude plugin marketplace add jgong5/skills
claude plugin install teams-channel@jgong5
```

### Launch

Run `/teams-channel` first; it walks through sign-in and chat creation and prints
these commands filled in. `launch.py` keeps the session in tmux:

```bash
python3 ~/.claude/plugins/cache/jgong5/teams-channel/<version>/skills/teams-channel/launch.py \
  install --agent <chat name> --workdir <project dir>
claude-teams start      # then: claude-teams attach | status | stop
```

- **One agent per machine.** `--agent` defaults to `<short hostname>-agent`; it
  names the tmux session `claude-<agent>` and, unless `TEAMS_STATE_DIR` is set,
  the state directory `~/.claude/channels/<agent>`.
- **A clean environment.** A tmux server hands new sessions the environment of
  whoever started it, so a session started from inside another Claude session
  inherited that session's id and took itself for its child. `claude` runs
  under `env -i` with a short allowlist of variables.
- **No manual steps.** The channel is outside Claude Code's channel allowlist,
  so every start asks to confirm the development-channels flag; `launch.py`
  confirms it, and when this session's `server.py` has not come up it
  reconnects the server from `/mcp`, exiting non-zero if it is still down.

Arguments after `--` go to `claude`. `install` writes `~/.local/bin/claude-teams`
with its options fixed in it, refuses to overwrite a file it did not write, and
needs running again after a plugin update. Without tmux, the bare command is
`TEAMS_STATE_DIR=<state dir> claude --dangerously-load-development-channels plugin:teams-channel@jgong5`.
The flag is for local channel development only.

### Requirements

- **A Microsoft Teams account** you can sign in to with a device code. Sign-in
  uses the Teams desktop client's public app id and needs no secret.
- **Python 3.10 or newer, standard library only.** The server and its Teams
  client are pure stdlib; there is nothing to install beyond Python and Claude
  Code itself.
- **tmux**, for `launch.py`.

## Update

```
/plugin marketplace update jgong5
/plugin update <bundle>@jgong5
```

where `<bundle>` is `pr-review-kit`, `amd-gpu`, `code-study`, `agent-team`, or
`teams-channel`.
Same commands work as `claude plugin ...` from the shell. Updates apply on
restart.

Qualify the plugin with `@jgong5` on update. The bare name resolves for
`install` and `details`, but `update` reports `Plugin "<bundle>" not found`
without it.

## Uninstall

```
/plugin uninstall <bundle>@jgong5
/plugin marketplace remove jgong5
```

`marketplace remove` drops all bundles at once, since one marketplace entry
covers this whole repo -- only run it once you want none of them installed.

## Layout

```
.claude-plugin/
  marketplace.json      # the only manifest: defines every bundle in this repo
skills/
  pr-explain/            # SKILL.md + diagrams.md (ASCII diagram recipes)
  pr-review-draft/       # SKILL.md
  pr-review-dossier/     # SKILL.md + render.md, pdf_forms.py, example.html
  asm-tutorial/          # SKILL.md + analysis.md, writing.md, rendering.md,
                         # verification.md, cdna-facts.md, annotate_asm.py,
                         # make_pdf.py, check_pdf.py, tutorial.css
  implementation-study/  # SKILL.md + phase docs, experiments.md,
                         # diagrams.md, pseudocode.md, make_pdf.py,
                         # check_pdf.py, check_evidence.py, tutorial.css
  architecture-study/    # SKILL.md + phase docs, paradigms.md,
                         # diagrams.md, make_pdf.py, check_pdf.py,
                         # check_evidence.py, tutorial.css
  kernel-perf/           # SKILL.md + ablation.md, modelling.md,
                         # bank_model.py, probe_roofline.py
  agent-team/            # SKILL.md + rules.md, overlay.md, plan.md,
                         # develop.md, review.md, land.md, overlay.py,
                         # pr_state.py, land.py
  gh-prose/              # SKILL.md + lint_gh_prose.py
  teams-channel/         # SKILL.md + server.py (inline MCP channel server),
                         # teams_client.py, launch.py (tmux launcher)
tests/
  asm_tutorial/          # pytest suite for asm-tutorial's scripts
  kernel_perf/           # pytest suite for the bank model and the probe
  implementation_study/  # pytest suite for implementation-study's scripts
                         # and reference docs
  architecture_study/    # pytest suite for architecture-study's scripts
                         # and reference docs
  agent_team/            # pytest suite for agent-team's scripts
  gh_prose/              # pytest suite for the lint, over real samples
  teams_channel/         # pytest suite for the Teams client, the server,
                         # the launcher, and the real manifest's inline
                         # MCP server
```

Skills live flat under `skills/`; nothing on disk records which bundle a skill
belongs to. Grouping is done entirely by the `skills` array of a `plugins[]`
entry in `marketplace.json`, so a skill can be re-bundled by editing the
manifest -- never by moving files, which would break the relative links between
a `SKILL.md` and its companion files.

There is deliberately no `plugin.json`. A marketplace entry with
`"source": "./"` and its own `skills` list is a complete plugin definition, and
it carries `version` too -- without one, Claude Code falls back to reporting the
commit SHA as the installed version.

## Adding to this repo

**A new skill in an existing bundle:** create `skills/<name>/SKILL.md`, then add
`"./skills/<name>"` to that bundle's `skills` array.

**A new bundle:** append another entry to `plugins[]`. It shares `"source": "./"`
with the others and simply names a different subset of `skills/`. A bundle
holding a single skill is perfectly normal. Bundles are the unit of
installation, so split by what someone would want *without* the rest -- every
installed skill's description occupies context in every session, whether or not
it fires.

Bundle names prefix their skills (`<bundle>:<skill>`), so keep them short.

**Before pushing**, confirm the manifest matches reality:

```bash
claude plugin validate .
claude plugin details <bundle>    # skill count must match the manifest
```

Run both. `validate` checks the schema only -- it accepts a `skills` entry
pointing at a directory that does not exist, and such a skill is then dropped
*silently* at install time, with no error and no warning. The inventory printed
by `details` is the only thing that catches a typo there.

## License

MIT -- see [LICENSE](LICENSE).
