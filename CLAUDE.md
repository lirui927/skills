# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Claude Code **plugin marketplace**, not an application. The product is prose:
each skill is a `SKILL.md` (instructions Claude follows) plus companion
reference docs it reads on demand, plus a few Python scripts the skill shells
out to. Bundles ship from one marketplace entry -- `pr-review-kit`
(`pr-explain`, `pr-review-draft`, `pr-review-dossier`), `amd-gpu`
(`asm-tutorial`, `kernel-perf`), `code-study` (`implementation-study`,
`architecture-study`), and `agent-team` (`agent-team`, `gh-prose`).
`README.md` is the user-facing documentation for every bundle; keep it in
sync when a bundle's skills, requirements, or install story change.

## Commands

The suite is pytest, run from the repository root:

```bash
python3 -m pytest tests -q                                  # whole suite
python3 -m pytest tests/asm_tutorial/test_check_pdf.py -v   # one file
python3 -m pytest tests -k wrapped_lines -v                 # one test
```

Python 3.10 or newer: the skills' scripts use `X | None` annotations.

**The interpreter that runs those commands may not be the one on your `PATH`.**
Anything specific to this checkout -- a container or wrapper the commands go
through, the host/container path mapping, how a missing dependency gets
reinstalled -- belongs in `CLAUDE.local.md`, which `.gitignore` excludes. Read
it before running anything here; if it is absent, run the commands as written.
Nothing tracked in this repository may depend on its contents.

Each suite's `conftest.py` puts its own skill directory on `sys.path`, so tests
import the scripts by bare module name (`import check_pdf`). There is no
package install step and no test runner config.

`asm-tutorial`, `implementation-study` and `architecture-study` each ship a
`check_pdf.py` and a `make_pdf.py`, all under those same bare names, so each
of their suites' conftests also evicts a cached module whose
`__file__` is not its own -- otherwise the first of those suites to collect
(`tests/architecture_study/`, alphabetically) would hand its copies to the
others, and the failure can be silently green rather than loud. Each suite
asserts the provenance of what it imported for the same reason. Run the whole
`tests/` directory or one suite at a time; naming two suite directories on one
command line is unsupported, because both conftests then load before any test
module is imported and there is nothing cached for the eviction to catch. The
full analysis is in `tests/implementation_study/conftest.py`.

The render tests -- `test_renders_a_tiny_markdown_to_pdf`,
`test_footnotes_render_as_a_notes_chapter`,
`test_a_conforming_report_passes_every_check` -- need `pandoc`, a Chrome-family
binary, `poppler-utils`, and the `websockets` package actually present. A
missing Chrome is the usual failure, and it is the one most likely to need a
reinstall step recorded in `CLAUDE.local.md`.

Before pushing a manifest change, run both from the repository root:

```bash
claude plugin validate .
claude plugin details <bundle>     # skill count must match the manifest
```

`validate` checks schema only -- it accepts a `skills` entry pointing at a
directory that does not exist, and such a skill is dropped *silently* at
install time. `details` is the only thing that catches that typo.

## Architecture

### Bundling is manifest-only

`.claude-plugin/marketplace.json` is the sole manifest -- there is deliberately
no `plugin.json`, because a marketplace entry with `"source": "./"`, its own
`skills` array, and a `version` is a complete plugin definition (without
`version`, Claude Code reports the commit SHA instead).

Skills live flat under `skills/`; nothing on disk records bundle membership.
Re-bundle by editing the `skills` array, **never** by moving a skill directory
-- that breaks the relative links between a `SKILL.md` and its companion files.
Bundles are the installation unit, so split by what someone would want without
the rest: every installed skill's description costs context in every session.

### Skill anatomy

- `SKILL.md` frontmatter is `name` + `description`. The description is the
  trigger -- it is all Claude sees when deciding whether to fire, so it names
  concrete phrasings and states where the skill degrades.
- `disable-model-invocation: true` marks a skill as user-invoked only
  (`/pr-review-draft`, `/pr-review-dossier`, `/implementation-study`,
  `/architecture-study`, `/agent-team`); its description becomes human-facing
  rather than a trigger, and costs no context. `pr-explain`, `asm-tutorial`,
  `kernel-perf`, and `gh-prose` omit it and fire automatically.
- Companion `.md` files (`analysis.md`, `writing.md`, `render.md`,
  `diagrams.md`, ...) are read on demand, one per phase or topic, to keep the
  always-loaded `SKILL.md` small.
- Every skill resolves its own directory at runtime (`${CLAUDE_PLUGIN_ROOT}/skills/<name>`,
  falling back to where the file was read from) and defers to the project's
  own command wrapper -- never a bare `python3` -- because the toolchain may be
  containerized.

### asm-tutorial: a phased pipeline with cross-file contracts

`SKILL.md` is the spine; each phase has its own reference doc and, mostly, its
own script: Analyze (`analysis.md`, `annotate_asm.py`) -> Write (`writing.md`)
-> Render (`rendering.md`, `make_pdf.py` + `tutorial.css`) -> Verify
(`verification.md`, `check_pdf.py`). A Phase 4 finding loops back to **Phase 2**
(fix the markdown or the CSS), not Phase 3.

Several couplings span files and will break silently if edited one-sidedly:

- `writing.md`'s heading and cross-reference conventions (`## N. Title`,
  "section N") exist to match `check_pdf.py`'s `HEADING_RE` and `XREF_RE`.
- `rendering.md`'s code-sizing inequality ties `tutorial.css`'s `pre`
  font-size to `make_pdf.py`'s `MARGIN_X` and the document's longest code line.
  Changing a margin reflows every page.
- `cdna-facts.md` is the human-checkable mirror of `annotate_asm.py`'s `ARCH`
  table; `test_cdna_facts.py` asserts every `ARCH` key has a section there.
- `test_reference_docs.py` and `test_skill_md.py` assert specific sentences and
  every `<skill-dir>/...` reference in the prose. Rewording a rule can fail the
  suite -- that is intentional, the tests pin the invariants.

The skill's discipline is "derive or cite, never guess": every asserted number
goes in a fact ledger (`<doc>.notes.md`) with a source before it reaches the
prose. Missing inputs cause **omission, never estimation** -- no `.resources`
sidecar means the occupancy section is dropped, and an architecture without
sourced constants (today: gfx90a, gfx950) gets instruction commentary with no
MFMA-cost or occupancy claims. Adding an architecture means sourcing the
constant, cross-checking it against a real listing, documenting the arithmetic
in `cdna-facts.md`, *then* adding the `ARCH` entry.

### code-study: deliberate forks, one discipline

`implementation-study` studies one algorithm (`path/to/file:symbol`);
`architecture-study` studies a whole system (a directory). Both run a phased
pipeline into a diagram-first verified PDF, and both enforce "cite, derive, or
omit" through a ledger in `<stem>_*.notes.md` that `check_evidence.py` checks
mechanically.

They are **forks, not a shared library**. `make_pdf.py`, `tutorial.css`,
`check_pdf.py`, and `check_evidence.py` exist as separate copies in each skill
directory, so either can be vendored alone; the cost is that a fix to one is
not a fix to the other. Decide consciously which copies a change belongs in,
and say so in the commit message.

What differs beyond scope:

- `architecture-study` runs Survey -> Trace -> Weigh -> Write -> Render ->
  Verify, and its **fixed section spine** is matched verbatim by
  `check_pdf.py`'s `SPINE`. `implementation-study` runs Analyze ->
  Investigate -> Write -> Render -> Verify, and its section spine is enforced
  only by prose.
- `architecture-study` runs nothing -- no experiments, no `measure:` evidence
  class. A claim needing a benchmark becomes one of its closing questions.
  `implementation-study` runs approved experiments and has `experiments.md`.
- Its central rule is the **witness rule**: a paradigm, contract, constraint,
  or coupling reaches the report only with the code structure that
  instantiates it, cited to `path:line`. Section 5 exists to attract "Python
  has a GIL"-class claims that are true of the world and say nothing about the
  system; the rule is what stops them.

Its cross-file couplings, all pinned by `tests/architecture_study/`:

- `check_pdf.py`'s `SPINE`, `BACK_MATTER`, `REQUIRED_DIAGRAMS`,
  `DECISION_MARKERS`, `MIN_QUESTIONS`/`MAX_QUESTIONS`, and
  `PSEUDOCODE_MAX_STEPS` are each restated in `writing.md` or `diagrams.md`.
  `test_reference_docs.py` compares the two ends, so rewording a section title
  in prose without changing the constant fails the suite.
- The abbreviation contract (`GIL (Global Interpreter Lock)[^gil]`, expanded
  once, footnote defined once and used) lives in `check_pdf.py`'s
  `glossary_problems` and in `writing.md`. Never bracket an acronym in prose:
  `check_evidence.py` reads `[BRACKETED-UPPERCASE]` as a ledger citation.
- `tutorial.css` styles the endnote chapter on `.footnotes` with no element
  name, because pandoc emits `<section>` before 3.x and `<aside>` after; an
  element-qualified selector stops applying across that upgrade and the
  chapter renders untitled with no error anywhere.

### pr-review-kit: separate review helpers, publish last

Complementary skills: `pr-explain` is a read-only briefing that traces
blast radius beyond the diff; `pr-review-draft` delegates the review to *your
project's* `pr-review` skill and adds GitHub pending-review submission; and
`pr-review-dossier` builds a printable PDF case file with form fields, then
applies the marked-up PDF's decisions. `pr-review-draft` does not invoke
`pr-explain`; use the briefing separately when wanted.

`pr-review-draft` deliberately ships no review standards of its own -- if no
`pr-review` skill is available it stops rather than substituting a generic
review. Nothing reaches a PR author by accident: drafts stay pending,
approving a comment's *wording* is a separate question from approving its
*publication*, and a bare "looks good" never publishes.

`pdf_forms.py` is pure standard library on purpose (no pypdf, reportlab, or
weasyprint) so `pr-review-dossier` stays zero-install beyond Chrome and Python.
`asm-tutorial` is the deliberate exception: it needs pandoc and poppler, and
that is a property of that one skill, not a rule for the repo.

### agent-team: standing rules as an orchestrator

`agent-team` was extracted from one project's standing rules for autonomous
agents (ATOM Compass's `AI_DEV_RULES.md`); every project-specific value moved
into a per-project overlay, `<repo>/.claude/agent-team.md`, parsed by
`overlay.py`. `SKILL.md` is the orchestrator's spine (`plan`, `run`,
`status`); `rules.md` binds every role; `plan.md`, `develop.md`, `review.md`
and `land.md` are per-role docs handed to dispatched agents. It ships no
prompt templates on purpose: a dispatched agent gets its role doc, the issue
and the overlay, so nothing stored can drift from the brief.

`run` is stateless: each pass rereads GitHub, and `pr_state.py` derives a
PR's turn from its thread. Contracts that span files:

- A verdict's first line names the head it covers (`APPROVE @ <sha>.`).
  `pr_state.py`'s `VERDICT`/`SHA`, `review.md`'s Post step, and `gh-prose`'s
  rule 1 must agree; `lint_gh_prose.py`'s `STATE` must accept each state line
  the docs prescribe, including the `Claimed:` comment.
- `land.py`'s exit codes are what `land.md` branches on.

Both skills test **behaviour only** -- no test reads their prose -- because
`rules.md` tells target projects that prose is not a test subject, and this
repo's own suites for those skills follow the rule they ship. `land.py` is
tested against real temporary git repositories with `gh` replaced by a
recording fake; the merge-async endpoint itself has no offline test.

## Conventions

- **Every tracked file is ASCII.** Use `--` for dashes, straight quotes. Tests
  enforce this for `asm-tutorial`'s, `implementation-study`'s, and
  `architecture-study`'s docs; the rest of the repo follows it too. Two
  exceptions: `tests/gh_prose/samples/bad/`, real GitHub text kept verbatim
  because its em dashes and CJK are what the lint must catch; and
  `docs/claude-setup-guide.md`, this fork owner's setup guide, written in
  Chinese for its reader.
- Prose in skills explains *why* a mechanism exists, not just what it does --
  the "Why this pipeline and not something simpler" sections are load-bearing,
  because a future editor who does not know why Chrome is driven over CDP
  rather than `--print-to-pdf` will simplify it back into a bug.
- Development runs through the superpowers SDD workflow; `docs/superpowers/`
  holds per-task briefs, reports, and review diffs and is entirely git-ignored.
