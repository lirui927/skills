---
name: gh-prose
description: >
  Use when writing a GitHub issue, PR body, comment, review record or handoff.
  Ponytail for prose: state and ask first, each fact once, evidence folded.
  Lint before posting.
---

# gh-prose

Keep every section your process requires (under `agent-team`, the task record
in its `rules.md`); this skill only decides how it is written. Write `none`
for an empty section.

`<skill-dir>` is the directory holding this `SKILL.md`, where
`lint_gh_prose.py` lives. Installed as a plugin it is
`${CLAUDE_PLUGIN_ROOT}/skills/gh-prose`; resolve the real path rather than
guessing.

## Rules

1. **First line: state, then the ask.** `Closes #N. Ready for review. No blocking issues.` /
   `Needs owner ruling: <question>.` / `APPROVE @ <sha>.` /
   `REQUEST CHANGES @ <sha>: <n> blocking.` / `Round N pushed <sha>.`
   These openers, `Claimed: branch task-<n>.` and `Depends on #N` stay
   verbatim English whatever language the rest is in: scripts and agents
   match them literally. Write the rest of the line in the publish language.
2. **Each fact once.** Do not restate the title, the diff, the issue or another comment; link it.
3. **Results, not process.** Commit and result, not how the tree was staged or verified.
4. **Current state, not history.** Rewrite the PR body in place each round;
   the round comment carries what changed.
5. **No private vocabulary or line numbers** (inline review comments excepted):
   link it or say it plainly.
6. **No rhetoric.** Do not grade the reviewer, defend a choice, or predict how
   others will get it wrong. State the fact.
7. **Evidence in `<details>`.** Outside it, one line per result:
   `branch @ 67117f7: 5286 passed, 1 failed (the blocked pair). ruff, black clean.`
   Revert-red tables, probe dumps, crosstabs and cost breakdowns go inside.
8. **Form:** at most 3 bold, 5-item lists, no em dashes or emoji.

Word budgets per kind are `BUDGET` in `lint_gh_prose.py`. Over budget means
something is restated, narrated or argued. Cut it; folding it is not a fix.

## Shapes

**Finding / bug issue**: symptom in one sentence and who found it; evidence
(path + symbol, or at most 5 lines of probe output); fix; file set; exit;
named result; effort; depends on.

**Ruling request**: the question; the conflict in 2-3 sentences; options
(a)-(c) with their cost, one line each; your recommendation in one clause;
branch and PR state.

**PR body**
```
Closes #N. <state line>
<Why, 1-3 sentences.>
## Dev record
Found / Decided / Surprised / Left undone: one-line bullets or `none`.
Decided = what the reviewer should check first.
## Named result
## Gates
<details><summary>Evidence</summary>revert-red, probes, cost vs estimate</details>
```

**Review record**: verdict line, `Checked:`, `Accepted with reservation:`,
`Watch next:`, one line each; inline findings read `<problem>. <fix>.` and
lint as `--kind inline`.

## Before posting

```
python3 <skill-dir>/lint_gh_prose.py --kind issue|brief|pr|ruling|review|comment|inline [--lang <language>] body.md
```
`--lang` is the language the text is published in; under `agent-team` it is
the overlay's `publish_language`. Without it the lint expects English and
counts any CJK as a hit. Fix every hit, then `gh ... --body-file body.md`. The lint checks form only;
rules 2, 3, 4 and 6 still need a reread as the owner.
