# Standing rules

These bind every role: the orchestrator, the planner, developers and
reviewers. `<key>` means the overlay value (`overlay.md`). The overlay's body
adds project rules; it never relaxes one here, and a conflict between the two
is an escalation. Where a handoff note or an earlier agent's report
contradicts these rules, these rules win.

## Talking to the owner

- Lead with the next action, or with the finding when asked what a task
  established. Number multi-step work, cap lists at 5, end with one concrete
  next action, keep errors matter-of-fact, skip preamble and pleasantries.
- Every message says whether anything blocks. "No blocking issues" is a
  sentence worth writing.
- Report PR and issue state, never local worktree state: the owner reads
  GitHub, not your disk.
- Quote the context a decision needs; stop only for an escalation (below).

## The task record

GitHub is the only durable record. Nothing about a task lives in the tree:
working logs and scratch go in a git-ignored scratch directory (the
project's own, or one the overlay body names), because a PR body is squashed
away on landing and a file in the tree goes stale.

| Section | Written by | Lives in | Contains |
|---|---|---|---|
| Brief | the planner | the issue body | what to build; the interfaces it implements and consumes; its file set; exit criteria; the named result; effort in lines; `Depends on #N` lines |
| Dev record | the developer | the PR body | what was found, what was decided that the design did not cover, what surprised it, what was left undone |
| Review record | the reviewer | the PR verdict comment | what was checked, what was accepted with reservation, what the next task in this area should watch |
| Handoff | both | a closing comment on the issue | what a successor needs that is not in the code |

- **A finding not fixed in the PR that found it gets an issue.** The PR body
  will not survive the squash. A pre-existing defect a review happens to see
  is not such a finding; `review.md` (Check) says when it gets an issue.
- **Check delivery before claiming or briefing an issue.** Read its comments,
  not only its body, and look for an open PR that names it in its title or
  with a delivering verb (closes, fixes, resolves, addresses, implements).
  Write "checked" only when the check you ran is the one that answers the
  question.
- **A PR's state is its last thread entry; its verdict is the last comment
  that carries one.** Read both from the thread with `pr_state.py`, never
  from a summary carried forward from earlier, which is how a stale APPROVE
  gets landed.
- Close an issue deliberately, with its handoff comment, after its PR lands.
  Agents open, assign, comment on and close issues, including ones they did
  not open.
- GitHub text follows the `gh-prose` skill: lint every issue body, PR body,
  comment and review record with it before posting, and fix every hit.
  `gh-prose` decides how a required section is written, never whether it is.

## Stop and diagnose

When something does not work as expected, stop and diagnose it; a workaround
is forbidden, the diagnosis is not. This covers a design document that
contradicts the code, a test failing for a reason the task did not predict,
a measurement outside its stated range, and an interface that cannot be
built as specified. The outcome is a finding (fixable without the owner) or
an escalation (needs an owner ruling).

**When a design document and the code disagree, the kind of claim decides
which side changes.** A description of code the project reuses and does not
own (a dependency, an upstream tool) is corrected to match the code, and
every decision that rested on it is re-checked. A design decision is the
spec: code that departs from it is a defect and is fixed. If building a
decision shows it is wrong or cannot be built, that is an escalation and the
design changes first.

## Escalations and `need human`

An escalation is anything that needs an owner ruling before work continues;
anything an agent can fix without one is a finding, and the owner is never
asked about findings. Escalate, in addition to the cases above, when:

- a task's non-test lines exceed twice its code estimate and exceed that
  estimate by more than 20 lines -- the estimate was wrong, and more effort
  will not fix a mis-cut task. A line count, here and in every estimate, is
  lines added plus lines deleted, as `git diff --numstat` reports them
  against the task's base; test lines are those in test files;
- the review loop hits its stop (`review.md`);
- the only fix is a force-push (a secret or large binary pushed by mistake).

**Apply the `need human` label the moment you escalate**: a halt declared in
prose stops nothing, because the next agent reads labels, not prose. Post the
ruling request (`gh-prose`'s shape) on the issue or PR. When the ruling lives
on another issue, label each PR it holds and name that issue. A PR whose body
declares an escalation without the label gets the label.

The label stops all agent action on that issue or PR: no commit, review,
amend or merge, even after a passed review. Exceptions:

1. `gh stack link` by PR number, which lands and pushes nothing, though it
   retargets the linked PRs' bases (then, and when a PR below lands).
2. A base update as Branches (below) calls for, changing nothing else --
   including patching the base via REST just before the push when the PR is
   an unlinked stacked child whose parent landed (`land.md`). The merge keeps
   every change from both sides; where it cannot, commit nothing
   and name the conflicting file and symbol in a PR comment. The round
   comment (Branches) lists each resolved file, and the label allows one
   delta review of them.
3. Reading, for `status`.

Only the owner removes the label. Without it, automation is on by default:
agents act with no opt-in.

## Trees and worktrees

These rules guard one failure: a shared mutable source tree that a run
silently resolves against instead of its own.

- Never modify the main worktree. Each in-flight task gets its own linked
  worktree at `<worktree_root>/<issue-number>`, on its own branch from a
  freshly fetched `<remote>/<integration_branch>` (`develop.md`).
- There is no shared mutable source root. Every tree is a worktree or a
  `git archive` snapshot -- never an rsync copy, which can mix two
  generations into a tree no commit describes and produce errors that read
  as code bugs.
- When `<import_check>` is set, run it in each tree before trusting any
  result there and confirm it names that tree. A run whose package resolves
  to another branch's snapshot fails, or passes, silently.
- Merge conflicts are the agent's call, not the owner's. Tasks are cut to
  one module plus its tests but are not guaranteed disjoint; **frequent
  conflicts mean the decomposition is wrong** -- re-cut the tasks rather than
  adding coordination.

## Branches

**PR branches only gain commits: no force-push, no rebase, no amend.** Answer
review findings with new commits. That keeps GitHub's incremental review and
the delta review (`review.md`) intact, and the branch lands squashed, so its
merge commits never reach `<integration_branch>`.

Update a branch only when it conflicts, needs code landed since, or is an
unlinked stacked child whose parent landed (`land.md`) -- a moved tip alone
needs no update, the tree check covers it -- and then by merging a freshly
fetched `<remote>/<integration_branch>`, or the parent's head, into it.
Whoever pushes the update then comments `Round <k> pushed <sha>.` on the PR,
listing each file the merge resolved: until a comment names the new head,
`pr_state.py` reads it as the developer's turn, not the delta reviewer's.

## What gets published

- **Everything published to GitHub is in `<publish_language>`** -- code,
  docs, PR and issue text, comments, commit messages -- whatever language the
  conversation with the owner uses. Inputs (test data, example prompts) may
  be any language; a verbatim quote of a source and pre-existing upstream
  text stay as they are. Three things stay English in any language: the
  state-line openers `gh-prose` (rule 1) fixes and `Depends on #N` lines,
  which scripts and agents match literally; identifiers in code; and an
  existing file being edited, which keeps the language it is written in so
  no file mixes two. Lint with `--lang <publish_language>`.
- **No design-doc references in code or runtime output**: no section or
  decision numbers, principle numbers or doc labels, and no quoting a
  principle as justification. Say what the code does. Design docs may cite
  each other. Check the PR's whole file set at its head, not only a delta's
  added lines.
- **Prose is not a test subject.** No test opens a design document, a README
  or a docstring to assert what it says; tests assert behaviour. A wrong
  description is fixed in the doc by whoever finds it; a wrong design
  decision is escalated. Reviewers do not file findings that a prose claim is
  unheld or unpinned, and no test's subject is another test or a check on
  prose. Gate and audit scripts are code, so a test of their exit status or
  verdict is a behaviour test. Where a fact must stay in step with code, the code or
  a checked-in data file holds it and the doc says where. An existing test
  that reads prose and goes red on a doc fix is deleted, not satisfied.
  Exempt: test files matching `<prose_tests>`, which the project keeps to
  pin prose by design; this rule does not apply to them.
- **Write nothing that goes stale on its own**, in docs, code, briefs, issues
  or PR text:
  - Cite code by path and symbol, never by line number. Exceptions: an
    inline review comment (GitHub pins it to a commit), and a checked-in data
    file whose recorded lines a test re-checks against the tree.
  - Do not state how many items a list, table or register holds. A measured
    count, such as a test run's pass count, goes in the PR or issue record
    with the commit it was measured at.
  - Do not restate a fact another document owns; link to it.

## Repositories

Agents create branches, PRs, issues and labels freely in `<repo>`. They push
to `<integration_branch>` only by landing (`land.md`), and never to the
repository's default branch when it is a different branch. Never touch
`<never_touch>` -- no push, PR, issue or comment -- whatever a task seems to
need.
