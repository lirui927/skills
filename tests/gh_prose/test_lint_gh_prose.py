"""Every bad sample must draw a hit and every good one none. Name: <name>.<kind>.md"""

from pathlib import Path

import pytest

import lint_gh_prose

SAMPLES = sorted((Path(__file__).parent / "samples").glob("*/*.md"))


def test_imported_this_skills_copy():
    assert Path(lint_gh_prose.__file__).parent.name == "gh-prose"


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_sample(path):
    kind = path.stem.rsplit(".", 1)[1]
    hits = lint_gh_prose.lint(path.read_text(encoding="utf-8"), kind)
    assert bool(hits) == (path.parent.name == "bad"), hits


def test_a_claim_comment_states_its_state():
    assert lint_gh_prose.lint("Claimed: worktree ../r-worktrees/12, branch task-12.", "comment") == []


def test_a_review_verdict_names_the_head_it_covers():
    assert lint_gh_prose.lint("REQUEST CHANGES: 1 blocking.", "review")
    assert lint_gh_prose.lint("**APPROVE** @ c0ffee1.", "review") == []


def test_chinese_is_a_hit_only_when_the_language_is_english():
    body = "Closes #3. 准备好 review，没有阻塞问题。"
    assert any("non-English" in h for h in lint_gh_prose.lint(body, "pr"))
    assert lint_gh_prose.lint(body, "pr", lang="Chinese") == []


def test_a_chinese_verdict_still_names_its_head_in_english():
    assert lint_gh_prose.lint("APPROVE @ c0ffee1. 已检查全部 gate。", "review", lang="Chinese") == []
    assert lint_gh_prose.lint("批准 @ c0ffee1。", "review", lang="Chinese")


def test_chinese_text_counts_toward_its_budget():
    # Split on whitespace alone, this is five words and never over budget.
    body = "Round 2 pushed c0ffee1. " + "修复了阻塞问题" * 40
    hits = lint_gh_prose.lint(body, "comment", lang="Chinese")
    assert any("budget" in h for h in hits), hits
