#!/usr/bin/env python3
"""Form checks for GitHub prose. Usage: lint_gh_prose.py --kind KIND [--lang LANG] FILE. Exit 1 on hits."""

import argparse
import re
import sys

BUDGET = {
    "issue": 150,
    "brief": 250,
    "pr": 300,
    "ruling": 150,
    "review": 200,
    "comment": 80,
    "inline": 60,
}
STATE = re.compile(
    r"\b(Closes|Fixes|Part of|Blocked|Claimed|Needs owner ruling|Ready|No blocking|APPROVE|REQUEST CHANGES|Round \d+ pushed)\b"
)
PATTERNS = {
    "private ref (link it or say it)": r"\u00a7|\b[DPTW]\d+(\.\d+)?\b|[Pp]rinciple \d|\bthe brief\b|\b[Dd]oc \d+",
    "line number (cite path + symbol)": r"\.(py|sh|json|md|ya?ml|toml|cpp|h)`?:\d+|`:\d+`",
    "process narration": r"(?i)md5|docker cp|git archive|__file__|tarball",
    "round history in body": r"(?i)\bround \d",
    "non-English text": r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]",
    "em dash": "\u2014",
    "emoji": r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
}
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def count_words(text):
    """Whitespace words, with two CJK characters counted as one.

    CJK is written without spaces, so splitting alone counts a whole Chinese
    sentence as one word and no budget is ever reached; a CJK character carries
    about half an English word.
    """
    return len(CJK.sub(" ", text).split()) + len(CJK.findall(text)) // 2


def visible(text):
    """Text outside <details> and fenced code: where budgets apply."""
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.S)
    return re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)


def lint(text, kind, lang="English"):
    out = visible(text)
    lines = out.splitlines()
    hits = []
    words = count_words(out)
    if words > BUDGET[kind]:
        hits.append(f"{words} words outside <details>/code, budget {BUDGET[kind]}")
    bold = re.sub(r"`[^`]*`", "", out).count("**") // 2
    if bold > 3:
        hits.append(f"{bold} bold spans, max 3")
    first = next((l for l in lines if l.strip()), "")
    if kind not in ("issue", "brief", "inline") and not STATE.search(first):
        hits.append(f"first line states no state or ask: {first[:60]!r}")
    verdict = r"[\s#>*_]*(APPROVE|REQUEST CHANGES)\b.*\b[0-9a-f]{7,40}\b"
    if kind == "review" and not re.match(verdict, first):
        hits.append(f"first line is no verdict naming its head sha: {first[:60]!r}")
    for n, line in enumerate(lines, 1):
        line = re.sub(r"\[[^\]]*\]\([^)]*\)", "", line)  # a linked ref is fine
        for name, pat in PATTERNS.items():
            if name == "non-English text" and lang.lower() != "english":
                continue  # the project publishes in that language
            if name == "round history in body" and kind in ("comment", "review"):
                continue
            if name.startswith("line number") and kind == "inline":
                continue  # GitHub pins an inline comment to a commit
            m = re.search(pat, line)
            if m:
                hits.append(f"line {n}: {name}: {m.group(0)!r}")
    for block in re.findall(r"(?m)(?:^[ \t]*(?:[-*] |\d+\. |\|).*\n?)+", out):
        run = sum(not re.match(r"\s*\|[-| :]+\|\s*$", l) for l in block.splitlines())
        if run > 6:  # ponytail: 5 items + a table header, lists get the same slack
            hits.append(f"list/table of {run} lines, max 5 items")
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=BUDGET, required=True)
    ap.add_argument("--lang", default="English",
                    help="the language the text is published in; CJK is a hit only in English")
    ap.add_argument("file")
    a = ap.parse_args()
    hits = lint(open(a.file, encoding="utf-8").read(), a.kind, a.lang)
    print("\n".join(hits) or "clean")
    sys.exit(1 if hits else 0)
