"""SKILL.md 和配套文件之间的约定：引用的文件都存在、pitfalls 编号连续、shell 脚本语法正确、
采集脚本不带子命令时只打印用法、extract_trace.py 与 NOTICE 记录的 md5 一致、清单里有这个 skill。"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from m3_synth import SKILL_DIR

REPO = SKILL_DIR.parents[1]
SKILL_MD = (SKILL_DIR / "SKILL.md").read_text()


def test_frontmatter_names_the_skill():
    fm = SKILL_MD.split("---")[1]
    assert re.search(r"^name: minimax-m3-agentx-trace$", fm, re.M)
    assert re.search(r"^description: .+", fm, re.M)
    assert len(SKILL_MD.splitlines()) < 500


def test_every_referenced_file_exists():
    refs = set(re.findall(r"<skill-dir>/([\w.\-]+)", SKILL_MD)) | set(re.findall(r"`([\w\-]+\.(?:py|sh))`", SKILL_MD))
    assert refs, "SKILL.md references no companion file"
    missing = [r for r in sorted(refs) if not (SKILL_DIR / r).exists()]
    assert not missing, missing


def test_pitfalls_are_numbered_contiguously_and_cited_items_exist():
    text = (SKILL_DIR / "pitfalls.md").read_text()
    nums = [int(n) for n in re.findall(r"^## (\d+)\. ", text, re.M)]
    assert nums == list(range(1, len(nums) + 1))
    for doc in (SKILL_MD, (SKILL_DIR / "reference.md").read_text()):
        for n in re.findall(r"第 (\d+) 条", doc):
            assert int(n) in nums, n


def test_shell_scripts_parse():
    for sh in SKILL_DIR.glob("*.sh"):
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        assert r.returncode == 0, (sh.name, r.stderr)


def test_capture_without_a_mode_prints_usage_and_touches_nothing():
    r = subprocess.run(["bash", str(SKILL_DIR / "capture_agentx.sh")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2
    assert "capture_agentx.sh check" in r.stdout and "capture_agentx.sh run" in r.stdout


def test_extract_trace_is_the_unmodified_upstream_copy():
    md5 = hashlib.md5((SKILL_DIR / "extract_trace.py").read_bytes()).hexdigest()
    assert md5 in (SKILL_DIR / "NOTICE").read_text()


def test_manifest_ships_the_skill():
    plugins = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())["plugins"]
    entry = [p for p in plugins if "./skills/minimax-m3-agentx-trace" in p["skills"]]
    assert len(entry) == 1
    for s in entry[0]["skills"]:
        assert (REPO / s / "SKILL.md").exists(), s
