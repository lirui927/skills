"""minimax-m3-agentx-trace 的测试：skill 目录放到 sys.path 上（脚本按裸模块名导入），合成 trace 见 m3_synth.py。"""

import json
import sys

import pytest

from m3_synth import SKILL_DIR, build_window, m3_config

sys.path.insert(0, str(SKILL_DIR))


@pytest.fixture
def m3_cfg(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(m3_config()))
    return p


@pytest.fixture
def window_trace(tmp_path):
    return build_window(tmp_path / "rank_0.pt.trace.json.gz")


@pytest.fixture
def build(tmp_path):
    def _b(name="t.pt.trace.json.gz", **kw):
        return build_window(tmp_path / name, **kw)
    return _b
