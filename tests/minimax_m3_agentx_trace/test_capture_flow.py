"""capture_agentx.sh run 的完整流程，不用 GPU：docker 和 curl 换成桩（只认脚本用到的子命令，其余返回 99），
KFD 换成临时目录。检查窗口的时序和记录、传给服务的环境、只动自己的容器、退出时删掉容器，以及接上的 analyze.sh
和用它的输出做的 Excel（m3_xlsx.py）。"""

import json
import os
import re
import socket
import subprocess
import sys

import pytest

import m3_common as mc
from m3_synth import PREFILL_NAME, SKILL_DIR, build_window, m3_config, write_aiperf

NAME = "m3trace_test"
TRACE_NAME = "node_4242.1790857041234.pt.trace.json.gz"

DOCKER = r"""#!/bin/bash
S=$STUB_STATE
echo "docker $*" >> "$S/calls"
host() { echo "${1/#\/out/$STUB_OUT}"; }
cmd=$1; shift
case "$cmd" in
  info) exit 0 ;;
  image)
    [ "$1" = inspect ] || exit 99
    shift
    if [ "$1" = -f ]; then
      case "$2" in
        *Labels*) echo '{"com.amd.atom.commit": "1234abcdef00000"}' ;;
        *Id*) echo "sha256:0123456789abcdef0123456789" ;;
        *Created*) echo "2026-10-30T13:33:00.000000000Z" ;;
      esac
    fi
    exit 0 ;;
  inspect)
    [ "$1" = -f ] && shift 2
    [ -f "$S/container" ] && [ "$1" = "$(cat "$S/container")" ] || exit 1
    echo 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    exit 0 ;;
  run)
    if [ "$1" = --rm ]; then
      [ "$2 $3" = "--network none" ] || exit 99
      [[ " $* " == *" --device "* ]] && exit 98
      printf '%s\n' atom_dir=/app/ATOM atom_rev=1234abc aiter=ok aiperf=ok \
        dataset=/root/.cache/huggingface/datasets/semianalysisai___cc-traces-weka-062126 \
        curl=ok rocminfo=ok detailed_annotation=ok profiler_options=ok
      exit 0
    fi
    printf '%s\n' "$@" > "$S/run_args"
    while [ $# -gt 0 ]; do [ "$1" = --name ] && echo "$2" > "$S/container"; shift; done
    exit 0 ;;
  rm)
    [ "$1" = -f ] && [ "$2" = "$(cat "$S/container" 2>/dev/null)" ] || exit 1
    rm -f "$S/container" "$S/server_alive"
    exit 0 ;;
  exec)
    envs=()
    while [ "$1" = -e ]; do envs+=("$2"); shift 2; done
    [ "$1" = "$(cat "$S/container" 2>/dev/null)" ] || { echo "no such container: $1" >&2; exit 1; }
    shift
    if [ "$1 $2" = "bash /skill/m3_server.sh" ]; then
      n=$(( $(ls "$S" | grep -c '^server_env_') + 1 ))
      printf '%s\n' "${envs[@]}" > "$S/server_env_$n"
      rm -f "$S/killed"
      log=$(host "$3"); echo "$log" > "$S/server_log"
      for e in "${envs[@]}"; do case "$e" in PROFILE_DIR=*) host "${e#PROFILE_DIR=}" > "$S/profile_dir" ;; esac; done
      echo "ARCH=gfx1250 CONC=10 KV=fp8 env: ATOM_ENABLE_DETAILED_ANNOTATION=1 ATOM_PROFILER_MORE=0" > "$log"
      if [ -n "${STUB_SERVER_DIES:-}" ]; then
        echo "RuntimeError: HIP out of memory (stub)" >> "$log"
      else
        touch "$S/server_alive"
      fi
      echo "server pid 4242 log $3"
      exit 0
    fi
    if [ "$1 $2" = "bash /skill/m3_aiperf.sh" ]; then
      n=$(( $(ls "$S" | grep -c '^aiperf_env_') + 1 ))
      printf '%s\n' "${envs[@]}" > "$S/aiperf_env_$n"
      echo "NOTICE   Phase warmup (warmup) started"
      echo "NOTICE   Phase profiling (profiling) started"
      for i in $(seq 1 600); do [ -f "$S/stopped_$n" ] || [ -f "$S/killed" ] && break; sleep 0.1; done
      [ -f "$S/stopped_$n" ] || exit 130
      mkdir -p "$(host "$3")" && cp -r "$STUB_FIXTURE/aiperf/." "$(host "$3")/"
      echo "Benchmark Duration: 70.0 sec"
      exit 0
    fi
    case "$1" in
      pgrep) [ -f "$S/server_alive" ]; exit ;;
      chown) exit 0 ;;
      bash)
        case "$3" in
          *pkill*) rm -f "$S/server_alive"; touch "$S/killed"; exit 0 ;;
          *"ps -eo args"*) [ -f "$S/server_alive" ] && echo 1 || echo 0; exit 0 ;;
        esac ;;
    esac
    echo "unhandled docker exec: $*" >&2; exit 99 ;;
esac
echo "unhandled docker $cmd $*" >&2
exit 99
"""

CURL = r"""#!/bin/bash
S=$STUB_STATE
out= w= url=
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift 2 ;;
    -w) w=$2; shift 2 ;;
    -m|-H|-d|-X) shift 2 ;;
    -*) shift ;;
    *) url=$1; shift ;;
  esac
done
echo "curl $url" >> "$S/calls"
reply() { if [ -n "$out" ]; then printf '%s' "$2" > "$out"; else printf '%s' "$2"; fi; [ -n "$w" ] && printf '%s' "$1"; return 0; }
alive() { [ -f "$S/server_alive" ]; }
case "$url" in
  */health) alive && reply 200 "" || reply 000 "" ;;
  */v1/chat/completions) alive && reply 200 '{"choices": [{"message": {"content": "17 * 23 = 391"}}]}' || exit 7 ;;
  */start_profile)
    alive || exit 7
    echo "[atom $(date +%T)] Engine Core: executing utility command: start_profile" >> "$(cat "$S/server_log")"
    reply 200 '{"status": "ok"}' ;;
  */stop_profile)
    alive || exit 7
    log=$(cat "$S/server_log"); pd=$(cat "$S/profile_dir")
    n=$(( $(ls "$S" | grep -c '^stopped_') + 1 ))
    echo "[atom $(date +%T)] [MTP Stats Interval] Average toks/fwd: 2.80, Accepted/Total Draft tokens: 1800/3000, x" >> "$log"
    echo "[atom $(date +%T)] Engine Core: executing utility command: stop_profile" >> "$log"
    for r in 0 1; do
      mkdir -p "$pd/rank_$r"
      cp "$STUB_FIXTURE/rank_$r.pt.trace.json.gz" "$pd/rank_$r/$STUB_TRACE_NAME"
      echo "[atom $(date +%T)] Rank $r: trace exported to ${pd/#$STUB_OUT//out}/rank_$r/$STUB_TRACE_NAME (1.0 MB, 0.5s)" >> "$log"
    done
    echo "[atom $(date +%T)] Engine Core: utility command 'stop_profile' finished in 0.60s" >> "$log"
    touch "$S/stopped_$n"
    reply 200 '{"status": "ok"}' ;;
  *) echo "unhandled curl $url" >&2; exit 99 ;;
esac
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    a = build_window(d / "rank_0.pt.trace.json.gz")
    build_window(d / "rank_1.pt.trace.json.gz", wait_us=5.0)
    write_aiperf(d / "aiperf", mc.Trace(a))
    return d


@pytest.fixture
def stub(tmp_path, fixture_dir):
    bin_dir, state, out, models, kfd = (tmp_path / n for n in ("bin", "state", "out", "models", "kfd"))
    for d in (bin_dir, state, kfd / "proc"):
        d.mkdir(parents=True)
    for name, text in (("docker", DOCKER), ("curl", CURL)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    for i in range(5):
        p = kfd / "topology" / "nodes" / str(i)
        p.mkdir(parents=True)
        (p / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {0 if i == 0 else 120500}\n")
    for m, cfg in (("MiniMax-M3-MXFP4", m3_config()), ("MiniMax-M3-EAGLE3-GQA", {})):
        (models / m).mkdir(parents=True)
        (models / m / "config.json").write_text(json.dumps(cfg))
    env = {k: v for k, v in os.environ.items() if k not in ("ACCEPT_RATE", "GPUS", "EXTRA_ENV", "SHAPES")}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", STUB_STATE=str(state), STUB_OUT=str(out), STUB_FIXTURE=str(fixture_dir),
               STUB_TRACE_NAME=TRACE_NAME, IMAGE="stub/minimax-m3:test", MODEL_DIR=str(models), OUT=str(out), CONC="10",
               START_OFFSET="1", WINDOW="1", DURATION="70", NAME=NAME, KFD=str(kfd), TICK="0.1", MIN_FREE_GB="0",
               SERVER_PORT=str(_free_port()), ENGINE_PORT=str(_free_port()), PY=sys.executable)

    def run(**extra):
        r = subprocess.run(["bash", str(SKILL_DIR / "capture_agentx.sh"), "run"], env={**env, **extra},
                           capture_output=True, text=True, timeout=300)
        calls = (state / "calls").read_text().splitlines() if (state / "calls").exists() else []
        return r, calls

    return run, state, out, kfd


def _only_our_container(calls):
    for c in calls:
        f = c.split()
        if f[:2] == ["docker", "exec"]:
            i = 2
            while f[i] == "-e":
                i += 2
            assert f[i] == NAME, c
        elif f[:2] == ["docker", "rm"]:
            assert f[-1] == NAME, c
        assert f[:2] not in (["docker", "kill"], ["docker", "stop"]), c


def test_run_captures_both_windows_and_releases_the_gpus(stub):
    run, state, out, _ = stub
    r, calls = run(SHAPES="1", SHAPES_OFFSET="1", SHAPES_WINDOW="1", ACCEPT_RATE="")
    log = (out / "capture.log").read_text()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]

    run_args = (state / "run_args").read_text().splitlines()
    for a in ("/dev/kfd", "/dev/dri", f"{SKILL_DIR}:/skill:ro", f"{out}:/out", "--name"):
        assert a in run_args, a
    assert run_args[run_args.index("--entrypoint") + 1:] == ["sleep", "stub/minimax-m3:test", "infinity"]

    main_env, shapes_env = ((state / f"server_env_{n}").read_text().splitlines() for n in (1, 2))
    assert {"CONC=10", "SHAPES=0", "PROFILE_DIR=/out/traces", "PROFILER_TIMEOUT=900", "ACCEPT_RATE="} <= set(main_env)
    assert {"SHAPES=1", "PROFILE_DIR=/out/traces_shapes"} <= set(shapes_env)
    assert {"DURATION=70"} <= set((state / "aiperf_env_1").read_text().splitlines())
    assert {"DURATION=122"} <= set((state / "aiperf_env_2").read_text().splitlines())

    # 每个窗口：aiperf 进入 profiling 阶段 -> 偏移之后 start_profile -> 窗口之后 stop_profile
    meta = json.loads((out / "capture_meta.json").read_text())
    assert meta["image"] == "stub/minimax-m3:test" and meta["conc"] == 10
    assert meta["image_labels"]["com.amd.atom.commit"].startswith("1234abc")
    for tag in ("main", "shapes"):
        w = meta["windows"][tag]
        assert w["start_profile_sent"] - w["phase_start_epoch"] >= 1.0
        assert w["stop_profile_sent"] - w["start_profile_returned"] >= 1.0
        assert w["stop_profile_returned"] >= w["stop_profile_sent"]
        assert json.loads((out / f"stop_profile_{tag}.json").read_text()) == {"status": "ok"}
    order = [c.split()[-1].rsplit("/", 1)[-1] for c in calls if c.startswith("curl") and "profile" in c]
    assert order == ["start_profile", "stop_profile"] * 2

    # 容器在分析之前就删了（GPU 释放），只碰过自己的容器
    assert not (state / "container").exists()
    _only_our_container(calls)
    assert f"docker rm -f {NAME}" in calls
    assert "exit rc=0; KFD processes on this host now: 0" in log

    s = (out / "breakdown" / "SUMMARY.txt").read_text()
    assert "RESULT: 41 of 41 forwards whole" in s
    assert "allreduce+rmsnorm_aiter" in s and "allreduce_aiter" in s
    assert (out / "breakdown" / "wide.csv").exists()
    assert (out / "breakdown_shapes" / "decode_console.txt").exists()
    w = json.loads((out / "breakdown" / "window_summary.json").read_text())
    assert w["toks_per_fwd"] == pytest.approx(1 + 3 * 1800 / 3000)
    assert w["stop_profile_s"] == pytest.approx(0.6)
    assert w["window_offset_s"] == pytest.approx(650, abs=1)


def test_xlsx_tab_from_the_analyzed_run(stub, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    import m3_xlsx
    run, _, out, _ = stub
    r, _ = run()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    base = ["--run", str(out), "--stack", "11-04 stack: ATOM 1234abc, aiter 5678def, FlyDSL 0.4.1"]
    assert m3_xlsx.main(base + ["--check"]) == 0
    assert not list((out / "breakdown").glob("*.xlsx"))

    master = tmp_path / "master.xlsx"
    assert m3_xlsx.main(base + ["--master", str(master)]) == 0          # 不存在就新建
    wb = openpyxl.load_workbook(master)
    tab = wb.sheetnames[0]
    assert re.fullmatch(r"MI455-M3-TP4-\d{4}", tab) and len(wb.sheetnames) == 1
    wb.create_sheet("later")
    wb.save(master)
    assert m3_xlsx.main(base + ["--master", str(master)]) == 0          # 同名 tab 原位替换
    assert openpyxl.load_workbook(master).sheetnames == [tab, "later"]

    ws = openpyxl.load_workbook(out / "breakdown" / f"M3_TP4_layer_breakdown_{tab}.xlsx")[tab]
    assert ws["A3"].value.startswith("MI455 MiniMax-M3 TP4 breakdown (11-04 stack, in-run AgentX c10 trace of ")
    notes = ws["A4"].value
    assert notes.startswith("MiniMax-M3-MXFP4 TP4 + EAGLE3 K3, rank 0\n") and "11-04 stack: ATOM 1234abc" in notes
    heads = [(c.row, c.value) for c in ws["B"] if isinstance(c.value, str) and "  DENSE  kernel sum=" in c.value]
    assert [h.split("  DENSE")[0] for _, h in heads] == [PREFILL_NAME.split(" sqsq")[0] + "]",
                                                         "decode[bs=3 tok=12 d=3 spec=3]"]
    for row, h in heads:
        sparse = ws[f"J{row}"].value
        assert " x1 = " in h and " x3 = " in sparse                  # 合成 config：1 个 dense 层、3 个 sparse 层
        for col, sum_col, title in (("B", "D", h), ("J", "L", sparse)):
            i = row + 2
            times = []
            while ws[f"{col}{i}"].value:
                times.append(ws[f"{sum_col}{i}"].value)
                i += 1
            assert ws[f"{chr(ord(col) + 1)}{i}"].value == "layer_time"
            assert ws[f"{sum_col}{i}"].value == f"=SUM({sum_col}{row + 2}:{sum_col}{i - 1})"
            assert float(title.split("kernel sum=")[1].split("us")[0]) == pytest.approx(sum(times), abs=0.1)
    qkv = next(c.row for c in ws["J"] if c.value == "attn_qkv_gemm")
    assert ws[f"N{qkv}"].value == "2517x6144x2560" and ws[f"P{qkv}"].value.startswith("=")


def test_xlsx_without_analysis_says_what_is_missing(tmp_path):
    pytest.importorskip("openpyxl")
    import m3_xlsx
    with pytest.raises(SystemExit, match="run analyze.sh on"):
        m3_xlsx.main(["--run", str(tmp_path), "--stack", "11-04 stack: ATOM 1234abc"])


def test_busy_gpus_and_no_waiting_starts_nothing(stub):
    run, state, out, kfd = stub
    (kfd / "proc" / str(os.getpid())).mkdir()
    r, calls = run(WAIT_GPUS="0")
    log = (out / "capture.log").read_text()
    assert r.returncode == 3
    assert "busy  GPUs held by:" in log and "WAIT_GPUS=0: not starting" in log
    assert not (state / "run_args").exists()
    assert not any(c.startswith(("docker exec", "docker rm", "curl")) for c in calls)


def test_a_server_that_dies_aborts_and_removes_only_our_container(stub):
    run, state, out, _ = stub
    r, calls = run(STUB_SERVER_DIES="1")
    log = (out / "capture.log").read_text()
    assert r.returncode == 1
    assert "ABORT server died: RuntimeError: HIP out of memory (stub)" in log
    assert f"cleanup: container {NAME} removed" in log
    assert not (state / "container").exists()
    assert not any("profile" in c for c in calls if c.startswith("curl"))
    assert not (out / "breakdown").exists()
    _only_our_container(calls)


def test_check_mode_opens_no_device_and_creates_no_container(stub):
    _, state, out, _ = stub
    r = subprocess.run(["bash", str(SKILL_DIR / "capture_agentx.sh"), "check"],
                       env={**os.environ, "PATH": f"{state.parent / 'bin'}:{os.environ['PATH']}", "STUB_STATE": str(state),
                            "STUB_OUT": str(out), "IMAGE": "stub/minimax-m3:test", "MODEL_DIR": str(state.parent / "models"),
                            "OUT": str(out), "KFD": str(state.parent / "kfd"), "MIN_FREE_GB": "0", "NAME": NAME,
                            "SERVER_PORT": str(_free_port()), "ENGINE_PORT": str(_free_port())},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "preflight passed" in r.stdout and "4 GPUs visible to KFD (gfx_target_version 120500 )" in r.stdout
    calls = (state / "calls").read_text().splitlines()
    assert [c for c in calls if c.startswith("docker run")] == [c for c in calls if c.startswith("docker run --rm --network none")]
    assert not (state / "run_args").exists()
