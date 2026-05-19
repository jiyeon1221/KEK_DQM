#!/usr/bin/env python3
"""
Web Server
----------
FastAPI + WebSocket bridge between browser and AgentRunner.
"""

import asyncio
import os
import queue
import re
import json
import signal
import threading
import tempfile
from pathlib import Path

from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agents.agent_runner import AgentRunner
from tools.hv_control_tool import HVControlTool


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="autoTB Control Panel")

STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent
DQM_DIR = PROJECT_ROOT / "DQM"
DQM_OUTPUT_DIR = DQM_DIR / "output"
DQM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = PROJECT_ROOT / "dqm_dashboards.yml"
PLOT_DIR = Path(tempfile.gettempdir())

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/plots", StaticFiles(directory=str(PLOT_DIR)), name="plots")
# JSON canvases + ROOT files emitted by monit
app.mount("/dqm-output", StaticFiles(directory=str(DQM_OUTPUT_DIR)), name="dqm_output")
# JSROOT bundle (single jsroot.js file shipped with DQM)
app.mount("/jsroot", StaticFiles(directory=str(DQM_DIR)), name="jsroot")

runner = AgentRunner()

# ── BrainAgent (loaded on startup, stays resident) ───────────────────────────
@app.on_event("startup")
def _startup_brain():
    """Load BrainAgent at server start so it's ready for ad-hoc requests."""
    try:
        runner.start_brain(use_base_model=False)
        print("✅ BrainAgent loaded and ready")
    except Exception as e:
        print(f"⚠️ BrainAgent failed to load (will use fallback): {e}")

from web.whisper_prompt import get_model as _get_whisper, PROMPT as _WHISPER_PROMPT, fix_physics as _fix_physics


# ── Parsing helpers ────────────────────────────────────────────────────────────

# Updatable column display names (matches RunLogTool.UPDATABLE_COLUMNS keys)
_COL_DISPLAY = {
    "program":       "Program (프로그램)",
    "notes":         "Notes (노트)",
    "config":        "Config (설정)",
    "beam_energy":   "Beam Energy (빔 에너지)",
    "beam_type":     "Beam Type (빔 타입)",
    "trigger_setup": "Trigger Setup (트리거)",
    "hv_drc":        "HV DRC",
    "hv_aux":        "HV Aux",
}

_COL_KEYWORDS = {
    "program":       [r'프로그램', r'program'],
    "notes":         [r'노트', r'note', r'비고', r'메모'],
    "config":        [r'config', r'설정'],
    "beam_energy":   [r'빔\s*에너지', r'beam.?energy', r'에너지'],
    "beam_type":     [r'빔\s*타입', r'beam.?type', r'타입'],
    "trigger_setup": [r'트리거', r'trigger'],
    "hv_drc":        [r'hv\s*drc', r'drc'],
    "hv_aux":        [r'hv\s*aux', r'aux'],
}

_COL_ASK_MSG = (
    "어느 열에 추가할까요?\n"
    "  program / notes / config / beam_energy / beam_type\n"
    "  trigger_setup / hv_drc / hv_aux"
)

def _parse_log_column(text: str):
    """Return column key if a known column is mentioned, else None."""
    t = text.lower()
    for col, patterns in _COL_KEYWORDS.items():
        if any(re.search(p, t) for p in patterns):
            return col
    return None




def _extract_log_value(text: str, column: str):
    """
    Try to extract the value to write from the original command.
    e.g. "run 12345 프로그램에 EM 추가해줘" → "EM"
    Returns None if not extractable.
    """
    patterns = _COL_KEYWORDS.get(column, [])
    for kw in patterns:
        m = re.search(
            rf'{kw}\s*(?:에|으로|에다)?\s+(.+?)\s*(?:추가|add|입력|써|넣|update)',
            text, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
    return None


# ── Direct tool commands (no agent running) ───────────────────────────────────

def _parse_direct_command(text: str):
    """
    Parse simple direct commands when no agent is running.
    Returns a dict describing the command, or None.
    """
    t = text.strip()

    # Log update: run number + any log/column keyword
    log_kw = r'로그|log|노트|note|비고|메모|추가|기록|수정|update|program|프로그램|config|설정|에너지|energy|트리거|trigger|hv'
    m = re.search(rf'(?:run\s*)?(\d{{4,6}}).*?(?:{log_kw})', t, re.IGNORECASE)
    if m and not re.search(r'waveform|wave|파형|plot|그래프|그려|peakadc|intadc', t, re.IGNORECASE):
        run_number = int(m.group(1))
        column = _parse_log_column(t)
        value = _extract_log_value(t, column) if column else None
        return {"tool": "log_update", "run_number": run_number, "column": column, "value": value}


    # DAQ run: "100개 돌려줘" / "run 100"
    m = re.search(r'(?:run\s+)?(\d+)\s*(?:개|events?)', t, re.IGNORECASE)
    if m:
        return {"tool": "daq_run", "events": int(m.group(1))}

    return None


def _run_direct_tool(cmd: dict, output_queue: queue.Queue, stop_event: threading.Event):
    """Execute a direct tool call in a background thread."""
    from agents.io_handler import WebSocketIO
    import queue as q

    dummy_input = q.Queue()
    io = WebSocketIO(dummy_input, output_queue, stop_event)

    try:
        if cmd["tool"] == "daq_run":
            from tools.daq_tool import DAQRunTool
            io.send_status("DAQ 실행 중...")
            DAQRunTool().execute({"events": cmd["events"]}, line_callback=io.send_tool_output)
            io.send_status("대기 중")


    except Exception as e:
        output_queue.put({"type": "error", "content": str(e)})
        output_queue.put({"type": "status", "content": "오류 발생"})


def _update_run_log(run_number: int, column: str, value: str,
                    output_queue: queue.Queue, stop_event: threading.Event):
    """Update a single column of a run log row in Google Sheets."""
    from agents.io_handler import WebSocketIO
    import queue as q

    dummy_input = q.Queue()
    io = WebSocketIO(dummy_input, output_queue, stop_event)

    io.send_status("로그 업데이트 중...")
    try:
        from tools.run_log_tool import RunLogTool
        result = RunLogTool().execute({"command": "update", "run_num": run_number, column: value})
        io.send_tool_output(result)
        label = _COL_DISPLAY.get(column, column)
        io.send_ai_message(f"Run {run_number}  {label} 열이 업데이트되었습니다.")
    except Exception as e:
        output_queue.put({"type": "error", "content": str(e)})
    io.send_status("대기 중")


_HELP_MSG = (
    "실행 중인 에이전트가 없습니다.\n\n"
    "직접 실행 가능한 명령:\n"
    "  • 100개 돌려줘  →  DAQ 100 events\n"
    "  • run 12345 프로그램에 EM 추가해줘  →  Google Sheets 열 업데이트\n"
    "  • run 12345 waveform 그려줘  →  Waveform 플롯\n"
    "  • run 12345 그려줘  →  플롯 종류 선택 후 그리기\n\n"
    "또는 상단 버튼으로 에이전트를 선택하세요."
)




@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Receive browser audio blob, run faster-whisper, return transcript."""
    data = await audio.read()
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(
            tmp_path, language="ko", beam_size=5, initial_prompt=_WHISPER_PROMPT
        )
        text = _fix_physics("".join(s.text for s in segments).strip())
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def api_status():
    return {"running": runner.is_running, "brain_ready": runner.brain_ready}


@app.get("/api/dqm/manifest")
async def api_dqm_manifest():
    """Return the per-agent DQM dashboard manifest as JSON."""
    import yaml
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with open(MANIFEST_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/dqm/canvases/{run_number}")
async def api_dqm_canvases(run_number: int):
    """List all per-canvas JSON files emitted for a run.
    Used by the freeform viewer's left-pane picker."""
    import re as _re
    pattern = _re.compile(r'^Run(\d+)_(.+?)_(.+?)_((?:AuxCut_)?)(.+)$')
    files = sorted(DQM_OUTPUT_DIR.glob(f"Run{run_number}_*.json"))
    items = []
    for p in files:
        m = pattern.match(p.stem)
        if m:
            type_ = m.group(2)
            method = m.group(3)
            canvas = m.group(5)
        else:
            type_ = ""
            method = ""
            canvas = p.stem
        items.append({
            "filename": p.name,
            "canvas": canvas,
            "type": type_,
            "method": method,
            "mtime": int(p.stat().st_mtime * 1000),
        })
    return items


@app.get("/api/dqm/runs")
async def api_dqm_runs():
    """List all runs available in the DQM output directory, grouped by run number."""
    import re as _re
    pattern = _re.compile(r'^Run(\d+)_(.+?)_(.+?)_((?:AuxCut_)?)(.+)\.json$')
    runs: dict[int, list] = {}
    for p in sorted(DQM_OUTPUT_DIR.glob("Run*_*.json")):
        m = pattern.match(p.name)
        if not m:
            continue
        run_num = int(m.group(1))
        type_ = m.group(2)
        method = m.group(3)
        auxcut = bool(m.group(4))
        canvas = m.group(5)
        runs.setdefault(run_num, []).append({
            "filename": p.name,
            "canvas": canvas,
            "type": type_,
            "method": method,
            "auxcut": auxcut,
            "mtime": int(p.stat().st_mtime * 1000),
        })
    result = []
    for run_num in sorted(runs.keys(), reverse=True):
        canvases = runs[run_num]
        methods = sorted(set(c["method"] for c in canvases))
        has_auxcut = any(c["auxcut"] for c in canvases)
        result.append({
            "run_number": run_num,
            "methods": methods,
            "auxcut": has_auxcut,
            "count": len(canvases),
            "canvases": canvases,
        })
    return result


@app.get("/dqm/freeform")
async def dqm_freeform():
    """Standalone JSROOT viewer: list all canvases for a run, click to draw.
    Open in a separate window for the dual-monitor workflow."""
    return FileResponse(str(STATIC_DIR / "dqm_freeform.html"))


@app.get("/hv/check")
async def hv_check_page():
    """Standalone HV status viewer page."""
    return FileResponse(str(STATIC_DIR / "hv_check.html"))


@app.get("/api/hv/status-all")
async def api_hv_status_all(expert: bool = False):
    """Fetch HV status for all channels using HVControlTool.

    expert=True: try extended fields (ramp up/down/max) as well.
    """
    try:
        tool = HVControlTool()
        if not expert:
            result = tool.execute({"command": "status", "channels": "all"})
            return {"ok": True, "output": result, "expert": False}

        # Expert mode: single-shot command so all fields share same timestamp.
        if not tool._ensure_connection():
            return JSONResponse({"ok": False, "error": "HV SSH connection failed"}, status_code=500)

        cmd = "./HVWrappdemo --ch all --Status --VMon --IMon --V0Set --I0Set --RUp --RDWn --SVMax"
        stdout, stderr = tool._run_remote_command(cmd)
        if not stdout or not stdout.strip():
            return JSONResponse(
                {"ok": False, "error": (stderr.strip() if stderr else "No output"), "command": cmd},
                status_code=500,
            )

        lines = [
            "📊 HV Status Query (Expert)",
            "📋 Request: Channels all",
            f"💻 Command: {cmd}",
            "",
            "📄 Output:",
            *stdout.strip().split('\n'),
        ]
        if stderr and stderr.strip():
            lines.extend(["", "⚠️ Stderr:", *stderr.strip().split('\n')])
        return {"ok": True, "output": "\n".join(lines), "expert": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/hv/expert-metrics")
async def api_hv_expert_metrics():
    """Return only expert metrics (RampUp/RampDown/Max) for all channels.

    This is intentionally separate from /api/hv/status-all so the frontend can
    refresh ramp/max less frequently than the main status.
    """
    try:
        tool = HVControlTool()

        if not tool._ensure_connection():
            return JSONResponse({"ok": False, "error": "HV SSH connection failed"}, status_code=500)

        # CAEN wrapper(MainWrapp.c) 기준 정확 파라미터명:
        #  - Ramp up:   RUp
        #  - Ramp down: RDWn
        #  - Max:       SVMax
        rup_cmd = "./HVWrappdemo --ch all --RUp"
        rdown_cmd = "./HVWrappdemo --ch all --RDWn"
        vmax_cmd = "./HVWrappdemo --ch all --SVMax"

        def _run(cmd: str):
            stdout, stderr = tool._run_remote_command(cmd)
            return {
                "ok": bool(stdout and stdout.strip()),
                "command": cmd,
                "output": stdout.strip() if stdout else "",
                "stderr": stderr.strip() if stderr else "",
            }

        return {
            "ok": True,
            "expert_outputs": {
                "rup": _run(rup_cmd),
                "rdown": _run(rdown_cmd),
                "vmax": _run(vmax_cmd),
            },
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class MonitRequest(BaseModel):
    run_number: int
    type: str = "full"
    method: str = "IntADC"
    modules: List[str] = []
    max_event: Optional[int] = None
    flags: List[str] = []
    # AUXcut mode chosen in the freeform UI dropdown:
    #   "none"   → no AUX cut (no --AUXcut)
    #   "WC"     → WC-only beam-spot cut    (--AUXcut --AUXCutMode WC)
    #   "WCHodo" → WC + hodoscope correlation cut (--AUXcut --AUXCutMode WCHodo)
    # The flags list also carries "AUXcut" when mode != "none" so the rest
    # of the pipeline (filename suffixes, run-history badges) keeps working.
    aux_cut_mode: Optional[str] = None

# ── Freeform LIVE process tracker ─────────────────────────────────────────────
import subprocess as _subprocess
from collections import deque as _deque
_freeform_live_proc: Optional[_subprocess.Popen] = None
_freeform_live_run: Optional[int] = None
_freeform_live_lock = threading.Lock()
_freeform_live_log: _deque = _deque(maxlen=500)
_freeform_live_log_lock = threading.Lock()

# Tracker for the (blocking) non-LIVE monit run, so the browser can issue
# /api/dqm/kill-blocking to abort it (SIGINT to the process group, like
# Ctrl+C in a terminal). Kept separate from _freeform_live_proc so that
# /api/dqm/live-status keeps reporting alive=False during a non-LIVE run.
_freeform_blocking_proc: Optional[_subprocess.Popen] = None
_freeform_blocking_run: Optional[int] = None
_freeform_blocking_lock = threading.Lock()


import re as _re
_ANSI_ESC = _re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colours, cursor moves, etc.)."""
    return _ANSI_ESC.sub('', text)


def _read_live_stdout(proc: "_subprocess.Popen") -> None:
    """Read monit stdout into the log deque.

    Always appends — never replaces — so the polling client's `since` offset
    advances with every new line and no update is silently dropped.
    ANSI escape codes and leading carriage-returns are stripped; the
    browser JS handles the "overwrite previous progress line" visual.
    """
    try:
        for raw in proc.stdout:
            line = _strip_ansi(raw.rstrip("\n")).lstrip("\r")
            if line:
                with _freeform_live_log_lock:
                    _freeform_live_log.append(line)
    except Exception:
        pass


@app.post("/api/dqm/run-monit")
async def api_run_monit(req: MonitRequest):
    """Execute monit with custom parameters and return generated canvases."""
    global _freeform_live_proc, _freeform_live_run

    monit_bin = str(DQM_DIR / "monit")
    config_path = str(PROJECT_ROOT / "config_general.yml")

    if not Path(monit_bin).exists():
        return JSONResponse({"error": f"monit not found: {monit_bin}"}, status_code=500)

    cmd = [
        monit_bin,
        "--RunNumber", str(req.run_number),
        "--Config", config_path,
        "--type", req.type,
        "--method", req.method,
    ]
    if req.modules:
        cmd.extend(["--module"] + req.modules)
    if req.max_event and req.max_event > 0:
        cmd.extend(["--MaxEvent", str(req.max_event)])
    for flag in req.flags:
        if flag in ("LIVE", "AUXcut", "AUX"):
            cmd.append(f"--{flag}")
    # Forward the AUXcut mode when an actual cut is requested. The C++ side
    # currently parses this flag as a no-op (until the position-correlation
    # cut is wired into TBaux::IsPassing), so all modes still produce the
    # same cut for now; the plumbing exists so the next step can simply
    # consume the mode without further server-side changes.
    if req.aux_cut_mode and req.aux_cut_mode != "none":
        cmd.extend(["--AUXCutMode", req.aux_cut_mode])

    generated_cmd = " ".join(cmd)

    from tools.dqm_live_worker import _build_monit_env
    monit_env = _build_monit_env()

    # LIVE mode: spawn without blocking, return immediately so the browser can
    # show the Kill Live button.  The process runs until the user calls
    # /api/dqm/kill-live (which touches the sentinel) or it exits on its own.
    if "LIVE" in req.flags:
        with _freeform_live_lock:
            # Stop any previously running freeform live first
            if _freeform_live_proc is not None and _freeform_live_proc.poll() is None:
                _sentinel = DQM_OUTPUT_DIR / f"Run{_freeform_live_run}_END"
                try:
                    _sentinel.touch()
                except OSError:
                    pass
                try:
                    _freeform_live_proc.wait(timeout=10)
                except _subprocess.TimeoutExpired:
                    _freeform_live_proc.kill()
                    _freeform_live_proc.wait()
                try:
                    _sentinel.unlink(missing_ok=True)
                except OSError:
                    pass

            # Clear stale sentinel and log for new run
            sentinel = DQM_OUTPUT_DIR / f"Run{req.run_number}_END"
            sentinel.unlink(missing_ok=True)
            with _freeform_live_log_lock:
                _freeform_live_log.clear()

            try:
                proc = _subprocess.Popen(
                    cmd,
                    cwd=str(DQM_DIR),
                    stdout=_subprocess.PIPE,
                    stderr=_subprocess.STDOUT,
                    env=monit_env,
                    preexec_fn=os.setpgrp,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as e:
                return JSONResponse({"error": str(e), "command": generated_cmd}, status_code=500)

            _freeform_live_proc = proc
            _freeform_live_run = req.run_number

            threading.Thread(
                target=_read_live_stdout, args=(proc,),
                daemon=True, name="FreeformLiveLog",
            ).start()

        return {
            "command": generated_cmd,
            "live": True,
            "run_number": req.run_number,
            "pid": proc.pid,
            "canvases": [],
        }

    # Non-LIVE: blocking run, but stream stdout+stderr line-by-line into the
    # shared live-log deque so the browser's progress pane updates while the
    # request is still in flight. The HTTP response stays open until monit
    # exits; we then return canvases as before.
    #
    # We deliberately do NOT touch _freeform_live_proc here: that global
    # tracks LIVE-mode runs, and /api/dqm/live-status must keep reporting
    # alive=False so the page-reload restorer doesn't mistake a non-LIVE
    # run for a LIVE one. Instead we use _freeform_blocking_proc, which the
    # /api/dqm/kill-blocking endpoint targets when the user clicks STOP.
    global _freeform_blocking_proc, _freeform_blocking_run

    with _freeform_live_log_lock:
        _freeform_live_log.clear()

    try:
        proc = _subprocess.Popen(
            cmd,
            cwd=str(DQM_DIR),
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            env=monit_env,
            preexec_fn=os.setpgrp,  # own process group → killpg works for STOP
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e), "command": generated_cmd}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e), "command": generated_cmd}, status_code=500)

    with _freeform_blocking_lock:
        _freeform_blocking_proc = proc
        _freeform_blocking_run = req.run_number

    reader = threading.Thread(
        target=_read_live_stdout, args=(proc,),
        daemon=True, name="FreeformBlockingLog",
    )
    reader.start()

    loop = asyncio.get_event_loop()
    try:
        exit_code = await loop.run_in_executor(None, lambda: proc.wait(timeout=300))
    except _subprocess.TimeoutExpired:
        proc.terminate()
        try:
            await loop.run_in_executor(None, lambda: proc.wait(timeout=5))
        except _subprocess.TimeoutExpired:
            proc.kill()
            await loop.run_in_executor(None, proc.wait)
        reader.join(timeout=2)
        with _freeform_blocking_lock:
            if _freeform_blocking_proc is proc:
                _freeform_blocking_proc = None
                _freeform_blocking_run = None
        return JSONResponse(
            {"error": "monit timed out (5 min)", "command": generated_cmd},
            status_code=504,
        )
    except Exception as e:
        with _freeform_blocking_lock:
            if _freeform_blocking_proc is proc:
                _freeform_blocking_proc = None
                _freeform_blocking_run = None
        return JSONResponse({"error": str(e), "command": generated_cmd}, status_code=500)

    reader.join(timeout=2)

    with _freeform_blocking_lock:
        if _freeform_blocking_proc is proc:
            _freeform_blocking_proc = None
            _freeform_blocking_run = None

    # Snapshot the tail of the streamed log for the response body. The
    # full stream is already in the browser via /api/dqm/live-log, this
    # is just so callers that don't poll the log still get something useful.
    with _freeform_live_log_lock:
        _tail = list(_freeform_live_log)[-50:]
    output = "\n".join(_tail)

    prefix = f"Run{req.run_number}_{req.type}_{req.method}"
    if "AUXcut" in req.flags:
        prefix += "_AuxCut"
    files = sorted(DQM_OUTPUT_DIR.glob(f"{prefix}_*.json"))
    canvases = []
    pfx = f"{prefix}_"
    for p in files:
        name = p.name
        if not name.endswith(".json"):
            continue
        canvas = name[len(pfx):-len(".json")]
        canvases.append({
            "filename": name,
            "canvas": canvas,
            "type": req.type,
            "method": req.method,
        })

    # When --AUX is set, TBaux dumps several auxiliary canvases as JSON
    # under the prefix Run<N>_AUX_<method>[_AuxCut]_<canvas>.json:
    #   method=WC         → wire-chamber position (fCanvas_WC)
    #   method=Hodoscope  → 16x16 IntADC hit map (fCanvas_HodoIntADC)
    #   method=Hodoscope  → 16x16 PeakADC hit map (fCanvas_HodoPeakADC)
    # We collect them all so they appear as separate entries in the run
    # browser, grouped by method.
    if "AUX" in req.flags:
        auxcut_set = "AUXcut" in req.flags
        # group(1) = method (no underscores), group(2) = remainder after the
        # mandatory underscore. We then check for the optional AuxCut_ infix.
        aux_re = re.compile(
            rf"^Run{req.run_number}_AUX_([^_]+)_(.+)\.json$"
        )
        for p in sorted(DQM_OUTPUT_DIR.glob(f"Run{req.run_number}_AUX_*.json")):
            m = aux_re.match(p.name)
            if not m:
                continue
            method = m.group(1)
            rest = m.group(2)
            has_auxcut = rest.startswith("AuxCut_")
            if has_auxcut != auxcut_set:
                # Skip files that don't match the current --AUXcut state
                # (the directory may still contain leftovers from a previous
                # run of the same RunNumber with the opposite AUXcut flag).
                continue
            canvas = rest[len("AuxCut_"):] if has_auxcut else rest
            canvases.append({
                "filename": p.name,
                "canvas": canvas,
                "type": "AUX",
                "method": method,
            })

    return {
        "command": generated_cmd,
        "exit_code": proc.returncode,
        "output": output[-500:] if len(output) > 500 else output,
        "canvases": canvases,
    }


@app.post("/api/dqm/kill-live")
async def api_kill_live():
    """Stop the freeform LIVE monit process by touching the sentinel file."""
    global _freeform_live_proc, _freeform_live_run

    with _freeform_live_lock:
        proc = _freeform_live_proc
        run_number = _freeform_live_run

        if proc is None or proc.poll() is not None:
            _freeform_live_proc = None
            _freeform_live_run = None
            return {"ok": True, "msg": "no live process running"}

        sentinel = DQM_OUTPUT_DIR / f"Run{run_number}_END"
        try:
            sentinel.touch()
        except OSError:
            pass

        def _wait_and_cleanup():
            try:
                proc.wait(timeout=15)
            except _subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except _subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            try:
                sentinel.unlink(missing_ok=True)
            except OSError:
                pass

        await asyncio.get_event_loop().run_in_executor(None, _wait_and_cleanup)

        _freeform_live_proc = None
        _freeform_live_run = None

    return {"ok": True, "run_number": run_number}


@app.post("/api/dqm/kill-blocking")
async def api_kill_blocking():
    """Abort the in-flight non-LIVE monit run with SIGINT (Ctrl+C equivalent).

    Sends SIGINT to the whole process group (the child was spawned with
    setpgrp so it has its own group). If the process doesn't die within
    a few seconds, escalate to SIGTERM and finally SIGKILL.

    The /api/dqm/run-monit handler is still awaiting proc.wait() in an
    executor thread; killing the process unblocks that wait, the handler
    cleans up _freeform_blocking_proc itself, and the original POST
    request returns to the browser with the (non-zero) exit code.
    """
    with _freeform_blocking_lock:
        proc = _freeform_blocking_proc
        run_number = _freeform_blocking_run

    if proc is None or proc.poll() is not None:
        return {"ok": True, "msg": "no blocking process running"}

    def _signal_chain():
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=5)
            return
        except _subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            return
        except (ProcessLookupError, _subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass

    await asyncio.get_event_loop().run_in_executor(None, _signal_chain)

    return {"ok": True, "run_number": run_number}


def _enumerate_monit_processes() -> list[dict]:
    """Find every running ``./monit`` process (excluding this web server).

    Uses psutil instead of shell-parsing ``ps aux`` so we look at the actual
    argv vector rather than a substring of the textual command line — that
    keeps ``./monitor``, ``demonit_helper``, etc. from being mistaken for
    monit. A process counts as ./monit iff:
      * argv[0] is ``./monit`` exactly, OR
      * argv[0]'s basename is ``monit`` AND its path is a relative ``./monit``
        form (sometimes ps captures the resolved absolute path; in that case
        we additionally accept the case where the executable's name is
        ``monit`` to handle background runs launched via the project's
        Makefile/shell wrappers).
    Always skips our own pid so the kill endpoint can never SIGKILL the web
    server itself.
    """
    try:
        import psutil
    except ImportError:
        return []

    own_pid = os.getpid()
    procs: list[dict] = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "username"]):
        try:
            info = p.info
            pid = info.get("pid")
            if pid is None or pid == own_pid:
                continue
            cmdline = info.get("cmdline") or []
            if not cmdline:
                continue
            argv0 = cmdline[0]
            name = info.get("name") or ""
            is_relative_monit = argv0 == "./monit"
            is_absolute_monit = (
                os.path.basename(argv0) == "monit" and name == "monit"
            )
            if not (is_relative_monit or is_absolute_monit):
                continue
            procs.append({
                "pid": pid,
                "username": info.get("username") or "",
                "cmdline": " ".join(cmdline),
                "argv0": argv0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


@app.get("/api/dqm/find-monit-processes")
async def api_find_monit_processes():
    """List every running ./monit process for the confirmation modal."""
    procs = _enumerate_monit_processes()
    return {"ok": True, "count": len(procs), "processes": procs}


@app.post("/api/dqm/kill-all-monit")
async def api_kill_all_monit():
    """SIGTERM (then SIGKILL after a short grace period) every ./monit.

    Re-enumerates inside this handler so the modal preview and the actual
    kill list are validated against the same psutil snapshot (no TOCTOU
    against a stale modal that the user left open for minutes). Skips our
    own pid, never touches non-./monit processes, and reports a per-PID
    success/failure result for the UI.
    """
    try:
        import psutil
        import signal as _signal_mod
    except ImportError:
        return {"ok": False, "error": "psutil not available", "killed_count": 0}

    own_pid = os.getpid()
    targets = _enumerate_monit_processes()

    killed: list[int] = []
    failed: list[dict] = []

    for proc_info in targets:
        pid = proc_info["pid"]
        if pid == own_pid:
            continue
        try:
            p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue

        try:
            p.send_signal(_signal_mod.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            failed.append({"pid": pid, "reason": str(e)})
            continue

        try:
            p.wait(timeout=2.0)
            killed.append(pid)
            continue
        except psutil.TimeoutExpired:
            pass

        try:
            p.send_signal(_signal_mod.SIGKILL)
            p.wait(timeout=2.0)
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied) as e:
            failed.append({"pid": pid, "reason": str(e)})

    msg = f"Killed {len(killed)} ./monit process(es)"
    if failed:
        msg += f"; {len(failed)} failed"

    return {
        "ok": True,
        "message": msg,
        "killed_count": len(killed),
        "failed_count": len(failed),
        "killed_pids": killed,
        "failed": failed,
    }


@app.get("/api/dqm/live-status")
async def api_live_status():
    """Return whether a freeform LIVE process is currently running."""
    with _freeform_live_lock:
        proc = _freeform_live_proc
        run_number = _freeform_live_run
        alive = proc is not None and proc.poll() is None
    return {"alive": alive, "run_number": run_number if alive else None}


@app.get("/api/dqm/live-log")
async def api_live_log(since: int = 0):
    """Return monit stdout lines accumulated since index `since`."""
    with _freeform_live_log_lock:
        lines = list(_freeform_live_log)
    return {"lines": lines[since:], "total": len(lines)}


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # confirm_mode: True while the 완료 button is visible on screen.
    # When the scenario agent asks for a physical action (move stage, set beam
    # energy, check plots …) it sends "awaiting_input".  Any FREE TEXT the
    # operator types at that moment is an ad-hoc request (not a "완료" answer)
    # and should go to BrainAgent instead of the scenario agent.
    # We use a single-element list so the nested pump_output coroutine can
    # mutate it without 'nonlocal' across tasks.
    _cm = [False]       # _cm[0] == confirm_mode (완료 button)
    _hv_cm = [False]    # _hv_cm[0] == HV confirm mode (완료 + 수정 buttons)
    _retry_cm = [False] # _retry_cm[0] == retry mode (다시 시도 button)

    async def pump_output():
        """
        Drain output_queue and forward to browser.
        - tool_output lines are batched per cycle to reduce send() calls
          (DAQ can produce hundreds of lines; each send() is a potential failure)
        - ai_message / plot / status / awaiting_input are sent immediately
        """
        from starlette.websockets import WebSocketState

        tool_buf = []

        async def flush_tool():
            nonlocal tool_buf
            if tool_buf:
                combined = "\n".join(tool_buf)
                try:
                    await ws.send_json({"type": "tool_output", "content": combined})
                except Exception:
                    pass
                # "Received termination" is printed by the DAQ executable when
                # a run finishes successfully.  Emit daq_complete so the browser
                # can play a notification sound for the remote operator.
                if "received termination" in combined.lower():
                    try:
                        await ws.send_json({"type": "daq_complete"})
                    except Exception:
                        pass
                tool_buf = []

        while True:
            if ws.client_state != WebSocketState.CONNECTED:
                break

            drained_any = False
            # Drain everything currently in queue
            while True:
                try:
                    msg = runner.output_queue.get_nowait()
                    drained_any = True
                    is_brain = msg.get("source") == "brain"

                    # Track confirm mode: set when 완료 button should be shown
                    if msg.get("type") == "awaiting_input":
                        _cm[0] = True
                    elif msg.get("type") == "awaiting_hv_confirm":
                        _cm[0] = True
                        _hv_cm[0] = True
                    elif msg.get("type") == "awaiting_retry":
                        _cm[0] = True
                        _retry_cm[0] = True

                    if msg.get("type") == "tool_output" and not is_brain:
                        # Batch scenario tool_output (DAQ stdout etc.)
                        tool_buf.append(msg["content"])
                    elif is_brain:
                        # Brain messages: flush scenario buffer first,
                        # then send immediately WITH source tag preserved
                        await flush_tool()
                        try:
                            await ws.send_json(msg)
                        except Exception:
                            pass
                        # After the brain's final ai_message, if the scenario
                        # agent is still waiting for confirmation, re-show
                        # the appropriate button(s).
                        if (msg.get("type") == "ai_message"
                                and _cm[0]
                                and runner.waiting_flag.is_set()):
                            try:
                                retype = "awaiting_hv_confirm" if _hv_cm[0] else "awaiting_input"
                                await ws.send_json({"type": retype})
                            except Exception:
                                pass
                    else:
                        # Non-tool messages: flush buffer first, then send immediately
                        await flush_tool()
                        try:
                            await ws.send_json(msg)
                        except Exception:
                            pass
                except queue.Empty:
                    break

            # Flush any accumulated tool lines
            await flush_tool()

            if not drained_any:
                await asyncio.sleep(0.05)

    pump_task = asyncio.create_task(pump_output())

    # Per-session state for multi-step direct commands
    # step values:
    #   "log_ask_column" → waiting for user to say which column
    #   "log_ask_value"  → column known, waiting for value to write
    #   "plot_ask_mode"  → waiting for user to say which plot type
    pending: dict | None = None

    async def _send(msg_type: str, content: str):
        await ws.send_json({"type": msg_type, "content": content})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            # ── User sends text / clicks 완료 ─────────────────────────────
            if msg_type == "user_input":
                content = data.get("content", "").strip()
                if runner.is_running:
                    # Route decision:
                    #
                    # waiting_flag + confirm_mode (완료 button visible):
                    #   → "완료"/"종료"/"exit"  → scenario (physical confirmation)
                    #   → anything else         → Brain ad-hoc (operator request)
                    #
                    # waiting_flag only (config input, e.g. energy/events prompt):
                    #   → all text             → scenario (typed config value)
                    #
                    # neither flag (DAQ/tool running):
                    #   → "완료"/"종료"/"exit"  → scenario (queue for next get_input)
                    #   → anything else         → Brain ad-hoc
                    if runner.waiting_flag.is_set():
                        if content == "retry" and _retry_cm[0]:
                            _cm[0] = False
                            _retry_cm[0] = False
                            runner.send_input(content)
                        elif content in ("완료", "종료", "exit"):
                            _cm[0] = False
                            _hv_cm[0] = False
                            _retry_cm[0] = False
                            runner.send_input(content)
                        elif _hv_cm[0]:
                            # HV modify mode: free text → scenario (not Brain)
                            runner.send_input(content)
                        elif _cm[0] and runner.brain_ready:
                            # 완료 button was shown → free text is an ad-hoc request
                            runner.send_adhoc(content)
                        else:
                            # Config phase (energy/events prompts) → goes to scenario
                            runner.send_input(content)
                    elif content in ("완료", "종료", "exit"):
                        runner.send_input(content)
                    elif runner.brain_ready:
                        # Ad-hoc request → BrainAgent (background)
                        runner.send_adhoc(content)
                    else:
                        # BrainAgent not available → forward to scenario agent
                        runner.send_input(content)

                elif pending is not None:
                    step = pending["step"]

                    if step == "log_ask_column":
                        col = _parse_log_column(content)
                        if col is None:
                            await _send("ai_message", f"열을 인식하지 못했습니다.\n{_COL_ASK_MSG}")
                        else:
                            pending = {"step": "log_ask_value",
                                       "run_number": pending["run_number"], "column": col}
                            await _send("ai_message",
                                        f"{_COL_DISPLAY[col]} 열에 어떤 내용을 입력할까요?")

                    elif step == "log_ask_value":
                        run_number = pending["run_number"]
                        column = pending["column"]
                        pending = None
                        threading.Thread(
                            target=_update_run_log,
                            args=(run_number, column, content,
                                  runner.output_queue, runner.stop_event),
                            daemon=True,
                        ).start()


                else:
                    cmd = _parse_direct_command(content)
                    if cmd is None:
                        # Try BrainAgent for unrecognized commands
                        if runner.brain_ready:
                            runner.send_adhoc(content)
                            continue
                        await _send("ai_message", _HELP_MSG)

                    elif cmd["tool"] == "log_update":
                        run_number = cmd["run_number"]
                        column = cmd.get("column")
                        value = cmd.get("value")

                        if column is None:
                            # Don't know which column
                            pending = {"step": "log_ask_column", "run_number": run_number}
                            await _send("ai_message",
                                        f"Run {run_number} 로그를 수정합니다.\n{_COL_ASK_MSG}")
                        elif value is None:
                            # Column known, but value not in message
                            pending = {"step": "log_ask_value",
                                       "run_number": run_number, "column": column}
                            await _send("ai_message",
                                        f"{_COL_DISPLAY[column]} 열에 어떤 내용을 입력할까요?")
                        else:
                            # Both column and value extracted from message → execute directly
                            threading.Thread(
                                target=_update_run_log,
                                args=(run_number, column, value,
                                      runner.output_queue, runner.stop_event),
                                daemon=True,
                            ).start()


                    else:
                        threading.Thread(
                            target=_run_direct_tool,
                            args=(cmd, runner.output_queue, runner.stop_event),
                            daemon=True,
                        ).start()

            # ── Start a specialized agent ──────────────────────────────────
            elif msg_type == "start_agent":
                if runner.is_running:
                    await ws.send_json({
                        "type": "error",
                        "content": "다른 에이전트가 이미 실행 중입니다."
                    })
                else:
                    agent_name = data.get("agent")
                    params = data.get("params", {})
                    _cm[0] = False      # reset confirm-mode from any previous session
                    _hv_cm[0] = False
                    _retry_cm[0] = False
                    try:
                        runner.start(agent_name, params)
                        await ws.send_json({
                            "type": "status",
                            "content": f"{agent_name} 에이전트 시작됨"
                        })
                    except Exception as e:
                        await ws.send_json({"type": "error", "content": str(e)})

            # ── Stop running agent ─────────────────────────────────────────
            elif msg_type == "stop_agent":
                if runner.is_running:
                    runner.stop()
                    await ws.send_json({"type": "status", "content": "에이전트 중지 요청됨"})

            # ── Ad-hoc confirmation (from popup Yes/No buttons) ────────────
            elif msg_type == "adhoc_confirm":
                confirmed = bool(data.get("confirmed", False))
                runner.send_confirm(confirmed)

            # ── Kill current DAQ run ───────────────────────────────────────
            elif msg_type == "kill_run":
                try:
                    from tools.daq_tool import WORKDIR, STUDIO_HOST, STUDIO_USER, STUDIO_KEY
                    import subprocess as _sp
                    studio_killme = WORKDIR + "/KILLME"
                    _sp.run(
                        [
                            "ssh", "-i", STUDIO_KEY,
                            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
                            "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                            f"{STUDIO_USER}@{STUDIO_HOST}",
                            f"touch {studio_killme}",
                        ],
                        timeout=10, check=True,
                    )
                    await ws.send_json({"type": "tool_output", "content": "🛑 KILLME 생성 → DAQ 중지 요청"})
                except Exception as e:
                    await ws.send_json({"type": "error", "content": f"KILLME 생성 실패: {e}"})

    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
