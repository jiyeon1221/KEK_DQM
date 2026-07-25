#!/usr/bin/env python3
"""
Minimal Web Server (NO AI)
-------------------------
Serve only standalone viewers (HV check / DQM freeform) and their APIs.

This intentionally does NOT load AgentRunner / BrainAgent / WebSocket bridge.
So it can be used in environments where AI can't run.
"""

import asyncio
import os
import re
import signal
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tools.hv_control_tool import HVControlTool


app = FastAPI(title="autoTB Minimal Viewers (no AI)")

STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent

# DQM paths
DQM_DIR = PROJECT_ROOT / "DQM"
DQM_OUTPUT_DIR = DQM_DIR / "output"
DQM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# static mounts needed by viewers
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/dqm-output", StaticFiles(directory=str(DQM_OUTPUT_DIR)), name="dqm_output")
app.mount("/jsroot", StaticFiles(directory=str(DQM_DIR)), name="jsroot")


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    # Provide a simple landing page so users don't land on a 404.
    return JSONResponse({
        "ok": True,
        "message": "autoTB minimal viewers (no AI). Use /hv/check or /dqm/freeform",
        "hv_check_url": "/hv/check",
        "dqm_freeform_url": "/dqm/freeform",
    })

@app.get("/hv/check")
async def hv_check_page():
    return FileResponse(str(STATIC_DIR / "hv_check.html"))


@app.get("/dqm/freeform")
async def dqm_freeform():
    return FileResponse(str(STATIC_DIR / "dqm_freeform.html"))


# ──────────────────────────────────────────────────────────────────────────────
# HV APIs
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/hv/status-all")
async def api_hv_status_all(expert: bool = False):
    try:
        tool = HVControlTool()
        if not expert:
            result = tool.execute({"command": "status", "channels": "all"})
            return {"ok": True, "output": result, "expert": False}

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


@app.get("/api/hv/hodoscope")
async def api_hv_hodoscope():
    """Return current hodoscope HV setting read from the DAQ set file."""
    try:
        from tools.hodoscope_hv_tool import read_hv_from_setfile
        hv = read_hv_from_setfile()
        if hv is None:
            return JSONResponse({"ok": False, "error": "Set file을 읽을 수 없습니다."}, status_code=500)
        return {"ok": True, "hv": hv}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class HvSetRequest(BaseModel):
    command: str           # "voltage" | "on" | "off" | "i0set" | "svmax" | "rup" | "rdown" | "name"
    channels: object       # str | list
    voltage: float = None
    current: float = None
    svmax: float = None
    rup: float = None
    rdown: float = None
    name: str = None


_hv_cmd_lock = asyncio.Lock()


@app.post("/api/hv/set")
async def api_hv_set(req: HvSetRequest):
    async with _hv_cmd_lock:
        try:
            tool = HVControlTool()
            params: dict = {"command": req.command, "channels": req.channels}
            if req.command == "voltage":
                if req.voltage is None:
                    return JSONResponse({"ok": False, "error": "voltage 값이 필요합니다"}, status_code=400)
                params["voltage"] = req.voltage
            if req.command == "i0set":
                if req.current is None:
                    return JSONResponse({"ok": False, "error": "current 값이 필요합니다"}, status_code=400)
                params["command"] = "current"
                params["current"] = req.current
            if req.command == "svmax":
                if req.svmax is None:
                    return JSONResponse({"ok": False, "error": "svmax 값이 필요합니다"}, status_code=400)
                params["svmax"] = req.svmax
            if req.command == "rup":
                if req.rup is None:
                    return JSONResponse({"ok": False, "error": "rup 값이 필요합니다"}, status_code=400)
                params["rup"] = req.rup
            if req.command == "rdown":
                if req.rdown is None:
                    return JSONResponse({"ok": False, "error": "rdown 값이 필요합니다"}, status_code=400)
                params["rdown"] = req.rdown
            if req.command == "name":
                if not req.name:
                    return JSONResponse({"ok": False, "error": "name 값이 필요합니다"}, status_code=400)
                params["name"] = req.name
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.execute, params)
            return {"ok": True, "output": result}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/hv/expert-metrics")
async def api_hv_expert_metrics():
    """Ramp up/down and SVMax only (exact CAEN wrapper param names)."""
    try:
        tool = HVControlTool()
        if not tool._ensure_connection():
            return JSONResponse({"ok": False, "error": "HV SSH connection failed"}, status_code=500)

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


# ──────────────────────────────────────────────────────────────────────────────
# DQM APIs (for freeform viewer)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/dqm/runs")
async def api_dqm_runs():
    """List all runs available in the DQM output directory, grouped by run number."""
    import re as _re
    pattern = _re.compile(r'^Run(\d+)_(.+?)_(.+?)_((?:AuxCut_)?)(.+)\.json$')
    # TBsingleWaveform's manifest doesn't fit the <type>_<method>_<canvas>
    # scheme above (see the run-monit handler for details), so it's matched
    # separately and folded into the same `runs` dict below.
    waveform_pattern = _re.compile(r'^Run(\d+)_SingleWaveform\.json$')
    runs: dict[int, list] = {}
    for p in sorted(DQM_OUTPUT_DIR.glob("Run*_*.json")):
        m = pattern.match(p.name)
        if m:
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
            continue
        wm = waveform_pattern.match(p.name)
        if wm:
            run_num = int(wm.group(1))
            runs.setdefault(run_num, []).append({
                "filename": p.name,
                "canvas": "SingleWaveform",
                "type": "single",
                "method": "Waveform",
                "auxcut": False,
                "kind": "waveform",
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


class MonitRequest(BaseModel):
    run_number: int
    type: str = "full"
    method: str = "IntADC"
    modules: List[str] = []
    max_event: Optional[int] = None
    flags: List[str] = []
    # AUXcut mode (none / PID / DWC / DWCPID; legacy WC / WCHodo still
    # supported by TBaux but hidden from the UI) chosen in the freeform UI
    # dropdown. Anything other than "none" turns on --AUXcut and forwards
    # --AUXCutMode <value> to monit.
    aux_cut_mode: Optional[str] = None
    # AUX scope mode — a comma-separated subset of "Hodo,DWC,PID" built
    # from the freeform UI's subsystem checkboxes, only meaningful when
    # "AUX" is in flags. Forwarded verbatim as --AUXMode so TBaux::init()
    # can skip loading the unused subsystems' MIDs (e.g. no MID 17 read
    # when hodoscope is physically absent from the setup). "WC" is still a
    # valid token for TBaux but has no UI checkbox — no wire chamber at
    # CERN this year.
    aux_mode: Optional[str] = None
    # --particle PION|KAON|PROTON, forwarded to TBaux::SetParticle(). Only
    # affects the "DWCPID" --AUXCutMode; harmless (and forwarded) otherwise.
    particle: Optional[str] = None
    # --SkipEvent N, used by `--type single --method Waveform` to start the
    # event-by-event waveform dump partway through the run.
    skip_event: Optional[int] = None


# ── Freeform LIVE process tracker (same behavior as main server) ──────────────
import subprocess as _subprocess
from collections import deque as _deque
_freeform_live_proc: Optional[_subprocess.Popen] = None
_freeform_live_run: Optional[int] = None
_freeform_live_lock = threading.Lock()
# Live-log deque sizing — kept in sync with web/server.py:
#   - monit writes one progress line per ~10 events (~10 lines/s).
#   - Sequence-counter bookkeeping below means the browser never gets
#     stuck regardless of how much monit prints. maxlen only controls
#     how far back we can replay history to a polling browser that
#     stopped consuming for a while (backgrounded tab, SSH tunnel
#     hiccup, etc.).
#   - At 150 B/line, 200_000 entries ≈ 30 MB resident — negligible.
_freeform_live_log: _deque = _deque(maxlen=200_000)
_freeform_live_log_lock = threading.Lock()
# Monotonically increasing counter of ALL lines ever appended (never
# resets on deque rotation, only on a new run). The browser polls
# /api/dqm/live-log with `since=<last total>` and we return lines whose
# sequence number is > `since`. Using len(deque) as the counter was
# broken: once the deque filled (maxlen) it stayed at maxlen forever,
# so the browser's `since` became equal to `total` and every subsequent
# poll returned nothing — the "stuck at 4510 / 5000" symptom.
_freeform_live_log_seq: int = 0

_freeform_blocking_proc: Optional[_subprocess.Popen] = None
_freeform_blocking_run: Optional[int] = None
_freeform_blocking_lock = threading.Lock()

import re as _re
_ANSI_ESC = _re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _strip_ansi(text: str) -> str:
    return _ANSI_ESC.sub('', text)


def _read_live_stdout(proc: "_subprocess.Popen") -> None:
    global _freeform_live_log_seq
    try:
        for raw in proc.stdout:
            line = _strip_ansi(raw.rstrip("\n")).lstrip("\r")
            if line:
                with _freeform_live_log_lock:
                    _freeform_live_log.append(line)
                    _freeform_live_log_seq += 1
    except Exception:
        pass


@app.post("/api/dqm/run-monit")
async def api_run_monit(req: MonitRequest):
    """Execute DQM monit with custom parameters and return generated canvases."""
    global _freeform_live_proc, _freeform_live_run, _freeform_live_log_seq

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
    if req.skip_event is not None and req.skip_event > 0:
        cmd.extend(["--SkipEvent", str(req.skip_event)])
    for flag in req.flags:
        # "Astro" is currently dormant (deactivated in the freeform UI) but is
        # forwarded anyway to stay identical to web/server.py if it comes back.
        if flag in ("LIVE", "AUXcut", "AUX", "Astro"):
            cmd.append(f"--{flag}")
    if req.aux_cut_mode and req.aux_cut_mode != "none":
        cmd.extend(["--AUXCutMode", req.aux_cut_mode])

    if req.particle and req.particle != "none":
        cmd.extend(["--particle", req.particle])

    # Forward AUX scope only when --AUX is on. With this, TBaux can skip
    # resolving unused CIDs and pushing their MIDs into the reader's MID
    # list when a subsystem is picked out of the checkbox list.
    if "AUX" in req.flags and req.aux_mode:
        cmd.extend(["--AUXMode", req.aux_mode])

    generated_cmd = " ".join(cmd)

    from tools.dqm_live_worker import _build_monit_env
    monit_env = _build_monit_env()

    if "LIVE" in req.flags:
        with _freeform_live_lock:
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

            sentinel = DQM_OUTPUT_DIR / f"Run{req.run_number}_END"
            sentinel.unlink(missing_ok=True)
            with _freeform_live_log_lock:
                _freeform_live_log.clear()
                _freeform_live_log_seq = 0

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

    global _freeform_blocking_proc, _freeform_blocking_run

    with _freeform_live_log_lock:
        _freeform_live_log.clear()
        _freeform_live_log_seq = 0

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
        # No wall-clock timeout: long DQM runs (12+ hours on full datasets)
        # are normal, and the user can always abort via /api/dqm/kill-blocking
        # (the STOP button on the freeform page).
        await loop.run_in_executor(None, proc.wait)
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

    if "AUX" in req.flags:
        auxcut_set = "AUXcut" in req.flags
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
                continue
            canvas = rest[len("AuxCut_"):] if has_auxcut else rest
            canvases.append({
                "filename": p.name,
                "canvas": canvas,
                "type": "AUX",
                "method": method,
            })

    # `--type single --method Waveform` writes a manifest file instead of
    # a JSROOT canvas JSON — surface it as a synthetic "waveform" canvas
    # entry so the freeform viewer knows to render an image-sequence
    # viewer instead. See web/server.py's run-monit handler for details.
    if req.type == "single" and req.method == "Waveform":
        wf_json = DQM_OUTPUT_DIR / f"Run{req.run_number}_SingleWaveform.json"
        if wf_json.exists():
            canvases.append({
                "filename": wf_json.name,
                "canvas": "SingleWaveform",
                "type": req.type,
                "method": req.method,
                "kind": "waveform",
            })

    return {
        "command": generated_cmd,
        "exit_code": proc.returncode,
        "output": output[-500:] if len(output) > 500 else output,
        "canvases": canvases,
    }


@app.post("/api/dqm/kill-live")
async def api_kill_live():
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
        except Exception:
            continue
    return procs


@app.get("/api/dqm/find-monit-processes")
async def api_find_monit_processes():
    procs = _enumerate_monit_processes()
    return {"ok": True, "count": len(procs), "processes": procs}


@app.post("/api/dqm/kill-all-monit")
async def api_kill_all_monit():
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
    with _freeform_live_lock:
        proc = _freeform_live_proc
        run_number = _freeform_live_run
        alive = proc is not None and proc.poll() is None
    return {"alive": alive, "run_number": run_number if alive else None}


@app.get("/api/dqm/live-log")
async def api_live_log(since: int = 0):
    """Return monit stdout lines accumulated since index `since`.

    `since` and `total` are *monotonically increasing* sequence numbers
    counting every line ever appended to the deque — they do NOT reset
    when the bounded-length deque rotates. The deque itself holds only
    the most recent `maxlen` lines, so if a polling browser falls more
    than `maxlen` behind we clamp to the oldest line we still have.
    """
    with _freeform_live_log_lock:
        lines = list(_freeform_live_log)
        total = _freeform_live_log_seq
    # Sequence number of the first line still in the deque:
    #   first_seq + 1, first_seq + 2, ..., total
    first_seq = total - len(lines)
    if since >= total:
        new_lines: list = []
    elif since <= first_seq:
        # Browser is behind by more than maxlen — return everything we
        # still have (older lines were already lost to deque rotation).
        new_lines = lines
    else:
        new_lines = lines[since - first_seq:]
    return {"lines": new_lines, "total": total}

