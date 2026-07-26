#!/usr/bin/env python3
"""DAQ Tool — Test-beam DAQ 시스템 제어"""

import os
import re
import time
import shlex
import shutil
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .base_tool import BaseTool
from .run_log_tool import RunLogTool
from .config_loader import get_path_config, get_data_directory, load_config
from .position_calculator_tool import tower_to_dqm_canvas_number


# ===== DAQ 설정 (Config from YAML) =====
WORKDIR = get_path_config("DaqWorkDir")
RUN_SCRIPT = get_path_config("DaqScript")
RUNNUM_FILE = get_path_config("RunNumberFile")
KILLME_FILE = get_path_config("KillMeFile")
NOTICE_BASE = get_path_config("NoticeBase")

_studio = load_config()["StudioSSH"]
STUDIO_HOST = _studio["Host"]
STUDIO_USER = _studio["User"]
STUDIO_PASSWORD = _studio["Password"]

# DAQ 시작 시 읽은 run number — agents는 종료 후 runnum.txt 재읽기(-1) 대신 이 값 사용
DAQ_RUN_NUMBER_MARKER = "__AUTO_TB_RUN_NUMBER__="


def parse_run_number_from_daq_output(daq_output: Optional[str]) -> Optional[int]:
    """DAQ tool 출력에서 시작 시 확정된 run number 추출."""
    if not daq_output:
        return None
    for line in reversed(daq_output.splitlines()):
        stripped = line.strip()
        if stripped.startswith(DAQ_RUN_NUMBER_MARKER):
            try:
                return int(stripped.split("=", 1)[1].strip())
            except (ValueError, IndexError):
                break
    match = re.search(r"Run:?\s*(\d+)", daq_output)
    return int(match.group(1)) if match else None


class DAQRunTool(BaseTool):
    """DAQ 실행 Tool"""
    
    def __init__(self):
        super().__init__(
            name="daq_run_tool",
            description="Execute DAQ run with specified events and configuration"
        )
    
    def execute(self, params: Dict[str, Any], line_callback=None) -> str:
        """
        DAQ 실행
        
        Args:
            params:
                - events (int): 수집할 이벤트 수
                - config (str, optional): 설정 파일 (기본값: "setup")
                - pos_h, pos_v, pos_rot, pos_tilt, beam_energy (optional): 로깅용 정보
        """
        valid, error = self.validate_params(params, ["events"])
        if not valid:
            raise RuntimeError(f"파라미터 오류: {error}")

        events = params["events"]
        config = params.get("config", "pre1")

        try:
            events = int(events)
            if events <= 0:
                raise RuntimeError("이벤트 수는 양수여야 합니다")
        except (TypeError, ValueError):
            raise RuntimeError(f"잘못된 이벤트 수: {events}")

        try:
            with open(RUNNUM_FILE, "r") as f:
                runnum = f.read().strip()
        except FileNotFoundError:
            raise RuntimeError("runnum.txt 파일을 찾을 수 없습니다. DAQ start를 먼저 실행하세요.")

        # ── DQM live session: spawn monit --LIVE alongside DAQ ────────────────
        # Best-effort: if anything fails (no manifest, missing binary, etc.)
        # we still run DAQ — DQM is auxiliary.
        dqm_started = False
        try:
            from tools.dqm_live_worker import dqm_session
            from agents.agent_runner import shared_state
            agent_type = shared_state.get("agent_type")
            output_queue = shared_state.get("_output_queue")
            if agent_type and output_queue is not None:
                # Manifest cell templates use ${current_tower} as the DQM live
                # canvas number (e.g. fCanvas_Tower17 for tower "M5T1" — DQM
                # assigns this number by sorting the mapping, not from the tower
                # name itself; see tower_to_dqm_canvas_number for the algorithm).
                tower_raw = shared_state.get("current_tower") or "5"
                tower_num = tower_to_dqm_canvas_number(tower_raw)
                dqm_session.start(
                    run_number=int(runnum),
                    agent_type=agent_type,
                    output_queue=output_queue,
                    context={
                        "current_tower": tower_num,
                        "current_energy": shared_state.get("current_energy", ""),
                    },
                )
                dqm_started = True
        except Exception as _dqm_err:
            print(f"⚠️ DQM live start skipped: {_dqm_err}", flush=True)

        cmd = (
            f"sshpass -p {shlex.quote(STUDIO_PASSWORD)} ssh"
            f" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
            f" -o PreferredAuthentications=password -o PubkeyAuthentication=no"
            f" {STUDIO_USER}@{STUDIO_HOST}"
            f" 'cd {WORKDIR} && bash {RUN_SCRIPT} {config} {events}'"
        )

        try:
            def _emit(line: str):
                """Print and stream to web UI via callback."""
                print(line, flush=True)
                output_lines.append(line)
                if line_callback:
                    line_callback(line)

            output_lines = []
            _emit("🟡 DAQ Run Started")
            _emit(f"📋 Command: {cmd}")
            _emit(f"📝 Run: {runnum} | Config: {config} | Events: {events}")
            _emit("💡 Tip: Create 'KILLME' file to stop execution")
            _emit("")

            start_time = datetime.now()

            # Write initial log row immediately so the run is recorded even if killed.
            _log_tool = RunLogTool()
            try:
                _log_tool.execute({
                    "command": "write",
                    "run_num": runnum,
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": config,
                    "program": params.get("program", ""),
                    "pos_h": params.get("pos_h", ""),
                    "pos_v": params.get("pos_v", ""),
                    "pos_rot": params.get("pos_rot", ""),
                    "pos_tilt": params.get("pos_tilt", ""),
                    "beam_energy": params.get("beam_energy", ""),
                })
                _emit("📝 초기 로그 기록 완료 (Run 시작 기준)")
            except Exception as _le:
                _emit(f"⚠️ 초기 로그 기록 실패: {_le}")

            process = subprocess.Popen(
                cmd,
                shell=True,
                preexec_fn=os.setpgrp,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            _emit("📄 Execution Output:")

            while True:
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        clean_line = line.strip()
                        if clean_line:
                            print(clean_line, flush=True)
                            output_lines.append(clean_line)
                            if line_callback:
                                line_callback(clean_line)
                        continue

                if process.poll() is not None:
                    break

                # KILLME 파일 모니터링
                if os.path.exists(KILLME_FILE):
                    print("🛑 KILLME file detected - Stopping execution...", flush=True)
                    output_lines.append("🛑 KILLME file detected - Stopping execution...")
                    process.terminate()
                    process.wait()
                    try:
                        subprocess.run(
                            [
                                "sshpass", "-p", STUDIO_PASSWORD,
                                "ssh",
                                "-o", "StrictHostKeyChecking=no",
                                "-o", "UserKnownHostsFile=/dev/null",
                                "-o", "PreferredAuthentications=password",
                                "-o", "PubkeyAuthentication=no",
                                f"{STUDIO_USER}@{STUDIO_HOST}",
                                f"rm -f {WORKDIR}/KILLME",
                            ],
                            timeout=10, check=False,
                        )
                    except Exception:
                        pass
                    try:
                        _log_tool.execute({
                            "command": "update",
                            "run_num": runnum,
                            "notes": "RUN KILLED",
                        })
                    except Exception:
                        pass
                    output_lines.append("❌ DAQ execution stopped by user")
                    output_lines.append(f"{DAQ_RUN_NUMBER_MARKER}{runnum}")
                    return "\n".join(output_lines)

                time.sleep(0.1)

            if process.stdout:
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.strip().split('\n'):
                        if line.strip():
                            print(line.strip(), flush=True)
                            output_lines.append(line.strip())

            exit_code = process.wait()

            end_time = datetime.now()
            duration = round((end_time - start_time).total_seconds(), 2)
            timestamp = start_time.strftime("%H:%M")

            output_lines.append("")

            total, used, free = shutil.disk_usage(".")
            free_gb = round(free / (1024 ** 3), 2)
            percent_used = used / total * 100

            if exit_code == 0:
                # BaseDirectory(config_general.yml)를 단일 진실 소스로 사용 — DQM/HV/dat_plot과 동일.
                try:
                    from pathlib import Path as _Path
                    data_base = get_data_directory()
                    run_wave_dir = (
                        _Path(data_base) / f"Run_{runnum}"
                        / f"Run_{runnum}_Wave"
                        / f"Run_{runnum}_Wave_MID_8"
                    )
                    dat_files = sorted(run_wave_dir.glob(f"Run_{runnum}_Wave_MID_8_FILE_*.dat"))
                    if dat_files:
                        pass
                except Exception:
                    pass

                _duration_secs = int((end_time - start_time).total_seconds())
                _rate = round(events / _duration_secs, 1) if _duration_secs > 0 else ""
                try:
                    log_result = _log_tool.execute({
                        "command": "update",
                        "run_num": runnum,
                        "evts": events,
                        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration": _duration_secs,
                        "rate": _rate,
                    })
                    output_lines.append(f"📝 {log_result}")
                except RuntimeError as _log_err:
                    output_lines.append(f"⚠️ 로그 업데이트 실패: {_log_err}")

                output_lines.append(f"Config: {config} | Events: {events}")
                output_lines.append(f"Duration: {duration}s | Started: {timestamp}")
                output_lines.append(f"Free Space: {free_gb}GB")
                
                if percent_used >= 90:
                    output_lines.append("⚠️ Warning: Disk usage exceeds 90%!")
                output_lines.append(f"{DAQ_RUN_NUMBER_MARKER}{runnum}")
            else:
                output_lines.append(f"❌ DAQ Run Failed (Exit Code: {exit_code})")
                output_lines.append(f"Duration: {duration}s | Started: {timestamp}")
                raise RuntimeError(f"DAQ Run Failed (Exit Code: {exit_code})")

            return "\n".join(output_lines)

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"DAQ 실행 중 오류: {str(e)}") from e
        finally:
            # Always stop DQM live session (touches sentinel, waits for monit
            # to flush a final chunk and exit).
            if dqm_started:
                try:
                    from tools.dqm_live_worker import dqm_session
                    dqm_session.stop()
                except Exception as _dqm_stop_err:
                    print(f"⚠️ DQM live stop error: {_dqm_stop_err}", flush=True)
