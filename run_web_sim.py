#!/usr/bin/env python3
"""
autoTB Web UI — Simulation mode (Mock)
DAQ, HV, Motor 모두 실제 하드웨어 없이 동작.

Usage:
  python run_web_sim.py              # localhost:8001
  python run_web_sim.py --port 8000  # run_web.py와 같은 포트

run_web.py는 수정하지 않음.
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── 1. Motor mock ─────────────────────────────────────────────────────
import tools.motor_control_tool as _motor


def _mock_move_x(x_scan: float):
    return True, f"[MOCK] X축 이동 완료: {x_scan:.3f} mm"


def _mock_move_relative(delta_mm: float):
    direction = "오른쪽" if delta_mm >= 0 else "왼쪽"
    return True, f"[MOCK] X축 상대 이동 완료: {direction} {abs(delta_mm):.3f} mm"


_motor.move_x = _mock_move_x
_motor.move_relative = _mock_move_relative

# ── 2. DAQ mock ───────────────────────────────────────────────────────
from tools.daq_tool import DAQRunTool

_mock_run_counter = [90000]


def _mock_daq_execute(self, params, line_callback=None):
    events = params.get("events", 1000)
    run_num = _mock_run_counter[0]
    _mock_run_counter[0] += 1
    lines = [
        f"[MOCK] DAQ 시작 — Run {run_num}",
        f"[MOCK] 이벤트 수집 중: {events} events",
        "[MOCK] DAQ 완료",
        f"Run: {run_num}",
    ]
    for line in lines:
        if line_callback:
            line_callback(line)
        time.sleep(0.05)
    return "\n".join(lines)


DAQRunTool.execute = _mock_daq_execute

# ── 3. HV mock ────────────────────────────────────────────────────────
from tools.hv_control_tool import HVControlTool

_mock_hv_voltages: dict = {}  # {channel: voltage}


def _mock_hv_execute(self, params):
    cmd = params.get("command", "").lower()

    if cmd == "status":
        channels = params.get("channels", "all")
        if channels == "all":
            targets = [f"T{n}{ch}" for n in range(1, 10) for ch in ("C", "S")]
        else:
            targets = channels if isinstance(channels, list) else [channels]
        lines = []
        for ch in targets:
            v = _mock_hv_voltages.get(ch, 850.0)
            lines.append(f"[MOCK] ({ch}) V0Set = {v:.1f} V")
        return "\n".join(lines)

    elif cmd == "voltage":
        cv = params.get("channel_values", {})
        for ch, v in cv.items():
            _mock_hv_voltages[ch] = float(v)
        return f"[MOCK] HV 전압 설정 완료: {cv}"

    else:
        return f"[MOCK] HV {cmd} 완료"


HVControlTool.execute = _mock_hv_execute

# ── 서버 실행 ─────────────────────────────────────────────────────────
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoTB Web UI (Simulation)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8001, type=int, help="Bind port (default: 8001)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change (dev only)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  autoTB Control Panel  [SIMULATION MODE]")
    print(f"  http://localhost:{args.port}")
    print(f"  DAQ / HV / Motor → mock")
    print(f"{'='*55}\n")

    uvicorn.run(
        "web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
