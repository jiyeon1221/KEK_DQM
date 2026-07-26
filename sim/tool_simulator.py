#!/usr/bin/env python3
"""
Tool Simulator
--------------
Web sim 모드에서 하드웨어/원격 연결 없이 tool calling 흐름을 검증하기 위한 mock 응답.
"""

from __future__ import annotations

import json
import math
import random
import threading
from typing import Any, Callable, Dict, Optional


class ToolSimulator:
    """시나리오/Brain agent tool 호출에 대한 가짜 응답 생성."""

    def __init__(self):
        self._lock = threading.Lock()
        self._run_counter = 99_000
        # 채널명(예: T5C/T5S) → V0Set. 미설정 채널은 hv_status에서 775.0 기본값.
        self._hv: Dict[str, float] = {}
        self._hodoscope_hv = 1200.0
        self._adc_params: Dict[str, Dict[str, float]] = {}

    def _next_run_number(self) -> int:
        with self._lock:
            self._run_counter += 1
            return self._run_counter

    def _ensure_adc_params(self, tower: str, target_adc: float = 1230.0):
        if tower in self._adc_params:
            return
        B_c = random.uniform(0.0060, 0.0080)
        B_s = random.uniform(0.0060, 0.0080)
        ref_hv_c = random.uniform(750.0, 800.0)
        ref_hv_s = random.uniform(750.0, 800.0)
        frac_c = random.uniform(0.35, 0.70)
        frac_s = random.uniform(0.35, 0.70)
        A_c = frac_c * target_adc / math.exp(B_c * ref_hv_c)
        A_s = frac_s * target_adc / math.exp(B_s * ref_hv_s)
        self._adc_params[tower] = {
            "C": {"A": A_c, "B": B_c, "noise": random.uniform(0.015, 0.040)},
            "S": {"A": A_s, "B": B_s, "noise": random.uniform(0.015, 0.040)},
        }

    def _simulate_adc(self, tower: str, channel: str, hv: float, iteration: int = 0) -> float:
        self._ensure_adc_params(tower)
        p = self._adc_params[tower][channel]
        base = p["A"] * math.exp(p["B"] * hv)
        effective_noise = p["noise"] / (1.0 + 0.5 * iteration)
        return max(0.0, base + random.normalvariate(0.0, base * effective_noise))

    @staticmethod
    def _emit(lines: list[str], line_callback: Optional[Callable[[str], None]] = None) -> str:
        if line_callback:
            for line in lines:
                line_callback(line)
        return "\n".join(lines)

    def daq_run(
        self,
        params: Dict[str, Any],
        line_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        run_number = self._next_run_number()
        events = params.get("events", "?")
        config = params.get("config", "pre1")
        program = params.get("program", "?")
        beam = params.get("beam_energy", "?")
        lines = [
            "🔧 [SIM] DAQ Run (no hardware)",
            f"   Program: {program} | Config: {config} | Beam: {beam} GeV | Events: {events}",
            f"   pos_h={params.get('pos_h')} pos_v={params.get('pos_v')}",
            f"Run: {run_number}",
            "Received termination",
        ]
        return self._emit(lines, line_callback)

    def hv_equalization_start(self, target_c: float, target_s: float, tower: str) -> str:
        from tools.hv_equalization_tool import _session_manager
        _session_manager.start_session("default", target_c, target_s, tower)
        return (
            f"🔬 [SIM] HV Equalization session started\n"
            f"   Tower: {tower} | Target ADC C={target_c}, S={target_s}"
        )

    def hv_status(self, channels=None) -> str:
        lines = ["📊 [SIM] HV Status Query"]
        if isinstance(channels, list) and channels:
            for ch in channels:
                v = self._hv.get(ch, 775.0)
                lines.append(f"({ch})  V0Set = {v:.1f} V  [SIM]")
        elif isinstance(channels, str) and channels and channels.lower() not in ("all", "전체"):
            lines.append(f"  channels = {channels}  [SIM]")
        else:
            lines.append("  channels = all  [SIM]")
        return "\n".join(lines)

    def hv_voltage(self, channel_values: Dict[str, float]) -> str:
        lines = ["⚡ [SIM] HV Voltage Set"]
        for ch, v in channel_values.items():
            self._hv[ch] = float(v)
            lines.append(f"  {ch} → {v:.1f} V")
        return "\n".join(lines)

    def hv_on_off(self, command: str, channels) -> str:
        return f"[SIM] HV {command.upper()} — channels={channels}"

    def hv_suggest(
        self,
        *,
        tower: str,
        run_number: int,
        hv_c: float,
        hv_s: float,
        target_adc_c: float,
        target_adc_s: float,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        adc_c = self._simulate_adc(tower, "C", hv_c, iteration)
        adc_s = self._simulate_adc(tower, "S", hv_s, iteration)
        tol = 0.02
        c_done = abs(adc_c - target_adc_c) / target_adc_c < tol
        s_done = abs(adc_s - target_adc_s) / target_adc_s < tol

        self._ensure_adc_params(tower, target_adc_c)
        p_c = self._adc_params[tower]["C"]
        p_s = self._adc_params[tower]["S"]
        next_hv_c = int(round(hv_c)) if c_done else int(round(math.log(target_adc_c / p_c["A"]) / p_c["B"]))
        next_hv_s = int(round(hv_s)) if s_done else int(round(math.log(target_adc_s / p_s["A"]) / p_s["B"]))

        return {
            "status": "success",
            "run_number": run_number,
            "current": {
                "C": {"hv": hv_c, "adc": round(adc_c, 1)},
                "S": {"hv": hv_s, "adc": round(adc_s, 1)},
            },
            "suggested": {
                "C": {"hv": next_hv_c, "done": c_done},
                "S": {"hv": next_hv_s, "done": s_done},
            },
        }

    def hv_done_channel(self, tower: str) -> str:
        return f"[SIM] {tower} HV equalization channel marked done"

    def dqm_plot(self, params: Dict[str, Any]) -> str:
        run_number = params.get("run_number", "?")
        method = params.get("method", "IntADC")
        type_ = params.get("type", "full")
        return (
            f"📈 [SIM] DQM plot generated (no monit)\n"
            f"   Run {run_number} | type={type_} | method={method}"
        )

    def run_log(self, params: Dict[str, Any]) -> str:
        cmd = params.get("command", "?")
        run_num = params.get("run_num", "?")
        if cmd == "read":
            return f"[SIM] Run log read for Run {run_num} (Google Sheets not connected)"
        cols = {k: v for k, v in params.items() if k not in ("command", "run_num")}
        return f"[SIM] Run log update Run {run_num}: {cols}"

    def hodoscope_hv_read(self) -> str:
        return f"[SIM] Hodoscope HV (set file): {self._hodoscope_hv:.1f} V"

    def hodoscope_hv_write(self, value: float) -> str:
        self._hodoscope_hv = float(value)
        return f"[SIM] Hodoscope HV set to {self._hodoscope_hv:.1f} V"

    def format_tool_call(self, tool_name: str, params: Dict[str, Any]) -> str:
        return (
            f"🧪 [SIM TOOL CALL] {tool_name}\n"
            f"   params: {json.dumps(params, ensure_ascii=False)}"
        )


_simulator: Optional[ToolSimulator] = None
_sim_lock = threading.Lock()


def get_simulator() -> ToolSimulator:
    global _simulator
    with _sim_lock:
        if _simulator is None:
            _simulator = ToolSimulator()
        return _simulator
