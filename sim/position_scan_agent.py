#!/usr/bin/env python3
"""Position Scan Agent — simulation mode (no hardware).

agents.position_scan_agent를 상속해 워크플로우/cross 판정/보간/결과저장은 그대로 공유하고,
하드웨어 호출(daq)과 peakADC 측정(_measure_peakadc)만 mock한다.

peakADC 모델: 센터 타워와 이웃 타워를 두 개의 가우시안으로 가정.
센터 타워의 빔 중심(true center)은 estimated center 근처(±interval)로 한 번만 정하고
두 스윕에서 공유한다. 이웃 타워 중심은 거기서 pitch만큼 떨어진 위치.
스캔축 좌표가 두 중심 사이로 이동하면 peakADC가 자연스럽게 교차한다.
"""

import math
import random
from typing import Dict, Optional

from agents.position_scan_agent import PositionScanAgent
from sim.sim_base import SimExecMixin
from sim.tool_simulator import get_simulator


class PositionScanSimAgent(SimExecMixin, PositionScanAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sim = get_simulator()
        self.agent_name = f"{self.agent_name} [SIM]"

        # ── sim peakADC 모델 파라미터 ──
        interval = self.state["interval"]
        # 타워 간격(mm) — sim 전용 근사값 (position_calculator 미사용)
        self._sim_pitch = 46.333 if self.state["direction"] == "horizontal" else 50.0
        # 센터 타워 빔 true center: estimated center 근처로 한 번만 결정 (두 스윕 공유)
        base = self.state["est_center"][self._axis_key()]
        self._sim_true_center = base + random.uniform(-0.4, 0.4) * interval
        self._sim_peak = random.uniform(1000.0, 1500.0)
        self._sim_sigma = self._sim_pitch * 0.7
        self._sim_noise = 0.01  # 1% 가우시안 노이즈
        self.log(
            f"[SIM] model: true_center({self._axis_key()})={self._sim_true_center:.3f}, "
            f"pitch={self._sim_pitch}, peak={self._sim_peak:.1f}, sigma={self._sim_sigma:.2f}"
        )

    def _measure_peakadc(self, run_number: int, tower: str, channel: str) -> Optional[float]:
        coord = self._current_axis_coord()
        if tower == self.state["center_tower"]:
            mu = self._sim_true_center
        else:
            # 현재 스윕 이웃 타워는 center로부터 current_sign*pitch 위치
            mu = self._sim_true_center + self._current_sign() * self._sim_pitch
        base = self._sim_peak * math.exp(-((coord - mu) ** 2) / (2.0 * self._sim_sigma ** 2))
        val = base + random.normalvariate(0.0, self._sim_peak * self._sim_noise)
        return max(0.0, val)

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        if tool_name == "none":
            return "no_tool_executed"

        if tool_name == "daq_run_tool":
            # 공통부(_sim_run_daq): params override → 표시 → 실행 → run# 추출 → plot 대기.
            # 표시가 override "뒤"에 찍혀 실제 실행 위치/에너지와 일치한다.
            result, run_number = self._sim_run_daq(
                params,
                events=self.state.get("target_events"),
                beam_energy=self.state.get("beam_energy"),
                program="Position Scan",
                pos=self._position_for_current_step(),
            )
            if run_number:
                self.state["last_run_number"] = run_number
                self.log(f"[SIM] DAQ Run {run_number} 완료: {self._position_for_current_step()}, "
                         f"{params.get('events', 0)} events")
            return result

        self._sim_emit_tool_call(tool_name, params)
        return f"Error: Unknown tool {tool_name}"
