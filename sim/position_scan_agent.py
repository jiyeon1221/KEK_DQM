#!/usr/bin/env python3
"""Position Scan Agent — 완전 simulation mode (no hardware, run_web_sim.py).

agents.position_scan_agent를 상속해 워크플로우/cross 판정/보간/결과저장은 그대로 공유하고,
하드웨어 호출(daq)과 peakADC 측정(_measure_peakadc)을 모두 mock한다.

peakADC 모델은 sim.position_scan_sim_model.PositionScanPeakADCSimMixin에서 가져온다
(run_web.py의 Sim-ADC 변종 agents/position_scan_sim_agent.py와 공유 — drift 방지).
DAQ까지 실제로 돌리고 peakADC만 mock하는 변종은 agents/position_scan_sim_agent.py에 있다.
"""

from typing import Dict

from agents.position_scan_agent import PositionScanAgent
from sim.position_scan_sim_model import PositionScanPeakADCSimMixin
from sim.sim_base import SimExecMixin
from sim.tool_simulator import get_simulator


class PositionScanSimAgent(SimExecMixin, PositionScanPeakADCSimMixin, PositionScanAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sim = get_simulator()
        self.agent_name = f"{self.agent_name} [SIM]"
        self._init_peakadc_sim_model()

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
