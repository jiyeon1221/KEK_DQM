#!/usr/bin/env python3
"""Position Scan Agent — Sim-ADC 모드 (run_web.py 경로).

HV Equalization Sim-ADC(agents/hv_equalization_sim_agent.py)와 동일한 설계:
DAQ·motor 등 워크플로우는 실제 PositionScanAgent 그대로 돌리고,
peakADC 측정(_measure_peakadc)만 시뮬레이션으로 대체한다.

모든 걸 "했다 치고" 넘어가는 완전 sim은 sim/position_scan_agent.py(run_web_sim.py)에 있다.
peakADC 모델은 두 변종이 공유(sim.position_scan_sim_model)한다.
"""

from agents.position_scan_agent import PositionScanAgent
from sim.position_scan_sim_model import PositionScanPeakADCSimMixin


class PositionScanSimADCAgent(PositionScanPeakADCSimMixin, PositionScanAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_name = f"{self.agent_name} [Sim-ADC]"
        self._init_peakadc_sim_model()

    # _execute_tool은 override하지 않는다 → 실제 daq_run_tool 실행 (하드웨어 그대로).
