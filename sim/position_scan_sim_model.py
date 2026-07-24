#!/usr/bin/env python3
"""Position Scan peakADC 시뮬레이션 모델 (공유 mixin).

두 sim 변종이 이 모델을 공유한다 (서로 byte-sync가 깨지지 않도록 한 곳에 둔다):
  - sim.position_scan_agent.PositionScanSimAgent
        → 모든 하드웨어(daq 포함)를 mock. run_web_sim.py 경로.
  - agents.position_scan_sim_agent.PositionScanSimADCAgent
        → daq/motor 등은 실제로 돌리고 peakADC 측정만 mock. run_web.py 경로.
        (hv_equalization_sim_agent가 _measure_adc만 override하는 것과 동일한 설계.)

peakADC 모델: 센터 타워와 이웃 타워를 두 개의 가우시안으로 가정.
센터 타워의 빔 중심(true center)은 estimated center 근처(±interval)로 한 번만 정하고
두 스윕에서 공유한다. 이웃 타워 중심은 거기서 pitch만큼 떨어진 위치.
스캔축 좌표가 두 중심 사이로 이동하면 peakADC가 자연스럽게 교차한다.
"""

import math
import random
from typing import Optional


class PositionScanPeakADCSimMixin:
    """PositionScanAgent와 함께 다중상속해 _measure_peakadc만 시뮬레이션으로 대체한다.

    MRO에서 이 mixin이 PositionScanAgent보다 앞에 와야 _measure_peakadc가 override된다:
        class X(PositionScanPeakADCSimMixin, PositionScanAgent): ...
    __init__ 끝에서 self._init_peakadc_sim_model()을 호출해야 파라미터가 준비된다.
    """

    def _init_peakadc_sim_model(self):
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
            f"[SIM] peakADC model: true_center({self._axis_key()})={self._sim_true_center:.3f}, "
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
