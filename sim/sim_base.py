#!/usr/bin/env python3
"""Sim agent 공통부 — 세 시나리오 sim agent(calib/energy/hv)가 공유하는 tool 실행 로직.

원칙: agent마다 흐름은 조금씩 달라도 DAQ를 돌리는 방식은 동일하다.
  params를 state 기준으로 확정(override) → 확정된 params로 tool-call 표시 →
  sim DAQ 실행 → run number 추출 → plot 확인 대기 플래그.
이 공통부를 한 곳에서 통제한다. agent별로 다른 것(어느 타워/에너지에 run을 기록할지 등
부킹)은 호출부에서 처리한다.

중요: 우측 패널의 "[SIM TOOL CALL]" params 표시는 반드시 _apply_daq_params_from_state로
override한 "뒤에" 찍어야 실제 실행 위치/에너지와 일치한다. LLM 원본 params는 무시되므로
override 전에 찍으면 표시값과 실제 실행값이 어긋난다.
"""

from typing import Any, Dict, Optional, Tuple


class SimExecMixin:
    """sim agent 공통 tool 실행 헬퍼. self._sim / self.io / self.state 를 가진
    sim agent에 믹스인한다."""

    def _sim_emit_tool_call(self, tool_name: str, params: Dict[str, Any]) -> None:
        """확정된 params로 tool-call 헤더 출력 (우측 패널 표시용)."""
        self.io.send_tool_output(self._sim.format_tool_call(tool_name, params))

    def _sim_run_daq(
        self,
        params: Dict[str, Any],
        *,
        events: Any,
        beam_energy: Any,
        program: str,
        pos: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, Optional[int]]:
        """DAQ 공통 실행: params override → 표시 → 실행 → run# 추출 → plot 대기.
        (result, run_number)를 반환. agent별 부킹은 호출부에서 run_number로 처리."""
        self._apply_daq_params_from_state(
            params,
            events=events,
            beam_energy=beam_energy,
            program=program,
            pos=pos,
        )
        # override 후 출력 — 표시 params가 실제 실행값과 일치.
        self._sim_emit_tool_call("daq_run_tool", params)
        result = self._sim.daq_run(params, line_callback=self.io.send_tool_output)
        run_number = self._extract_run_number(result)
        # 사용자 plot 확인 전까지 completed=True 차단 (실제 agent와 동일 부킹)
        self.state["needs_plot_confirm"] = True
        return result, run_number
