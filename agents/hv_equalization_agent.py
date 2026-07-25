#!/usr/bin/env python3
"""
HV Equalization Agent
단일 타워의 HV Equalization 수행. calib_scan_agent 구조를 그대로 따름.
컨트롤러(hv_equalization_scan.py)가 타워를 순서대로 호출함.
"""

import json
import re
import sys
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from tools.daq_tool import DAQRunTool
from tools.hv_control_tool import HVControlTool
from tools.position_calculator_tool import calculate_position
from tools.hv_equalization_tool import (
    hv_equalization_suggest,
    hv_equalization_done_channel,
    generate_fitting_summary,
)

from .base_agent import BaseAgent
sys.path.append(str(Path(__file__).parent.parent))
from config import AGENT_MODELS, MSG_PLOT_CONFIRM, MSG_HV_CONFIRM


class HVEqualizationAgent(BaseAgent):
    def __init__(
        self,
        tower: str,
        beam_energy: float,
        target_events: int,
        target_adc: float,
        use_base_model: bool = False,
        io_handler=None,
    ):
        model_name = "hv_equalization"
        if model_name not in AGENT_MODELS:
            model_name = "calibration"
        model_config = AGENT_MODELS[model_name]

        if use_base_model:
            model_path = model_config["base_model"]
            print(f"⚠️  Base model 사용 ({model_path})")
        else:
            fine_tuned_path = Path(model_config["fine_tuned_path"])
            if fine_tuned_path.exists() and (fine_tuned_path / "config.json").exists():
                model_path = str(fine_tuned_path)
                print(f"✅ Fine-tuned model 사용 ({model_path})")
            else:
                model_path = model_config["base_model"]
                print(f"⚠️  Fine-tuned 모델 없음. Base model 사용 ({model_path})")

        super().__init__(
            model_path=model_path,
            agent_name=f"HV Equalization [{tower}]",
            io_handler=io_handler,
        )

        self.tower = tower
        self.daq_tool = DAQRunTool()
        self.hv_control_tool = HVControlTool()

        pos = calculate_position(tower)
        self.tower_pos = pos

        self.state = {
            "phase": "idle",
            "beam_energy": beam_energy,
            "target_events": target_events,
            "target_adc_c": target_adc,
            "target_adc_s": target_adc,
            "current_tower": tower,
            "tower_pos": {"x": pos["x"], "y": pos["y"]},
            "last_hv_c": None,
            "last_hv_s": None,
            "last_suggested_hv_c": None,
            "last_suggested_hv_s": None,
            "last_adc_c": None,
            "last_adc_s": None,
            "channel_done_c": False,
            "channel_done_s": False,
            "last_run_number": None,
            "iterations": 0,
            "done": False,
            "y_confirmed": False,
            "needs_suggest": False,
            "needs_plot_confirm": False,
            # voltage 적용(1f) 직후 사용자에게 확인 메시지(1g, MSG_HV_CONFIRM)를
            # 반드시 보여주기 위한 게이트. 이게 없으면 needs_plot_confirm/needs_suggest가
            # 이미 False/reset된 상태라 step hint가 곧장 daq_run_tool(1c)로 넘어가버려서
            # 모델이 1g 메시지를 그냥 건너뛰어도 아무도 막지 못한다(사용자에게 전압 변경
            # 확인이 전혀 출력되지 않는 버그의 원인).
            "needs_hv_confirm": False,
            # 승인 단계에서 사용자가 실제로 '완료'를 눌렀는지(코드가 판정).
            # phase=="approving"만으로 voltage 적용을 강제하면 수동 조정 입력이 무시된다.
            "approval_confirmed": False,
        }
        self.log(f"Agent 초기화: {tower}, E={beam_energy}GeV, Events={target_events}, Target ADC={target_adc}")

    def _position_for_current_step(self) -> Optional[Dict[str, float]]:
        """이 인스턴스의 타워 위치 (runner가 타워마다 새 Agent 생성)."""
        return self.tower_pos

    def _get_system_prompt(self) -> str:
        t = self.tower
        x = self.tower_pos["x"]
        y = self.tower_pos["y"]
        return f"""You are HV Equalization Agent for tower {t} (x:{x:.3f}, y:{y:.3f}).
Your task: adjust HV for {t}C and {t}S channels to reach the target peakADC value.

Follow these steps EXACTLY:

=== Workflow for {t} ===
1a. Ask user to move to tower position:
  {{"message": "x = {x:.3f} mm, y = {y:.3f} mm 으로 이동해주세요."}}

After user says "완료":
The SYSTEM marks position confirmed automatically — do NOT output any state update for it.
1b. Check HV status:
  {{"tool": "hv_execute_tool", "params": {{"command": "status", "channels": ["{t}C", "{t}S"]}}}}

[INNER LOOP — repeat 1c→1g until CONVERGED]
1c. Execute DAQ:
  {{"tool": "daq_run_tool", "params": {{"events": <events>, "pos_h": <x>, "pos_v": <y>, "beam_energy": <energy>}}}}

1c-plot. Show plot confirmation (IMMEDIATELY after DAQ, before suggest):
  {{"message": "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."}}
  Wait for user 완료 — the SYSTEM then sets needs_suggest=True automatically.

1d. Suggest HV:
  {{"tool": "hv_equalization_suggest", "params": {{"run_number": <run>, "tower": "{t}"}}}}

1e. Ask approval (only NOT-done channels):
  Both not done: {{"message": "분석 결과, 현재 ADC: {t}C=<adc_c>, {t}S=<adc_s> (목표: <target>). HV 변경 제안: {t}C <old_c>V→<new_c>V, {t}S <old_s>V→<new_s>V. 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  Only C not done: {{"message": "분석 결과, 현재 ADC: {t}C=<adc_c> (목표: <target>). HV 변경 제안: {t}C <old_c>V→<new_c>V, {t}S 완료(변경 없음). 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  Only S not done: {{"message": "분석 결과, 현재 ADC: {t}S=<adc_s> (목표: <target>). HV 변경 제안: {t}C 완료(변경 없음), {t}S <old_s>V→<new_s>V. 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  CRITICAL: Copy the state's "HV 변경 제안" line EXACTLY, always in C-then-S order. Keep each arrow's old→new order (do NOT swap the two numbers). A done channel appears as "완료(변경 없음)" in its own C/S slot — NEVER relabel which channel is done. Use EXACT ADC values (last_adc_c/s). NEVER fabricate numbers.
  If user requests manual HV adjustment (e.g. "C를 800으로", "S 10 올려줘"):
    Update suggested values via update_state and re-send approval message:
    {{"message": "...(updated approval)...", "update_state": {{"last_suggested_hv_c": <new_c>, "last_suggested_hv_s": <new_s>}}}}

After user says "완료":
1f. Apply voltage (only NOT-done channels):
  {{"tool": "hv_execute_tool", "params": {{"command": "voltage", "channel_values": {{"{t}C": <new_c>, "{t}S": <new_s>}}}}, "update_state": {{"phase": "equalizing"}}}}
  NEVER include a done channel in channel_values.

1g. Confirmation:
  {{"message": "전압이 변경되었습니다. 확인 후 '완료'를 눌러주세요."}}

After user says "완료":
  - State shows NOT CONVERGED → back to step 1c
  - State shows CONVERGED (C=True, S=True) → proceed to step 1h

1h. Done:
  {{"tool": "hv_equalization_done_channel", "params": {{"channels": "all"}}}}

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip Step 1e (Approval).
2. Step 1a ALWAYS comes before 1b.
3. Output JSON ONLY. No natural language.
4. NEVER include a done channel in channel_values.
5. ALWAYS use EXACT numbers from state — never invent values.
6. When CONVERGED (state C=True, S=True), call hv_equalization_done_channel IMMEDIATELY.
7. All "message" field values MUST be written in Korean (한국어) only. Never use Chinese characters (한자).
"""

    def _get_step_hint(self) -> str:
        tower = self.tower
        adc_known = self.state.get("last_adc_c") is not None
        suggest_pending = self.state.get("last_suggested_hv_c") is not None
        done_c = self.state.get("channel_done_c", False)
        done_s = self.state.get("channel_done_s", False)
        phase = self.state.get("phase", "idle")
        base = f"Phase: {phase} | Tower: {tower}"

        if self.state.get("last_hv_c") is None:
            if not self.state.get("y_confirmed"):
                return f"{base} | REQUIRED NEXT: position move message (step 1a)"
            else:
                return f"{base} | REQUIRED NEXT: hv_execute_tool status (step 1b — position confirmed)"
        elif adc_known and done_c and done_s:
            return f"{base} | CONVERGED → call hv_equalization_done_channel (step 1h)"
        elif adc_known and suggest_pending and self.state.get("approval_confirmed"):
            return f"{base} | REQUIRED NEXT: hv_execute_tool voltage (step 1f — user already confirmed)"
        elif adc_known and suggest_pending:
            return f"{base} | REQUIRED NEXT: approval message (step 1e)"
        elif self.state.get("needs_hv_confirm"):
            return f"{base} | REQUIRED NEXT: HV confirmation message (step 1g — voltage just applied, tell the user before next DAQ)"
        elif self.state.get("needs_plot_confirm"):
            return f"{base} | REQUIRED NEXT: plot confirmation message (step 1c-plot — DAQ done, send plot confirm before suggest)"
        elif self.state.get("needs_suggest"):
            return f"{base} | REQUIRED NEXT: hv_equalization_suggest (step 1d — plot confirmed, analyze now)"
        elif adc_known:
            return f"{base} | REQUIRED NEXT: daq_run_tool (step 1c)"
        else:
            return f"{base} | REQUIRED NEXT: daq_run_tool (step 1c — first DAQ)"

    def _build_state_context(self) -> str:
        lines = []
        tower = self.tower
        adc_known = self.state.get("last_adc_c") is not None
        suggest_pending = self.state.get("last_suggested_hv_c") is not None
        phase = self.state.get("phase", "idle")

        if self.state.get("last_hv_c") is None:
            if not self.state.get("y_confirmed"):
                lines.append(f"*** REQUIRED NEXT: position move message (step 1a) — ask user to move to {tower} position ***")
            else:
                lines.append(f"*** REQUIRED NEXT: hv_execute_tool status (step 1b) — position confirmed, check HV now ***")
            lines.append("")

        if self.state.get("needs_hv_confirm"):
            lines.append(f"*** REQUIRED NEXT: HV confirmation message (step 1g) — voltage was just applied, tell the user BEFORE the next DAQ ***")
            lines.append(f'*** output: {{"message": "{MSG_HV_CONFIRM}"}} ***')
            lines.append("")
        elif self.state.get("needs_plot_confirm"):
            lines.append(f"*** REQUIRED NEXT: plot confirmation message (step 1c-plot) — DAQ done, send plot confirm BEFORE suggest ***")
            lines.append(f'*** output: {{"message": "{MSG_PLOT_CONFIRM}"}} ***')
            lines.append("")
        elif self.state.get("needs_suggest"):
            lines.append(f"*** REQUIRED NEXT: hv_equalization_suggest (step 1d) — plot confirmed, analyze NOW ***")
            lines.append(f"*** DO NOT call daq_run_tool again — call hv_equalization_suggest first ***")
            lines.append("")
        elif adc_known:
            done_c = self.state.get("channel_done_c", False)
            done_s = self.state.get("channel_done_s", False)
            adc_c = self.state["last_adc_c"]
            adc_s = self.state["last_adc_s"]
            target = self.state.get("target_adc_c")
            if done_c and done_s:
                lines.append(f"*** CONVERGENCE: C=True, S=True — CALL hv_equalization_done_channel NOW ***")
            elif suggest_pending and self.state.get("approval_confirmed"):
                lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
                lines.append(f"*** REQUIRED NEXT: hv_execute_tool voltage (step 1f) — user confirmed ***")
            elif suggest_pending:
                lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
                lines.append(f"*** REQUIRED NEXT: approval message (step 1e) — if user gave manual HV values, update last_suggested_hv_c/s and re-send approval ***")
            else:
                lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
                lines.append(f"*** REQUIRED NEXT: daq_run_tool (step 1c) ***")
            lines.append("")

        lines.append(f"Phase: {phase}")
        lines.append(f"Tower: {tower} (x:{self.state['tower_pos']['x']:.3f}, y:{self.state['tower_pos']['y']:.3f})  [Position confirmed: {self.state.get('y_confirmed', False)}]")
        lines.append(f"Beam Energy: {self.state['beam_energy']} GeV")
        lines.append(f"Target Events: {self.state['target_events']}")
        lines.append(f"Target ADC: {self.state['target_adc_c']}")
        lines.append(f"Last HV: C={self.state.get('last_hv_c')}V, S={self.state.get('last_hv_s')}V")
        lines.append(f"needs_plot_confirm: {self.state.get('needs_plot_confirm', False)}")
        lines.append(f"needs_hv_confirm: {self.state.get('needs_hv_confirm', False)}")
        if self.state.get("last_suggested_hv_c") is not None:
            dc = self.state.get("channel_done_c", False)
            ds = self.state.get("channel_done_s", False)
            oc, nc = self.state.get("last_hv_c"), self.state["last_suggested_hv_c"]
            os_, ns = self.state.get("last_hv_s"), self.state["last_suggested_hv_s"]
            c_str = "C 완료(변경 없음)" if dc else f"C {oc:.0f}V→{nc}V"
            s_str = "S 완료(변경 없음)" if ds else f"S {os_:.0f}V→{ns}V"
            lines.append(f"HV 변경 제안 (현재→제안, 이 화살표를 그대로 승인 메시지에 복사): {c_str}, {s_str}")
        if self.state.get("last_run_number"):
            lines.append(f"Last Run Number: {self.state['last_run_number']}")
        lines.append(f"Iterations: {self.state.get('iterations', 0)}")
        return "\n".join(lines)

    def build_full_context(self, current_input: Optional[str] = None) -> str:
        if current_input is None and self.conversation_history:
            if self.conversation_history[-1]["role"] == "user":
                current_input = self.conversation_history[-1]["content"]
                temp_history = self.conversation_history[:-1]
            else:
                temp_history = self.conversation_history
        else:
            temp_history = self.conversation_history

        parts = []
        parts.append("=== Current State ===")
        parts.append(self._build_state_context())
        parts.append("")
        parts.append("=== Recent Conversation ===")
        history_lines = []
        if not temp_history:
            history_lines.append("(No conversation yet)")
        else:
            for msg in temp_history[-10:]:
                role = "User" if msg["role"] == "user" else "Agent"
                history_lines.append(f"{role}: {msg['content']}")
        parts.append("\n".join(history_lines))
        parts.append("")
        if current_input:
            parts.append("=== Current User Input ===")
            parts.append(current_input)
            parts.append("")
        parts.append("=== Your Task ===")
        parts.append(self._get_step_hint())
        parts.append("")
        parts.append("Output JSON with tool name and parameters.")
        return "\n".join(parts)

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        try:
            if tool_name == "none":
                return "no_tool_executed"

            elif tool_name == "daq_run_tool":
                self._apply_daq_params_from_state(
                    params,
                    events=self.state.get("target_events"),
                    beam_energy=self.state.get("beam_energy"),
                    program="HV Equalization",
                    pos=self._position_for_current_step(),
                )
                # DAQ 실행. daq_tool 내부의 dqm_session.start()이 monit --LIVE를 띄워
                # DAQ 동안 우측 하단 DQM 패널이 실시간 갱신된다 — 여기가 유일한 플롯 경로.
                result = self._run_tool_with_retry(
                    lambda: self.daq_tool.execute(params, line_callback=self.io.send_tool_output),
                    "daq_run_tool",
                )
                run_number = self._extract_run_number(result)
                if run_number:
                    self.state["last_run_number"] = run_number
                    self.state["iterations"] = self.state.get("iterations", 0) + 1
                    self.log(f"DAQ Run {run_number} 완료: {self.tower}, {params.get('events', 0)} events")
                self.state["needs_suggest"] = False       # suggest는 plot confirm 후
                self.state["needs_plot_confirm"] = True   # DAQ 후 먼저 plot 확인
                return result

            elif tool_name == "hv_execute_tool":
                cmd = params.get("command", "").lower()

                if cmd == "voltage":
                    if self.state.get("last_suggested_hv_c") is not None:
                        cv = {}
                        if not self.state.get("channel_done_c", False):
                            cv[f"{self.tower}C"] = self.state["last_suggested_hv_c"]
                        if not self.state.get("channel_done_s", False):
                            cv[f"{self.tower}S"] = self.state["last_suggested_hv_s"]
                        if cv:
                            self._apply_hv_voltage_params_from_state(params, cv)

                result = self._run_tool_with_retry(
                    lambda: self.hv_control_tool.execute(params),
                    "hv_execute_tool",
                )
                self.io.send_tool_output(result)

                if cmd == "status":
                    v_c, v_s = self._extract_voltages(result)
                    if v_c is not None:
                        self.state["last_hv_c"], self.state["last_hv_s"] = v_c, v_s
                        self.log(f"HV Status: C={v_c}V, S={v_s}V")
                elif cmd == "voltage":
                    if self.state.get("last_suggested_hv_c") is not None:
                        self.state["last_hv_c"] = self.state["last_suggested_hv_c"]
                        self.state["last_hv_s"] = self.state["last_suggested_hv_s"]
                    self.state["last_suggested_hv_c"] = None
                    self.state["last_suggested_hv_s"] = None
                    self.state["last_adc_c"] = None
                    self.state["last_adc_s"] = None
                    self.state["approval_confirmed"] = False  # 다음 승인 라운드용 리셋

                    self.io.send_tool_output(f"🔍 HV 적용 확인 중 ({self.tower})...")
                    try:
                        verify = self.hv_control_tool.execute({
                            "command": "status",
                            "channels": [f"{self.tower}C", f"{self.tower}S"],
                        })
                        self.io.send_tool_output(verify)
                    except RuntimeError as _ve:
                        verify = ""
                        self.io.send_tool_output(f"⚠️ HV 확인 실패 (전압 설정은 완료됨): {_ve}")
                    v_c, v_s = self._extract_voltages(verify)
                    if v_c is not None:
                        self.state["last_hv_c"] = v_c
                        self.state["last_hv_s"] = v_s
                        self.log(f"HV Verified: C={v_c}V, S={v_s}V")
                    # 전압 적용 직후엔 반드시 1g(MSG_HV_CONFIRM) 메시지를 사용자에게
                    # 보여준 다음에야 다음 DAQ로 넘어가게 강제한다 (_guard_tool/_guard_ai_message).
                    self.state["needs_hv_confirm"] = True
                return result

            elif tool_name == "hv_equalization_suggest":
                result_dict = self._run_tool_with_retry(self._do_suggest, "hv_equalization_suggest")
                self.state["needs_suggest"] = False  # suggest 완료
                self._emit_suggest_summary()
                self._emit_fitting_history()
                return json.dumps(result_dict, ensure_ascii=False) if isinstance(result_dict, dict) else str(result_dict)

            elif tool_name == "hv_equalization_done_channel":
                run_number = self.state.get("last_run_number", 0) or 0
                result = hv_equalization_done_channel.invoke(params) if hasattr(hv_equalization_done_channel, "invoke") else hv_equalization_done_channel(**params)
                try:
                    fit_result = generate_fitting_summary(
                        session_id="default", tower=self.tower, run_number=run_number
                    )
                    summary = (
                        f"── {self.tower} HV Equalization 완료 ──\n"
                        f"{fit_result['table']}\n"
                        f"Eq: {fit_result['equation']}"
                    )
                    itr = self.state.get("iterations", 0)
                    done_msg = f"{self.tower} HV Equalization 완료 ({itr}회 반복)"
                    self.io.send_tool_output(summary)
                    self.io.send_ai_message(done_msg)
                    if fit_result.get("plot_path"):
                        self.io.send_plots([fit_result["plot_path"]])
                except Exception as e:
                    self.log(f"완료 fitting summary 실패: {e}")
                self.state["done"] = True
                self.log(f"HV Equalization Done: {self.tower}")
                return result

            self.log(f"Unknown tool: {tool_name}")
            return f"Error: Unknown tool {tool_name}"
        except Exception as e:
            self.log(f"Tool 실행 오류 ({tool_name}): {str(e)}")
            return f"Error: {str(e)}"

    # ===== Suggest 처리 (ADC 측정만 sim이 오버라이드) =====

    def _measure_adc(self, hv_c: float, hv_s: float, run_number: int) -> Tuple[Optional[float], Optional[float]]:
        """실제 run 데이터에서 (adc_c, adc_s) peakADC 측정.
        HV Equalization Sim agent는 이 메서드만 오버라이드해 ADC를 시뮬레이션한다.
        나머지 워크플로우(suggest 계산/state 갱신/hv_execute/daq)는 전부 공유."""
        from tools.hv_equalization_tool import calculate_valley_cut_average
        avg_c, _ = calculate_valley_cut_average(run_number, "C", self.tower)
        avg_s, _ = calculate_valley_cut_average(run_number, "S", self.tower)
        return avg_c, avg_s

    def _do_suggest(self) -> Dict[str, Any]:
        """ADC 측정 → process_suggestion → state 갱신 (찐/ADC-sim 공통). 실패 시 RuntimeError."""
        from tools.hv_equalization_tool import _session_manager
        run_number = int(self.state.get("last_run_number") or 0)
        hv_c = float(self.state.get("last_hv_c") or 775.0)
        hv_s = float(self.state.get("last_hv_s") or 775.0)

        adc_c, adc_s = self._measure_adc(hv_c, hv_s, run_number)
        if adc_c is None or adc_s is None:
            raise RuntimeError(f"Run {run_number}에서 ADC 데이터를 가져올 수 없습니다.")

        result_dict = _session_manager.process_suggestion(
            "default", run_number, float(adc_c), float(adc_s), hv_c, hv_s
        )
        if result_dict.get("status") != "success":
            raise RuntimeError(result_dict.get("message", "hv_equalization_suggest 실패"))

        cur = result_dict.get("current", {})
        sug = result_dict.get("suggested", {})
        self.state["last_adc_c"] = cur.get("C", {}).get("adc", adc_c)
        self.state["last_adc_s"] = cur.get("S", {}).get("adc", adc_s)
        raw_hv_c = sug.get("C", {}).get("hv")
        raw_hv_s = sug.get("S", {}).get("hv")
        self.state["last_suggested_hv_c"] = int(round(raw_hv_c)) if raw_hv_c is not None else None
        self.state["last_suggested_hv_s"] = int(round(raw_hv_s)) if raw_hv_s is not None else None
        self.state["channel_done_c"] = bool(sug.get("C", {}).get("done", False))
        self.state["channel_done_s"] = bool(sug.get("S", {}).get("done", False))
        self.state["last_hv_c"] = round(hv_c, 1)
        self.state["last_hv_s"] = round(hv_s, 1)
        return result_dict

    def _emit_suggest_summary(self):
        adc_c = self.state.get("last_adc_c")
        adc_s = self.state.get("last_adc_s")
        adc_c_str = f"{adc_c:.1f}" if adc_c is not None else "N/A"
        adc_s_str = f"{adc_s:.1f}" if adc_s is not None else "N/A"
        summary = (
            f"🔬 HV Suggest — {self.tower} | "
            f"ADC: C={adc_c_str}, S={adc_s_str} | "
            f"HV 제안: C→{self.state.get('last_suggested_hv_c')}V, S→{self.state.get('last_suggested_hv_s')}V | "
            f"Done: C={self.state.get('channel_done_c')}, S={self.state.get('channel_done_s')}"
        )
        self.io.send_tool_output(summary)
        self.log(summary)

    def _emit_fitting_history(self):
        try:
            run_number = self.state.get("last_run_number", 0) or 0
            fit_result = generate_fitting_summary(
                session_id="default", tower=self.tower, run_number=run_number
            )
            self.io.send_tool_output(
                f"── HV Fitting History ({self.tower}) ──\n{fit_result['table']}\nEq: {fit_result['equation']}"
            )
            if fit_result.get("plot_path"):
                self.io.send_plots([fit_result["plot_path"]])
        except Exception as e:
            self.log(f"fitting summary 실패: {e}")

    def _extract_voltages(self, status_output: str) -> Tuple[Optional[float], Optional[float]]:
        # 채널명은 타워별 (M1T1C/M1T1S … M9T4C/M9T4S). status 출력의 "(<name>) ... V0Set = <v>" 형식에서 추출.
        t = re.escape(self.tower)
        match_c = re.search(rf"\({t}C\).*?V0Set\s*=\s*([\d.]+)", status_output, re.I)
        match_s = re.search(rf"\({t}S\).*?V0Set\s*=\s*([\d.]+)", status_output, re.I)
        return (
            float(match_c.group(1)) if match_c else None,
            float(match_s.group(1)) if match_s else None,
        )

    # Fields the LLM must not overwrite.
    # - Config values set at init: beam_energy, target_events, target_adc_*
    # - Hardware-read values (set by _execute_tool): last_adc_*, channel_done_*,
    #   last_hv_*, last_run_number
    # - Code-managed counters: iterations, done
    # - last_suggested_hv_*: _do_suggest()가 설정하고, 사용자의 수동 조정 요청은
    #   _on_user_input()이 _parse_manual_hv_adjustment로 결정론적으로 반영한다.
    #   LLM이 자유 텍스트에서 숫자를 잘못 읽어 여기 덮어쓰면 방금 코드가 반영한
    #   수동 조정값이 사라지므로(반영 안 되는 버그의 원인) 코드 소유로 보호한다.
    _PROTECTED_FIELDS = frozenset({
        "beam_energy", "target_events", "target_adc_c", "target_adc_s",
        "current_tower",
        "last_adc_c", "last_adc_s",
        "last_suggested_hv_c", "last_suggested_hv_s",
        "channel_done_c", "channel_done_s",
        "last_hv_c", "last_hv_s",
        "last_run_number",
        "iterations", "done",
        "needs_suggest", "needs_hv_confirm",
        "y_confirmed",  # 위치 확인은 코드 소유 (_on_user_input)
        "approval_confirmed",  # 승인 확인은 코드 소유 (_on_user_input)
    })

    def _update_state(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            if key in self._PROTECTED_FIELDS:
                self.log(f"WARNING: LLM tried to update protected field '{key}' = {value} — rejected")
            else:
                self.state[key] = value
                self.log(f"State updated: {key} = {value}")

    # ===== 공용 드라이버 hooks (run()은 BaseAgent에서 제공) =====

    def _print_banner(self):
        print(f"\n{'='*60}\n⚡ HV Equalization — {self.tower}\n{'='*60}")

    def _is_complete(self) -> bool:
        # 단일 타워 — done_channel 실행 시 state['done']=True (runner가 타워를 순회)
        return bool(self.state.get("done"))

    # 순수 확인 응답에 나타나는 단어들. 수동 HV 조정 입력은 항상 숫자를 포함하므로
    # (예: "C 850 S 850", "C 300 올리고 S는 850으로") 숫자가 있으면 확인이 아니다.
    _CONFIRM_WORDS = (
        "완료", "네", "예", "확인", "적용", "응", "그래", "좋아", "진행",
        "ok", "okay", "yes", "y", "apply", "confirm",
    )

    def _is_confirmation(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if any(ch.isdigit() for ch in t):
            return False  # 숫자 포함 → 수동 조정 요청
        return any(w in t for w in self._CONFIRM_WORDS)

    # 상대 조정 키워드. 절대값 지정("...으로", "...V")과 구분해 baseline(현재 제안값)에
    # 더하거나 뺀다.
    _REL_UP_WORDS = ("올려", "올림", "높여", "높임", "증가")
    _REL_DOWN_WORDS = ("내려", "내림", "낮춰", "낮춤", "감소")

    @staticmethod
    def _parse_manual_hv_adjustment(
        text: str, current_c: Optional[float], current_s: Optional[float]
    ) -> Dict[str, float]:
        """수동 HV 조정 자연어("c 850으로 s 900으로", "C를 800으로", "S 10 올려줘")를
        코드가 결정론적으로 파싱한다. LLM이 update_state로 이 값을 파싱/반영하는 걸
        신뢰하면 파인튜닝 모델이 자릿수를 틀리거나 아예 반영을 누락할 때 화면에
        수동 조정 이전 값이 그대로 남는다(사용자가 값을 바꿔도 반영 안 되는 버그의
        원인) — 그래서 이벤트 개수/승인 메시지와 동일하게 코드가 직접 파싱해 state를
        갱신하고, LLM의 update_state는 _PROTECTED_FIELDS로 차단한다.
        반환: 감지된 채널만 담은 {"C": <new_hv>, "S": <new_hv>}."""
        result: Dict[str, float] = {}
        # trailing은 lookahead(비소비)로 캡처한다 — 소비 그룹으로 두면 "c 850으로 s 900으로"처럼
        # 트레일링이 다음 채널 글자까지 먹어치워 두 번째 채널을 놓친다.
        for m in re.finditer(r'([cCsS])\D{0,8}?(\d+(?:\.\d+)?)(?=(\D{0,6}))', text or ""):
            ch = m.group(1).upper()
            val = float(m.group(2))
            trailing = m.group(3) or ""
            current = current_c if ch == "C" else current_s
            if current is not None and any(w in trailing for w in HVEqualizationAgent._REL_UP_WORDS):
                result[ch] = current + val
            elif current is not None and any(w in trailing for w in HVEqualizationAgent._REL_DOWN_WORDS):
                result[ch] = current - val
            else:
                result[ch] = val  # "...으로", "...V" 등 → 절대값 지정
        return result

    def _on_user_input(self, user_input: str):
        # 이동 확인
        if (not self.state.get("y_confirmed")
                and self.state.get("last_hv_c") is None):
            self.state["y_confirmed"] = True
            self.log("Position confirmed by user")
            return
        # 전압 적용(1f) 후 확인 메시지(1g)에 대한 응답 → 다음 DAQ로 진행
        if self.state.get("needs_hv_confirm"):
            self.state["needs_hv_confirm"] = False
            self.log("HV 변경 확인됨 → 다음 DAQ로 진행 (step 1c)")
            return
        # DAQ 후 plot 확인 → suggest 단계로 전환
        if self.state.get("needs_plot_confirm"):
            self.state["needs_plot_confirm"] = False
            self.state["needs_suggest"] = True
            self.log("Plot confirmed → proceed to hv_equalization_suggest")
            return
        # 승인 단계(제안 존재 + ADC 측정됨): 사용자가 '완료'로 승인했는지,
        # 아니면 수동 HV 조정을 요청했는지 판정한다.
        # - 확인(완료 등) → approval_confirmed=True → step 1f(voltage 적용)
        # - 수동 조정(숫자 포함) → approval_confirmed=False → step 1e 재진입해
        #   LLM이 last_suggested_hv_c/s를 갱신하고 승인 메시지를 재전송하게 둔다.
        if (self.state.get("last_suggested_hv_c") is not None
                and self.state.get("last_adc_c") is not None):
            if self._is_confirmation(user_input):
                self.state["approval_confirmed"] = True
                self.log("HV 승인 확인됨 → voltage 적용 (step 1f)")
            else:
                self.state["approval_confirmed"] = False
                manual = self._parse_manual_hv_adjustment(
                    user_input,
                    self.state.get("last_suggested_hv_c"),
                    self.state.get("last_suggested_hv_s"),
                )
                if "C" in manual and not self.state.get("channel_done_c", False):
                    self.state["last_suggested_hv_c"] = manual["C"]
                if "S" in manual and not self.state.get("channel_done_s", False):
                    self.state["last_suggested_hv_s"] = manual["S"]
                self.log(f"승인 단계 수동 조정 요청 감지: '{user_input}' → 파싱 결과 {manual} → 승인 메시지 재전송 (step 1e)")
            return

    def _guard_tool(self, tool_name: str, decision: Dict[str, Any]) -> Optional[str]:
        # 전압 적용(1f) 직후엔 확인 메시지(1g)를 먼저 보여줘야 한다 — 이게 없으면
        # 모델이 확인 메시지 없이 바로 다음 daq_run_tool을 호출해도 아무도 못 막아서
        # 사용자에게 "전압이 변경되었습니다" 안내가 전혀 출력되지 않는다.
        if self.state.get("needs_hv_confirm"):
            return (
                f"needs_hv_confirm=True — send HV confirmation message first: "
                f'{{"message": "{MSG_HV_CONFIRM}"}}'
            )
        # plot confirm 필요 시 DAQ/suggest/hv 차단
        if self.state.get("needs_plot_confirm"):
            return (
                f"needs_plot_confirm=True — send plot confirmation message first: "
                f'{{"message": "{MSG_PLOT_CONFIRM}"}}'
            )
        # done_channel은 두 채널 모두 수렴한 경우에만 허용
        if tool_name == "hv_equalization_done_channel":
            done_c = self.state.get("channel_done_c", False)
            done_s = self.state.get("channel_done_s", False)
            if not (done_c and done_s):
                return (f"수렴 미완료 (C={done_c}, S={done_s}). "
                        f"승인 메시지(step 1e)를 먼저 출력하세요.")
        # voltage 적용(1f)은 사용자가 승인 메시지(1e)에 실제로 '완료'한 뒤에만 허용.
        # 이게 없으면 hv_equalization_suggest 직후 모델이 승인 메시지를 아예 건너뛰고
        # 곧장 전압을 적용해버려도 아무도 못 막는다(사용자에게 "적용하시겠습니까?"가
        # 전혀 안 뜨는 버그의 원인).
        if tool_name == "hv_execute_tool":
            cmd = (decision.get("params") or {}).get("command", "").lower()
            if cmd == "voltage" and not self.state.get("approval_confirmed"):
                return (
                    "approval_confirmed=False — voltage를 적용하기 전에 승인 메시지(step 1e)를 "
                    "먼저 출력하고 사용자가 '완료'로 확인할 때까지 기다리세요. "
                    f"{self._get_step_hint()}"
                )
        return None

    def _guard_ai_message(self, message: str) -> Optional[str]:
        # 전압 적용 직후엔 확인 메시지(1g)만 유효하다 — 승인 메시지 canonicalization
        # (아래 _finalize_ai_message)보다 먼저 걸려야 하므로 canonical=None인 이 구간에서
        # 별도로 강제한다.
        if self.state.get("needs_hv_confirm") and message != MSG_HV_CONFIRM:
            return (f"needs_hv_confirm=True — the ONLY valid message now is the HV "
                    f'confirmation: {{"message": "{MSG_HV_CONFIRM}"}}. Do not send any other '
                    f"message (e.g. jumping straight to the next DAQ). {self._get_step_hint()}")
        # DAQ 직후엔 plot 확인 메시지만 유효하다. 모델이 가끔 한 단계 되돌아가 다른
        # 메시지(예: 위치 이동)를 다시 내보내는 걸 막지 않으면 "가끔 이상한 메시지가
        # 뜬다"는 증상으로 그대로 사용자에게 노출된다.
        if self.state.get("needs_plot_confirm") and message != MSG_PLOT_CONFIRM:
            return (f"needs_plot_confirm=True — the ONLY valid message now is the plot "
                    f'confirmation: {{"message": "{MSG_PLOT_CONFIRM}"}}. Do not send any other '
                    f"message (e.g. a position move message). {self._get_step_hint()}")
        if MSG_PLOT_CONFIRM in message and not self.state.get("needs_plot_confirm"):
            return f"needs_plot_confirm=False — DO NOT send plot confirmation before DAQ runs. {self._get_step_hint()}"
        return None

    def _build_approval_message(self) -> Optional[str]:
        """승인 메시지(step 1e)를 state로부터 결정론적으로 생성한다.
        실제 적용 전압은 state.last_suggested_hv_*에서 나오므로(LLM 출력 무시),
        화면 메시지도 반드시 state와 일치해야 '보이는 값 = 적용되는 값'이 보장된다.
        LLM이 자주 C/S done 라벨을 뒤바꾸거나 done 채널을 x→x 화살표로 렌더링하는
        문제를 근본 차단한다. 승인 단계가 아니면 None."""
        adc_c = self.state.get("last_adc_c")
        adc_s = self.state.get("last_adc_s")
        nc = self.state.get("last_suggested_hv_c")
        ns = self.state.get("last_suggested_hv_s")
        if adc_c is None or nc is None:
            return None
        t = self.tower
        dc = self.state.get("channel_done_c", False)
        ds = self.state.get("channel_done_s", False)
        if dc and ds:
            return None  # 둘 다 완료 → 승인 메시지 없음 (done_channel로)
        oc = self.state.get("last_hv_c")
        os_ = self.state.get("last_hv_s")
        target = self.state.get("target_adc_c")
        c_part = f"{t}C 완료(변경 없음)" if dc else f"{t}C {oc:.0f}V→{nc}V"
        s_part = f"{t}S 완료(변경 없음)" if ds else f"{t}S {os_:.0f}V→{ns}V"
        if not dc and not ds:
            adc_part = f"{t}C={adc_c:.1f}, {t}S={adc_s:.1f}"
        elif dc:  # C 완료, S만 조정
            adc_part = f"{t}S={adc_s:.1f}"
        else:     # S 완료, C만 조정
            adc_part = f"{t}C={adc_c:.1f}"
        return (f"분석 결과, 현재 ADC: {adc_part} (목표: {target}). "
                f"HV 변경 제안: {c_part}, {s_part}. 적용하시겠습니까?")

    def _finalize_ai_message(self, message: str) -> str:
        # 승인 메시지는 LLM 텍스트를 신뢰하지 않고 state 기준으로 재구성한다.
        # 이전엔 message에 "적용하시겠습니까" 문자열이 정확히 들어있을 때만 교정했는데,
        # 반복된 승인 라운드에서 모델이 그 문구를 깨뜨리거나 다른 형식으로 출력하면
        # 교정 없이 그대로 사용자에게 노출돼 "출력이 이상해서 확인이 안 되는" 문제가
        # 생겼다. _build_approval_message()가 None이 아니라는 것 자체가 지금이
        # 승인 단계(step 1e)라는 뜻이므로, 문자열 매칭 대신 그 여부로 판단한다.
        canonical = self._build_approval_message()
        if canonical is not None:
            if canonical != message:
                self.log(f"승인 메시지 보정(state 기준): {message!r} → {canonical!r}")
            return canonical
        return super()._finalize_ai_message(message)
