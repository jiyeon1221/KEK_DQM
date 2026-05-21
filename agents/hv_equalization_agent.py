#!/usr/bin/env python3
"""
HV Equalization Agent
단일 타워의 HV Equalization 수행. calib_scan_agent 구조를 그대로 따름.
컨트롤러(hv_equalization_scan.py)가 타워를 순서대로 호출함.
"""

import json
import sys
import re
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from tools.daq_tool import DAQRunTool
from tools.hv_control_tool import HVControlTool
import tools.motor_control_tool as motor
from tools.position_calculator_tool import calculate_position
from tools.hv_equalization_tool import (
    hv_equalization_suggest,
    hv_equalization_done_channel,
    generate_fitting_summary,
)

from .base_agent import BaseAgent
sys.path.append(str(Path(__file__).parent.parent))
from config import AGENT_MODELS


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
            "x_moved": False,
            "y_confirmed": False,
            "needs_suggest": False,
        }
        self.log(f"Agent 초기화: {tower}, E={beam_energy}GeV, Events={target_events}, Target ADC={target_adc}")

    def _get_system_prompt(self) -> str:
        t = self.tower
        x = self.tower_pos["x"]
        y = self.tower_pos["y"]
        return f"""You are HV Equalization Agent for tower {t} (x:{x:.1f}, y:{y:.1f}).
Your task: adjust HV for {t}C and {t}S channels to reach the target peakADC value.

Follow these steps EXACTLY:

=== Workflow for {t} ===
1a-i. Move X-axis automatically (no user input needed):
  {{"tool": "motor_x_move_tool", "params": {{"x": {x:.1f}}}}}

1a-ii. After motor completes, request Y-axis from user:
  {{"message": "X축 자동 이동 완료 ({x:.1f} mm). Y축을 {y:.1f}으로 이동해주세요."}}

After user says "완료":
1b. Check HV status:
  {{"tool": "hv_execute_tool", "params": {{"command": "status", "channels": ["{t}C", "{t}S"]}}}}

[INNER LOOP — repeat 1c→1g until CONVERGED]
1c. Execute DAQ:
  {{"tool": "daq_run_tool", "params": {{"events": <events>, "pos_h": <x>, "pos_v": <y>, "beam_energy": <energy>}}}}
  (Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

1d. Suggest HV:
  {{"tool": "hv_equalization_suggest", "params": {{"run_number": <run>, "tower": "{t}"}}}}

1e. Ask approval (only NOT-done channels):
  Both not done: {{"message": "분석 결과, 현재 ADC: {t}C=<adc_c>, {t}S=<adc_s> (목표: <target>). HV 변경 제안: {t}C <old_c>V→<new_c>V, {t}S <old_s>V→<new_s>V. 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  Only C not done: {{"message": "분석 결과, 현재 ADC: {t}C=<adc_c> (목표: <target>). HV 변경 제안: {t}C <old_c>V→<new_c>V. ({t}S 완료) 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  Only S not done: {{"message": "분석 결과, 현재 ADC: {t}S=<adc_s> (목표: <target>). HV 변경 제안: {t}S <old_s>V→<new_s>V. ({t}C 완료) 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  CRITICAL: Use EXACT values from state — NEVER fabricate numbers.

After user says "완료":
1f. Apply voltage (only NOT-done channels):
  {{"tool": "hv_execute_tool", "params": {{"command": "voltage", "channel_values": {{"{t}C": <new_c>, "{t}S": <new_s>}}}}, "update_state": {{"phase": "equalizing"}}}}
  NEVER include a done channel in channel_values.

1g. Confirmation:
  {{"message": "HV 전압이 변경되었습니다. 결과를 확인하고 '완료'를 입력하면 다음 DAQ를 시작합니다."}}

After user says "완료":
  - State shows NOT CONVERGED → back to step 1c
  - State shows CONVERGED (C=True, S=True) → proceed to step 1h

1h. Done:
  {{"tool": "hv_equalization_done_channel", "params": {{"channels": "all"}}}}

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip Step 1e (Approval).
2. Step 1a-i ALWAYS comes before 1a-ii.
3. Output JSON ONLY. No natural language.
4. NEVER include a done channel in channel_values.
5. ALWAYS use EXACT numbers from state — never invent values.
6. When CONVERGED (state C=True, S=True), call hv_equalization_done_channel IMMEDIATELY.
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
            if not self.state.get("x_moved"):
                return f"{base} | REQUIRED NEXT: motor_x_move_tool (step 1a-i)"
            elif not self.state.get("y_confirmed"):
                return f"{base} | REQUIRED NEXT: Y-axis move message (step 1a-ii)"
            else:
                return f"{base} | REQUIRED NEXT: hv_execute_tool status (step 1b — Y-axis confirmed)"
        elif adc_known and done_c and done_s:
            return f"{base} | CONVERGED → call hv_equalization_done_channel (step 1h)"
        elif adc_known and suggest_pending and phase == "approving":
            return f"{base} | REQUIRED NEXT: hv_execute_tool voltage (step 1f — user already confirmed)"
        elif adc_known and suggest_pending:
            return f"{base} | REQUIRED NEXT: approval message (step 1e)"
        elif self.state.get("needs_suggest"):
            return f"{base} | REQUIRED NEXT: hv_equalization_suggest (step 1d — DAQ done, analyze now)"
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
            if not self.state.get("x_moved"):
                lines.append(f"*** REQUIRED NEXT: motor_x_move_tool (step 1a-i) — auto-move X to {tower} ***")
            elif not self.state.get("y_confirmed"):
                lines.append(f"*** REQUIRED NEXT: Y-axis move message (step 1a-ii) — ask user to move Y to {tower} ***")
            else:
                lines.append(f"*** REQUIRED NEXT: hv_execute_tool status (step 1b) — Y-axis confirmed, check HV now ***")
            lines.append("")

        if self.state.get("needs_suggest"):
            lines.append(f"*** REQUIRED NEXT: hv_equalization_suggest (step 1d) — DAQ run {self.state.get('last_run_number')} complete, analyze NOW ***")
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
            elif suggest_pending and phase == "approving":
                lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
                if done_c:
                    lines.append(f"*** C=DONE: NEVER include {tower}C in channel_values ***")
                if done_s:
                    lines.append(f"*** S=DONE: NEVER include {tower}S in channel_values ***")
                lines.append(f"*** REQUIRED NEXT: hv_execute_tool voltage (step 1f) ***")
            elif suggest_pending:
                lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
                if done_c:
                    lines.append(f"*** C=DONE: NEVER include {tower}C in channel_values ***")
                if done_s:
                    lines.append(f"*** S=DONE: NEVER include {tower}S in channel_values ***")
                lines.append(f"*** REQUIRED NEXT: approval message (step 1e) ***")
            else:
                lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
                if done_c:
                    lines.append(f"*** C=DONE: NEVER include {tower}C in channel_values ***")
                if done_s:
                    lines.append(f"*** S=DONE: NEVER include {tower}S in channel_values ***")
                lines.append(f"*** REQUIRED NEXT: daq_run_tool (step 1c) ***")
            lines.append("")

        lines.append(f"Phase: {phase}")
        lines.append(f"Tower: {tower} (x:{self.state['tower_pos']['x']:.1f}, y:{self.state['tower_pos']['y']:.1f})  [X moved: {self.state.get('x_moved', False)}]")
        lines.append(f"Beam Energy: {self.state['beam_energy']} GeV")
        lines.append(f"Target Events: {self.state['target_events']}")
        lines.append(f"Target ADC: {self.state['target_adc_c']}")
        lines.append(f"Last HV: C={self.state.get('last_hv_c')}V, S={self.state.get('last_hv_s')}V")
        if self.state.get("last_suggested_hv_c") is not None:
            dc = self.state.get("channel_done_c", False)
            ds = self.state.get("channel_done_s", False)
            c_str = f"C={self.state['last_suggested_hv_c']}V" + (" [DONE-skip]" if dc else "")
            s_str = f"S={self.state['last_suggested_hv_s']}V" + (" [DONE-skip]" if ds else "")
            lines.append(f"Suggested HV: {c_str}, {s_str}  <- use EXACT values in approval message")
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
                content = msg["content"]
                if content.strip() == "완료":
                    history_lines.append(f"{role}: 완료 [IMPORTANT: User confirmed completion]")
                else:
                    history_lines.append(f"{role}: {content}")
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

            elif tool_name == "motor_x_move_tool":
                x = float(params.get("x", self.tower_pos["x"]))
                self.io.send_tool_output(f"[Motor] X축 이동 시작: {x:.3f} mm")
                def _do_move():
                    ok, msg = motor.move_x(x)
                    if not ok:
                        raise RuntimeError(msg)
                    return msg
                result = self._run_tool_with_retry(_do_move, "motor_x_move_tool")
                self.io.send_tool_output(f"[Motor] {result}")
                self.state["x_moved"] = True
                return result

            elif tool_name == "daq_run_tool":
                if self.state.get("target_events") is not None:
                    params["events"] = self.state["target_events"]
                params["program"] = "HV Equalization"
                # Calibration과 동일하게 위치/자세 정보를 항상 함께 기록한다.
                params.setdefault("pos_h", self.tower_pos["x"])
                params.setdefault("pos_v", self.tower_pos["y"])
                params.setdefault("pos_rot", 0.0)
                params.setdefault("pos_tilt", 0.0)
                params.setdefault("beam_energy", self.state.get("beam_energy", ""))
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
                self.state["needs_suggest"] = True  # DAQ 후 반드시 suggest 호출
                return result

            elif tool_name == "hv_execute_tool":
                cmd = params.get("command", "").lower()

                if cmd == "voltage":
                    # LLM 출력 대신 state의 last_suggested 값으로 override
                    if self.state.get("last_suggested_hv_c") is not None:
                        cv = {}
                        if not self.state.get("channel_done_c", False):
                            cv[f"{self.tower}C"] = self.state["last_suggested_hv_c"]
                        if not self.state.get("channel_done_s", False):
                            cv[f"{self.tower}S"] = self.state["last_suggested_hv_s"]
                        if cv:
                            params["channel_values"] = cv

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
                return result

            elif tool_name == "hv_equalization_suggest":
                params.setdefault("tower", self.tower)
                if self.state.get("last_run_number"):
                    params.setdefault("run_number", self.state["last_run_number"])
                # 하드웨어에서 읽은 실제 HV 값을 전달 (세션 초기값 800 사용 방지)
                if self.state.get("last_hv_c") is not None:
                    params.setdefault("hv_c", self.state["last_hv_c"])
                if self.state.get("last_hv_s") is not None:
                    params.setdefault("hv_s", self.state["last_hv_s"])
                def _call_suggest():
                    res = hv_equalization_suggest.invoke(params) if hasattr(hv_equalization_suggest, "invoke") else hv_equalization_suggest(**params)
                    r = json.loads(res) if isinstance(res, str) else res
                    if isinstance(r, dict) and r.get("status") == "error":
                        raise RuntimeError(r.get("message", "hv_equalization_suggest 실패"))
                    return res
                result = self._run_tool_with_retry(_call_suggest, "hv_equalization_suggest")
                self.state["needs_suggest"] = False  # suggest 완료
                try:
                    r = json.loads(result) if isinstance(result, str) else result
                    cur = r.get("current", {})
                    sug = r.get("suggested", {})
                    self.state["last_adc_c"] = cur.get("C", {}).get("adc")
                    self.state["last_adc_s"] = cur.get("S", {}).get("adc")
                    raw_hv_c = sug.get("C", {}).get("hv")
                    raw_hv_s = sug.get("S", {}).get("hv")
                    self.state["last_suggested_hv_c"] = int(round(raw_hv_c)) if raw_hv_c is not None else None
                    self.state["last_suggested_hv_s"] = int(round(raw_hv_s)) if raw_hv_s is not None else None
                    self.state["channel_done_c"] = bool(sug.get("C", {}).get("done", False))
                    self.state["channel_done_s"] = bool(sug.get("S", {}).get("done", False))
                    adc_c = self.state["last_adc_c"]
                    adc_s = self.state["last_adc_s"]
                    summary = (
                        f"🔬 HV Suggest — {self.tower} | "
                        f"ADC: C={adc_c:.1f if adc_c else 'N/A'}, S={adc_s:.1f if adc_s else 'N/A'} | "
                        f"HV 제안: C→{self.state['last_suggested_hv_c']}V, S→{self.state['last_suggested_hv_s']}V | "
                        f"Done: C={self.state['channel_done_c']}, S={self.state['channel_done_s']}"
                    )
                    self.io.send_tool_output(summary)
                    self.log(summary)
                except Exception:
                    self.io.send_tool_output(str(result)[:400])
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
                return result

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

    def _parse_hv_modify(self, text: str) -> Optional[dict]:
        """
        Parse relative/absolute HV modification requests.
        Returns {"C": delta, "S": delta} for relative or {"C_abs": v, "S_abs": v} for absolute.
        Returns None if text doesn't look like a modification.
        """
        t = text.strip()
        has_dir = bool(re.search(r'올려|올리|내려|내리', t))
        has_abs = bool(re.search(r'[CS]\s*=\s*\d', t, re.I))
        if not (has_dir or has_abs):
            return None

        result: dict = {}

        # "모두 N 올려/내려" or just "N 올려" (no channel → both)
        all_m = re.search(r'(?:모두|둘\s*다|all)?\s*(\d+(?:\.\d+)?)\s*(올려|올리|내려|내리)', t)
        explicit_c = re.search(r'[Cc]\s+(\d+(?:\.\d+)?)\s*(올려|올리|내려|내리)', t)
        explicit_s = re.search(r'[Ss]\s+(\d+(?:\.\d+)?)\s*(올려|올리|내려|내리)', t)

        if explicit_c or explicit_s:
            if explicit_c:
                v = float(explicit_c.group(1))
                result["C"] = v if '올' in explicit_c.group(2) else -v
            if explicit_s:
                v = float(explicit_s.group(1))
                result["S"] = v if '올' in explicit_s.group(2) else -v
        elif all_m:
            v = float(all_m.group(1))
            delta = v if '올' in all_m.group(2) else -v
            result["C"] = delta
            result["S"] = delta

        # Absolute: "C=790" / "S=800"
        for ch, key in (('[Cc]', 'C_abs'), ('[Ss]', 'S_abs')):
            m = re.search(rf'{ch}\s*=\s*(\d+(?:\.\d+)?)', t)
            if m:
                result[key] = float(m.group(1))

        return result if result else None

    def _build_approval_message(self) -> str:
        t = self.tower
        adc_c = self.state.get("last_adc_c")
        adc_s = self.state.get("last_adc_s")
        target = self.state.get("target_adc_c")
        hv_c = self.state.get("last_hv_c") or 0
        hv_s = self.state.get("last_hv_s") or 0
        sug_c = self.state.get("last_suggested_hv_c")
        sug_s = self.state.get("last_suggested_hv_s")
        done_c = self.state.get("channel_done_c", False)
        done_s = self.state.get("channel_done_s", False)
        if not done_c and not done_s:
            return (
                f"분석 결과, 현재 ADC: {t}C={adc_c:.1f}, {t}S={adc_s:.1f} (목표: {target}). "
                f"HV 변경 제안: {t}C {hv_c:.0f}V→{sug_c}V, {t}S {hv_s:.0f}V→{sug_s}V. 적용하시겠습니까?"
            )
        elif not done_c:
            return (
                f"분석 결과, 현재 ADC: {t}C={adc_c:.1f} (목표: {target}). "
                f"HV 변경 제안: {t}C {hv_c:.0f}V→{sug_c}V. ({t}S 완료) 적용하시겠습니까?"
            )
        else:
            return (
                f"분석 결과, 현재 ADC: {t}S={adc_s:.1f} (목표: {target}). "
                f"HV 변경 제안: {t}S {hv_s:.0f}V→{sug_s}V. ({t}C 완료) 적용하시겠습니까?"
            )

    def _extract_voltages(self, status_output: str) -> Tuple[Optional[float], Optional[float]]:
        tower = self.tower
        match_c = re.search(rf"\({tower}C\).*?V0Set\s*=\s*([\d.]+)", status_output, re.I)
        match_s = re.search(rf"\({tower}S\).*?V0Set\s*=\s*([\d.]+)", status_output, re.I)
        return (
            float(match_c.group(1)) if match_c else None,
            float(match_s.group(1)) if match_s else None,
        )

    def _extract_run_number(self, daq_output: str = None) -> Optional[int]:
        try:
            from tools.config_loader import get_path_config
            runnum_file = Path(get_path_config("RunNumberFile"))
            if runnum_file.exists():
                with open(runnum_file, "r") as f:
                    return int(f.read().strip()) - 1
        except Exception as e:
            self.log(f"Run number 읽기 실패: {e}")
        if daq_output:
            match = re.search(r"Run:?\s*(\d+)", daq_output)
            if match:
                return int(match.group(1))
        return None

    # Fields the LLM must not overwrite.
    # - Config values set at init: beam_energy, target_events, target_adc_*
    # - Hardware-read values (set by _execute_tool): last_adc_*, last_suggested_hv_*,
    #   channel_done_*, last_hv_*, last_run_number
    # - Code-managed counters: iterations, done
    _PROTECTED_FIELDS = frozenset({
        "beam_energy", "target_events", "target_adc_c", "target_adc_s",
        "current_tower",
        "last_adc_c", "last_adc_s",
        "last_suggested_hv_c", "last_suggested_hv_s",
        "channel_done_c", "channel_done_s",
        "last_hv_c", "last_hv_s",
        "last_run_number",
        "iterations", "done",
        "needs_suggest",
    })

    def _update_state(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            if key in self._PROTECTED_FIELDS:
                self.log(f"WARNING: LLM tried to update protected field '{key}' = {value} — rejected")
            else:
                self.state[key] = value
                self.log(f"State updated: {key} = {value}")

    def run(self):
        print(f"\n{'='*60}")
        print(f"⚡ HV Equalization — {self.tower}")
        print(f"{'='*60}")
        self.log(f"Tower {self.tower} 시작")

        _error_count = 0
        _MAX_ERRORS = 3

        while True:
            try:
                context = self.build_full_context()
                decision = self.decide(context)
                print(f"\n🔍 Agent Decision:")
                print(json.dumps(decision, indent=2, ensure_ascii=False))

                if "error" in decision:
                    _error_count += 1
                    self.log(f"Agent error ({_error_count}/{_MAX_ERRORS}): {decision['error']}")
                    if _error_count >= _MAX_ERRORS:
                        print(f"\n❌ 연속 오류 {_MAX_ERRORS}회 — 종료합니다.")
                        break
                    self.add_to_history("user", "Output valid JSON only. No other text.")
                    continue

                _error_count = 0

                if "update_state" in decision:
                    self._update_state(decision["update_state"])

                message = decision.get("message")
                tool_name = decision.get("tool")

                if message:
                    # Approval messages must show exact state values — LLM can hallucinate
                    # the suggested HV (especially on first iteration with low starting HV).
                    if "적용하시겠습니까" in message and self.state.get("last_suggested_hv_c") is not None:
                        message = self._build_approval_message()
                    self.io.send_ai_message(message)
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    user_input = self.io.get_input()
                    if user_input in ["종료", "exit"]:
                        break

                    # HV approval: handle modify requests before proceeding
                    if "적용하시겠습니까" in message:
                        while True:
                            modify = self._parse_hv_modify(user_input)
                            if modify is None:
                                break  # "완료" or unrecognized → proceed normally
                            # Apply modifications: delta는 현재 HV(last_hv) 기준, 절댓값은 그대로
                            if "C" in modify and not self.state.get("channel_done_c"):
                                self.state["last_suggested_hv_c"] = int(round(
                                    (self.state["last_hv_c"] or 0) + modify["C"]
                                ))
                            if "C_abs" in modify and not self.state.get("channel_done_c"):
                                self.state["last_suggested_hv_c"] = int(round(modify["C_abs"]))
                            if "S" in modify and not self.state.get("channel_done_s"):
                                self.state["last_suggested_hv_s"] = int(round(
                                    (self.state["last_hv_s"] or 0) + modify["S"]
                                ))
                            if "S_abs" in modify and not self.state.get("channel_done_s"):
                                self.state["last_suggested_hv_s"] = int(round(modify["S_abs"]))

                            self.add_to_history("user", user_input)
                            updated_msg = self._build_approval_message()
                            self.add_to_history("assistant", json.dumps(
                                {"message": updated_msg}, ensure_ascii=False
                            ))
                            self.io.send_ai_message(updated_msg)
                            user_input = self.io.get_input()
                            if user_input in ["종료", "exit"]:
                                return

                    if "이동해주세요" in message and user_input.strip() == "완료":
                        self.state["y_confirmed"] = True
                        self.log("Y-axis confirmed")
                    self.add_to_history("user", user_input)
                    continue

                if tool_name and tool_name != "none":
                    # 가드: done_channel은 두 채널 모두 수렴한 경우에만 허용
                    if tool_name == "hv_equalization_done_channel":
                        done_c, done_s = self.state.get("channel_done_c", False), self.state.get("channel_done_s", False)
                        if not (done_c and done_s):
                            self.log(f"done_channel 조기 호출 차단: C={done_c}, S={done_s}")
                            self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                            self.add_to_history("user",
                                f"수렴 미완료 (C={done_c}, S={done_s}). "
                                f"승인 메시지(step 1e)를 먼저 출력하세요.")
                            continue

                    print(f"\n🤖 Executing Tool: {tool_name}")
                    result = self._execute_tool(tool_name, decision.get("params", {}))
                    print(f"📝 Tool Result: {str(result)[:200]}...")
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    if self.state.get("done"):
                        break
                    continue

                if "update_state" in decision:
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    continue

                _error_count += 1
                self.log(f"Unrecognized decision ({_error_count}/{_MAX_ERRORS}): {decision}")
                if _error_count >= _MAX_ERRORS:
                    print(f"\n❌ 연속 인식 불가 응답 {_MAX_ERRORS}회 — 종료합니다.")
                    break
                self.add_to_history("user", "Output valid JSON only. No other text.")
                continue

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                break
