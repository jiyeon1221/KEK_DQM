#!/usr/bin/env python3
"""Calibration Scan Agent — 모든 타워(T1-T9)를 돌며 데이터 수집 자동화"""

import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from tools.daq_tool import DAQRunTool
import tools.motor_control_tool as motor

from .base_agent import BaseAgent
sys.path.append(str(Path(__file__).parent.parent))
from config import AGENT_MODELS


class CalibScanAgent(BaseAgent):
    def __init__(
        self,
        beam_energy: Optional[float] = None,
        target_events: Optional[int] = None,
        daq_config: str = "setup",
        use_base_model: bool = True,
        io_handler=None,
    ):
        model_config = AGENT_MODELS["calibration"]
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
            agent_name="Calibration",
            io_handler=io_handler,
        )
        
        self.daq_tool = DAQRunTool()
        
        # Zigzag 순서
        self.tower_order = ["T1", "T2", "T3", "T6", "T5", "T4", "T7", "T8", "T9"]
        
        # Calibration은 rot=0, tilt=0 고정
        self.tower_positions = {}
        from tools.position_calculator_tool import get_calculator
        calc = get_calculator()
        for tower in self.tower_order:
            pos = calc.calculate_tower_position(tower, rotation=0.0, tilting=0.0)
            if pos:
                self.tower_positions[tower] = pos
            else:
                print(f"⚠️  Warning: {tower} 위치 계산 실패")

        self.state = {
            "phase": "config" if beam_energy is None or target_events is None else "idle",
            "beam_energy": beam_energy,
            "target_events": target_events,
            "daq_config": daq_config,
            
            "tower_order": self.tower_order,
            "current_tower_idx": 0,
            
            "tower_status": {
                tower: {
                    "collected_events": 0,
                    "runs": [],
                    "completed": False,
                    "completed_at": None,
                    "x_moved": False,
                    "y_confirmed": False,
                }
                for tower in self.tower_order
            },
            
            "start_time": datetime.now().isoformat(),
            "plot_method": "PeakADC"
        }
        
        self.log(f"Calibration Scan Agent 초기화: Energy={beam_energy}, Events={target_events}")

    def _get_system_prompt(self) -> str:
        """System prompt (workflow 정의 - EnergyScanAgent와 통일)"""
        return """You are Calibration Scan Agent for test beam experiments.

Follow these steps EXACTLY:

=== STEP 0: Configuration ===
- If 'beam_energy' is null: Request energy from user.
  Output: {"message": "에너지를 입력하세요."}
- After energy is provided: Update 'beam_energy' and request event count.
  CRITICAL: beam_energy is in GeV. Store the number exactly as user inputs. (e.g. user inputs "2" → beam_energy: 2, NOT 2000)
  Output: {"message": "이벤트를 몇개 받을까요?", "update_state": {"beam_energy": <number in GeV>, "phase": "config_events"}}
- After event count is provided: Update 'target_events' and set phase to 'idle'.
  Output: {"tool": "none", "update_state": {"target_events": <number>, "phase": "idle"}}

=== STEP 1: For Each Tower in tower_order (REPEAT for T1-T9) ===
Repeat steps 1a-1c for each tower in tower_order until all towers are completed.

1a-i. Move X-axis automatically (no user input needed):
  Output: {"tool": "motor_x_move_tool", "params": {"x": <x from state>}}

1a-ii. After motor tool completes, ask user to move Y-axis manually:
  Output: {"message": "X축 자동 이동 완료 (<x> mm). Y축을 <y>으로 이동해주세요.", "update_state": {"current_tower": "<tower>"}}
  (Replace <x>, <y>, <tower> with values from state)

After user says "완료":
CRITICAL: When you see "완료" in conversation history, proceed to next step immediately.
1b. Execute DAQ
Tool: "daq_run_tool"
Params: {
    "events": <target_events from state>,
    "config": "setup",
    "pos_h": <x>,
    "pos_v": <y>,
    "pos_rot": 0.0,
    "pos_tilt": 0.0,
    "beam_energy": <beam_energy from state>
}
(Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

1c. Request Plot Confirmation
Output: {"message": "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."}

After user says "완료":
CRITICAL: When you see "완료" in conversation history, you MUST mark current tower as completed.
Output: {"tool": "none", "update_state": {"tower_status": {"{tower}": {"completed": true}}}}
(Note: replace {tower} with the CURRENT tower name shown in state. Do NOT include current_tower_idx.)

=== STEP 2: Completion ===
When ALL towers are completed, output:
{"message": "모든 타워에 대한 스캔이 완료되었습니다."}

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip or reorder steps.
2. Step 1a-i ALWAYS comes before 1a-ii for every tower.
3. Output JSON format (CHOOSE ONE):
   - {"tool": "...", "params": {...}, "update_state": {...}}
   - {"message": "...", "update_state": {...}}
   - {"tool": "none", "update_state": {...}}
4. When you see "완료" in conversation history, process it IMMEDIATELY.
"""

    def _get_step_hint(self) -> str:
        """현재 상태 요약 - AI가 학습을 통해 다음 단계를 스스로 결정"""
        phase = self.state.get("phase", "config")
        tower_idx = self.state.get("current_tower_idx", 0)
        total = len(self.tower_order)
        if tower_idx < total:
            tower = self.tower_order[tower_idx]
            status = self.state["tower_status"].get(tower, {})
            if not status.get("x_moved"):
                return f"Phase: {phase} | Tower: {tower} ({tower_idx+1}/{total}) | NEXT: motor_x_move_tool (step 1a-i)"
            elif not status.get("y_confirmed"):
                return f"Phase: {phase} | Tower: {tower} ({tower_idx+1}/{total}) | NEXT: Y-axis move message (step 1a-ii)"
            elif not status.get("runs"):
                return f"Phase: {phase} | Tower: {tower} ({tower_idx+1}/{total}) | NEXT: daq_run_tool (step 1b)"
            return f"Phase: {phase} | Tower: {tower} ({tower_idx+1}/{total})"
        return f"Phase: {phase} | All towers completed"

    # Fields that can never be overwritten by the LLM under any circumstances.
    _ALWAYS_PROTECTED = frozenset({
        "tower_order", "start_time", "plot_method", "daq_config",
    })
    # Fields that become read-only once set (not None/0).
    # During STEP 0 config phase they're None → LLM is allowed to initialise them.
    # After STEP 0 they hold real values → any further LLM change is rejected.
    _ONCE_SET_PROTECTED = frozenset({"beam_energy", "target_events"})

    def _update_state(self, updates: Dict[str, Any]):
        """State 업데이트"""
        for key, value in updates.items():
            if key == "tower_status" and isinstance(value, dict):
                for t, v in value.items():
                    if t in self.state["tower_status"]:
                        current_runs = self.state["tower_status"][t].get("runs", [])
                        # Don't allow completed=True if no runs have been recorded
                        if v.get("completed") and not current_runs:
                            safe_v = {k: val for k, val in v.items() if k != "completed"}
                            self.state["tower_status"][t].update(safe_v)
                            self.log(f"WARNING: {t} completion rejected (no runs) — ignored")
                        else:
                            self.state["tower_status"][t].update(v)
                        self.log(f"State updated: tower_status[{t}] = {v}")
                completed_count = sum(
                    1 for s in self.state["tower_status"].values() if s.get("completed")
                )
                self.state["current_tower_idx"] = completed_count
                self.log(f"current_tower_idx 자동 갱신: {completed_count}")
            elif key == "current_tower_idx":
                # tower_status 기반 자동 관리 — LLM 직접 설정 무시
                self.log(f"current_tower_idx 직접 설정 무시 (tower_status 기반 자동 관리)")
            elif key in self._ALWAYS_PROTECTED:
                self.log(f"WARNING: LLM tried to update protected field '{key}' = {value} — rejected")
            elif key in self._ONCE_SET_PROTECTED and self.state.get(key) is not None:
                # Already initialised → reject mid-scan changes
                self.log(f"WARNING: LLM tried to overwrite already-set '{key}' = {value} — rejected")
            else:
                self.state[key] = value
                self.log(f"State updated: {key} = {value}")

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """Tool 실행"""
        if tool_name == "none":
            return "no_tool_executed"

        elif tool_name == "motor_x_move_tool":
            x = float(params.get("x", 0))
            self.io.send_tool_output(f"[Motor] X축 이동 시작: {x:.3f} mm")
            def _do_move():
                ok, msg = motor.move_x(x)
                if not ok:
                    raise RuntimeError(msg)
                return msg
            result = self._run_tool_with_retry(_do_move, "motor_x_move_tool")
            self.io.send_tool_output(f"[Motor] {result}")
            tower = self.tower_order[self.state["current_tower_idx"]]
            self.state["tower_status"][tower]["x_moved"] = True
            return result

        elif tool_name == "daq_run_tool":
            # Override events with configured target (don't trust LLM's value)
            if self.state.get('target_events') is not None:
                params['events'] = self.state['target_events']

            params['program'] = "Calibration"

            # DAQ 실행. daq_tool 내부에서 dqm_session.start()이 monit --LIVE를 띄워
            # DAQ 동안 우측 하단 DQM 패널이 실시간 갱신된다 — 여기가 유일한 플롯 경로.
            result = self._run_tool_with_retry(
                lambda: self.daq_tool.execute(params, line_callback=self.io.send_tool_output),
                "daq_run_tool",
            )

            run_number = self._extract_run_number(result)
            if run_number:
                self.state['last_run_number'] = run_number
                tower = self.tower_order[self.state['current_tower_idx']]
                self.state['tower_status'][tower]['runs'].append(run_number)
                self.state['tower_status'][tower]['collected_events'] = params.get('events', 0)
                self.log(f"DAQ Run {run_number} 완료: {tower} 타워, {params.get('events', 0)} events")
            return result

        return f"Error: Unknown tool {tool_name}"

    def _extract_run_number(self, daq_output: str = None) -> Optional[int]:
        """Run number 추출"""
        try:
            from tools.config_loader import get_path_config
            runnum_file = Path(get_path_config("RunNumberFile"))
            if runnum_file.exists():
                with open(runnum_file, 'r') as f:
                    val = f.read().strip()
                    return int(val) - 1
        except: pass
        if daq_output:
            import re
            match = re.search(r'Run:?\s*(\d+)', daq_output)
            if match: return int(match.group(1))
        return None

    def _format_progress(self) -> str:
        """현재 진행 상황을 문자열로 반환 (AI 메시지용)"""
        done = sum(1 for s in self.state['tower_status'].values() if s.get('completed'))
        total = len(self.tower_order)
        lines = [
            f"📊 Calibration Scan  —  {done} / {total} 타워 완료",
            "─" * 36,
        ]
        for i, tower in enumerate(self.tower_order):
            status = self.state['tower_status'][tower]
            if status['completed']:
                runs_str = ', '.join(str(r) for r in status['runs']) if status['runs'] else '-'
                lines.append(f"  ✅  {tower}   Run {runs_str}")
            else:
                lines.append(f"       {tower}")
        lines.append("─" * 36)
        return "\n".join(lines)

    def _print_summary(self):
        """현재 진행 상황 요약 출력 (CLI용)"""
        print(f"\n📊 Calibration Progress Summary:")
        print("-" * 70)
        print(f"Energy: {self.state['beam_energy']} GeV | Target: {self.state['target_events']} events/tower")
        print("-" * 70)
        for i, tower in enumerate(self.tower_order):
            status = self.state['tower_status'][tower]
            pos = self.tower_positions.get(tower, {'x': 0, 'y': 0})
            mark = "✅" if status['completed'] else ("➡️ " if i == self.state['current_tower_idx'] else "  ")
            run_info = f" (Runs: {status['runs']})" if status['runs'] else ""
            print(f"  {mark} {tower} (x:{pos['x']:.1f}, y:{pos['y']:.1f}): {'Completed' if status['completed'] else 'Pending'}{run_info}")
        print("-" * 70)

    def run(self):
        """에이전트 메인 루프 (EnergyScanAgent와 통일)"""
        print(f"\n{'='*70}")
        print(f"⚡ Calibration Scan Agent Started")
        print(f"{'='*70}")
        self.log("Calibration Scan 시작")

        while True:
            try:
                self._print_summary()
                context = self.build_full_context()
                decision = self.decide(context)
                
                print(f"\n🔍 Agent Decision:")
                print(json.dumps(decision, indent=2, ensure_ascii=False))
                
                if "error" in decision:
                    print(f"\n❌ Agent Error: {decision['error']}")
                    self.log(f"Error: {decision}")
                    break

                if "update_state" in decision:
                    before = sum(1 for s in self.state["tower_status"].values() if s.get("completed"))
                    self._update_state(decision["update_state"])
                    after = sum(1 for s in self.state["tower_status"].values() if s.get("completed"))
                    if after > before:
                        self.io.send_ai_message(self._format_progress())

                message = decision.get("message")
                tool_name = decision.get("tool")

                if message:
                    self.io.send_ai_message(message)
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    if "모든 타워에 대한 스캔이 완료되었습니다" in message:
                        self.log("모든 타워 스캔 완료 - 자동 종료")
                        break
                    user_input = self.io.get_input()
                    if user_input in ["종료", "exit"]: break
                    if "이동해주세요" in message and user_input.strip() == "완료":
                        tower_idx = self.state["current_tower_idx"]
                        if tower_idx < len(self.tower_order):
                            tower = self.tower_order[tower_idx]
                            self.state["tower_status"][tower]["y_confirmed"] = True
                            self.log(f"Y-axis confirmed for {tower}")
                    self.add_to_history("user", user_input)
                    continue

                if tool_name and tool_name != "none":
                    params = decision.get("params", {})
                    print(f"\n🤖 Executing Tool: {tool_name}")
                    result = self._execute_tool(tool_name, params)
                    print(f"📝 Tool Result: {result[:200]}...")
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    continue
                
                if "update_state" in decision:
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    continue

                if all(s['completed'] for s in self.state['tower_status'].values()):
                    print("\n✅ 모든 타워 스캔이 완료되었습니다.")
                    break
                
                break
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
                break

    def build_full_context(self, current_input: Optional[str] = None) -> str:
        """전체 context 생성 (EnergyScanAgent와 통일)"""
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
                if content == "완료" or content.strip() == "완료":
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

    def _build_state_context(self) -> str:
        """State를 문자열로 변환 (EnergyScanAgent와 통일)"""
        lines = []
        lines.append(f"Phase: {self.state['phase']}")
        lines.append(f"Beam Energy: {self.state['beam_energy']} GeV")
        lines.append(f"Target Events: {self.state['target_events']}")
        lines.append("")
        lines.append("Tower Progress:")
        for i, tower in enumerate(self.tower_order):
            status = self.state['tower_status'][tower]
            pos = self.tower_positions.get(tower, {'x': 0, 'y': 0})
            if status['completed']:
                lines.append(f"  ✅ {tower} (x:{pos['x']:.1f}, y:{pos['y']:.1f}): Completed (Runs: {status['runs']})")
            elif i == self.state['current_tower_idx']:
                x_tag = " [X moved]" if status.get("x_moved") else ""
                y_tag = " [Y confirmed - proceed to DAQ]" if status.get("y_confirmed") else ""
                lines.append(f"  ➡️  {tower} (x:{pos['x']:.1f}, y:{pos['y']:.1f}): Pending{x_tag}{y_tag}  <- CURRENT (target: {self.state['target_events']} events)")
            else:
                lines.append(f"     {tower} (x:{pos['x']:.1f}, y:{pos['y']:.1f}): Pending")
        return "\n".join(lines)
