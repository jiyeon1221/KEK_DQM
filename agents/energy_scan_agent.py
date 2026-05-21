#!/usr/bin/env python3
"""Energy Scan Agent — 다양한 빔 에너지에서 데이터 수집 자동화"""

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


class EnergyScanAgent(BaseAgent):
    def __init__(
        self,
        energy_config: Dict[float, int],
        tower: str = "T5",
        position: Optional[Dict[str, float]] = None,
        daq_config: str = "setup",
        use_base_model: bool = True,  # Fine-tuning 전에는 base model 사용
        io_handler=None,
    ):
        model_config = AGENT_MODELS["energy_scan"]
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
            agent_name="EnergyScan",
            io_handler=io_handler,
        )
        
        self._init_energy_config = energy_config if energy_config else {}
        self.daq_tool = DAQRunTool()
        
        from tools.position_calculator_tool import get_calculator
        t5_pos = get_calculator().calculate_tower_position("T5", rotation=1.5, tilting=1.0)
        self.t5_x = t5_pos['x']
        self.t5_y = t5_pos['y']
        
        self.state = {
            "phase": "config" if not self._init_energy_config else "idle",
            "tower": tower,
            "position": position,
            "daq_config": daq_config,
            
            "energy_config": {
                energy: {
                    "target_events": events,
                    "collected_events": 0,
                    "runs": [],
                    "completed": False,
                    "completed_at": None
                }
                for energy, events in self._init_energy_config.items()
            },
            
            "scan_order": sorted(list(self._init_energy_config.keys())),
            "current_energy": None,
            "current_energy_idx": 0,
            
            "start_time": datetime.now().isoformat(),
            "plot_method": "PeakADC",
            "plot_max_event": None,
            "x_moved": False,
            "y_confirmed": False,
        }
        
        self.log(f"Energy Scan Agent 초기화: {list(self._init_energy_config.keys())} GeV")
    
    # ===== System Prompt =====
    
    def _get_system_prompt(self) -> str:
        """System prompt (workflow 정의)"""
        return """You are Energy Scan Agent for test beam experiments.

Follow these steps EXACTLY:

=== STEP 0: Get Energy Config (only when phase is "config") ===
0a. Ask user for energy settings:
  {"message": "에너지 설정을 입력해주세요.\n예) 1GeV 100개 2GeV 200개  또는  1GeV 2000 3GeV 3000"}

After user responds, parse their input:
0b. Update state with parsed config:
  {"tool": "none", "update_state": {"energy_config": {<energy_int>: {"target_events": <n>, "collected_events": 0, "runs": [], "completed": false, "completed_at": null}, ...}, "scan_order": [<sorted ints>], "phase": "idle"}}
  CRITICAL: energy keys must be INTEGERS (e.g., 1, 2, 3). scan_order must be sorted ascending.
  CRITICAL: beam_energy in GeV → store as integer (e.g. "2GeV" → 2, NOT 2000)
  CRITICAL: If user says "모두", "각각", or "씩" with one number (e.g., "모두 500개"), apply that number to ALL energies.

=== STEP 1: Move to T5 ===
CRITICAL RULE: After STEP 0, when phase is "idle" and energy_config is NOT empty, start STEP 1.
DO NOT repeat STEP 0. DO NOT skip to phase "scanning".

1a-i. Move X-axis automatically (no user input needed):
  Output: {"tool": "motor_x_move_tool", "params": {"x": <x from state>}}

1a-ii. After motor tool completes, ask user to move Y-axis manually:
  Output: {"message": "X축 자동 이동 완료 (<x> mm). Y축을 <y>으로 이동해주세요."}
  (Replace <x>, <y> with T5 Position values from state)

After user says "완료":
CRITICAL: When you see "완료" in conversation history, you MUST NOT repeat the same message.
You MUST proceed to next step immediately by outputting:
{"tool": "none", "update_state": {"phase": "scanning"}}
DO NOT output the same message again.

Then go to STEP 2.

=== STEP 2: For Each Energy in scan_order (REPEAT for ALL energies) ===
Repeat steps 2a-2c for each energy in scan_order until all energies are completed.

2a. Request Energy Setting
Output: {"message": "빔 에너지를 {energy} GeV로 설정해주세요."} (replace {energy} with number, e.g., "빔 에너지를 10 GeV로 설정해주세요.")

After user says "완료":
CRITICAL: When you see "완료" in conversation history, proceed to next step immediately.
2b. Execute DAQ
Tool: "daq_run_tool"
Params: {
    "events": <target_events from energy_config>,
    "config": "setup",
    "pos_h": <x_from_state>,
    "pos_v": <y_from_state>,
    "pos_rot": 1.5,
    "pos_tilt": 1.0,
    "beam_energy": <energy>
}
(Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

2c. Request Plot Confirmation
Output: {"message": "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."}

After user says "완료":
CRITICAL: When you see "완료" in conversation history, you MUST mark current energy as completed.
Output: {"tool": "none", "update_state": {"energy_config": {"{energy}": {"completed": true}}}}
Then proceed to next energy or STEP 3.

=== STEP 3: Completion ===
When ALL energies in energy_config have "completed": true, you MUST output:
{"message": "모든 에너지 스캔이 완료되었습니다."}

After this message, if user provides NEW energy settings, you MUST reset and go back to STEP 0.
If user says "종료", the session ends.

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip or reorder steps.
2. Use EXACT messages above. DO NOT change or paraphrase.
3. Output JSON format (CHOOSE ONE, NEVER BOTH):
   - {"tool": "...", "params": {...}, "update_state": {...}}  (for tool execution)
   - {"message": "...", "update_state": {...}}  (for user message)
   - {"tool": "none", "update_state": {...}}  (for internal state update only)
   CRITICAL: NEVER output both "tool" and "message" in the same JSON. NEVER put "message" inside "update_state".
4. Use energy_config[energy].target_events for DAQ events
5. STEP TRANSITION RULES (MOST CRITICAL):
   - phase="config", no history → output STEP 0a (ask message). DO NOT skip to parse.
   - phase="config", user just answered → output STEP 0b (parse + update_state). DO NOT ask again.
   - After STEP 0b (energy_config parsed, phase="idle"): You MUST go to STEP 1 (motor_x_move_tool). DO NOT repeat STEP 0.
   - After STEP 1a-i (motor done): Send Y-axis message. After user "완료", go to STEP 2.
   - NEVER skip STEP 1. NEVER output STEP 0 decision twice in a row.
6. MOST IMPORTANT: When you see "완료" in conversation history, you MUST process it IMMEDIATELY:
   - Y-axis "완료" → Output {"tool": "none", "update_state": {"phase": "scanning"}} and proceed to STEP 2
   - Energy setting "완료" → Execute DAQ immediately ({"tool": "daq_run_tool", ...})
   - Plot confirmation "완료" → Update energy status to "completed": true.
7. AFTER COMPLETION: If all energies are done and you have sent the completion message, you are ready for a new task or exit.
"""
    
    # ===== State Context =====
    
    def _build_state_context(self) -> str:
        """State를 문자열로 변환"""
        lines = []
        lines.append(f"Phase: {self.state['phase']}")
        lines.append(f"Tower: {self.state['tower']}")
        lines.append(f"T5 Position: x={self.t5_x:.1f}, y={self.t5_y:.1f}, rot=1.5, tilt=1.0")
        lines.append(f"x_moved: {self.state.get('x_moved', False)}")
        lines.append(f"y_confirmed: {self.state.get('y_confirmed', False)}")
        if self.state['position']:
            lines.append(f"Position: {self.state['position']}")
        lines.append("")

        ec = self.state.get("energy_config", {})
        so = self.state.get("scan_order", [])
        if ec:
            lines.append("Energy Config:")
            for e in so:
                cfg = ec.get(e, {})
                status = "✅" if cfg.get("completed") else "➡️" if e == self.state.get("current_energy") else "  "
                lines.append(
                    f"  {status} {e} GeV: target={cfg.get('target_events', '?')} "
                    f"collected={cfg.get('collected_events', 0)} "
                    f"runs={cfg.get('runs', [])} completed={cfg.get('completed', False)}"
                )

        return "\n".join(lines)
    
    def _get_step_hint(self) -> str:
        """현재 상태 요약 - AI가 학습을 통해 다음 단계를 스스로 결정"""
        phase = self.state.get("phase", "config")
        if phase == "config":
            if self.conversation_history:
                return "Phase: config | REQUIRED NEXT: parse user input and update state (step 0b)"
            return "Phase: config | REQUIRED NEXT: ask for energy settings (step 0a)"
        if phase == "idle":
            if not self.state.get("x_moved"):
                return f"Phase: idle | NEXT: motor_x_move_tool (step 1a-i, x={self.t5_x:.1f})"
            elif not self.state.get("y_confirmed"):
                return f"Phase: idle | NEXT: Y-axis move message (step 1a-ii, y={self.t5_y:.1f})"
            else:
                return f"Phase: idle | Y-axis CONFIRMED — NEXT: set phase=scanning (step 1a-ii complete)"
        current_energy = self.state.get("current_energy")
        scan_order = self.state.get("scan_order", [])
        idx = scan_order.index(current_energy) + 1 if current_energy in scan_order else 0
        total = len(scan_order)
        return f"Phase: {phase} | Energy: {current_energy} GeV ({idx}/{total})"

    def build_full_context(self, current_input: Optional[str] = None) -> str:
        """전체 context 생성 (단계별 힌트 포함)"""
        
        # data_gen과 형식 통일: 마지막 user 메시지는 current_input으로 분리
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
            recent_history = temp_history[-10:]
            for msg in recent_history:
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
    
    # ===== Main Loop =====
    
    def _print_summary(self):
        """현재 진행 상황 요약 출력 (Dash보드 스타일)"""
        print(f"\n📊 Energy Scan Progress Summary:")
        print("-" * 70)
        tower = self.state.get('tower', 'T5')
        print(f"Tower: {tower} | Position: x={self.t5_x:.1f}, y={self.t5_y:.1f}")
        print("-" * 70)
        
        for energy in self.state['scan_order']:
            if energy is None: continue
            config = self.state['energy_config'][energy]
            completed = config.get('completed', False)
            collected = config.get('collected_events', 0)
            target = config.get('target_events', 0)
            mark = "✅" if completed else ("➡️ " if energy == self.state['current_energy'] else "  ")
            status_text = "Completed" if completed else f"{collected}/{target} events"
            run_info = f" (Runs: {config['runs']})" if config['runs'] else ""
            print(f"  {mark} {energy} GeV: {status_text}{run_info}")
        print("-" * 70)

    def run(self):
        """
        Energy Scan 실행 (대화형)
        """
        print(f"\n{'='*70}")
        print(f"⚡ Energy Scan Agent Started")
        print(f"{'='*70}")
        
        self.log("Energy Scan 시작")
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Use 'with agent:' statement.")
        
        while True:
            try:
                self._print_summary()

                # data_gen과 동작 일치: scanning 시작 시 다음 미완료 에너지로 자동 전환
                if self.state.get('phase') == 'scanning':
                    _cur = self.state.get('current_energy')
                    _cur_done = (
                        _cur is not None and
                        self.state['energy_config'].get(_cur, {}).get('completed', False)
                    )
                    if _cur is None or _cur_done:
                        for _e in self.state['scan_order']:
                            if not self.state['energy_config'][_e].get('completed', False):
                                self.state['current_energy'] = _e
                                self.state['current_energy_idx'] = self.state['scan_order'].index(_e)
                                break

                context = self.build_full_context()
                decision = self.decide(context)
                
                print(f"\n🔍 Agent Decision:")
                print(json.dumps(decision, indent=2, ensure_ascii=False))
                
                if "error" in decision:
                    print(f"\n❌ Agent Error: {decision['error']}")
                    self.log(f"Error: {decision}")
                    break
                
                if "update_state" in decision:
                    before = sum(1 for c in self.state["energy_config"].values() if c.get("completed"))
                    self._update_state(decision["update_state"])
                    after = sum(1 for c in self.state["energy_config"].values() if c.get("completed"))
                    if after > before:
                        self.io.send_ai_message(self._format_progress())

                message = decision.get("message")
                tool_name = decision.get("tool")

                if message:
                    self.io.send_ai_message(message)
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    if "모든 에너지 스캔이 완료되었습니다" in message:
                        self.log("모든 에너지 스캔 완료 - 자동 종료")
                        break
                    user_input = self.io.get_input()
                    if user_input in ["종료", "exit"]: break
                    if "이동해주세요" in message and user_input.strip() == "완료":
                        self.state["y_confirmed"] = True
                        self.log("Y-axis confirmed for T5")
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

                if all(c.get("completed", False) for c in self.state["energy_config"].values()) and self.state["energy_config"]:
                    print("\n✅ 모든 에너지 스캔이 완료되었습니다.")
                    break

                break
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}")
                break

    # ===== Tool 실행 =====
    
    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """Tool 실행"""
        print(f"\n🤖 Agent Decision:")
        print(f"   Tool: {tool_name}")
        print(f"   Params: {json.dumps(params, ensure_ascii=False)}")
        print()
        
        if tool_name == "none":
            return "no_tool_executed"

        elif tool_name == "motor_x_move_tool":
            x = float(params.get("x", self.t5_x))
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
            # Determine the correct energy (don't blindly trust LLM's beam_energy)
            energy_key = params.get('beam_energy')
            if energy_key is not None:
                try:
                    energy_key = int(float(energy_key))
                except Exception:
                    energy_key = None
            # Reject if LLM gave an energy not in our config
            if energy_key is not None and energy_key not in self.state['energy_config']:
                self.log(f"WARNING: LLM beam_energy={energy_key} not in config — falling back")
                energy_key = None
            if energy_key is None:
                energy_key = self.state.get('current_energy')
            if energy_key is None:
                for e in self.state['scan_order']:
                    if not self.state['energy_config'][e].get('completed', False):
                        energy_key = e
                        break

            # Override both events AND beam_energy with authoritative Python values
            if energy_key is not None and energy_key in self.state['energy_config']:
                params['events'] = self.state['energy_config'][energy_key]['target_events']
                params['beam_energy'] = energy_key

            params['program'] = "EM Scan"

            # DAQ 실행. daq_tool 내부의 dqm_session.start()이 monit --LIVE를 띄워서
            # DAQ 동안 우측 하단 DQM 패널이 실시간 갱신된다 — 여기가 유일한 플롯 경로.
            result = self._run_tool_with_retry(
                lambda: self.daq_tool.execute(params, line_callback=self.io.send_tool_output),
                "daq_run_tool",
            )

            run_number = self._extract_run_number(result)
            if run_number:
                self.state['last_run_number'] = run_number
                if energy_key is not None and energy_key in self.state['energy_config']:
                    self.state['current_energy'] = energy_key
                    self.state['energy_config'][energy_key]['runs'].append(run_number)
                    self.state['energy_config'][energy_key]['collected_events'] = params.get('events', 0)
                    self.log(f"DAQ Run {run_number} 완료: {energy_key} GeV, {params.get('events', 0)} events")

            return result

        else:
            print(f"⚠️  Unknown tool: {tool_name}")
            self.log(f"Unknown tool: {tool_name}")
            return f"Error: Unknown tool {tool_name}"
    
    # ===== Helper 함수 =====
    
    # Fields the LLM must not overwrite (set once by Python at init)
    _PROTECTED_FIELDS = frozenset({
        "tower", "daq_config", "start_time", "plot_method", "plot_max_event",
    })

    def _update_state(self, updates: Dict[str, Any]):
        """State 업데이트 (energy_config는 deep merge로 기존 필드 보존)"""
        for key, value in updates.items():
            if key == "energy_config" and isinstance(value, dict):
                for energy_key, config_value in value.items():
                    try:
                        int_key = int(energy_key)
                    except:
                        int_key = energy_key
                    if int_key in self.state['energy_config'] and isinstance(config_value, dict):
                        # target_events는 LLM이 변경 불가 (Python 파서가 설정)
                        safe_update = {k: v for k, v in config_value.items() if k != 'target_events'}
                        # completed=True는 실제 런이 기록된 경우에만 허용
                        current_runs = self.state['energy_config'][int_key].get('runs', [])
                        if safe_update.get('completed') and not current_runs:
                            safe_update.pop('completed', None)
                            self.log(f"WARNING: LLM tried to mark {int_key} GeV complete without runs — rejected")
                        self.state['energy_config'][int_key].update(safe_update)
                        if self.state['energy_config'][int_key].get('completed') and not self.state['energy_config'][int_key].get('completed_at'):
                            self.state['energy_config'][int_key]['completed_at'] = datetime.now().strftime("%H:%M:%S")
                    else:
                        self.state['energy_config'][int_key] = config_value
                self.state['scan_order'] = sorted(
                    [e for e in self.state['energy_config'].keys() if e is not None]
                )
                self.log(f"State updated: energy_config (deep merge), scan_order={self.state['scan_order']}")

            elif key == "current_energy" and value is not None:
                try:
                    self.state[key] = int(value)
                except:
                    self.state[key] = value
                self.log(f"State updated: {key} = {self.state[key]}")

            elif key in self._PROTECTED_FIELDS:
                self.log(f"WARNING: LLM tried to update protected field '{key}' = {value} — rejected")
            else:
                self.state[key] = value
                self.log(f"State updated: {key} = {value}")
    
    def _extract_run_number(self, daq_output: str = None) -> Optional[int]:
        """Run number 추출 (runnum.txt → fallback: DAQ output 파싱)"""
        try:
            from tools.config_loader import get_path_config
            runnum_file = Path(get_path_config("RunNumberFile"))
            if runnum_file.exists():
                with open(runnum_file, 'r') as f:
                    # Run이 종료된 후 runnum.txt가 다음 번호로 업데이트되므로, 
                    # 방금 종료된 Run 정보를 위해 -1을 수행함
                    val = f.read().strip()
                    run_number = int(val) - 1
                    return run_number
        except Exception as e:
            self.log(f"Run number 읽기 실패: {e}")
        
        if daq_output:
            import re
            match = re.search(r'Run:?\s*(\d+)', daq_output)
            if match:
                return int(match.group(1))
        
        return None
    
    def _format_progress(self) -> str:
        """현재 진행 상황을 문자열로 반환 (AI 메시지용)"""
        total = len(self.state['scan_order'])
        done = sum(1 for c in self.state['energy_config'].values() if c.get('completed'))
        lines = [
            f"📊 Energy Scan  —  {done} / {total} 완료",
            f"T5 위치:  x = {self.t5_x:.1f},  y = {self.t5_y:.1f}",
            "─" * 36,
        ]
        for energy in self.state['scan_order']:
            if energy is None:
                continue
            config = self.state['energy_config'].get(energy, {})
            if config.get('completed'):
                runs_str = ', '.join(str(r) for r in config['runs']) if config['runs'] else '-'
                lines.append(f"  ✅  {energy} GeV   {config['target_events']} events   Run {runs_str}")
            else:
                lines.append(f"       {energy} GeV   {config.get('target_events',0)} events")
        lines.append("─" * 36)
        return "\n".join(lines)

