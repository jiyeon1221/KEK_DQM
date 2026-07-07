#!/usr/bin/env python3
"""Energy Scan Agent — 다양한 빔 에너지에서 데이터 수집 자동화"""

import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from tools.daq_tool import DAQRunTool

from .base_agent import BaseAgent
sys.path.append(str(Path(__file__).parent.parent))
from config import AGENT_MODELS, MSG_PLOT_CONFIRM


class EnergyScanAgent(BaseAgent):
    def __init__(
        self,
        energy_config: Dict[float, int],
        tower: str = "M5T3",
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
        t5_pos = get_calculator().calculate_tower_position("M5T3", rotation=1.5, tilting=1.0)
        self.t5_x = t5_pos['x']
        self.t5_y = t5_pos['y']
        
        self.state = {
            "phase": "config" if not self._init_energy_config else "idle",
            "tower": tower,
            "position": position,
            "daq_config": daq_config,
            "t5_x": self.t5_x,
            "t5_y": self.t5_y,

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
            "y_confirmed": False,
            "needs_plot_confirm": False,
        }
        
        self.log(f"Energy Scan Agent 초기화: {list(self._init_energy_config.keys())} GeV")
    
    # ===== System Prompt =====
    
    def _get_system_prompt(self) -> str:
        """System prompt (workflow 정의)"""
        return """You are Energy Scan Agent for test beam experiments.

Follow these steps EXACTLY:

=== STEP 0: Get Energy Config (only when phase is "config") ===
0a. Ask user for energy settings:
  {"message": "에너지 설정을 입력해주세요.\n예) 1GeV 50000개 2GeV 200000개 5GeV 100000개  또는  1GeV 80000 3GeV 500000 5GeV 300000"}

After user responds, parse their input:
0b. Update state with parsed config:
  {"tool": "none", "update_state": {"energy_config": {<energy_int>: {"target_events": <n>, "collected_events": 0, "runs": [], "completed": false, "completed_at": null}, ...}, "scan_order": [<sorted ints>], "phase": "idle"}}
  CRITICAL: energy keys must be INTEGERS (e.g., 1, 2, 3). scan_order must be sorted ascending.
  CRITICAL: beam_energy in GeV → store as number (integer if whole: "2GeV" → 2; float if decimal: "2.5GeV" → 2.5). NEVER convert to MeV.
  CRITICAL: If user says "모두", "각각", or "씩" with one number (e.g., "모두 500개"), apply that number to ALL energies.

=== STEP 1: Move to M5T3 ===
CRITICAL RULE: After STEP 0, when phase is "idle" and energy_config is NOT empty, start STEP 1.
DO NOT repeat STEP 0. DO NOT skip to phase "scanning".

1a. Ask user to move to M5T3 position:
  Output: {"message": "x = <x> mm, y = <y> mm 으로 이동해주세요."}
  (Replace <x>, <y> with M5T3 Position values from state)

After user says "완료":
The SYSTEM marks position confirmed and switches to scanning automatically — you do NOT output any state update.
Just proceed to STEP 2 (the step hint will say "set-beam message").

=== STEP 2: For Each Energy in scan_order (REPEAT for ALL energies) ===
Repeat steps 2a-2c for each energy in scan_order until all energies are completed.

2a. Request Energy Setting
Output: {"message": "빔 에너지를 {energy} GeV로 설정해주세요."} (replace {energy} with number, e.g., "빔 에너지를 10 GeV로 설정해주세요.")

After user says "완료":
2b. Execute DAQ immediately:
Tool: "daq_run_tool"
Params: {
    "events": <target_events from energy_config>,
    "pos_h": <x_from_state>,
    "pos_v": <y_from_state>,
    "pos_rot": 1.5,
    "pos_tilt": 1.0,
    "beam_energy": <energy>
}
(Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

2c. Request Plot Confirmation (only AFTER the DAQ tool has run):
Output: {"message": "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."}

After user says "완료" to the plot message:
The SYSTEM marks the current energy completed and advances automatically — you do NOT output any state update.
Proceed to the next energy's STEP 2a (or, if all done, the SYSTEM ends the session).

=== STEP 3: Completion ===
When ALL energies are completed, the SYSTEM sends the completion message and ends the session automatically.

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip or reorder steps.
2. Use EXACT messages above. DO NOT change or paraphrase.
3. The SYSTEM (not you) owns all bookkeeping: y_confirmed, phase→scanning, energy "completed", and session termination. NEVER output update_state for these — only the step hint tells you the next action.
4. Output JSON format (CHOOSE ONE, NEVER BOTH):
   - {"tool": "...", "params": {...}}  (for tool execution)
   - {"message": "..."}  (for user message)
   - {"tool": "none", "update_state": {...}}  (ONLY for STEP 0b config parsing)
   CRITICAL: NEVER output both "tool" and "message" in the same JSON. NEVER put "message" inside "update_state".
5. Use energy_config[energy].target_events for DAQ events
6. STEP TRANSITION RULES:
   - phase="config", no history → output STEP 0a (ask message). DO NOT skip to parse.
   - phase="config", user just answered → output STEP 0b (parse + update_state). DO NOT ask again.
   - After STEP 0b (energy_config parsed, phase="idle"): go to STEP 1 (position move message). DO NOT repeat STEP 0.
   - After STEP 1a (position message sent): wait for user "완료", the system advances — go to STEP 2a.
   - After DAQ tool runs: send STEP 2c plot message. DO NOT call daq_run_tool again for the same energy.
   - NEVER skip STEP 1. NEVER output the same message twice in a row.
7. All "message" field values MUST be written in Korean (한국어) only. Never use Chinese characters (한자).
"""
    
    # ===== State Context =====
    
    def _build_state_context(self) -> str:
        """State를 문자열로 변환"""
        lines = []
        lines.append(f"Phase: {self.state['phase']}")
        lines.append(f"Tower: {self.state['tower']}")
        lines.append(f"M5T3 Position: x={self.t5_x:.3f}, y={self.t5_y:.3f}, rot=1.5, tilt=1.0")
        lines.append(f"Position confirmed: {self.state.get('y_confirmed', False)}")
        lines.append(f"needs_plot_confirm: {self.state.get('needs_plot_confirm', False)}")
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
            return f"Phase: idle | REQUIRED NEXT: position move message (step 1a, x={self.t5_x:.3f}, y={self.t5_y:.3f})"
        current_energy = self.state.get("current_energy")
        scan_order = self.state.get("scan_order", [])
        idx = scan_order.index(current_energy) + 1 if current_energy in scan_order else 0
        total = len(scan_order)
        ec = self.state.get("energy_config", {})
        if ec and all(c.get("completed", False) for c in ec.values()):
            return f"Phase: {phase} | ALL ENERGIES COMPLETE — system will terminate automatically"
        cfg = ec.get(current_energy, {})
        if cfg.get("runs") and not cfg.get("completed"):
            last_run = cfg["runs"][-1]
            return (
                f"Phase: {phase} | Energy: {current_energy} GeV ({idx}/{total}) | "
                f"DAQ done (Run {last_run}) — "
                f"REQUIRED NEXT: plot confirmation message (step 2c). DO NOT call daq_run_tool again. "
                f"완료 시 시스템이 자동으로 완료 처리한다."
            )
        return (
            f"Phase: {phase} | Energy: {current_energy} GeV ({idx}/{total}) | "
            f"needs_plot_confirm=False — DO NOT output plot confirmation. "
            f"REQUIRED NEXT: set-beam message (step 2a) then daq_run_tool (step 2b)"
        )

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

    # ===== 공용 드라이버 hooks (run()은 BaseAgent에서 제공) =====

    def _print_banner(self):
        print(f"\n{'='*70}\n⚡ Energy Scan Agent Started\n{'='*70}")

    def _print_summary(self):
        """현재 진행 상황 요약 출력 (Dash보드 스타일)"""
        print(f"\n📊 Energy Scan Progress Summary:")
        print("-" * 70)
        tower = self.state.get('tower', 'M5T3')
        print(f"Tower: {tower} | Position: x={self.t5_x:.3f}, y={self.t5_y:.3f}")
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

    def _pre_iteration(self):
        self._print_summary()
        # scanning 시작 시 다음 미완료 에너지로 자동 전환
        if self.state.get('phase') == 'scanning':
            _cur = self.state.get('current_energy')
            _cur_done = _cur is not None and self.state['energy_config'].get(_cur, {}).get('completed', False)
            if _cur is None or _cur_done:
                for _e in self.state['scan_order']:
                    if not self.state['energy_config'][_e].get('completed', False):
                        self.state['current_energy'] = _e
                        self.state['current_energy_idx'] = self.state['scan_order'].index(_e)
                        break

    def _is_complete(self) -> bool:
        ec = self.state.get("energy_config", {})
        return bool(ec) and self.state.get("phase") == "scanning" and all(
            c.get("completed", False) for c in ec.values()
        )

    def _completion_message(self) -> Optional[str]:
        return "모든 에너지 스캔이 완료되었습니다."

    def _completed_count(self) -> int:
        return sum(1 for c in self.state.get("energy_config", {}).values() if c.get("completed"))

    def _progress_message(self) -> Optional[str]:
        return self._format_progress()

    def _on_user_input(self, user_input: str):
        # 1) 이동 확인 (M5T3 고정 위치이므로 스캔당 1회). 코드가 직접 처리.
        if self.state.get("phase") == "idle" and not self.state.get("y_confirmed"):
            self.state["y_confirmed"] = True
            self.state["phase"] = "scanning"
            self.log("Position confirmed by user → phase=scanning")
            return
        # 2) DAQ 후 plot 확인 → 현재 에너지 완료 처리 (코드가 소유)
        if self.state.get("needs_plot_confirm"):
            self.state["needs_plot_confirm"] = False
            e = self.state.get("current_energy")
            cfg = self.state.get("energy_config", {}).get(e)
            if cfg is not None and cfg.get("runs"):
                cfg["completed"] = True
                cfg.setdefault("completed_at", datetime.now().strftime("%H:%M:%S"))
                self.log(f"{e} GeV plot 확인 완료 → completed")

    def _position_for_current_step(self) -> Optional[Dict[str, float]]:
        """EM Scan은 M5T3 고정 위치."""
        return {"x": self.t5_x, "y": self.t5_y}

    def _resolve_daq_energy_key(self):
        """DAQ용 에너지 — state/scan_order 기준 (LLM params 무시)."""
        energy_key = self.state.get("current_energy")
        ec = self.state.get("energy_config", {})
        if energy_key is not None and energy_key in ec:
            if not ec[energy_key].get("completed", False):
                return energy_key
        for e in self.state.get("scan_order", []):
            if not ec.get(e, {}).get("completed", False):
                return e
        return None

    # ===== Tool 실행 =====

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """Tool 실행"""
        print(f"\n🤖 Agent Decision:")
        print(f"   Tool: {tool_name}")
        print(f"   Params: {json.dumps(params, ensure_ascii=False)}")
        print()
        
        if tool_name == "none":
            return "no_tool_executed"

        elif tool_name == "daq_run_tool":
            energy_key = self._resolve_daq_energy_key()
            events = None
            if energy_key is not None and energy_key in self.state['energy_config']:
                events = self.state['energy_config'][energy_key]['target_events']
            self._apply_daq_params_from_state(
                params,
                events=events,
                beam_energy=energy_key,
                program="EM Scan",
                pos=self._position_for_current_step(),
                pos_rot=1.5,
                pos_tilt=1.0,
            )

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
            # 사용자 plot 확인 전까지 completed=True 차단
            self.state['needs_plot_confirm'] = True

            return result

        else:
            print(f"⚠️  Unknown tool: {tool_name}")
            self.log(f"Unknown tool: {tool_name}")
            return f"Error: Unknown tool {tool_name}"
    
    # ===== Helper 함수 =====
    
    def _guard_tool(self, tool_name: str, params) -> Optional[str]:
        if tool_name == "daq_run_tool" and self.state.get("needs_plot_confirm"):
            return (
                f'needs_plot_confirm=True — DAQ already ran. '
                f'Send: {{"message": "{MSG_PLOT_CONFIRM}"}}'
            )
        return None

    def _guard_ai_message(self, message: str) -> Optional[str]:
        if MSG_PLOT_CONFIRM in message and not self.state.get("needs_plot_confirm"):
            return f"needs_plot_confirm=False — DO NOT send plot confirmation. {self._get_step_hint()}"
        return None

    # Fields the LLM must not overwrite.
    # - init-only config: tower, daq_config, start_time, plot_method, plot_max_event
    # - code-owned bookkeeping (driver/_on_user_input/_execute_tool set these):
    #   y_confirmed, needs_plot_confirm, current_energy_idx, last_run_number
    _PROTECTED_FIELDS = frozenset({
        "tower", "daq_config", "start_time", "plot_method", "plot_max_event",
        "y_confirmed", "needs_plot_confirm", "current_energy_idx",
        "last_run_number",
    })

    def _update_state(self, updates: Dict[str, Any]):
        """State 업데이트 (energy_config는 deep merge로 기존 필드 보존)"""
        for key, value in updates.items():
            if key == "energy_config" and isinstance(value, dict):
                for energy_key, config_value in value.items():
                    try:
                        f = float(energy_key)
                        int_key = int(f) if f == int(f) else f
                    except (ValueError, TypeError):
                        int_key = energy_key
                    if int_key in self.state['energy_config'] and isinstance(config_value, dict):
                        # target_events(파서 설정), completed/runs/collected_events(코드 소유)는
                        # LLM이 변경 불가. 완료 표시는 _on_user_input(plot 확인)에서만 일어난다.
                        safe_update = {
                            k: v for k, v in config_value.items()
                            if k not in ('target_events', 'completed', 'completed_at', 'runs', 'collected_events')
                        }
                        if 'completed' in config_value:
                            self.log(f"WARNING: LLM tried to set completed for {int_key} GeV — rejected (code-owned)")
                        self.state['energy_config'][int_key].update(safe_update)

                    else:
                        self.state['energy_config'][int_key] = config_value
                self.state['scan_order'] = sorted(
                    [e for e in self.state['energy_config'].keys() if e is not None],
                    key=lambda x: float(x)
                )
                self.log(f"State updated: energy_config (deep merge), scan_order={self.state['scan_order']}")

            elif key == "current_energy" and value is not None:
                try:
                    f = float(value)
                    self.state[key] = int(f) if f == int(f) else f
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
            f"M5T3 위치:  x = {self.t5_x:.3f},  y = {self.t5_y:.3f}",
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

