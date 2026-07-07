#!/usr/bin/env python3
"""Base Agent — abstract base class for all scenario agents (EnergyScan, CalibScan, HVEqualization)."""

import json
import time
import torch
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MAX_CONVERSATION_HISTORY, MAX_NEW_TOKENS


class ToolFatalError(Exception):
    """Tool이 max_retries 이후에도 실패했을 때 발생."""
    pass



class BaseAgent(ABC):

    def __init__(self, model_path: str, agent_name: str, io_handler=None):
        self.model_path = Path(model_path)
        self.agent_name = agent_name

        if io_handler is None:
            from agents.io_handler import TerminalIO
            self.io = TerminalIO()
        else:
            self.io = io_handler
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.state = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = MAX_CONVERSATION_HISTORY
    
    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    def load(self):
        if self.model is not None:
            return
        
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"  ✅ [{self.agent_name}] MPS 사용")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"  ✅ [{self.agent_name}] CUDA 사용")
        else:
            self.device = "cpu"
            print(f"  ✅ [{self.agent_name}] CPU 사용")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        
        print(f"  ✅ [{self.agent_name}] 모델 로드 완료: {self.model_path}")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print(f"  🗑️  [{self.agent_name}] 모델 언로드 완료")

    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            entry.update(metadata)
        
        self.conversation_history.append(entry)
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_recent_history(self, n: Optional[int] = None) -> List[Dict]:
        if n is None:
            n = self.max_history
        return self.conversation_history[-n:]
    
    @abstractmethod
    def _build_state_context(self) -> str:
        pass

    def _build_history_context(self) -> str:
        if not self.conversation_history:
            return "(No conversation yet)"
        
        lines = []
        recent_history = self.conversation_history[-self.max_history:]
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Agent"
            content = msg["content"]

            if role == "Agent":
                try:
                    decision = json.loads(content)
                    if "message" in decision:
                        content = decision["message"]
                    elif "tool" in decision:
                        tool = decision["tool"]
                        params = decision.get("params", {})
                        summary = f"[Tool Call: {tool}]"
                        if tool in ("dqm_plot", "run_log") and params.get("run_number") or params.get("run_num"):
                            run = params.get("run_number") or params.get("run_num")
                            summary += f" run={run}"
                        if tool == "dqm_plot" and params.get("type"):
                            summary += f" type={params['type']}"
                            if params.get("modules"):
                                summary += f" modules={params['modules']}"
                        if tool in ("hv_write", "hodoscope_hv_write"):
                            cmd = params.get("command", "")
                            ch = params.get("channels", "")
                            v = params.get("voltage") or params.get("value", "")
                            summary += f" cmd={cmd} ch={ch}" + (f" v={v}" if v != "" else "")
                        if "update_state" in decision:
                            summary += f" (Update State: {list(decision['update_state'].keys())})"
                        content = summary
                except:
                    pass

            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def build_full_context(self, current_input: Optional[str] = None) -> str:
        parts = []
        
        parts.append("=== Current State ===")
        parts.append(self._build_state_context())
        parts.append("")
        
        parts.append("=== Recent Conversation ===")
        parts.append(self._build_history_context())
        parts.append("")
        
        if current_input:
            parts.append("=== Current User Input ===")
            parts.append(current_input)
            parts.append("")
        
        parts.append("=== Your Task ===")
        parts.append("Based on the current state and conversation, decide the next action.")
        parts.append("Output JSON with tool name and parameters.")
        
        return "\n".join(parts)
    
    def decide(self, context: str, max_retries: int = 3) -> Dict[str, Any]:
        """LLM inference → JSON. 첫 시도 greedy, 재시도 sampling."""
        if self.model is None:
            raise RuntimeError(f"[{self.agent_name}] Model not loaded. Use with statement or call load().")

        system_prompt = self._get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        last_err: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(max_retries):
            gen_kwargs = dict(do_sample=False) if attempt == 0 else dict(do_sample=True, temperature=0.7, top_p=0.9)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **gen_kwargs,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            try:
                text = generated_text.strip()
                if '{' not in text:
                    raise json.JSONDecodeError("No JSON found", text, 0)

                decoder = json.JSONDecoder()
                start = text.index('{')
                decision, _ = decoder.raw_decode(text, start)
                if attempt > 0:
                    self.log(f"JSON 파싱 재시도 성공 (attempt {attempt + 1}/{max_retries})")
                return decision

            except (json.JSONDecodeError, ValueError) as e:
                last_err = str(e)
                last_raw = generated_text
                self.log(f"JSON 파싱 실패 (attempt {attempt + 1}/{max_retries}): {e}")
                continue

        return {
            "error": f"JSON parsing failed after {max_retries} attempts: {last_err}",
            "raw_output": last_raw,
        }
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        pass

    # ── Tool params: LLM 출력 무시, state가 source of truth ──

    def _position_for_current_step(self) -> Optional[Dict[str, float]]:
        return None

    def _apply_daq_params_from_state(
        self,
        params: Dict[str, Any],
        *,
        events: Optional[int] = None,
        beam_energy: Any = None,
        program: str,
        pos: Optional[Dict[str, float]] = None,
        pos_rot: float = 0.0,
        pos_tilt: float = 0.0,
    ) -> None:
        if events is not None:
            params["events"] = events
        if beam_energy is not None:
            params["beam_energy"] = beam_energy
        params["program"] = program
        if pos is not None:
            params["pos_h"] = pos["x"]
            params["pos_v"] = pos["y"]
        params["pos_rot"] = pos_rot
        params["pos_tilt"] = pos_tilt

    def _apply_hv_voltage_params_from_state(
        self,
        params: Dict[str, Any],
        channel_values: Dict[str, float],
    ) -> None:
        params["channel_values"] = channel_values

    def _apply_hv_suggest_params_from_state(
        self,
        params: Dict[str, Any],
        *,
        tower: str,
        run_number: Optional[int] = None,
        hv_c: Optional[float] = None,
        hv_s: Optional[float] = None,
    ) -> None:
        params["tower"] = tower
        if run_number is not None:
            params["run_number"] = run_number
        if hv_c is not None:
            params["hv_c"] = hv_c
        if hv_s is not None:
            params["hv_s"] = hv_s

    def _extract_run_number(self, daq_output: Optional[str] = None) -> Optional[int]:
        from tools.daq_tool import parse_run_number_from_daq_output

        run_number = parse_run_number_from_daq_output(daq_output)
        if run_number is None:
            self.log("WARNING: DAQ output에서 run number를 찾지 못함")
        return run_number

    def _run_tool_with_retry(self, tool_fn: Callable, tool_name: str, max_retries: int = 3) -> str:
        while True:
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    return tool_fn()
                except RuntimeError as e:
                    last_error = e
                    self.log(f"[Retry {attempt}/{max_retries}] Tool '{tool_name}' 실패: {e}")
                    if attempt < max_retries:
                        time.sleep(2)
            self.io.send_tool_error(tool_name, str(last_error), max_retries)
            action = self.io.wait_for_retry()
            if action == "skip":
                return f"[SKIPPED] {tool_name} 건너뜀 (사용자 요청)"

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.agent_name}] {message}", flush=True)
    
    # ── Run loop hooks (서브클래스가 필요 시 override) ──

    def _print_banner(self):
        print(f"\n{'='*70}\n⚡ {self.agent_name} Started\n{'='*70}")

    def _pre_iteration(self): pass
    def _is_complete(self) -> bool: return bool(self.state.get("done"))
    def _completion_message(self) -> Optional[str]: return None
    def _completed_count(self) -> int: return 0
    def _progress_message(self) -> Optional[str]: return None
    def _on_user_input(self, user_input: str): pass

    def _guard_tool(self, tool_name: str, decision: Dict[str, Any]) -> Optional[str]:
        """거부 사유 반환 → 실행 차단 후 재시도. None이면 통과."""
        return None

    def _guard_ai_message(self, message: str) -> Optional[str]:
        """거부 사유 반환 → 출력 차단 후 재시도. None이면 통과."""
        return None

    def _stop_requested(self) -> bool:
        """WebSocketIO의 stop_event가 set이면 True (TerminalIO는 항상 False)."""
        ev = getattr(self.io, "stop_event", None)
        return ev is not None and ev.is_set()

    def run(self):
        self._print_banner()
        self.log(f"{self.agent_name} 시작")

        if self.model is None:
            raise RuntimeError(f"[{self.agent_name}] Model not loaded. Use 'with agent:' statement.")

        _error_count = 0
        _MAX_ERRORS = 3
        # 워치독: 사용자 상호작용(message)도 없고 실제 tool 실행도 없이
        # update_state만 반복하면(예: config에서 beam_energy=null 무한 반복)
        # get_input()을 절대 호출하지 않아 stop_event도 못 보고 무한 루프에 빠진다.
        _no_progress = 0
        _MAX_NO_PROGRESS = 5
        # 워치독2: guard(_guard_ai_message/_guard_tool)가 같은 결정을 계속 거부하면
        # get_input()이 호출되지 않아 state가 진전되지 않고, greedy 디코딩이 거부된
        # 출력(예: plot 확인 메시지)을 무한 반복한다. 실제 진행(메시지 전송/tool 실행/
        # 사용자 입력)이 있을 때 0으로 리셋되고, 연속 거부만 누적되면 종료한다.
        _guard_reject = 0
        _MAX_GUARD_REJECT = 6

        while True:
            try:
                # get_input()이 호출되지 않는 경로(아래 update_state-only 등)에서도
                # Stop 버튼(stop_event)에 반응해 깨끗이 빠져나가도록 매 반복 확인.
                if self._stop_requested():
                    self.log("Stop 요청 감지 — 종료합니다.")
                    break

                self._pre_iteration()

                if self._is_complete():
                    msg = self._completion_message()
                    if msg:
                        self.io.send_ai_message(msg)
                    break

                context = self.build_full_context()
                decision = self.decide(context)
                print(f"\n🔍 Decision: {json.dumps(decision, ensure_ascii=False)}")

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
                    before = self._completed_count()
                    self._update_state(decision["update_state"])
                    after = self._completed_count()
                    if after > before:
                        prog = self._progress_message()
                        if prog:
                            self.io.send_ai_message(prog)

                message = decision.get("message")
                tool_name = decision.get("tool")

                if message:
                    rejection = self._guard_ai_message(message)
                    if rejection:
                        self.log(f"message guard blocked: {rejection}")
                        self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                        self.add_to_history("user", rejection)
                        _guard_reject += 1
                        if _guard_reject >= _MAX_GUARD_REJECT:
                            self.log(f"Guard-reject 워치독 발동 ({_guard_reject}회 연속 거부) — 종료")
                            self.io.send_ai_message(
                                "에이전트가 올바른 다음 단계를 내지 못하고 같은 응답을 반복하고 있습니다. "
                                "세션을 종료합니다. 다시 시작해주세요."
                            )
                            break
                        continue
                    _guard_reject = 0
                    _no_progress = 0
                    self.io.send_ai_message(message)
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    user_input = self.io.get_input()
                    if user_input in ["종료", "exit"]:
                        break
                    before = self._completed_count()
                    self._on_user_input(user_input)
                    after = self._completed_count()
                    if after > before:
                        prog = self._progress_message()
                        if prog:
                            self.io.send_ai_message(prog)
                    self.add_to_history("user", user_input)
                    continue

                if tool_name and tool_name != "none":
                    rejection = self._guard_tool(tool_name, decision)
                    if rejection:
                        self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                        self.add_to_history("user", rejection)
                        _guard_reject += 1
                        if _guard_reject >= _MAX_GUARD_REJECT:
                            self.log(f"Guard-reject 워치독 발동 ({_guard_reject}회 연속 거부) — 종료")
                            self.io.send_ai_message(
                                "에이전트가 올바른 다음 단계를 내지 못하고 같은 응답을 반복하고 있습니다. "
                                "세션을 종료합니다. 다시 시작해주세요."
                            )
                            break
                        continue
                    _guard_reject = 0
                    _no_progress = 0
                    result = self._execute_tool(tool_name, decision.get("params", {}))
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    if self.state.get("done"):
                        break
                    continue

                if "update_state" in decision:
                    # message도 tool도 없이 update_state만 오는 턴. 정상 워크플로우에선
                    # config 파싱(target_events→idle) 직후 딱 1번 나오고 곧바로 message
                    # 턴으로 이어진다. 이게 연속으로 반복되면(모델이 config를 못 내보내는
                    # 경우) 사용자 입력 없이 무한 루프 → 워치독으로 차단.
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    _no_progress += 1
                    if _no_progress >= _MAX_NO_PROGRESS:
                        self.log(f"No-progress 워치독 발동 ({_no_progress}회 연속 update_state-only) — 종료")
                        self.io.send_ai_message(
                            "에이전트가 다음 단계를 진행하지 못하고 있습니다. 세션을 종료합니다. "
                            "다시 시작해주세요."
                        )
                        break
                    continue

                _error_count += 1
                self.log(f"Unrecognized decision ({_error_count}/{_MAX_ERRORS}): {decision}")
                if _error_count >= _MAX_ERRORS:
                    print(f"\n❌ 연속 인식 불가 응답 {_MAX_ERRORS}회 — 종료합니다.")
                    break
                self.add_to_history("user", "Output valid JSON only. No other text.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                # get_input()/wait_for_retry()가 Stop 요청 시 던지는 StopAgentException
                # 등, stop_event가 켜진 상태의 예외는 정상 종료로 처리(트레이스백 X).
                if self._stop_requested():
                    self.log("Stop 요청 감지 — 종료합니다.")
                    break
                print(f"\n❌ 오류 발생: {str(e)}")
                import traceback as _tb
                _tb.print_exc()
                break

    @abstractmethod
    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        pass
