#!/usr/bin/env python3
"""Base Agent — abstract base class for all scenario agents (EnergyScan, CalibScan, HVEqualization)."""

import json
import re
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


# ── 이벤트 개수 표시 유틸 ──
# 모델이 자릿수(0의 개수)를 자주 틀리므로, 사용자에게 보이는 이벤트 개수는
# 항상 코드가 state의 진짓값으로부터 렌더링/교정한다. LLM 출력 텍스트를 신뢰하지 않는
# 기존 원칙(_finalize_ai_message)의 이벤트-개수판.

_EVENT_KEYWORD = r'(?:이벤트|events?|evt)'
# 숫자(콤마 허용)가 이벤트 키워드 바로 앞: "10000 이벤트", "10,000개 events"
_RE_NUM_BEFORE_EVENT = re.compile(r'([\d,]+)(\s*개?\s*' + _EVENT_KEYWORD + r')', re.IGNORECASE)
# 이벤트 키워드(+ 수/콜론)가 숫자 바로 앞: "이벤트 수: 10000", "events: 1000", "이벤트 10000개"
_RE_NUM_AFTER_EVENT = re.compile(r'(' + _EVENT_KEYWORD + r'\s*수?\s*[:：]?\s*)([\d,]+)', re.IGNORECASE)


def format_event_count(n) -> str:
    """이벤트 개수를 천 단위 콤마 문자열로. 정수화 불가하면 원본을 문자열로."""
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


_RE_NUMBER = re.compile(r'\d+(?:\.\d+)?')
_RE_GEV_SUFFIX = re.compile(r'\s*(?:GeV|기가)', re.IGNORECASE)


# 천 단위 콤마 그룹 후보: "50,000" / "1,234,567" (콤마 뒤 정확히 3자리 반복)
_RE_COMMA_GROUP = re.compile(r'\d{1,3}(?:,\d{3})+(?!\d)')


def _normalize_thousands_commas(text: str) -> str:
    """"10,000" → "10000" (천 단위 콤마 제거 — 나열 구분 콤마는 보존).
    그룹 전체가 GeV/기가로 끝나면 에너지 나열("50,100,120GeV")이므로 병합하지 않는다
    (빔 에너지는 최대 수백 GeV라 천 단위 콤마가 필요한 경우가 없음)."""
    def _repl(m):
        if _RE_GEV_SUFFIX.match(text, m.end()):
            return m.group()  # 에너지 나열 → 콤마 보존
        return m.group().replace(',', '')
    return _RE_COMMA_GROUP.sub(_repl, text)


def extract_number_tokens(text: str) -> List[tuple]:
    """텍스트의 숫자 토큰을 (값, 시작위치, 끝위치, 에너지단위 여부) 리스트로 반환.
    위치는 천 단위 콤마 정규화 후 텍스트 기준. 에너지단위 여부 = 숫자 바로 뒤에
    GeV/기가가 붙는지 (예: '3GeV' → (3.0, s, e, True)).
    코드가 LLM 파싱의 짝(에너지↔이벤트 수)을 결정론적으로 재구성/교정할 때 사용."""
    if not isinstance(text, str):
        return []
    normalized = _normalize_thousands_commas(text)
    tokens = []
    for m in _RE_NUMBER.finditer(normalized):
        is_energy = bool(_RE_GEV_SUFFIX.match(normalized, m.end()))
        tokens.append((float(m.group()), m.start(), m.end(), is_energy))
    return tokens


def extract_numbers_from_text(text: str) -> set:
    """텍스트에 등장하는 모든 숫자를 float 집합으로 추출 (천 단위 콤마 정규화).
    LLM이 파싱해 state에 넣는 숫자가 사용자 입력에 실제로 존재하는지(지어낸/자릿수
    틀린 숫자가 아닌지) 검증하는 데 사용한다. 형식이 어떻게 섞여 있어도 동작."""
    return {v for v, _, _, _ in extract_number_tokens(text)}


def correct_event_count_in_text(text: str, n) -> str:
    """LLM이 문장에 쓴 이벤트 개수를 state 진짓값(콤마 포맷)으로 교정.
    이벤트 키워드에 인접한 숫자만 대상으로 하여 전압/에너지/ADC 등은 건드리지 않는다."""
    if not isinstance(text, str) or not text:
        return text
    try:
        target = int(n)
    except (TypeError, ValueError):
        return text
    if target <= 0:
        return text
    formatted = f"{target:,}"
    text = _RE_NUM_BEFORE_EVENT.sub(lambda m: formatted + m.group(2), text)
    text = _RE_NUM_AFTER_EVENT.sub(lambda m: m.group(1) + formatted, text)
    return text


# ── 이동 위치(x/y mm) 표시 교정 ──
# 모델이 좌표 자릿수(0의 개수)도 자주 틀리므로("1000 → 100"), 이동 메시지의
# x/y는 코드가 state의 진짓값(_position_for_current_step)으로부터 교정한다.
# "mm" 단위에 앵커링하여 전압/에너지/ADC 등 다른 숫자는 건드리지 않는다.
_RE_POS_X = re.compile(r'((?<![A-Za-z])x\s*[=:]\s*)(-?[\d,]+(?:\.\d+)?)(\s*mm)', re.IGNORECASE)
_RE_POS_Y = re.compile(r'((?<![A-Za-z])y\s*[=:]\s*)(-?[\d,]+(?:\.\d+)?)(\s*mm)', re.IGNORECASE)


def correct_position_in_text(text: str, x, y) -> str:
    """이동 메시지의 x/y 좌표를 state 진짓값(mm, 소수 셋째 자리)으로 교정.
    "이동" 키워드가 있는 메시지에만 적용해 위치 요약·배너 등 다른 좌표 나열을
    건드리지 않는다. x/y 각각 float 변환 가능할 때만 해당 축을 교정한다."""
    if not isinstance(text, str) or not text or '이동' not in text:
        return text
    for val, rgx in ((x, _RE_POS_X), (y, _RE_POS_Y)):
        try:
            formatted = f"{float(val):.3f}"
        except (TypeError, ValueError):
            continue
        rgx_formatted = formatted  # 클로저 캡처용
        text = rgx.sub(lambda m: m.group(1) + rgx_formatted + m.group(3), text)
    return text



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
        # 숫자 출처 검증용: LLM이 update_state에 넣는 숫자가 이 입력에 실제로
        # 등장했는지 _guard_update_state에서 확인한다.
        self._last_user_input: str = ""
    
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
        config: Optional[str] = None,
    ) -> None:
        if events is not None:
            params["events"] = events
        if beam_energy is not None:
            params["beam_energy"] = beam_energy
        if config is not None:
            params["config"] = config
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
    def _recover_decision(self, failed_decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """decide()가 파싱 실패로 error를 낸 턴에, state 진짓값으로 결정을 결정론적
        복구할 수 있으면 대체 decision을 반환. 기본은 복구 안 함(None)."""
        return None
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

    def _guard_update_state(self, updates: Dict[str, Any]) -> Optional[str]:
        """update_state 적용 직전 검증. 거부 사유 반환 → 적용 차단 후 재시도. None이면 통과.
        서브클래스가 LLM이 파싱한 숫자(이벤트 수·에너지 등)의 출처를
        _last_user_input과 대조할 때 사용 (자릿수 오류 방어)."""
        return None

    def _numbers_in_last_input(self) -> set:
        return extract_numbers_from_text(self._last_user_input)

    def _finalize_ai_message(self, message: str) -> str:
        """전송 직전 메시지를 보정할 기회.
        기본 동작: state의 target_events를 진짓값으로 삼아, 모델이 문장에 쓴
        이벤트 개수의 자릿수를 교정하고 콤마 포맷으로 통일한다.
        서브클래스가 오버라이드할 땐 super()를 호출해 이 교정을 유지할 것."""
        corrected = correct_event_count_in_text(message, self.state.get("target_events"))
        # 이동 메시지의 x/y 좌표도 현재 스텝 진짓값으로 자릿수 교정.
        # _position_for_current_step()이 None이면(예: brain, 활성 스텝 없음) no-op.
        pos = self._position_for_current_step()
        if pos is not None:
            corrected = correct_position_in_text(corrected, pos.get("x"), pos.get("y"))
        if corrected != message:
            self.log(f"메시지 표시 교정(state 기준): {message!r} → {corrected!r}")
        return corrected

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
                    recovered = self._recover_decision(decision)
                    if recovered is not None:
                        decision = recovered
                    else:
                        _error_count += 1
                        self.log(f"Agent error ({_error_count}/{_MAX_ERRORS}): {decision['error']}")
                        if _error_count >= _MAX_ERRORS:
                            print(f"\n❌ 연속 오류 {_MAX_ERRORS}회 — 종료합니다.")
                            break
                        self.add_to_history("user", "Output valid JSON only. No other text.")
                        continue

                _error_count = 0

                if "update_state" in decision:
                    # 숫자 출처 검증 등 — 틀린 값이 state(진짓값)에 들어가기 전에 차단.
                    rejection = self._guard_update_state(decision["update_state"])
                    if rejection:
                        self.log(f"update_state guard blocked: {rejection}")
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
                    _guard_reject = 0  # state가 실제로 진전 → 연속 거부 리셋
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
                    message = self._finalize_ai_message(message)
                    decision["message"] = message
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
                    self._last_user_input = user_input
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
