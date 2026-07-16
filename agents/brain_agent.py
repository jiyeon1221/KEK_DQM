#!/usr/bin/env python3
"""BrainAgent — 자연어 → tool JSON 변환 후 실행. 서버 시작부터 종료까지 상주."""

import json
import re
import time
import threading
import queue
import traceback
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from agents.io_handler import WebSocketIO


TOOL_LOCK_MAP = {
    "daq_run": "daq",
    "dqm_plot": None,
    "run_log": None,
    "hv_read": "hv",
    "hv_write": "hv",
}

TOOLS_NEED_CONFIRM = {"daq_run", "hv_write"}

# hv_write가 실제로 지원하는 command (모델이 "write" 등 엉뚱한 값을 내보낼 때 정규화용)
_HV_WRITE_COMMANDS = {"voltage", "current", "svmax", "rup", "rdown", "name", "on", "off"}



SYSTEM_PROMPT = """You are the Brain Agent for a test beam experiment (KEK/CERN).
Your job is to interpret the operator's ad-hoc request and call the right tool.

Available tools:
- daq_run: Run DAQ data collection. params: {"events": int, "config": str (optional)}
  - config is the DAQ config name passed to the run script. It can be ANY name the user says
    (e.g. "test", "setup", "setup1", "config1", "physics", ...). Default is "setup" —
    OMIT config unless the user explicitly names one
    (e.g. "setup1으로 10000개", "config1 설정으로", "test로 돌려줘").
    Copy the name EXACTLY as the user wrote it — never invent or normalize it.
- dqm_plot: Generate DQM plots for a run and display in the DQM panel.
  params: {"run_number": int, "method": "IntADC"|"PeakADC", "type": "full"|"heatmap"|"single", "modules": [list]}
  - type defaults to "full" (all towers + heatmap). No modules needed for full.
  - "heatmap": modules must be ["MCPPMT"]. method: IntADC or PeakADC.
  - "single": modules is a list of channel names, e.g. ["M1-T1-C"], ["M1-T1-S","M1-T1-C"], or ["M1"] (M1 auto-expands).
  - method has NO default. If the user does not say IntADC/적분 or PeakADC/피크, you MUST ask (tool:none) — never assume IntADC.
  - run_number has NO default. Never invent one. If not given and no relative reference, ask (tool:none).
- run_log: Google Sheets run log.
  Read:   params: {"command": "read", "run_num": int}
  Update: params: {"command": "update", "run_num": int, "<column>": "<value>"}
  Updatable columns: program, notes, config, beam_energy, beam_type, trigger_setup, hv_drc, hv_aux
- hv_read: Read current CAEN HV status.
  params: {"command": "status", "channels": <ch_spec>}
  "channels" is OPTIONAL and accepts the SAME <ch_spec> forms as hv_write (see below).
  - Omit "channels" (reads ALL channels) ONLY when there is no channel hint or the user says 전체/모든/all.
  - If the user restricts to a subset (짝수/홀수/C채널/S채널/타워/특정 모듈·채널/슬롯 등),
    put it in "channels" EXACTLY as you would for hv_write
    (짝수→"even", 홀수→"odd", S채널/S만→"S", C채널/C만→"C", 타워 T1→"T1",
     M3만→"M3", 특정 채널→["M3T2C"], 슬롯→"slot:12").
- hv_write: Change CAEN HV voltage or turn channels on/off. User confirmation required.
  Voltage: {"command": "voltage", "channels": <ch_spec>, "voltage": <V as float>}
  On/off:  {"command": "on"|"off", "channels": <ch_spec>}
  Valid channel names (ONLY these): M{1-9}T{1-4}{C,S} (e.g. M1T1C, M1T1S, M1T2C ... M9T4S), TRIG1, TRIG2, MCP-S, MCP-C
  Channel spec (<ch_spec>) options:
    "all"          — 전체 채널 (ONLY when user says 전체/모든/all)
    "S" / "C"      — 모든 S(신틸)/C(체렌코프) 채널
    "T1"~"T4"      — 타워 단위: 모든 모듈의 해당 타워 채널 (예: "T1")
    "M5"           — 모듈 단위: M5의 모든 채널 (M5T1C/S~M5T4C/S, 8채널). M{1-9} 형식.
    ["M3","M5"]    — 모듈 목록: 복수 모듈 지정 (각 모듈 8채널)
    "even"         — 짝수 번호 채널 전체
    "odd"          — 홀수 번호 채널 전체
    "N-M"          — ch 번호 N~M 범위 (예: "0-8", "2-12")
    "N,M,K"        — ch 번호 목록 (예: "1,2,5,6")
    ["M1T1C","M1T2C"]  — 이름 목록
    "slot:S"       — 슬롯 S 전체
    "slot:S:even/odd" — 슬롯 S 짝/홀수

Current experiment state is provided so you can resolve relative references
like "방금", "이번 런", "지금" to concrete run numbers or energies.

Respond with a single JSON object:
{"tool": "<tool_name>", "params": {<params>}, "reason": "<short explanation>"}

If the request is unclear or you cannot determine a tool, respond:
{"tool": "none", "message": "<ask the user for clarification>"}

RULES:
0. All "message" and "reason" field values MUST be written in Korean (한국어) only. Never use Chinese characters (한자).
1. Output ONLY valid JSON. No markdown, no explanation outside JSON.
2. Always resolve relative references using the provided state.
3. For run_log updates, extract column and value from the user's message.
4. run_log supports both READ and WRITE:
   - VIEW/CHECK a log (확인, 보여줘, 읽어줘) WITHOUT a value → {"command": "read", "run_num": ...}
   - WRITE with column+value (e.g. "프로그램에 EM 추가") → {"command": "update", "run_num": ..., "<column>": "<value>"}
5. hv_read for ANY HV status. "HV 확인", "HV 상태" → all use hv_read.
   If the request names a channel subset (짝수/홀수/C/S/타워/모듈/채널명/슬롯), pass it in "channels" just like hv_write;
   otherwise omit "channels" to read ALL channels.
6. DAQ requires an event count. If the user says "DAQ 돌려줘" without a number, ask how many events.
   If the user names a DAQ config (test, setup1, config1 등 — 어떤 이름이든), put it in "config";
   otherwise OMIT config (default "setup").
7. Channel names like M1T1C, M1T1S, M2T3C, ... are HV channels (format: M{1-9}T{1-4}{C,S}) — NOT log columns.
   A SINGLE channel name + voltage → channels: [that single channel].
   ONLY use channels: "all" when the input explicitly says 전체/모든/전 채널/all channels.
8. For daq_run and hv_write, the system asks the user to confirm before execution.
9. "플롯", "그려줘", "그래프" → dqm_plot. Default type: full.
   Method: ONLY set it when the user explicitly says IntADC/intADC/적분 (→ "IntADC") or PeakADC/peakADC/피크 (→ "PeakADC").
   If method is not mentioned, respond with tool:none asking "IntADC로 그릴까요, PeakADC로 그릴까요?"
10. Specific tower/channel (M1, M1-T1-C, M1-T1-S, M5 etc.) → type: single, modules: [name].
11. "heatmap" or "MCPPMT" mentioned → type: heatmap, modules: ["MCPPMT"].
12. dqm_plot requires run_number. NEVER invent or guess a run number.
    Only resolve it from state when the user uses a relative reference (방금/이번/지금/현재/마지막/최근/last).
    If the user gives NO explicit run number AND NO relative reference,
    respond tool:none asking "어떤 런 번호의 DQM 플롯을 그릴까요?".
"""


class BrainAgent(BaseAgent):

    def __init__(self, shared_state: Optional[Dict] = None,
                 shared_locks: Optional[Dict[str, threading.Lock]] = None,
                 io_handler=None, use_base_model: bool = False,
                 confirm_queue: Optional[queue.Queue] = None,
                 clarify_queue: Optional[queue.Queue] = None):
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from config import AGENT_MODELS

        model_cfg = AGENT_MODELS["brain"]
        model_path = model_cfg["base_model"] if use_base_model else model_cfg["fine_tuned_path"]

        super().__init__(model_path=model_path, agent_name="BrainAgent", io_handler=io_handler)

        self.shared_state = shared_state or {}
        self.shared_locks = shared_locks or {}
        self.confirm_queue = confirm_queue or queue.Queue()
        self.clarify_queue = clarify_queue or queue.Queue()
        self._last_daq_run: Optional[int] = None  # run number from last brain-initiated DAQ
        # 누락 파라미터를 물어본 뒤, 사용자의 답변을 "그 필드의 값"으로 채우기 위한 대기 상태.
        # {"tool": str, "params": dict, "reason": str}
        self._pending: Optional[Dict[str, Any]] = None

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_state_context(self) -> str:
        s = self.shared_state
        lines = []

        if s.get("agent_type"):
            lines.append(f"Running agent: {s['agent_type']}")
        if s.get("current_run"):
            lines.append(f"Current run number: {s['current_run']}")

        last_run = s.get("last_run") or s.get("last_run_number") or self._last_daq_run
        if last_run:
            lines.append(f"Last completed run: {last_run}")

        if s.get("current_tower"):
            lines.append(f"Current tower: {s['current_tower']}")
        if s.get("current_energy"):
            lines.append(f"Current energy: {s['current_energy']} GeV")
        if s.get("phase"):
            lines.append(f"Phase: {s['phase']}")
        return "\n".join(lines) if lines else "(No scenario agent running)"

    def run(self):
        """Not used — BrainAgent uses handle_request() for single-turn dispatch."""
        pass

    def handle_request(self, user_input: str, io: WebSocketIO) -> None:
        # ── 직전 턴에서 누락 파라미터를 물어봤다면, 이번 입력을 그 답으로 채운다.
        #    (모델을 다시 태워 이미 준 정보를 또 묻는 것을 방지) ──
        if self._pending is not None:
            resumed = self._resume_pending(user_input)
            if resumed is not None:
                tool_name, params, reason = resumed
                self.add_to_history("user", user_input)
                self._dispatch(tool_name, params, reason, io)
                return
            # 답이 해당 필드로 해석되지 않으면 → 새 요청으로 간주하고 정상 처리
            self._pending = None

        context = self.build_full_context(current_input=user_input)

        io.send_status("BrainAgent 처리 중...")
        decision = self.decide(context)
        self.log(f"[Brain] tool={decision.get('tool','?')} | params={decision.get('params',{})} | reason={decision.get('reason','')!r} | message={decision.get('message','')!r}")

        if "error" in decision:
            io.send_ai_message(f"요청을 이해하지 못했습니다: {decision.get('raw_output', '')[:200]}")
            io.send_status("대기 중")
            return

        self.add_to_history("user", user_input)
        self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))

        tool_name = decision.get("tool", "none")
        reason = decision.get("reason", "")

        if tool_name == "none":
            fallback = self._apply_fallback_rules(user_input)
            if fallback:
                self.log(f"[Brain] tool:none → fallback: {fallback['tool']}")
                decision = fallback
                tool_name = fallback["tool"]
                reason = fallback.get("reason", "")
            else:
                msg = decision.get("message", reason or "무엇을 도와드릴까요?")
                if not msg or msg.strip().startswith("<"):
                    msg = "요청을 좀 더 구체적으로 말씀해 주세요."
                io.output_queue.put({"type": "adhoc_clarify", "question": msg, "source": "brain"})
                io.send_status("대기 중")
                return

        params = decision.get("params", {})

        # HV command / channels 정규화 + 사용자 입력에서 누락 인자 backfill
        self._normalize_hv_params(tool_name, params)
        self._backfill_params(tool_name, params, user_input)

        # "방금"/"이번"/"last"/"지금" → infer run_number from state instead of asking
        if tool_name == "dqm_plot" and not params.get("run_number"):
            if re.search(r'(방금|이번|last|지금)', user_input.lower()):
                run_num = (self.shared_state.get("last_run")
                           or self.shared_state.get("last_run_number")
                           or self._last_daq_run)
                if run_num:
                    params["run_number"] = int(run_num)

        self._dispatch(tool_name, params, reason, io)

    def _dispatch(self, tool_name: str, params: dict, reason: str, io: WebSocketIO) -> None:
        """검증 → (필요 시)확인 → 잠금 → 실행. 누락 인자는 _pending에 저장 후 되묻는다."""
        missing = self._validate_params(tool_name, params)
        if missing:
            field, question = missing
            self._pending = {"tool": tool_name, "params": params, "reason": reason}
            io.output_queue.put({"type": "adhoc_clarify", "question": question, "source": "brain"})
            io.send_status("대기 중")
            return

        self._pending = None  # 모든 인자 충족 → 대기 상태 해제

        if tool_name in TOOLS_NEED_CONFIRM:
            preview = self._format_confirm_preview(tool_name, params)
            io.send_ai_message(self._format_dispatch_msg(tool_name, params, reason))
            while not self.confirm_queue.empty():  # drain stale replies
                try: self.confirm_queue.get_nowait()
                except queue.Empty: break
            io.output_queue.put({
                "type": "adhoc_confirm",
                "preview": preview,
                "tool": tool_name,
            })
            io.send_status("사용자 확인 대기 중...")
            try:
                confirmed = self.confirm_queue.get(timeout=120)
            except queue.Empty:
                io.send_ai_message("확인 대기 시간 초과 — 요청을 취소했습니다.")
                io.send_status("대기 중")
                return
            if not confirmed:
                io.send_ai_message("요청이 취소되었습니다.")
                io.send_status("대기 중")
                return
        else:
            io.send_ai_message(self._format_dispatch_msg(tool_name, params, reason))

        lock_key = TOOL_LOCK_MAP.get(tool_name)
        lock = self.shared_locks.get(lock_key) if lock_key else None

        if lock and not lock.acquire(blocking=False):
            resource = {"daq": "DAQ", "hv": "HV"}.get(lock_key, lock_key)
            io.send_ai_message(f"{resource}가 현재 사용 중이라 지금은 실행할 수 없습니다.")
            io.send_status("대기 중")
            return

        try:
            self._execute_tool(tool_name, params, io)
        finally:
            if lock:
                lock.release()

        io.send_status("대기 중")

    # ── 파라미터 정규화 / backfill / 클래리파이 재개 ──

    @staticmethod
    def _normalize_hv_params(tool_name: str, params: dict) -> None:
        """모델이 tool 이름(read/write)을 command로 잘못 넣거나, 시스템 프롬프트의
        예시 문구("모든 S 채널" 등)를 그대로 복사한 경우를 바로잡는다."""
        if tool_name in ("hv_read", "hv_status"):
            params["command"] = "status"
        elif tool_name == "hv_write":
            cmd = str(params.get("command", "")).lower()
            if cmd not in _HV_WRITE_COMMANDS:
                # "write"/"" 등 → 전압/전류 값이 있으면 voltage, 아니면 그대로 둠
                if "voltage" in params or "value" in params:
                    params["command"] = "voltage"
                elif "current" in params:
                    params["command"] = "current"

        # channels가 프롬프트 예시 문구를 그대로 복사한 한국어 문자열이면 토큰으로 치환
        ch = params.get("channels")
        if isinstance(ch, str):
            low = ch.strip().lower()
            if "s 채널" in low or "s채널" in low or "s 만" in low or low in ("s만", "s"):
                params["channels"] = "S"
            elif "c 채널" in low or "c채널" in low or low in ("c만", "c"):
                params["channels"] = "C"
            elif "짝수" in low or low == "even":
                params["channels"] = "even"
            elif "홀수" in low or low == "odd":
                params["channels"] = "odd"

    def _backfill_params(self, tool_name: str, params: dict, user_input: str) -> None:
        """모델이 놓친 인자를 사용자 입력 원문에서 직접 복구한다 (지어내지 않음)."""
        if tool_name == "dqm_plot":
            if not params.get("run_number"):
                rn = self._extract_run_number_from_text(user_input)
                if rn:
                    params["run_number"] = rn
            if not params.get("method"):
                m = self._extract_method_from_text(user_input)
                if m:
                    params["method"] = m
        elif tool_name == "daq_run":
            if not params.get("events"):
                ev = self._extract_events_from_text(user_input)
                if ev:
                    params["events"] = ev
            if not params.get("config"):
                cfg = self._extract_config_from_text(user_input)
                if cfg:
                    params["config"] = cfg

    def _resume_pending(self, user_input: str):
        """대기 중인 누락 필드를 이번 입력으로 채운다.
        성공 시 (tool, params, reason) 반환, 해석 불가면 None."""
        pend = self._pending
        tool_name = pend["tool"]
        params = dict(pend["params"])
        filled = False

        missing = self._validate_params(tool_name, params)
        if not missing:
            self._pending = None
            return tool_name, params, pend.get("reason", "")
        field, _ = missing

        if field in ("run_number", "run_num"):
            rn = self._extract_run_number_from_text(user_input)
            if rn:
                params[field] = rn
                filled = True
        elif field == "method":
            m = self._extract_method_from_text(user_input)
            if m:
                params["method"] = m
                filled = True
        elif field == "events":
            ev = self._extract_events_from_text(user_input)
            if ev:
                params["events"] = ev
                filled = True
                if not params.get("config"):
                    cfg = self._extract_config_from_text(user_input)
                    if cfg:
                        params["config"] = cfg
        elif field == "modules":
            mods = self._extract_modules_from_text(user_input)
            if mods:
                params["modules"] = mods
                filled = True

        if not filled:
            return None
        self._pending = {"tool": tool_name, "params": params, "reason": pend.get("reason", "")}
        return tool_name, params, pend.get("reason", "")

    @staticmethod
    def _extract_run_number_from_text(text: str) -> Optional[int]:
        m = re.search(r'(?:run|런|번)\s*(\d{3,})', text, re.IGNORECASE)
        if not m:
            m = re.search(r'(\d{4,})', text)  # 단독 4자리 이상 숫자
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_method_from_text(text: str) -> Optional[str]:
        t = text.lower()
        if re.search(r'peak|피크', t):
            return "PeakADC"
        if re.search(r'int|적분|integral', t):
            return "IntADC"
        return None

    # config 후보에서 제외할 일반 단어 (DAQ 관련 영어 표현들)
    _CONFIG_STOPWORDS = {
        "daq", "run", "runs", "event", "events", "evt", "evts", "data",
        "start", "take", "collect", "get", "fire", "acquire", "please", "now", "with",
    }

    @classmethod
    def _extract_config_from_text(cls, text: str) -> Optional[str]:
        # config 이름은 임의 문자열(test, setup1, config1, physics ...)일 수 있다.
        # [A-Za-z0-9_-]로 제한 — \w는 한글 조사("test로")까지 매칭하므로 사용 금지
        # 1) 명시 키워드: "config setup1", "컨피그 test", "config=abc"
        m = re.search(r'(?:config|컨피그|콘피그)\s*[:=]?\s*([A-Za-z][A-Za-z0-9_-]*)', text, re.IGNORECASE)
        if m and m.group(1).lower() not in cls._CONFIG_STOPWORDS:
            return m.group(1)
        # 2) "<이름> 설정으로 / <이름> 셋업으로" 형태
        m = re.search(r'([A-Za-z][A-Za-z0-9_-]*)\s*(?:설정|셋업|세팅)', text, re.IGNORECASE)
        if m and m.group(1).lower() not in cls._CONFIG_STOPWORDS:
            return m.group(1)
        # 3) "<이름>으로/로 ..." 형태 (예: "setup1로 10000개", "test로 돌려줘")
        m = re.search(r'([A-Za-z][A-Za-z0-9_-]*)(?:으로|로)(?=[\s,.!?]|$)', text, re.IGNORECASE)
        if m and m.group(1).lower() not in cls._CONFIG_STOPWORDS:
            return m.group(1)
        return None

    @staticmethod
    def _extract_events_from_text(text: str) -> Optional[int]:
        # (?<![A-Za-z]) — "setup1" 같은 config 이름 속 숫자를 이벤트 수로 오인하지 않도록
        m = re.search(r'(?<![A-Za-z])(\d+)\s*(k|천|만)?', text, re.IGNORECASE)
        if not m:
            return None
        n = int(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit == "k" or unit == "천":
            n *= 1000
        elif unit == "만":
            n *= 10000
        return n if n > 0 else None

    @staticmethod
    def _extract_modules_from_text(text: str) -> list:
        # M1-T1-C / M1T1S / M3 등
        mods = re.findall(r'M\d(?:-?T\d)?(?:-?[CS])?', text, re.IGNORECASE)
        return [m.upper() for m in mods] if mods else []

    @staticmethod
    def _format_dispatch_msg(tool_name: str, params: dict, reason: str) -> str:
        lines = [f"Tool: {tool_name}"]
        if params:
            for k, v in params.items():
                lines.append(f"  {k}: {v}")
        if reason:
            lines.append(reason)
        return "\n".join(lines)

    @staticmethod
    def _format_confirm_preview(tool_name: str, params: dict) -> str:
        if tool_name == "daq_run":
            events = params.get("events", "?")
            config = params.get("config", "setup")
            events_str = f"{events:,}" if isinstance(events, int) else f"{events}"
            return f"DAQ 실행\n  Config: {config}\n  이벤트 수: {events_str}"
        if tool_name == "hv_write":
            cmd = params.get("command", "?")
            ch = params.get("channels", "?")
            ch_str = ch if isinstance(ch, str) else ", ".join(ch)
            if cmd == "voltage":
                v = params.get("voltage", "?")
                return f"HV 전압 변경\n  채널: {ch_str}\n  전압: {v} V"
            if cmd == "current":
                c = params.get("current", "?")
                return f"HV 전류 변경\n  채널: {ch_str}\n  전류: {c} μA"
            if cmd == "on":
                return f"HV 켜기\n  채널: {ch_str}"
            if cmd == "off":
                return f"HV 끄기\n  채널: {ch_str}"
        return f"{tool_name}\n  params: {params}"

    @staticmethod
    def _validate_params(tool_name: str, params: dict):
        if tool_name == "daq_run":
            events = params.get("events")
            if not events or (isinstance(events, (int, float)) and events <= 0):
                return ("events", "몇 개의 이벤트를 수집할까요?")
        if tool_name == "dqm_plot":
            if not params.get("run_number"):
                return ("run_number", "어떤 런 번호의 DQM 플롯을 그릴까요?")
            if not params.get("method"):
                return ("method", "IntADC로 그릴까요, PeakADC로 그릴까요?")
            if params.get("type") == "single" and not params.get("modules"):
                return ("modules", "어떤 채널을 그릴까요? (예: M1, M1-T1-C, M1-T1-S)")
        if tool_name == "run_log":
            if not params.get("run_num"):
                return ("run_num", "어떤 런 번호의 로그를 처리할까요?")
        return None

    def _apply_fallback_rules(self, user_input: str) -> Optional[dict]:
        """모델이 tool:none을 출력했을 때 규칙 기반으로 재시도."""
        u = user_input.lower()

        # HV status check (English / Korean)
        if re.search(r'hv.*(status|check)', u) or re.search(r'(hv|고압).*(확인|상태)', u):
            return {"tool": "hv_read", "params": {"command": "status"}, "reason": "HV 상태 확인"}

        # 방금 / 이번 / last run + plot
        if re.search(r'(방금|이번|last)', u) and re.search(r'(plot|플롯|그려|그래프)', u):
            run_num = (self.shared_state.get("last_run")
                       or self.shared_state.get("last_run_number")
                       or self._last_daq_run)
            if run_num:
                return {"tool": "dqm_plot",
                        "params": {"run_number": int(run_num), "type": "full"},
                        "reason": f"Run {run_num} 플롯 (방금 런)"}

        return None

    def _ask_clarify(self, question: str, io, timeout: int = 120) -> Optional[str]:
        while not self.clarify_queue.empty():  # drain stale replies
            try: self.clarify_queue.get_nowait()
            except queue.Empty: break
        io.output_queue.put({"type": "adhoc_clarify", "question": question})
        io.send_status("추가 정보 입력 대기 중...")
        try:
            return self.clarify_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _execute_tool(self, tool_name: str, params: dict, io: WebSocketIO, max_retries: int = 3):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                if tool_name == "daq_run":
                    from tools.daq_tool import DAQRunTool
                    io.send_status("DAQ 실행 중...")
                    daq_lines: list[str] = []
                    def _daq_cb(line: str):
                        daq_lines.append(line)
                        io.send_tool_output(line)
                    DAQRunTool().execute(params, line_callback=_daq_cb)
                    for _line in daq_lines:
                        m = re.search(r'Run:\s*(\d+)', _line)
                        if m:
                            self._last_daq_run = int(m.group(1))
                            break

                elif tool_name == "dqm_plot":
                    from tools.dqm_tool import DQMPlotTool
                    from tools.dqm_live_worker import OUTPUT_DIR
                    run_number = int(params.get("run_number", 0))
                    method = params.get("method", "IntADC")
                    type_ = params.get("type", "full")
                    modules = params.get("modules", [])
                    io.send_status("DQM 플롯 생성 중...")

                    result = DQMPlotTool().execute(params)
                    io.send_tool_output(result)

                    if type_ == "full":
                        base_prefix = f"Run{run_number}_full_{method}"
                    elif type_ in ("heatmap", "module"):
                        mod = modules[0] if modules else "MCPPMT"
                        base_prefix = f"Run{run_number}_{type_}_{method}_{mod}"
                    else:
                        base_prefix = f"Run{run_number}_single_{method}_"

                    pfx = f"{base_prefix}_"
                    canvases = [
                        p.name[len(pfx):-len(".json")]
                        for p in sorted(OUTPUT_DIR.glob(f"{base_prefix}_*.json"))
                        if p.name.endswith(".json")
                    ]
                    if canvases:
                        io.output_queue.put({
                            "type": "dqm_canvases",
                            "base_prefix": base_prefix,
                            "canvases": canvases,
                            "run_number": run_number,
                        })
                    else:
                        io.send_ai_message(f"Run {run_number} DQM JSON 파일을 찾을 수 없습니다.")

                elif tool_name in ("run_log", "run_log_read"):
                    from tools.run_log_tool import RunLogTool
                    result = RunLogTool().execute(params)
                    io.send_tool_output(result)

                elif tool_name in ("hv_read", "hv_status"):
                    from tools.hv_control_tool import HVControlTool
                    # hv_read는 항상 status. 모델이 "read"/"write" 등 엉뚱한 command를
                    # 내보내도 무조건 status로 강제한다.
                    params["command"] = "status"
                    result = HVControlTool().execute(params)
                    io.send_tool_output(result)

                elif tool_name == "hv_write":
                    from tools.hv_control_tool import HVControlTool
                    io.send_status("HV 변경 중...")
                    result = HVControlTool().execute(params)
                    io.send_tool_output(result)

                else:
                    io.send_ai_message(f"알 수 없는 도구: {tool_name}")

                return

            except RuntimeError as e:
                last_error = e
                self.log(f"[Retry {attempt}/{max_retries}] Tool '{tool_name}' 실패: {e}")
                if attempt < max_retries:
                    time.sleep(2)

            except Exception as e:
                io.send_ai_message(f"도구 실행 중 오류: {e}")
                return

        io.send_tool_error(tool_name, str(last_error), max_retries)


class _BrainOutputQueue:
    """모든 메시지에 source='brain'을 태깅해서 output_queue로 전달."""

    def __init__(self, real_queue: queue.Queue):
        self._q = real_queue

    def put(self, item):
        if isinstance(item, dict):
            item = {**item, "source": "brain"}
        self._q.put(item)

    def get(self, *args, **kwargs):
        return self._q.get(*args, **kwargs)

    def get_nowait(self):
        return self._q.get_nowait()

    def empty(self):
        return self._q.empty()


def run_brain_thread(
    brain_agent: BrainAgent,
    adhoc_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    confirm_queue: Optional[queue.Queue] = None,
    clarify_queue: Optional[queue.Queue] = None,
):
    if confirm_queue is not None:
        brain_agent.confirm_queue = confirm_queue
    if clarify_queue is not None:
        brain_agent.clarify_queue = clarify_queue
    tagged_queue = _BrainOutputQueue(output_queue)
    io = WebSocketIO(
        input_queue=queue.Queue(),  # 입력 없음 — BrainAgent는 단방향 dispatch
        output_queue=tagged_queue,
        stop_event=stop_event,
    )

    while not stop_event.is_set():
        try:
            user_input = adhoc_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if user_input in ("종료", "exit"):
            break

        try:
            brain_agent.handle_request(user_input, io)
        except Exception:
            tb = traceback.format_exc()
            output_queue.put({"type": "error", "content": f"BrainAgent error:\n{tb}"})
