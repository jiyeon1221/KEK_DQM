#!/usr/bin/env python3
"""
Training data generator for HV Equalization Agent

build_full_context / _build_state_context / _get_step_hint 포맷이
HVEqualizationAgent(hv_equalization_agent.py)와 완전히 동일하도록 유지.
"""

import json
import random
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MSG_PLOT_CONFIRM, MSG_HV_CONFIRM

# HV 로직은 타워마다 동일 — 대표 타워 6개로 학습 (모듈·타워번호 다양성 확보)
TOWER_ORDER = ["M1T1", "M2T3", "M3T2", "M5T3", "M7T4", "M9T2"]

MESSAGE_MOVE_REQ     = "x = {x:.3f} mm, y = {y:.3f} mm 으로 이동해주세요."
MESSAGE_PLOT_CONFIRM = MSG_PLOT_CONFIRM
MESSAGE_HV_CONFIRM   = MSG_HV_CONFIRM
# {c}/{s}: 타워별 채널명 (예: M1T1C/M1T1S). 호출부에서 c=f"{tower}C", s=f"{tower}S" 전달.
MESSAGE_APPROVE_BOTH = "분석 결과, 현재 ADC: {c}={adc_c:.1f}, {s}={adc_s:.1f} (목표: {target}). HV 변경 제안: {c} {hv_c_old}V→{hv_c_new}V, {s} {hv_s_old}V→{hv_s_new}V. 적용하시겠습니까?"
MESSAGE_APPROVE_C    = "분석 결과, 현재 ADC: {c}={adc_c:.1f} (목표: {target}). HV 변경 제안: {c} {hv_c_old}V→{hv_c_new}V. ({s} 완료) 적용하시겠습니까?"
MESSAGE_APPROVE_S    = "분석 결과, 현재 ADC: {s}={adc_s:.1f} (목표: {target}). HV 변경 제안: {s} {hv_s_old}V→{hv_s_new}V. ({c} 완료) 적용하시겠습니까?"


def random_events() -> int:
    digits = random.randint(3, 6)
    return random.randint(10 ** (digits - 1), 10 ** digits - 1)


def _make_system_prompt(tower: str, x: float, y: float) -> str:
    t = tower
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
  Only C not done: {{"message": "분석 결과, 현재 ADC: {t}C=<adc_c> (목표: <target>). HV 변경 제안: {t}C <old_c>V→<new_c>V. ({t}S 완료) 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  Only S not done: {{"message": "분석 결과, 현재 ADC: {t}S=<adc_s> (목표: <target>). HV 변경 제안: {t}S <old_s>V→<new_s>V. ({t}C 완료) 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  CRITICAL: Copy the "현재→제안" arrow (e.g. C 775V→785V) EXACTLY from the state's "HV 변경 제안" line — keep the old→new order, do NOT swap the two numbers. Use EXACT ADC values (last_adc_c/s). NEVER fabricate numbers.
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


def _build_state_context(state: Dict) -> str:
    lines = []
    tower = state["current_tower"]
    pos_x = state.get("tower_pos", {}).get("x", 0.0)
    pos_y = state.get("tower_pos", {}).get("y", 0.0)
    adc_known = state.get("last_adc_c") is not None
    suggest_pending = state.get("last_suggested_hv_c") is not None
    phase = state.get("phase", "idle")

    # Step 1a detection
    if state.get("last_hv_c") is None:
        if not state.get("y_confirmed"):
            lines.append(f"*** REQUIRED NEXT: position move message (step 1a) — ask user to move to {tower} position ***")
        else:
            lines.append(f"*** REQUIRED NEXT: hv_execute_tool status (step 1b) — position confirmed, check HV now ***")
        lines.append("")

    if state.get("needs_plot_confirm"):
        lines.append(f"*** REQUIRED NEXT: plot confirmation message (step 1c-plot) — DAQ done, send plot confirm BEFORE suggest ***")
        lines.append(f'*** output: {{"message": "{MSG_PLOT_CONFIRM}"}} ***')
        lines.append("")
    elif state.get("needs_suggest"):
        lines.append(f"*** REQUIRED NEXT: hv_equalization_suggest (step 1d) — plot confirmed, analyze NOW ***")
        lines.append(f"*** DO NOT call daq_run_tool again — call hv_equalization_suggest first ***")
        lines.append("")
    elif adc_known:
        done_c = state.get("channel_done_c", False)
        done_s = state.get("channel_done_s", False)
        adc_c = state["last_adc_c"]
        adc_s = state["last_adc_s"]
        target = state.get("target_adc_c")
        if done_c and done_s:
            lines.append(f"*** CONVERGENCE: C=True, S=True — CALL hv_equalization_done_channel NOW ***")
        elif suggest_pending and phase == "approving":
            lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
            lines.append(f"*** REQUIRED NEXT: hv_execute_tool voltage (step 1f) ***")
        elif suggest_pending:
            lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
            lines.append(f"*** REQUIRED NEXT: approval message (step 1e) ***")
        else:
            lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
            lines.append(f"*** REQUIRED NEXT: daq_run_tool (step 1c) ***")
        lines.append("")

    lines.append(f"Phase: {phase}")
    lines.append(f"Tower: {tower} (x:{pos_x:.3f}, y:{pos_y:.3f})  [Position confirmed: {state.get('y_confirmed', False)}]")
    lines.append(f"Beam Energy: {state.get('beam_energy')} GeV")
    lines.append(f"Target Events: {state.get('target_events')}")
    lines.append(f"Target ADC: {state.get('target_adc_c')}")
    lines.append(f"Last HV: C={state.get('last_hv_c')}V, S={state.get('last_hv_s')}V")
    lines.append(f"needs_plot_confirm: {state.get('needs_plot_confirm', False)}")
    if state.get("last_suggested_hv_c") is not None:
        dc = state.get("channel_done_c", False)
        ds = state.get("channel_done_s", False)
        oc, nc = state.get("last_hv_c"), state["last_suggested_hv_c"]
        os_, ns = state.get("last_hv_s"), state["last_suggested_hv_s"]
        c_str = "C 완료" if dc else f"C {oc:.0f}V→{nc}V"
        s_str = "S 완료" if ds else f"S {os_:.0f}V→{ns}V"
        lines.append(f"HV 변경 제안 (현재→제안, 이 화살표를 그대로 승인 메시지에 복사): {c_str}, {s_str}")
    if state.get("last_run_number"):
        lines.append(f"Last Run Number: {state['last_run_number']}")
    lines.append(f"Iterations: {state.get('iterations', 0)}")
    return "\n".join(lines)


def _build_history_context(history: List[Dict]) -> str:
    if not history:
        return "(No conversation yet)"
    lines = []
    for msg in history[-10:]:
        role = "User" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _get_step_hint(state: Dict) -> str:
    tower = state["current_tower"]
    adc_known = state.get("last_adc_c") is not None
    suggest_pending = state.get("last_suggested_hv_c") is not None
    done_c = state.get("channel_done_c", False)
    done_s = state.get("channel_done_s", False)
    phase = state.get("phase", "idle")
    base = f"Phase: {phase} | Tower: {tower}"

    if state.get("last_hv_c") is None:
        if not state.get("y_confirmed"):
            return f"{base} | REQUIRED NEXT: position move message (step 1a)"
        else:
            return f"{base} | REQUIRED NEXT: hv_execute_tool status (step 1b — position confirmed)"
    elif adc_known and done_c and done_s:
        return f"{base} | CONVERGED → call hv_equalization_done_channel (step 1h)"
    elif adc_known and suggest_pending and phase == "approving":
        return f"{base} | REQUIRED NEXT: hv_execute_tool voltage (step 1f — user already confirmed)"
    elif adc_known and suggest_pending:
        return f"{base} | REQUIRED NEXT: approval message (step 1e)"
    elif state.get("needs_plot_confirm"):
        return f"{base} | REQUIRED NEXT: plot confirmation message (step 1c-plot — DAQ done, send plot confirm before suggest)"
    elif state.get("needs_suggest"):
        return f"{base} | REQUIRED NEXT: hv_equalization_suggest (step 1d — plot confirmed, analyze now)"
    elif adc_known:
        return f"{base} | REQUIRED NEXT: daq_run_tool (step 1c)"
    else:
        return f"{base} | REQUIRED NEXT: daq_run_tool (step 1c — first DAQ)"


def build_full_context(state: Dict, history: List[Dict], current_input: Optional[str] = None) -> str:
    if current_input is None and history and history[-1]["role"] == "user":
        current_input = history[-1]["content"]
        temp_history = history[:-1]
    else:
        temp_history = history

    parts = ["=== Current State ===", _build_state_context(state), ""]
    parts.append("=== Recent Conversation ===")
    parts.append(_build_history_context(temp_history))
    parts.append("")
    if current_input:
        parts.append("=== Current User Input ===")
        parts.append(current_input)
        parts.append("")
    parts.append("=== Your Task ===")
    parts.append(_get_step_hint(state))
    parts.append("")
    parts.append("Output JSON with tool name and parameters.")
    return "\n".join(parts)


def make_example(state: Dict, history: List[Dict], decision: Dict,
                 current_input: Optional[str] = None) -> Dict:
    tower = state["current_tower"]
    pos_x = state.get("tower_pos", {}).get("x", 0.0)
    pos_y = state.get("tower_pos", {}).get("y", 0.0)
    system_prompt = _make_system_prompt(tower, pos_x, pos_y)
    ctx = build_full_context(state, history, current_input)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": ctx},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ]
    }


CONVERGENCE_PATTERNS = [
    [(True,  True)],
    [(False, False), (True,  True)],
    [(True,  False), (True,  True)],
    [(False, True),  (True,  True)],
    [(False, False), (False, False), (True,  True)],
    [(False, False), (True,  False), (True,  True)],
    [(False, False), (False, True),  (True,  True)],
    [(False, False), (False, False), (False, False), (True, True)],
    [(True,  False), (True,  False), (True,  True)],
    [(False, False), (True,  True)],
    [(False, True),  (False, True),  (True,  True)],
    [(False, False), (False, False), (True,  False), (True, True)],
]

# Manual HV adjustment scenarios for LLM-based update_state flow
# (text, kind, value)
# kind: C_abs, S_abs, C_delta, S_delta, both_abs, cs_abs
MANUAL_ADJUST_SCENARIOS = [
    ("C를 810으로 바꿔줘",          "C_abs",   810),
    ("S를 820으로 설정해줘",         "S_abs",   820),
    ("C를 15 올려줘",               "C_delta",  15),
    ("S를 10 내려줘",               "S_delta", -10),
    ("둘 다 850으로 해줘",           "both_abs", 850),
    ("C 800 S 820으로 바꿔",        "cs_abs",  (800, 820)),
    ("C를 830으로 변경해줘",         "C_abs",   830),
    ("S를 840으로 해줘",             "S_abs",   840),
    ("C 20 올려줘",                 "C_delta",  20),
    ("S를 5 올려",                  "S_delta",   5),
    ("모두 860으로 바꿔줘",          "both_abs", 860),
    ("C 790 S 810으로 설정해",      "cs_abs",  (790, 810)),
    ("C를 25 내려줘",               "C_delta", -25),
    ("S 30 내려",                   "S_delta", -30),
    ("C 835 S 845로 해줘",          "cs_abs",  (835, 845)),
]


def _apply_manual_adjust(hv_c: int, hv_s: int, kind: str, value, done_c: bool, done_s: bool):
    """Apply manual adjustment request. Returns (new_hv_c, new_hv_s)."""
    new_c, new_s = hv_c, hv_s
    if kind == "C_abs" and not done_c:
        new_c = int(value)
    elif kind == "S_abs" and not done_s:
        new_s = int(value)
    elif kind == "C_delta" and not done_c:
        new_c = int(hv_c + value)
    elif kind == "S_delta" and not done_s:
        new_s = int(hv_s + value)
    elif kind == "both_abs":
        if not done_c:
            new_c = int(value)
        if not done_s:
            new_s = int(value)
    elif kind == "cs_abs":
        cv, sv = value
        if not done_c:
            new_c = int(cv)
        if not done_s:
            new_s = int(sv)
    return new_c, new_s


def _build_approval_msg(done_c: bool, done_s: bool,
                        adc_c: float, adc_s: float, target: int,
                        hv_c_old: int, hv_s_old: int, hv_c_new: int, hv_s_new: int,
                        tower: str) -> str:
    c, s = f"{tower}C", f"{tower}S"
    if not done_c and not done_s:
        return MESSAGE_APPROVE_BOTH.format(
            c=c, s=s, adc_c=adc_c, adc_s=adc_s, target=target,
            hv_c_old=hv_c_old, hv_c_new=hv_c_new, hv_s_old=hv_s_old, hv_s_new=hv_s_new,
        )
    elif done_c:
        return MESSAGE_APPROVE_S.format(
            c=c, s=s, adc_s=adc_s, target=target,
            hv_s_old=hv_s_old, hv_s_new=hv_s_new,
        )
    else:
        return MESSAGE_APPROVE_C.format(
            c=c, s=s, adc_c=adc_c, target=target,
            hv_c_old=hv_c_old, hv_c_new=hv_c_new,
        )


def _hv_delta(adc_frac: float, going_down: bool) -> int:
    if going_down:
        return random.randint(-15, -6)
    else:
        return random.randint(6, 18)


def generate_workflow(tower: str, x: float, y: float,
                      energy: float, events: int, target_adc: float,
                      pattern: List, going_down: bool = False,
                      start_iter: int = 0) -> List[Dict]:
    examples = []
    history = []

    hv_c = random.randint(820, 880)
    hv_s = random.randint(820, 880)
    run_number = random.randint(10000, 19999)

    state = {
        "phase": "idle",
        "beam_energy": energy,
        "target_events": events,
        "target_adc_c": target_adc,
        "target_adc_s": target_adc,
        "current_tower": tower,
        "tower_pos": {"x": x, "y": y},
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
    }

    # start_iter > 0: 이미 완료된 반복을 히스토리에 압축
    if start_iter > 0:
        for idx in range(min(start_iter, len(pattern) - 1)):
            iter_done_c, iter_done_s = pattern[idx]
            adc_frac = 0.80 + 0.06 * idx if not going_down else 1.10 - 0.05 * idx
            adc_c = round(target_adc * adc_frac + random.uniform(-5, 5), 1)
            adc_s = round(target_adc * adc_frac + random.uniform(-5, 5), 1)
            delta_c = _hv_delta(adc_frac, going_down)
            delta_s = _hv_delta(adc_frac, going_down)
            next_hv_c = hv_c + delta_c
            next_hv_s = hv_s + delta_s

            if idx >= start_iter - 2:
                history.append({"role": "assistant", "content": json.dumps(
                    {"tool": "daq_run_tool", "params": {"events": events, "pos_h": x, "pos_v": y, "beam_energy": energy}},
                    ensure_ascii=False)})
                history.append({"role": "assistant", "content": json.dumps(
                    {"message": MESSAGE_PLOT_CONFIRM}, ensure_ascii=False)})
                history.append({"role": "user", "content": "완료"})
                history.append({"role": "assistant", "content": json.dumps(
                    {"tool": "hv_equalization_suggest", "params": {"run_number": run_number, "tower": tower}},
                    ensure_ascii=False)})
                history.append({"role": "assistant", "content": json.dumps(
                    {"message": MESSAGE_APPROVE_BOTH.format(
                        c=f"{tower}C", s=f"{tower}S",
                        adc_c=adc_c, adc_s=adc_s, target=int(target_adc),
                        hv_c_old=hv_c, hv_c_new=next_hv_c, hv_s_old=hv_s, hv_s_new=next_hv_s),
                     "update_state": {"phase": "approving"}},
                    ensure_ascii=False)})
                history.append({"role": "user", "content": "완료"})
                history.append({"role": "assistant", "content": json.dumps(
                    {"tool": "hv_execute_tool",
                     "params": {"command": "voltage", "channel_values": {f"{tower}C": next_hv_c, f"{tower}S": next_hv_s}},
                     "update_state": {"phase": "equalizing"}},
                    ensure_ascii=False)})
                history.append({"role": "assistant", "content": json.dumps(
                    {"message": MESSAGE_HV_CONFIRM}, ensure_ascii=False)})
                history.append({"role": "user", "content": "완료"})

            hv_c = next_hv_c
            hv_s = next_hv_s
            run_number += 1

        state["last_hv_c"] = hv_c
        state["last_hv_s"] = hv_s
        state["last_run_number"] = run_number - 1
        state["iterations"] = start_iter
        state["y_confirmed"] = True  # 중간 상태에서는 이미 위치 확인 완료
        state["needs_suggest"] = False  # 압축된 히스토리는 suggest 완료 상태

    # ── 1a: Position move message ─────────────────────────────────────
    if start_iter == 0:
        dec = {"message": MESSAGE_MOVE_REQ.format(x=x, y=y)}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        state["y_confirmed"] = True

        # ── 1b: Status check (y_confirmed는 시스템이 설정 — update_state 없음) ──
        dec = {"tool": "hv_execute_tool", "params": {"command": "status", "channels": [f"{tower}C", f"{tower}S"]}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["last_hv_c"] = hv_c
        state["last_hv_s"] = hv_s

    # ── Inner loop (1c~1h) ───────────────────────────────────────────
    for iter_idx, (done_c, done_s) in enumerate(pattern):
        adc_frac = 0.80 + 0.07 * (iter_idx + start_iter) if not going_down else 1.12 - 0.06 * (iter_idx + start_iter)
        adc_c = round(target_adc * adc_frac + random.uniform(-8, 8), 1)
        adc_s = round(target_adc * adc_frac + random.uniform(-8, 8), 1)

        if done_c and done_s:
            adc_c = round(target_adc * random.uniform(0.990, 1.010), 1)
            adc_s = round(target_adc * random.uniform(0.990, 1.010), 1)
        elif done_c:
            adc_c = round(target_adc * random.uniform(0.990, 1.010), 1)
        elif done_s:
            adc_s = round(target_adc * random.uniform(0.990, 1.010), 1)

        delta_c = 0 if done_c else _hv_delta(adc_frac, going_down)
        delta_s = 0 if done_s else _hv_delta(adc_frac, going_down)
        next_hv_c = hv_c + delta_c
        next_hv_s = hv_s + delta_s

        # 1c: DAQ
        dec = {"tool": "daq_run_tool", "params": {"events": events, "pos_h": x, "pos_v": y, "beam_energy": energy}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["last_run_number"] = run_number
        state["iterations"] = (state.get("iterations") or 0) + 1
        state["needs_suggest"] = False
        state["needs_plot_confirm"] = True   # DAQ 완료 → 먼저 plot 확인

        # 1c-plot: plot confirmation message
        dec = {"message": MESSAGE_PLOT_CONFIRM}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        state["needs_plot_confirm"] = False
        state["needs_suggest"] = True        # plot 확인 완료 → suggest 진행

        # 1d: Suggest
        dec = {"tool": "hv_equalization_suggest", "params": {"run_number": run_number, "tower": tower}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["needs_suggest"] = False  # suggest 완료
        state["last_adc_c"] = adc_c
        state["last_adc_s"] = adc_s
        state["last_suggested_hv_c"] = next_hv_c
        state["last_suggested_hv_s"] = next_hv_s
        state["channel_done_c"] = done_c
        state["channel_done_s"] = done_s

        if done_c and done_s:
            # 1h: Done
            dec = {"tool": "hv_equalization_done_channel", "params": {"channels": "all"}}
            examples.append(make_example(state, history, dec))
            history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
            break

        # 1e: Approval message
        msg = _build_approval_msg(done_c, done_s, adc_c, adc_s,
                                  int(target_adc), hv_c, hv_s, next_hv_c, next_hv_s, tower)
        dec = {"message": msg, "update_state": {"phase": "approving"}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})

        apply_hv_c, apply_hv_s = next_hv_c, next_hv_s

        history.append({"role": "user", "content": "완료"})
        state["phase"] = "approving"

        # 1f: Apply voltage
        cv = {}
        if not done_c:
            cv[f"{tower}C"] = apply_hv_c
        if not done_s:
            cv[f"{tower}S"] = apply_hv_s
        dec = {"tool": "hv_execute_tool",
               "params": {"command": "voltage", "channel_values": cv},
               "update_state": {"phase": "equalizing"}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["last_hv_c"] = apply_hv_c
        state["last_hv_s"] = apply_hv_s
        state["last_suggested_hv_c"] = None
        state["last_suggested_hv_s"] = None
        state["last_adc_c"] = None
        state["last_adc_s"] = None
        state["channel_done_c"] = done_c
        state["channel_done_s"] = done_s
        state["phase"] = "equalizing"

        # 1g: Confirmation
        dec = {"message": MESSAGE_HV_CONFIRM}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        state["phase"] = "idle"

        hv_c = apply_hv_c
        hv_s = apply_hv_s
        run_number += 1

    return examples


def generate_manual_adjust_workflow(tower: str, x: float, y: float,
                                    energy: float, events: int, target_adc: float,
                                    pattern_step: tuple,
                                    adjust_scenario: tuple) -> List[Dict]:
    """
    Generate training examples for manual HV adjustment (LLM update_state flow).

    Flow after approval message is shown:
      User says adjustment text (e.g. "C를 810으로 바꿔줘")
      → LLM outputs updated approval message with update_state for new HV values
      → User says "완료"
      → LLM applies voltage
    """
    examples = []
    history = []

    hv_c = random.randint(820, 880)
    hv_s = random.randint(820, 880)
    run_number = random.randint(10000, 19999)

    done_c, done_s = pattern_step
    # Skip if both are done (no approval needed)
    if done_c and done_s:
        return examples

    state = {
        "phase": "idle",
        "beam_energy": energy,
        "target_events": events,
        "target_adc_c": target_adc,
        "target_adc_s": target_adc,
        "current_tower": tower,
        "tower_pos": {"x": x, "y": y},
        "y_confirmed": True,
        "last_hv_c": hv_c,
        "last_hv_s": hv_s,
        "last_suggested_hv_c": None,
        "last_suggested_hv_s": None,
        "last_adc_c": None,
        "last_adc_s": None,
        "channel_done_c": done_c,
        "channel_done_s": done_s,
        "last_run_number": run_number,
        "iterations": 1,
        "done": False,
        "needs_suggest": False,
        "needs_plot_confirm": False,
    }

    # Build partial history: position move, status, DAQ, suggest already done
    adc_frac = random.uniform(0.82, 0.95)
    adc_c = round(target_adc * adc_frac + random.uniform(-8, 8), 1)
    adc_s = round(target_adc * adc_frac + random.uniform(-8, 8), 1)
    delta_c = _hv_delta(adc_frac, False)
    delta_s = _hv_delta(adc_frac, False)
    next_hv_c = hv_c + (delta_c if not done_c else 0)
    next_hv_s = hv_s + (delta_s if not done_s else 0)

    # Prepopulate history with steps 1a through 1d
    history.append({"role": "assistant", "content": json.dumps(
        {"message": MESSAGE_MOVE_REQ.format(x=x, y=y)}, ensure_ascii=False)})
    history.append({"role": "user", "content": "완료"})
    history.append({"role": "assistant", "content": json.dumps(
        {"tool": "hv_execute_tool", "params": {"command": "status", "channels": [f"{tower}C", f"{tower}S"]}},
        ensure_ascii=False)})
    history.append({"role": "assistant", "content": json.dumps(
        {"tool": "daq_run_tool", "params": {"events": events, "pos_h": x, "pos_v": y, "beam_energy": energy}},
        ensure_ascii=False)})
    history.append({"role": "assistant", "content": json.dumps(
        {"message": MESSAGE_PLOT_CONFIRM}, ensure_ascii=False)})
    history.append({"role": "user", "content": "완료"})
    history.append({"role": "assistant", "content": json.dumps(
        {"tool": "hv_equalization_suggest", "params": {"run_number": run_number, "tower": tower}},
        ensure_ascii=False)})

    # State after suggest
    state["last_adc_c"] = adc_c
    state["last_adc_s"] = adc_s
    state["last_suggested_hv_c"] = next_hv_c
    state["last_suggested_hv_s"] = next_hv_s

    # Step 1e: approval message (agent output)
    approval_msg = _build_approval_msg(done_c, done_s, adc_c, adc_s,
                                       int(target_adc), hv_c, hv_s, next_hv_c, next_hv_s, tower)
    dec_approval = {"message": approval_msg, "update_state": {"phase": "approving"}}
    # This is NOT a training example itself; it's what the agent already said
    history.append({"role": "assistant", "content": json.dumps(dec_approval, ensure_ascii=False)})
    state["phase"] = "approving"

    # Now: user sends manual adjustment request
    adj_text, adj_kind, adj_value = adjust_scenario
    history.append({"role": "user", "content": adj_text})

    # Compute new adjusted HV values
    adj_hv_c, adj_hv_s = _apply_manual_adjust(
        next_hv_c, next_hv_s, adj_kind, adj_value, done_c, done_s
    )

    # Training example: LLM must output updated approval + update_state
    updated_approval_msg = _build_approval_msg(done_c, done_s, adc_c, adc_s,
                                               int(target_adc), hv_c, hv_s, adj_hv_c, adj_hv_s, tower)
    update_state_dict: Dict[str, Any] = {}
    if not done_c:
        update_state_dict["last_suggested_hv_c"] = adj_hv_c
    if not done_s:
        update_state_dict["last_suggested_hv_s"] = adj_hv_s

    dec_update = {"message": updated_approval_msg, "update_state": update_state_dict}
    examples.append(make_example(state, history, dec_update, current_input=adj_text))

    # Continue history: agent outputs updated approval
    history.append({"role": "assistant", "content": json.dumps(dec_update, ensure_ascii=False)})
    state["last_suggested_hv_c"] = adj_hv_c
    state["last_suggested_hv_s"] = adj_hv_s

    # User says 완료
    history.append({"role": "user", "content": "완료"})

    # Training example: LLM applies voltage (step 1f)
    cv = {}
    if not done_c:
        cv[f"{tower}C"] = adj_hv_c
    if not done_s:
        cv[f"{tower}S"] = adj_hv_s
    dec_voltage = {"tool": "hv_execute_tool",
                   "params": {"command": "voltage", "channel_values": cv},
                   "update_state": {"phase": "equalizing"}}
    examples.append(make_example(state, history, dec_voltage, current_input="완료"))

    return examples


def main():
    output_file = Path(__file__).parent / "data" / "hv_equalization_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_ex = []
    TARGET_ADC_CHOICES = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    ENERGY_CHOICES = [10, 20, 50, 100, 200, 250]

    # ── Main workflow generation ────────────────────────────────────────
    # All towers, all convergence patterns, multiple events/energy combos
    for tower in TOWER_ORDER:
        x = random.uniform(70.0, 130.0)
        y = random.uniform(70.0, 130.0)
        for pattern in CONVERGENCE_PATTERNS:
            for _ in range(1):  # 1 random event/energy combo per pattern (was 3, scaled for 36 towers)
                energy = random.choice(ENERGY_CHOICES)
                target_adc = random.choice(TARGET_ADC_CHOICES)
                events = random_events()
                all_ex.extend(generate_workflow(tower, x, y, energy, events, target_adc,
                                                pattern, going_down=False))
            # Going down variant (every other tower to reduce volume)
            if TOWER_ORDER.index(tower) % 2 == 0:
                energy = random.choice(ENERGY_CHOICES)
                target_adc = random.choice(TARGET_ADC_CHOICES)
                events = random_events()
                all_ex.extend(generate_workflow(tower, x, y, energy, events, target_adc,
                                                pattern, going_down=True))

    # ── Start-iter variants (mid-workflow training) ─────────────────────
    for tower in TOWER_ORDER:
        x = random.uniform(70.0, 130.0)
        y = random.uniform(70.0, 130.0)
        long_patterns = [
            CONVERGENCE_PATTERNS[4],   # 3-step
            CONVERGENCE_PATTERNS[7],   # 4-step
            CONVERGENCE_PATTERNS[11],  # 4-step
        ]
        for pattern in long_patterns:
            for start_iter in [1, 2, 3]:
                if start_iter >= len(pattern):
                    continue
                for _ in range(1):  # was 2
                    energy = random.choice(ENERGY_CHOICES)
                    target_adc = random.choice(TARGET_ADC_CHOICES)
                    events = random_events()
                    all_ex.extend(generate_workflow(tower, x, y, energy, events, target_adc,
                                                    pattern, going_down=False, start_iter=start_iter))

    # ── Manual HV adjustment examples ──────────────────────────────────
    partial_steps = [
        (False, False),  # both not done
        (True,  False),  # only S not done
        (False, True),   # only C not done
    ]
    for tower in TOWER_ORDER[::2]:  # every other tower (18 of 36) to control volume
        x = random.uniform(70.0, 130.0)
        y = random.uniform(70.0, 130.0)
        for step in partial_steps:
            for scenario in MANUAL_ADJUST_SCENARIOS:
                energy = random.choice(ENERGY_CHOICES)
                target_adc = random.choice(TARGET_ADC_CHOICES)
                events = random_events()
                all_ex.extend(generate_manual_adjust_workflow(
                    tower, x, y, energy, events, target_adc, step, scenario
                ))


    random.shuffle(all_ex)

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    lengths = [sum(len(m["content"]) for m in ex["messages"]) for ex in all_ex]
    print(f"Generated {len(all_ex)} samples -> {output_file}")
    print(f"   char len  max={max(lengths):,}  avg={sum(lengths)/len(lengths):,.0f}")


if __name__ == "__main__":
    main()
