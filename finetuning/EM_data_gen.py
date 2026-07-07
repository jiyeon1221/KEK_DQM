#!/usr/bin/env python3
"""
Training data generator for Energy Scan Agent

build_full_context / _build_state_context / _get_step_hint 포맷이
EnergyScanAgent(energy_scan_agent.py)와 완전히 동일하도록 유지.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MSG_PLOT_CONFIRM

MESSAGE_ASK_ENERGY  = "에너지 설정을 입력해주세요.\n예) 1GeV 50000개 2GeV 200000개 5GeV 100000개  또는  1GeV 80000 3GeV 500000 5GeV 300000"
MESSAGE_MOVE_POS    = "x = {x:.3f} mm, y = {y:.3f} mm 으로 이동해주세요."
MESSAGE_ENERGY_SET  = "빔 에너지를 {energy} GeV로 설정해주세요."
MESSAGE_PLOT_CONFIRM = MSG_PLOT_CONFIRM
MESSAGE_COMPLETE    = "모든 에너지 스캔이 완료되었습니다."

SYSTEM_PROMPT = """You are Energy Scan Agent for test beam experiments.

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


def random_events():
    """Return a uniformly distributed 3-6 digit integer."""
    digits = random.randint(3, 6)
    return random.randint(10**(digits-1), 10**digits - 1)


def _build_state_context(state):
    lines = []
    lines.append(f"Phase: {state['phase']}")
    lines.append(f"Tower: {state['tower']}")
    lines.append(f"M5T3 Position: x={state['t5_x']:.3f}, y={state['t5_y']:.3f}, rot=1.5, tilt=1.0")
    lines.append(f"Position confirmed: {state.get('y_confirmed', False)}")
    lines.append(f"needs_plot_confirm: {state.get('needs_plot_confirm', False)}")
    if state.get("position"):
        lines.append(f"Position: {state['position']}")
    lines.append("")

    ec = state.get("energy_config", {})
    so = state.get("scan_order", [])
    if ec:
        lines.append("Energy Config:")
        for e in so:
            cfg = ec.get(e, {})
            status = "✅" if cfg.get("completed") else "➡️" if e == state.get("current_energy") else "  "
            lines.append(f"  {status} {e} GeV: target={cfg.get('target_events','?')} "
                         f"collected={cfg.get('collected_events',0)} "
                         f"runs={cfg.get('runs',[])} completed={cfg.get('completed',False)}")
    return "\n".join(lines)


def _build_history_context(history):
    if not history:
        return "(No conversation yet)"
    lines = []
    for msg in history[-10:]:
        role = "User" if msg["role"] == "user" else "Agent"
        content = msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_step_hint(state, history):
    """EnergyScanAgent._get_step_hint와 문자 단위로 동일해야 한다."""
    phase = state.get("phase", "config")
    if phase == "config":
        if history:
            return "Phase: config | REQUIRED NEXT: parse user input and update state (step 0b)"
        return "Phase: config | REQUIRED NEXT: ask for energy settings (step 0a)"
    if phase == "idle":
        return f"Phase: idle | REQUIRED NEXT: position move message (step 1a, x={state.get('t5_x', 0):.3f}, y={state.get('t5_y', 0):.3f})"
    current_energy = state.get("current_energy")
    scan_order = state.get("scan_order", [])
    idx = scan_order.index(current_energy) + 1 if current_energy in scan_order else 0
    total = len(scan_order)
    ec = state.get("energy_config", {})
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


def build_full_context(state, history, current_input=None):
    if current_input is None and history and history[-1]["role"] == "user":
        current_input = history[-1]["content"]
        temp_history = history[:-1]
    else:
        temp_history = history

    parts = []
    parts.append("=== Current State ===")
    parts.append(_build_state_context(state))
    parts.append("")
    parts.append("=== Recent Conversation ===")
    parts.append(_build_history_context(temp_history))
    parts.append("")
    if current_input:
        parts.append("=== Current User Input ===")
        parts.append(current_input)
        parts.append("")
    parts.append("=== Your Task ===")
    parts.append(_get_step_hint(state, history))
    parts.append("")
    parts.append("Output JSON with tool name and parameters.")
    return "\n".join(parts)


def make_example(state, history, decision, current_input=None):
    ctx = build_full_context(state, history, current_input)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ctx},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ]
    }


def _emit_energy(examples, state, history, energy, events, t5_x, t5_y, run_number):
    """한 에너지의 모델 결정 턴(set-beam → DAQ → plot msg)을 생성.
    완료(completed) 표시는 코드(시스템)가 소유하므로 모델 턴으로 만들지 않는다."""
    # 2a: set-beam message
    dec = {"message": MESSAGE_ENERGY_SET.format(energy=energy)}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user", "content": "완료"})

    # 2b: DAQ
    dec = {"tool": "daq_run_tool", "params": {
        "events": events,
        "pos_h": t5_x, "pos_v": t5_y,
        "pos_rot": 1.5, "pos_tilt": 1.0, "beam_energy": energy,
    }}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    state["energy_config"][energy]["collected_events"] = events
    state["energy_config"][energy]["runs"].append(run_number)
    state["needs_plot_confirm"] = True

    # 2c: plot confirm message
    dec = {"message": MESSAGE_PLOT_CONFIRM}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user", "content": "완료"})
    # 시스템이 완료 처리 (모델 턴 없음)
    state["needs_plot_confirm"] = False
    state["energy_config"][energy]["completed"] = True


def generate_workflow_normal(energy_list, events_list, user_input):
    examples = []
    history = []
    t5_x = round(random.uniform(70.0, 130.0), 3)
    t5_y = round(random.uniform(70.0, 130.0), 3)
    state = {
        "phase": "config", "tower": "M5T3", "t5_x": t5_x, "t5_y": t5_y,
        "position": {"x": 0.2, "y": -0.3},
        "energy_config": {}, "scan_order": [],
        "current_energy": None, "current_energy_idx": 0,
        "plot_method": "PeakADC", "plot_max_event": None,
        "y_confirmed": False,
        "needs_plot_confirm": False,
    }

    # STEP 0a
    dec = {"message": MESSAGE_ASK_ENERGY}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user",      "content": user_input})

    # STEP 0b (config parse — model-owned; state still phase=config at decision time)
    energy_config_dict = {
        e: {"target_events": ev, "collected_events": 0, "runs": [], "completed": False, "completed_at": None}
        for e, ev in zip(energy_list, events_list)
    }
    dec = {"tool": "none", "update_state": {"energy_config": energy_config_dict, "scan_order": energy_list, "phase": "idle"}}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    state.update(dec["update_state"])

    # STEP 1a: 위치 이동 메시지
    dec = {"message": MESSAGE_MOVE_POS.format(x=t5_x, y=t5_y)}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user",      "content": "완료"})
    # 시스템이 위치 확인 + phase=scanning 처리 (모델 턴 없음)
    state["y_confirmed"] = True
    state["phase"] = "scanning"

    run_number = 100
    for i, energy in enumerate(energy_list):
        state["current_energy"] = energy
        state["current_energy_idx"] = i
        _emit_energy(examples, state, history, energy, events_list[i], t5_x, t5_y, run_number)
        run_number += 1

    # STEP 3 완료 메시지는 시스템이 보낸다 — 모델 턴으로 만들지 않는다.
    return examples


def generate_workflow_from_mid(energy_list, events_list, start_idx):
    """후반 에너지부터 시작하는 partial 워크플로우 (phase=scanning, 위치 이동 이미 완료)."""
    examples = []
    t5_x = round(random.uniform(70.0, 130.0), 3)
    t5_y = round(random.uniform(70.0, 130.0), 3)

    energy_config_dict = {
        e: {
            "target_events": ev,
            "collected_events": ev if j < start_idx else 0,
            "runs": [1000 + j] if j < start_idx else [],
            "completed": j < start_idx,
            "completed_at": "10:00:00" if j < start_idx else None,
        }
        for j, (e, ev) in enumerate(zip(energy_list, events_list))
    }

    state = {
        "phase": "scanning", "tower": "M5T3", "t5_x": t5_x, "t5_y": t5_y,
        "position": None,
        "energy_config": energy_config_dict,
        "scan_order": energy_list,
        "current_energy": energy_list[start_idx] if start_idx < len(energy_list) else None,
        "current_energy_idx": start_idx,
        "plot_method": "PeakADC", "plot_max_event": None,
        "y_confirmed": True,  # 이미 위치 이동 완료된 상태
        "needs_plot_confirm": False,
    }

    # 직전 1-2개 에너지의 모델 결정 턴만 히스토리로 미리 채운다
    # (completed 턴은 시스템 소유라 히스토리에도 넣지 않는다).
    history = []
    for j in range(max(0, start_idx - 2), start_idx):
        prev_e = energy_list[j]
        prev_ev = events_list[j]
        history.append({"role": "assistant", "content": json.dumps(
            {"message": MESSAGE_ENERGY_SET.format(energy=prev_e)}, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        history.append({"role": "assistant", "content": json.dumps(
            {"tool": "daq_run_tool", "params": {
                "events": prev_ev,
                "pos_h": t5_x, "pos_v": t5_y,
                "pos_rot": 1.5, "pos_tilt": 1.0, "beam_energy": prev_e,
            }}, ensure_ascii=False)})
        history.append({"role": "assistant", "content": json.dumps(
            {"message": MESSAGE_PLOT_CONFIRM}, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

    run_number = 1000 + start_idx
    for i in range(start_idx, len(energy_list)):
        energy = energy_list[i]
        state["current_energy"] = energy
        state["current_energy_idx"] = i
        _emit_energy(examples, state, history, energy, events_list[i], t5_x, t5_y, run_number)
        run_number += 1

    return examples


def main():
    output_file = Path(__file__).parent / "data" / "EM_scan_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_ex = []

    test_cases = [
        # ── 정수만, 각각 (10개) ──────────────────────────────────────────
        ([1, 5, 10, 20],         [50000, 200000, 300000, 500000],            "1GeV 50000개, 5GeV 200000개, 10GeV 300000개, 20GeV 500000개"),
        ([1, 2, 3, 4, 5],        [50000, 100000, 200000, 300000, 400000],    "1GeV 50000개 2GeV 100000개 3GeV 200000개 4GeV 300000개 5GeV 400000개"),
        ([2, 5, 10, 20, 40],     [50000, 100000, 200000, 300000, 400000],    "2GeV 50000 5GeV 100000 10GeV 200000 20GeV 300000 40GeV 400000"),
        ([10, 20, 30, 40],       [100000, 200000, 300000, 400000],           "10:100000 20:200000 30:300000 40:400000"),
        ([1, 2, 3, 4, 5, 6],     [50000, 100000, 150000, 200000, 250000, 300000], "1GeV 50000개, 2GeV 100000개, 3GeV 150000개, 4GeV 200000개, 5GeV 250000개, 6GeV 300000개"),
        ([5, 10, 20, 40, 80],    [50000, 100000, 200000, 300000, 500000],    "5gev 50000 10gev 100000개 20gev 200000 40gev 300000 80gev 500000"),
        ([1, 3, 5, 7, 10],       [80000, 100000, 150000, 200000, 300000],    "1GeV 80000개, 3GeV 100000개, 5GeV 150000개, 7GeV 200000개, 10GeV 300000개"),
        ([2, 4, 6, 8, 10, 12],   [100000, 150000, 200000, 250000, 300000, 350000], "2GeV 100000 4GeV 150000 6GeV 200000 8GeV 250000 10GeV 300000 12GeV 350000"),
        ([20, 50, 100, 120],     [50000, 100000, 200000, 300000],            "20GeV 50000개, 50GeV 100000개, 100GeV 200000개, 120GeV 300000개"),
        ([1, 2, 4, 6, 8],        [80000, 120000, 200000, 280000, 350000],    "1gev 80000 2gev 120000개 4gev 200000 6gev 280000개 8gev 350000"),

        # ── 정수만, 같은 (10개) ──────────────────────────────────────────
        ([1, 2, 3, 4],           [100000, 100000, 100000, 100000],           "1,2,3,4GeV 각각 100000개씩"),
        ([1, 5, 10, 20],         [80000, 80000, 80000, 80000],               "1,5,10,20GeV 모두 80000개"),
        ([1, 2, 3, 4, 5],        [200000, 200000, 200000, 200000, 200000],   "1,2,3,4,5GeV 모두 200000개"),
        ([10, 20, 30, 40, 50],   [100000, 100000, 100000, 100000, 100000],   "10,20,30,40,50GeV 100000개씩"),
        ([1, 2, 3, 4, 5, 6],     [100000, 100000, 100000, 100000, 100000, 100000], "1,2,3,4,5,6GeV 각각 100000개씩"),
        ([1, 2, 3, 4, 5],        [100000, 100000, 100000, 200000, 200000],   "1,2,3GeV 100000개, 4,5GeV 200000개"),
        ([5, 10, 20, 30],        [80000, 80000, 200000, 200000],             "5,10GeV 80000개, 20,30GeV 200000개"),
        ([1, 2, 3, 4, 5, 6],     [50000, 50000, 50000, 200000, 200000, 200000], "1,2,3GeV 50000개, 4,5,6GeV 200000개"),
        ([10, 20, 30, 40],       [100000, 100000, 200000, 200000],           "10,20GeV 100000개, 30,40GeV 200000개"),
        ([2, 4, 6, 8, 10],       [100000, 100000, 100000, 300000, 300000],   "2,4,6GeV 100000개, 8,10GeV 300000개"),

        # ── 정수+소수, 각각 (15개) ───────────────────────────────────────
        ([0.5, 1, 1.5, 2],       [50000, 100000, 150000, 200000],            "0.5GeV 50000개, 1GeV 100000개, 1.5GeV 150000개, 2GeV 200000개"),
        ([2.5, 5, 7.5, 10],      [80000, 100000, 150000, 200000],            "2.5GeV 80000개, 5GeV 100000개, 7.5GeV 150000개, 10GeV 200000개"),
        ([0.5, 1, 2, 2.5],       [50000, 100000, 200000, 300000],            "0.5GeV 50000개, 1GeV 100000개, 2GeV 200000개, 2.5GeV 300000개"),
        ([1, 1.5, 2, 2.5, 3],    [50000, 80000, 100000, 150000, 200000],     "1GeV 50000개, 1.5GeV 80000개, 2GeV 100000개, 2.5GeV 150000개, 3GeV 200000개"),
        ([1.5, 2, 2.5, 3, 3.5],  [50000, 100000, 150000, 200000, 300000],    "1.5GeV 50000개, 2GeV 100000개, 2.5GeV 150000개, 3GeV 200000개, 3.5GeV 300000개"),
        ([1, 2.5, 5, 7.5, 10],   [50000, 100000, 200000, 300000, 400000],    "1gev 50000 2.5gev 100000 5gev 200000 7.5gev 300000 10gev 400000"),
        ([0.5, 1.5, 3, 4.5, 6],  [80000, 100000, 150000, 200000, 300000],    "0.5GeV 80000개, 1.5GeV 100000개, 3GeV 150000개, 4.5GeV 200000개, 6GeV 300000개"),
        ([1, 1.5, 2, 2.5, 3, 3.5],[50000, 80000, 100000, 150000, 200000, 300000], "1GeV 50000개, 1.5GeV 80000개, 2GeV 100000개, 2.5GeV 150000개, 3GeV 200000개, 3.5GeV 300000개"),
        ([2, 2.5, 3, 3.5, 4, 4.5],[80000, 100000, 150000, 200000, 250000, 300000], "2GeV 80000 2.5GeV 100000 3GeV 150000 3.5GeV 200000 4GeV 250000 4.5GeV 300000"),
        ([1.5, 3, 4.5, 6],        [100000, 150000, 200000, 300000],          "1.5gev 100000개 3gev 150000개 4.5gev 200000개 6gev 300000개"),
        ([0.5, 1, 1.5, 2, 2.5],   [50000, 100000, 150000, 200000, 300000],   "0.5gev 50000 1gev 100000개 1.5gev 150000 2gev 200000개 2.5gev 300000"),
        ([1, 2.5, 4, 5.5, 7],     [80000, 120000, 180000, 250000, 350000],   "1GeV 80000개, 2.5GeV 120000개, 4GeV 180000개, 5.5GeV 250000개, 7GeV 350000개"),
        ([1.5, 2.5, 3.5, 4.5, 5.5, 6.5],[50000, 100000, 150000, 200000, 250000, 300000], "1.5GeV 50000 2.5GeV 100000 3.5GeV 150000 4.5GeV 200000 5.5GeV 250000 6.5GeV 300000"),
        ([2, 3.5, 5, 6.5, 8],     [80000, 100000, 200000, 300000, 400000],   "2gev 80000 3.5gev 100000 5gev 200000 6.5gev 300000 8gev 400000"),
        ([1, 2, 3.5, 5, 7.5, 10], [50000, 100000, 150000, 200000, 300000, 500000], "1GeV 50000개, 2GeV 100000개, 3.5GeV 150000개, 5GeV 200000개, 7.5GeV 300000개, 10GeV 500000개"),

        # ── 정수+소수, 같은 (15개) ───────────────────────────────────────
        ([2.5, 3, 3.5, 4],        [100000, 100000, 100000, 100000],          "2.5, 3, 3.5, 4GeV 모두 100000개"),
        ([1, 1.5, 2, 2.5, 3],     [100000, 100000, 100000, 100000, 100000],  "1,1.5,2,2.5,3GeV 100000개씩"),
        ([3.5, 4, 4.5, 5],        [200000, 200000, 200000, 200000],          "3.5,4,4.5,5GeV 각각 200000개"),
        ([1.5, 2, 2.5, 3, 3.5],   [80000, 80000, 80000, 80000, 80000],       "1.5,2,2.5,3,3.5GeV 모두 80000개"),
        ([1, 1.5, 2, 2.5, 3, 3.5],[100000, 100000, 100000, 100000, 100000, 100000], "1,1.5,2,2.5,3,3.5GeV 각각 100000개"),
        ([2, 3.5, 5, 6.5, 8],     [100000, 100000, 100000, 100000, 100000],  "2,3.5,5,6.5,8GeV 모두 100000개"),
        ([1.5, 3, 4.5, 6, 7.5],   [200000, 200000, 200000, 200000, 200000],  "1.5,3,4.5,6,7.5GeV 각각 200000개"),
        ([2.5, 3, 3.5, 4, 4.5, 5],[100000, 100000, 100000, 100000, 100000, 100000], "2.5,3,3.5,4,4.5,5GeV 모두 100000개"),
        ([1, 2, 3.5, 4, 5.5],     [100000, 100000, 200000, 200000, 200000],  "1,2GeV 100000개, 3.5,4,5.5GeV 200000개"),
        ([1.5, 2, 2.5, 3, 3.5, 4],[50000, 50000, 50000, 200000, 200000, 200000], "1.5,2,2.5GeV 50000개, 3,3.5,4GeV 200000개"),
        ([0.5, 1, 1.5, 2],        [80000, 80000, 200000, 200000],            "0.5,1GeV 80000개, 1.5,2GeV 200000개"),
        ([1, 2, 2.5, 3, 4, 4.5],  [100000, 100000, 100000, 200000, 200000, 200000], "1,2,2.5GeV 100000개, 3,4,4.5GeV 200000개"),
        ([1, 1.5, 2, 2.5, 3, 4],  [50000, 50000, 50000, 300000, 300000, 300000], "1,1.5,2GeV 50000개, 2.5,3,4GeV 300000개"),
        ([0.5, 1, 1.5, 2, 2.5],   [50000, 50000, 50000, 50000, 50000],       "0.5,1,1.5,2,2.5GeV 모두 50000개"),
        ([1, 2, 3, 4.5, 5.5],     [80000, 80000, 200000, 200000, 200000],    "1,2GeV 80000개, 3,4.5,5.5GeV 200000개"),

        # ── 대소문자 혼합 (gev + GeV 섞임) (10개) ─────────────────────────
        ([1, 3, 4],               [200000, 100000, 400000],                  "1gev 200000 3GeV 100000 4GeV 400000개"),
        ([1, 2, 5],               [50000, 100000, 200000],                   "1GeV 50000 2gev 100000 5GeV 200000개"),
        ([2, 4, 6, 8],            [100000, 150000, 200000, 250000],          "2gev 100000 4GeV 150000 6gev 200000 8GeV 250000"),
        ([1, 3, 5, 10],           [80000, 100000, 150000, 200000],           "1GEV 80000개 3gev 100000 5Gev 150000개 10GeV 200000"),
        ([5, 10, 20],             [50000, 100000, 200000],                   "5gev 50000개 10GeV 100000개 20gev 200000개"),
        ([1.5, 2.5, 3.5],         [80000, 100000, 150000],                   "1.5gev 80000 2.5GeV 100000 3.5gev 150000개"),
        ([2, 3, 4, 5, 6],         [50000, 100000, 150000, 200000, 250000],   "2gev 50000 3GeV 100000개 4gev 150000개 5GeV 200000 6gev 250000"),
        ([1, 4, 7],               [100000, 200000, 300000],                  "1GeV 100000개 4gev 200000개 7GeV 300000"),
        ([10, 30, 50, 80],        [50000, 100000, 200000, 300000],           "10gev 50000개 30GeV 100000 50gev 200000 80GeV 300000개"),
        ([1, 2.5, 5, 7.5],        [80000, 120000, 200000, 280000],           "1gev 80000개 2.5GEV 120000 5GeV 200000 7.5gev 280000개"),

        # ── 부분적 "개" (마지막만, 첫개만, 중간만 등) (10개) ───────────────
        ([1, 3, 5, 7],            [50000, 100000, 200000, 300000],           "1GeV 50000 3GeV 100000 5GeV 200000 7GeV 300000개"),
        ([2, 4, 6],               [80000, 100000, 200000],                   "2GeV 80000개 4GeV 100000 6GeV 200000"),
        ([1, 2, 4, 8],            [50000, 80000, 120000, 200000],            "1GeV 50000 2GeV 80000개 4GeV 120000 8GeV 200000"),
        ([5, 10, 20, 40],         [50000, 100000, 150000, 200000],           "5GeV 50000 10GeV 100000 20GeV 150000개 40GeV 200000"),
        ([1, 3, 5],               [100000, 200000, 300000],                  "1GeV 100000개 3GeV 200000개 5GeV 300000"),
        ([2.5, 5, 7.5, 10],       [50000, 80000, 120000, 200000],            "2.5GeV 50000 5GeV 80000 7.5GeV 120000 10GeV 200000개"),
        ([1, 2, 3, 4, 5],         [50000, 100000, 150000, 200000, 250000],   "1GeV 50000개 2GeV 100000 3GeV 150000 4GeV 200000 5GeV 250000"),
        ([0.5, 1.5, 2.5],         [80000, 120000, 200000],                   "0.5GeV 80000 1.5GeV 120000개 2.5GeV 200000"),
        ([1, 5, 10],              [100000, 200000, 300000],                  "1GeV 100000 5GeV 200000개 10GeV 300000개"),
        ([3, 6, 9],               [50000, 100000, 150000],                   "3GeV 50000개 6GeV 100000개 9GeV 150000"),

        # ── 콤마 묶음 + 개별 추가 (개 optional) (12개) ─────────────────────
        # "1,2,3GeV 100000 4GeV 50000" 같은 입력: 앞은 묶음, 뒤는 개별
        ([1, 2, 3, 4],            [100000, 100000, 100000, 50000],           "1,2,3gev 100000 4gev 50000"),
        ([1, 2, 3, 4],            [100000, 100000, 100000, 50000],           "1,2,3GeV 100000 4GeV 50000"),
        ([1, 2, 3, 4],            [100000, 100000, 100000, 50000],           "1,2,3GeV 100000개 4GeV 50000개"),
        ([1, 2, 5, 10],           [80000, 80000, 200000, 300000],            "1,2GeV 80000 5GeV 200000 10GeV 300000"),
        ([1, 2, 3, 5, 10],        [100000, 100000, 100000, 200000, 300000],  "1,2,3GeV 100000개 5GeV 200000 10GeV 300000개"),
        ([2, 4, 6, 10],           [50000, 50000, 50000, 200000],             "2,4,6gev 50000 10gev 200000"),
        ([1, 3, 5, 7, 10],        [80000, 80000, 80000, 80000, 300000],      "1,3,5,7GeV 80000 10GeV 300000"),
        ([5, 10, 20, 50],         [100000, 100000, 200000, 500000],          "5,10GeV 100000 20GeV 200000 50GeV 500000"),
        ([1, 2, 4, 8, 16],        [50000, 50000, 50000, 200000, 300000],     "1,2,4GeV 50000개 8GeV 200000개 16GeV 300000개"),
        ([0.5, 1, 1.5, 3],        [80000, 80000, 80000, 200000],             "0.5,1,1.5gev 80000 3gev 200000"),
        ([1, 2, 3, 4, 5, 10],     [100000, 100000, 100000, 100000, 100000, 300000], "1,2,3,4,5GeV 100000 10GeV 300000"),
        ([2, 4, 6, 8, 10],        [50000, 50000, 50000, 200000, 200000],     "2,4,6GeV 50000개 8,10GeV 200000개"),

        # ── 다양한 구분자/공백 변형 (10개) ────────────────────────────────
        ([1, 2, 3],               [100000, 200000, 300000],                  "1GeV,100000개  2GeV,200000개  3GeV,300000개"),
        ([5, 10, 15],             [50000, 100000, 150000],                   "5GeV  50000   10GeV  100000   15GeV  150000개"),
        ([1, 2, 3, 4],            [80000, 80000, 80000, 80000],              "1, 2, 3, 4 GeV 각각 80000개"),
        ([10, 20, 40],            [100000, 200000, 400000],                  "10GeV/100000개 20GeV/200000개 40GeV/400000개"),
        ([1, 3, 5],               [100000, 150000, 200000],                  "1GeV - 100000개, 3GeV - 150000개, 5GeV - 200000개"),
        ([2, 4, 6, 8],            [50000, 80000, 120000, 200000],            "2GeV=50000 4GeV=80000 6GeV=120000 8GeV=200000"),
        ([1, 2, 3, 4, 5],         [100000, 100000, 100000, 100000, 100000],  "1~5GeV (1,2,3,4,5) 모두 100000개"),
        ([0.5, 1, 2, 5],          [50000, 100000, 150000, 200000],           "에너지: 0.5GeV 50000개, 1GeV 100000개, 2GeV 150000개, 5GeV 200000개"),
        ([10, 30, 50],            [100000, 200000, 300000],                  "10GeV는 100000개, 30GeV는 200000개, 50GeV는 300000개"),
        ([1, 2, 4],               [80000, 120000, 200000],                   "1GeV는 80000, 2GeV는 120000개, 4GeV는 200000"),

        # ── 3자리 이벤트 (100-999) (12개) ───────────────────────────────
        ([1, 2, 3],               [500, 1000, 2000],                         "1GeV 500개 2GeV 1000개 3GeV 2000개"),
        ([1, 5, 10],              [100, 500, 1000],                          "1GeV 100개 5GeV 500개 10GeV 1000개"),
        ([1, 2, 3, 4],            [200, 300, 400, 500],                      "1GeV 200개 2GeV 300개 3GeV 400개 4GeV 500개"),
        ([2, 4, 6],               [100, 200, 300],                           "2gev 100 4gev 200 6gev 300"),
        ([1, 3, 5, 7],            [500, 500, 500, 500],                      "1,3,5,7GeV 각각 500개씩"),
        ([1, 2],                  [100, 200],                                "1GeV 100개 2GeV 200개"),
        ([5, 10, 20],             [300, 500, 800],                           "5GeV 300개, 10GeV 500개, 20GeV 800개"),
        ([1, 2, 3, 4, 5],         [100, 200, 300, 400, 500],                 "1GeV 100개 2GeV 200개 3GeV 300개 4GeV 400개 5GeV 500개"),
        ([0.5, 1, 1.5, 2],        [200, 300, 400, 500],                      "0.5GeV 200개 1GeV 300개 1.5GeV 400개 2GeV 500개"),
        ([10, 20],                [500, 900],                                "10GeV 500개 20GeV 900개"),
        ([1, 2, 3, 4, 5, 6],      [100, 100, 200, 200, 300, 300],            "1,2GeV 100개, 3,4GeV 200개, 5,6GeV 300개"),
        ([3, 6, 9],               [300, 600, 900],                           "3GeV 300개 6GeV 600개 9GeV 900개"),

        # ── 4자리 이벤트 (1000-9999) (14개) ─────────────────────────────
        ([2, 4, 6],               [3000, 5000, 8000],                        "2GeV 3000개 4GeV 5000개 6GeV 8000개"),
        ([1, 2, 3, 4],            [1000, 2000, 3000, 4000],                  "1,2,3,4GeV 각각 다른 이벤트 1000 2000 3000 4000"),
        ([1, 5, 10, 20],          [2000, 3000, 5000, 8000],                  "1GeV 2000개 5GeV 3000개 10GeV 5000개 20GeV 8000개"),
        ([1, 2, 3],               [5000, 5000, 5000],                        "1,2,3GeV 모두 5000개씩"),
        ([5, 10, 15, 20],         [1000, 2000, 3000, 4000],                  "5GeV 1000개 10GeV 2000개 15GeV 3000개 20GeV 4000개"),
        ([1, 2, 4, 8],            [1000, 2000, 4000, 8000],                  "1gev 1000 2gev 2000 4gev 4000 8gev 8000"),
        ([0.5, 1, 2, 4],          [1500, 2000, 3000, 5000],                  "0.5GeV 1500개, 1GeV 2000개, 2GeV 3000개, 4GeV 5000개"),
        ([1, 3, 5, 7, 10],        [2000, 3000, 4000, 5000, 7000],            "1GeV 2000개, 3GeV 3000개, 5GeV 4000개, 7GeV 5000개, 10GeV 7000개"),
        ([2, 4, 6, 8, 10],        [1000, 1000, 1000, 1000, 1000],            "2,4,6,8,10GeV 각각 1000개씩"),
        ([1, 2, 3, 4, 5, 6],      [1000, 2000, 3000, 4000, 5000, 6000],      "1GeV 1000 2GeV 2000 3GeV 3000 4GeV 4000 5GeV 5000 6GeV 6000"),
        ([10, 20, 30],            [3000, 5000, 9000],                        "10GeV 3000개, 20GeV 5000개, 30GeV 9000개"),
        ([1, 5, 10],              [5000, 5000, 5000],                        "1,5,10GeV 모두 5000개"),
        ([2, 3, 4, 5],            [2500, 3500, 4500, 6500],                  "2GeV 2500개 3GeV 3500개 4GeV 4500개 5GeV 6500개"),
        ([1, 2.5, 5, 7.5],        [1000, 2000, 4000, 8000],                  "1gev 1000개 2.5gev 2000개 5gev 4000개 7.5gev 8000개"),

        # ── 3-4자리 혼합 이벤트 (12개) ──────────────────────────────────
        ([1, 2, 3],               [500, 1500, 5000],                         "1GeV 500개 2GeV 1500개 3GeV 5000개"),
        ([1, 5, 10, 20],          [200, 1000, 5000, 20000],                  "1GeV 200개 5GeV 1000개 10GeV 5000개 20GeV 20000개"),
        ([2, 4, 6, 8],            [500, 1000, 5000, 10000],                  "2GeV 500개 4GeV 1000개 6GeV 5000개 8GeV 10000개"),
        ([1, 2, 3, 4, 5],         [100, 500, 1000, 5000, 10000],             "1GeV 100개 2GeV 500개 3GeV 1000개 4GeV 5000개 5GeV 10000개"),
        ([1, 3, 5],               [300, 3000, 30000],                        "1GeV 300개, 3GeV 3000개, 5GeV 30000개"),
        ([5, 10, 20],             [700, 7000, 70000],                        "5gev 700 10gev 7000 20gev 70000"),
        ([1, 2, 4, 8],            [250, 500, 2500, 25000],                   "1GeV 250개 2GeV 500개 4GeV 2500개 8GeV 25000개"),
        ([0.5, 1, 2, 5, 10],      [100, 500, 2000, 8000, 50000],             "0.5GeV 100개 1GeV 500개 2GeV 2000개 5GeV 8000개 10GeV 50000개"),
        ([1, 1.5, 2, 2.5, 3],     [500, 700, 1000, 3000, 5000],              "1GeV 500개 1.5GeV 700개 2GeV 1000개 2.5GeV 3000개 3GeV 5000개"),
        ([10, 20, 30, 40],        [800, 2000, 6000, 15000],                  "10GeV 800개 20GeV 2000개 30GeV 6000개 40GeV 15000개"),
        ([2, 4, 6],               [999, 4999, 49999],                        "2GeV 999개 4GeV 4999개 6GeV 49999개"),
        ([1, 2, 3, 4, 5, 6],      [100, 200, 1000, 2000, 10000, 20000],      "1,2GeV 각각 100,200개 3,4GeV 각각 1000,2000개 5,6GeV 각각 10000,20000개"),

        # ── 단일 에너지 (2개) ────────────────────────────────────────────
        ([5],                     [50000],                                   "5GeV 50000개"),
        ([10],                    [1000],                                    "10GeV 1000개"),

        # ── 6자리 이벤트 (100000-999999) — 소에너지 포함 (20개) ─────────
        ([1, 2, 3],               [100000, 200000, 400000],                  "1gev 100000개 2gev 200000 3gev 400000개"),
        ([1, 2, 3],               [100000, 200000, 300000],                  "1GeV 100000개 2GeV 200000개 3GeV 300000개"),
        ([1, 2, 3, 4],            [100000, 200000, 300000, 400000],          "1GeV 100000 2GeV 200000 3GeV 300000 4GeV 400000"),
        ([1, 2, 3, 4, 5],         [100000, 200000, 300000, 400000, 500000],  "1gev 100000개 2gev 200000개 3gev 300000개 4gev 400000개 5gev 500000개"),
        ([1, 2, 3, 4, 5],         [100000, 200000, 300000, 400000, 500000],  "1GeV 100000 2GeV 200000 3GeV 300000 4GeV 400000 5GeV 500000"),
        ([1, 2, 4, 6],            [100000, 200000, 400000, 600000],          "1GeV 100000개 2GeV 200000개 4GeV 400000개 6GeV 600000개"),
        ([2, 4, 6, 8, 10],        [100000, 200000, 300000, 400000, 500000],  "2GeV 100000 4GeV 200000 6GeV 300000 8GeV 400000 10GeV 500000"),
        ([1, 3, 5, 10],           [100000, 300000, 500000, 100000],          "1GeV 100000개 3GeV 300000개 5GeV 500000개 10GeV 100000개"),
        ([1, 2, 3, 4, 5, 6],      [100000, 100000, 100000, 100000, 100000, 100000], "1,2,3,4,5,6GeV 각각 100000개씩"),
        ([1, 2, 3],               [500000, 500000, 500000],                  "1,2,3GeV 모두 500000개"),
        ([5, 10, 20],             [100000, 200000, 500000],                  "5GeV 100000개 10GeV 200000개 20GeV 500000개"),
        ([1, 2, 3, 4, 5],         [200000, 200000, 200000, 200000, 200000],  "1,2,3,4,5GeV 각각 200000개"),
        ([1, 5, 10, 20],          [100000, 200000, 300000, 500000],          "1gev 100000 5gev 200000 10gev 300000 20gev 500000"),
        ([1, 2],                  [100000, 200000],                          "1GeV 100000개 2GeV 200000개"),
        ([1, 2, 3],               [300000, 400000, 500000],                  "1GeV 300000개 2GeV 400000개 3GeV 500000개"),
        ([10, 20, 30],            [100000, 200000, 300000],                  "10GeV 100000개 20GeV 200000개 30GeV 300000개"),
        ([1, 2, 3, 4],            [200000, 200000, 400000, 400000],          "1,2GeV 200000개 3,4GeV 400000개"),
        ([2, 4, 6],               [100000, 200000, 300000],                  "2gev 100000 4gev 200000 6gev 300000"),
        ([1, 3, 5, 7, 10],        [100000, 100000, 200000, 200000, 300000],  "1,3GeV 100000 5,7GeV 200000 10GeV 300000"),

        # ── 넓은 에너지 범위 + 소형 이벤트 (8개) ────────────────────────
        ([1, 10, 50, 100],        [500, 1000, 5000, 10000],                  "1GeV 500개 10GeV 1000개 50GeV 5000개 100GeV 10000개"),
        ([1, 2, 5, 10, 20, 50],   [200, 500, 1000, 3000, 8000, 20000],       "1GeV 200개 2GeV 500개 5GeV 1000개 10GeV 3000개 20GeV 8000개 50GeV 20000개"),
        ([5, 20, 50, 100],        [300, 2000, 8000, 30000],                  "5GeV 300개, 20GeV 2000개, 50GeV 8000개, 100GeV 30000개"),
        ([1, 5, 10, 50, 100],     [100, 500, 1000, 5000, 10000],             "1gev 100 5gev 500 10gev 1000 50gev 5000 100gev 10000"),
        ([2, 4, 8, 16, 32],       [1000, 2000, 4000, 8000, 16000],           "2GeV 1000개 4GeV 2000개 8GeV 4000개 16GeV 8000개 32GeV 16000개"),
        ([1, 3, 9, 27],           [300, 900, 2700, 8100],                    "1GeV 300개 3GeV 900개 9GeV 2700개 27GeV 8100개"),
        ([10, 30, 60, 90, 120],   [500, 1500, 3000, 6000, 9000],             "10GeV 500개 30GeV 1500개 60GeV 3000개 90GeV 6000개 120GeV 9000개"),
        ([1, 2, 3, 5, 8, 13],     [100, 200, 300, 500, 800, 1300],           "1GeV 100개 2GeV 200개 3GeV 300개 5GeV 500개 8GeV 800개 13GeV 1300개"),
    ]
    for energy_list, events_list, user_input in test_cases:
        all_ex.extend(generate_workflow_normal(energy_list, events_list, user_input))

    mid_cases = [
        # 정수만 각각
        ([1, 5, 10, 20],        [50000, 200000, 300000, 500000],   2),
        ([1, 2, 3, 4, 5],       [50000, 100000, 200000, 300000, 400000], 3),
        ([5, 10, 20, 40, 80],   [50000, 100000, 200000, 300000, 500000], 2),
        ([5, 10, 20, 40, 80],   [50000, 100000, 200000, 300000, 500000], 4),
        ([1, 2, 3, 4, 5, 6],    [50000, 100000, 150000, 200000, 250000, 300000], 2),
        ([1, 2, 3, 4, 5, 6],    [50000, 100000, 150000, 200000, 250000, 300000], 4),
        ([10, 20, 30, 40, 50],  [100000, 200000, 300000, 400000, 500000], 3),
        ([2, 4, 6, 8, 10, 12],  [100000, 150000, 200000, 250000, 300000, 350000], 3),
        # 정수만 같은
        ([1, 2, 3, 4],          [100000, 100000, 100000, 100000],  2),
        ([2, 5, 10, 20],        [100000, 100000, 100000, 100000],  3),
        ([5, 10, 20, 30],       [80000, 80000, 200000, 200000],    2),
        ([1, 2, 3, 4, 5, 6],    [100000, 100000, 100000, 100000, 100000, 100000], 3),
        ([10, 20, 30, 40, 50],  [100000, 100000, 100000, 100000, 100000], 2),
        ([1, 3, 5, 7, 10],      [80000, 100000, 150000, 200000, 300000], 2),
        # 정수+소수 각각
        ([0.5, 1, 1.5, 2],      [50000, 100000, 150000, 200000],   2),
        ([1, 2.5, 5, 7.5, 10],  [50000, 100000, 200000, 300000, 400000], 2),
        ([1, 1.5, 2, 2.5, 3],   [50000, 80000, 100000, 150000, 200000], 3),
        ([1, 2, 3.5, 5, 7.5, 10],[50000, 100000, 150000, 200000, 300000, 500000], 4),
        ([0.5, 1.5, 3, 4.5, 6], [80000, 100000, 150000, 200000, 300000], 3),
        ([2, 2.5, 3, 3.5, 4, 4.5],[80000, 100000, 150000, 200000, 250000, 300000], 3),
        # 정수+소수 같은
        ([2.5, 3, 3.5, 4],      [100000, 100000, 100000, 100000],  2),
        ([1, 1.5, 2, 2.5, 3],   [100000, 100000, 100000, 100000, 100000], 2),
        ([1, 2, 3, 4.5, 5.5],   [80000, 80000, 200000, 200000, 200000], 3),
        ([1.5, 2, 2.5, 3, 3.5, 4],[50000, 50000, 50000, 200000, 200000, 200000], 4),
        ([1, 1.5, 2, 2.5, 3, 3.5],[100000, 100000, 100000, 100000, 100000, 100000], 2),
        # 3자리 이벤트 mid cases
        ([1, 2, 3],             [500, 1000, 2000],  1),
        ([1, 5, 10],            [100, 500, 1000],   1),
        ([1, 2, 3, 4, 5],       [100, 200, 300, 400, 500], 2),
        ([5, 10, 20],           [300, 500, 800],    1),
        ([0.5, 1, 1.5, 2],      [200, 300, 400, 500], 2),
        # 4자리 이벤트 mid cases
        ([2, 4, 6],             [3000, 5000, 8000], 1),
        ([1, 2, 3, 4],          [1000, 2000, 3000, 4000], 2),
        ([1, 5, 10, 20],        [2000, 3000, 5000, 8000], 2),
        ([1, 3, 5, 7, 10],      [2000, 3000, 4000, 5000, 7000], 3),
        ([2, 4, 6, 8, 10],      [1000, 1000, 1000, 1000, 1000], 2),
        # 혼합 이벤트 mid cases
        ([1, 2, 3],             [500, 1500, 5000],  1),
        ([1, 5, 10, 20],        [200, 1000, 5000, 20000], 2),
        ([2, 4, 6, 8],          [500, 1000, 5000, 10000], 2),
        ([1, 2, 3, 4, 5],       [100, 500, 1000, 5000, 10000], 3),
        ([1, 3, 5],             [300, 3000, 30000], 1),
    ]
    for energy_list, events_list, start_idx in mid_cases:
        all_ex.extend(generate_workflow_from_mid(energy_list, events_list, start_idx))

    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    lengths = [sum(len(m["content"]) for m in ex["messages"]) for ex in all_ex]
    print(f"Generated {len(all_ex)} samples -> {output_file}")
    print(f"   char len  max={max(lengths):,}  avg={sum(lengths)/len(lengths):,.0f}")


if __name__ == "__main__":
    main()
