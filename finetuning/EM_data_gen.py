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

MESSAGE_ASK_ENERGY  = "에너지 설정을 입력해주세요.\n예) 10GeV 100000개 20GeV 200000개 50GeV 300000개  또는  10GeV 80000 30GeV 500000 120GeV 300000"
MESSAGE_MOVE_POS    = "x = {x:.3f} mm, y = {y:.3f} mm 으로 이동해주세요."
MESSAGE_ENERGY_SET  = "빔 에너지를 {energy} GeV로 설정해주세요."
MESSAGE_PLOT_CONFIRM = MSG_PLOT_CONFIRM
MESSAGE_COMPLETE    = "모든 에너지 스캔이 완료되었습니다."

SYSTEM_PROMPT = """You are Energy Scan Agent for test beam experiments.

Follow these steps EXACTLY:

=== STEP 0: Get Energy Config (only when phase is "config") ===
0a. Ask user for energy settings:
  {"message": "에너지 설정을 입력해주세요.\n예) 10GeV 100000개 20GeV 200000개 50GeV 300000개  또는  10GeV 80000 30GeV 500000 120GeV 300000"}

After user responds, parse their input:
0b. Update state with parsed config. Emit ONLY target_events per energy — the SYSTEM
  fills in collected_events/runs/completed/scan_order. Keep the JSON as SHORT as possible:
  {"tool": "none", "update_state": {"energy_config": {"<energy_int>": {"target_events": <n>}, ...}, "phase": "idle"}}
  Example: {"tool": "none", "update_state": {"energy_config": {"10": {"target_events": 100000}, "50": {"target_events": 300000}}, "phase": "idle"}}
  CRITICAL: energy keys must be INTEGERS (e.g., 1, 2, 3).
  CRITICAL: Emit ONLY "target_events" (and "config" if named) per energy. Do NOT emit
  collected_events, runs, completed, completed_at, or scan_order — the SYSTEM owns those.
  CRITICAL: beam_energy in GeV → store as number (integer if whole: "2GeV" → 2; float if decimal: "2.5GeV" → 2.5). NEVER convert to MeV.
  CRITICAL: If user says "모두", "각각", or "씩" with one number (e.g., "모두 500개"), apply that number to ALL energies.
  CRITICAL: If the user names a DAQ config for an energy (e.g. "3GeV setup1로 200000"),
  add "config": "<name>" to THAT energy's entry ONLY (copy the name EXACTLY as written).
  If a config name appears before ALL energies, apply it to every energy.
  OMIT "config" when the user does not name one — the default ("setup") applies.

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
    "pos_tilt": 0.0,
    "beam_energy": <energy>
}
(If energy_config[energy] has "config", also include "config": <that name> in Params. Omit otherwise.)
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
   - {"tool": "none", "update_state": {...}}  (ONLY for STEP 0b config parsing; target_events only)
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
    """주로 5-6자리(십만 단위 중심) 이벤트 수를 반환. 3-4자리는 소량만."""
    digits = random.choices([3, 4, 5, 6], weights=[5, 5, 45, 45])[0]
    return random.randint(10**(digits-1), 10**digits - 1)


def _build_state_context(state):
    lines = []
    lines.append(f"Phase: {state['phase']}")
    lines.append(f"Tower: {state['tower']}")
    lines.append(f"M5T3 Position: x={state['t5_x']:.3f}, y={state['t5_y']:.3f}, rot=1.5, tilt=0.0")
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
            cfg_suffix = f" config={cfg['config']}" if cfg.get("config") else ""
            lines.append(f"  {status} {e} GeV: target={cfg.get('target_events','?')} "
                         f"collected={cfg.get('collected_events',0)} "
                         f"runs={cfg.get('runs',[])} completed={cfg.get('completed',False)}{cfg_suffix}")
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


def _emit_energy(examples, state, history, energy, events, t5_x, t5_y, run_number, daq_config=None):
    """한 에너지의 모델 결정 턴(set-beam → DAQ → plot msg)을 생성.
    완료(completed) 표시는 코드(시스템)가 소유하므로 모델 턴으로 만들지 않는다."""
    # 2a: set-beam message
    dec = {"message": MESSAGE_ENERGY_SET.format(energy=energy)}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user", "content": "완료"})

    # 2b: DAQ (에너지별 DAQ config이 지정된 경우에만 "config" 포함 — 기본 setup은 생략)
    params = {
        "events": events,
        "pos_h": t5_x, "pos_v": t5_y,
        "pos_rot": 1.5, "pos_tilt": 0.0, "beam_energy": energy,
    }
    if daq_config:
        params["config"] = daq_config
    dec = {"tool": "daq_run_tool", "params": params}
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


def generate_workflow_normal(energy_list, events_list, user_input, config_list=None):
    """config_list: 에너지별 DAQ config 이름 리스트 (None 또는 에너지별 None/이름)."""
    if config_list is None:
        config_list = [None] * len(energy_list)
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

    # STEP 0b (config parse — model-owned; state still phase=config at decision time).
    # 학습 타깃은 짧은 형태(target_events만)로 낸다 — MAX_NEW_TOKENS 잘림 방지.
    # collected_events/runs/completed/scan_order는 SYSTEM(코드)이 채우므로 모델이 뱉지 않는다.
    target_config = {}
    for e, ev, cfg in zip(energy_list, events_list, config_list):
        entry = {"target_events": ev}
        if cfg:
            entry["config"] = cfg
        target_config[e] = entry
    dec = {"tool": "none", "update_state": {"energy_config": target_config, "phase": "idle"}}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    # state는 코드가 채우는 전체 필드 + scan_order로 진전(이후 STEP 1+ context 생성에 필요).
    full_config = {}
    for e, ev, cfg in zip(energy_list, events_list, config_list):
        entry = {"target_events": ev, "collected_events": 0, "runs": [], "completed": False, "completed_at": None}
        if cfg:
            entry["config"] = cfg
        full_config[e] = entry
    state.update({"energy_config": full_config, "scan_order": energy_list, "phase": "idle"})

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
        _emit_energy(examples, state, history, energy, events_list[i], t5_x, t5_y, run_number,
                     daq_config=config_list[i])
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
                "pos_rot": 1.5, "pos_tilt": 0.0, "beam_energy": prev_e,
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
        # 에너지: 10-120 GeV, 10 단위. 이벤트 수: 주로 5-6자리(십만 단위 중심).
        # ── 각각 지정 (10개) ─────────────────────────────────────────────
        ([10, 20, 30, 40],        [100000, 200000, 300000, 400000],           "10GeV 100000개, 20GeV 200000개, 30GeV 300000개, 40GeV 400000개"),
        ([10, 50, 100, 120],      [50000, 200000, 300000, 500000],            "10GeV 50000개, 50GeV 200000개, 100GeV 300000개, 120GeV 500000개"),
        ([20, 40, 60, 80, 100],   [50000, 100000, 200000, 300000, 400000],    "20GeV 50000 40GeV 100000 60GeV 200000 80GeV 300000 100GeV 400000"),
        ([10, 20, 30, 40],        [100000, 200000, 300000, 400000],           "10:100000 20:200000 30:300000 40:400000"),
        ([10, 20, 30, 40, 50, 60],[50000, 100000, 150000, 200000, 250000, 300000], "10GeV 50000개, 20GeV 100000개, 30GeV 150000개, 40GeV 200000개, 50GeV 250000개, 60GeV 300000개"),
        ([10, 30, 50, 70, 90],    [80000, 100000, 150000, 200000, 300000],    "10gev 80000 30gev 100000개 50gev 150000 70gev 200000 90gev 300000"),
        ([20, 50, 80, 110],       [80000, 120000, 200000, 280000],            "20GeV 80000개, 50GeV 120000개, 80GeV 200000개, 110GeV 280000개"),
        ([10, 40, 70, 100, 120],  [100000, 150000, 200000, 250000, 350000],   "10GeV 100000 40GeV 150000 70GeV 200000 100GeV 250000 120GeV 350000"),
        ([30, 60, 90, 120],       [50000, 100000, 200000, 300000],            "30GeV 50000개, 60GeV 100000개, 90GeV 200000개, 120GeV 300000개"),
        ([10, 20, 40, 60, 80],    [80000, 120000, 200000, 280000, 350000],    "10gev 80000 20gev 120000개 40gev 200000 60gev 280000개 80gev 350000"),

        # ── 같은/묶음 (10개) ─────────────────────────────────────────────
        ([10, 20, 30, 40],        [100000, 100000, 100000, 100000],           "10,20,30,40GeV 각각 100000개씩"),
        ([10, 50, 100, 120],      [80000, 80000, 80000, 80000],               "10,50,100,120GeV 모두 80000개"),
        ([10, 20, 30, 40, 50],    [200000, 200000, 200000, 200000, 200000],   "10,20,30,40,50GeV 모두 200000개"),
        ([20, 40, 60, 80, 100],   [100000, 100000, 100000, 100000, 100000],   "20,40,60,80,100GeV 100000개씩"),
        ([10, 20, 30, 40, 50],    [100000, 100000, 100000, 200000, 200000],   "10,20,30GeV 100000개, 40,50GeV 200000개"),
        ([10, 20, 60, 70],        [80000, 80000, 200000, 200000],             "10,20GeV 80000개, 60,70GeV 200000개"),
        ([10, 20, 30, 40, 50, 60],[50000, 50000, 50000, 200000, 200000, 200000], "10,20,30GeV 50000개, 40,50,60GeV 200000개"),
        ([70, 80, 90, 100],       [100000, 100000, 200000, 200000],           "70,80GeV 100000개, 90,100GeV 200000개"),
        ([20, 40, 60, 80, 100],   [100000, 100000, 100000, 300000, 300000],   "20,40,60GeV 100000개, 80,100GeV 300000개"),
        ([10, 20, 30, 40, 50, 60],[100000, 100000, 100000, 100000, 100000, 100000], "10,20,30,40,50,60GeV 각각 100000개씩"),

        # ── 대소문자 혼합 (gev + GeV 섞임) (8개) ─────────────────────────
        ([10, 30, 40],            [200000, 100000, 400000],                   "10gev 200000 30GeV 100000 40GeV 400000개"),
        ([10, 20, 50],            [50000, 100000, 200000],                    "10GeV 50000 20gev 100000 50GeV 200000개"),
        ([20, 40, 60, 80],        [100000, 150000, 200000, 250000],           "20gev 100000 40GeV 150000 60gev 200000 80GeV 250000"),
        ([10, 30, 50, 100],       [80000, 100000, 150000, 200000],            "10GEV 80000개 30gev 100000 50Gev 150000개 100GeV 200000"),
        ([50, 100, 120],          [50000, 100000, 200000],                    "50gev 50000개 100GeV 100000개 120gev 200000개"),
        ([20, 30, 40, 50, 60],    [50000, 100000, 150000, 200000, 250000],    "20gev 50000 30GeV 100000개 40gev 150000개 50GeV 200000 60gev 250000"),
        ([10, 40, 70],            [100000, 200000, 300000],                   "10GeV 100000개 40gev 200000개 70GeV 300000"),
        ([10, 30, 50, 80],        [50000, 100000, 200000, 300000],            "10gev 50000개 30GeV 100000 50gev 200000 80GeV 300000개"),

        # ── 부분적 "개" (마지막만, 첫개만, 중간만 등) (8개) ───────────────
        ([10, 30, 50, 70],        [50000, 100000, 200000, 300000],            "10GeV 50000 30GeV 100000 50GeV 200000 70GeV 300000개"),
        ([20, 40, 60],            [80000, 100000, 200000],                    "20GeV 80000개 40GeV 100000 60GeV 200000"),
        ([10, 20, 40, 80],        [50000, 80000, 120000, 200000],             "10GeV 50000 20GeV 80000개 40GeV 120000 80GeV 200000"),
        ([50, 100, 120],          [100000, 200000, 300000],                   "50GeV 100000개 100GeV 200000개 120GeV 300000"),
        ([10, 20, 30, 40, 50],    [50000, 100000, 150000, 200000, 250000],    "10GeV 50000개 20GeV 100000 30GeV 150000 40GeV 200000 50GeV 250000"),
        ([10, 50, 100],           [100000, 200000, 300000],                   "10GeV 100000 50GeV 200000개 100GeV 300000개"),
        ([30, 60, 90],            [50000, 100000, 150000],                    "30GeV 50000개 60GeV 100000개 90GeV 150000"),
        ([20, 50, 70, 110],       [50000, 80000, 120000, 200000],             "20GeV 50000 50GeV 80000 70GeV 120000개 110GeV 200000"),

        # ── 콤마 묶음 + 개별 추가 (개 optional) (10개) ────────────────────
        ([10, 20, 30, 40],        [100000, 100000, 100000, 50000],            "10,20,30gev 100000 40gev 50000"),
        ([10, 20, 30, 40],        [100000, 100000, 100000, 50000],            "10,20,30GeV 100000개 40GeV 50000개"),
        ([10, 20, 50, 100],       [80000, 80000, 200000, 300000],             "10,20GeV 80000 50GeV 200000 100GeV 300000"),
        ([10, 20, 30, 50, 100],   [100000, 100000, 100000, 200000, 300000],   "10,20,30GeV 100000개 50GeV 200000 100GeV 300000개"),
        ([20, 40, 60, 100],       [50000, 50000, 50000, 200000],              "20,40,60gev 50000 100gev 200000"),
        ([10, 30, 50, 70, 100],   [80000, 80000, 80000, 80000, 300000],       "10,30,50,70GeV 80000 100GeV 300000"),
        ([50, 100, 110, 120],     [100000, 100000, 200000, 500000],           "50,100GeV 100000 110GeV 200000 120GeV 500000"),
        ([10, 20, 40, 80, 120],   [50000, 50000, 50000, 200000, 300000],      "10,20,40GeV 50000개 80GeV 200000개 120GeV 300000개"),
        ([10, 20, 30, 40, 50, 100],[100000, 100000, 100000, 100000, 100000, 300000], "10,20,30,40,50GeV 100000 100GeV 300000"),
        ([20, 40, 60, 80, 100],   [50000, 50000, 50000, 200000, 200000],      "20,40,60GeV 50000개 80,100GeV 200000개"),

        # ── 다양한 구분자/공백 변형 (8개) ────────────────────────────────
        ([10, 20, 30],            [100000, 200000, 300000],                   "10GeV,100000개  20GeV,200000개  30GeV,300000개"),
        ([50, 100, 120],          [50000, 100000, 150000],                    "50GeV  50000   100GeV  100000   120GeV  150000개"),
        ([10, 20, 30, 40],        [80000, 80000, 80000, 80000],               "10, 20, 30, 40 GeV 각각 80000개"),
        ([10, 20, 40],            [100000, 200000, 400000],                   "10GeV/100000개 20GeV/200000개 40GeV/400000개"),
        ([10, 30, 50],            [100000, 150000, 200000],                   "10GeV - 100000개, 30GeV - 150000개, 50GeV - 200000개"),
        ([20, 40, 60, 80],        [50000, 80000, 120000, 200000],             "20GeV=50000 40GeV=80000 60GeV=120000 80GeV=200000"),
        ([10, 20, 30, 40, 50],    [100000, 100000, 100000, 100000, 100000],   "10~50GeV (10,20,30,40,50) 모두 100000개"),
        ([10, 30, 50],            [100000, 200000, 300000],                   "10GeV는 100000개, 30GeV는 200000개, 50GeV는 300000개"),

        # ── 5-6자리 경계 변별 — 같은 config에 5자리·6자리 혼재 (20개) ────
        # 모델이 0의 개수를 정확히 세어 옮기도록 강제하는 핵심 케이스
        ([10, 20],                [80000, 800000],                            "10GeV 80000개 20GeV 800000개"),
        ([10, 20, 30],            [50000, 500000, 100000],                    "10GeV 50000개 20GeV 500000개 30GeV 100000개"),
        ([20, 40, 60],            [90000, 900000, 99000],                     "20GeV 90000 40GeV 900000 60GeV 99000"),
        ([10, 50],                [10000, 100000],                            "10GeV 10000개 50GeV 100000개"),
        ([10, 20, 30, 40],        [20000, 200000, 30000, 300000],             "10GeV 20000 20GeV 200000 30GeV 30000 40GeV 300000"),
        ([50, 100],               [850000, 85000],                            "50GeV 850000개 100GeV 85000개"),
        ([10, 30, 50],            [40000, 400000, 44000],                     "10gev 40000 30gev 400000 50gev 44000"),
        ([20, 40],                [60000, 600000],                            "20GeV 60000개, 40GeV 600000개"),
        ([10, 20, 50, 100],       [95000, 950000, 105000, 15000],             "10GeV 95000개 20GeV 950000개 50GeV 105000개 100GeV 15000개"),
        ([10, 20],                [110000, 11000],                            "10GeV 110000개 20GeV 11000개"),
        ([30, 60, 90],            [70000, 700000, 77000],                     "30GeV 70000개 60GeV 700000개 90GeV 77000개"),
        ([10, 20, 30],            [125000, 12500, 250000],                    "10GeV 125000개 20GeV 12500개 30GeV 250000개"),
        ([10, 20, 40],            [55000, 550000, 65000],                     "10GeV 55000개 20GeV 550000개 40GeV 65000개"),
        ([10, 20, 30, 40, 50],    [10000, 100000, 20000, 200000, 500000],     "10GeV 10000 20GeV 100000 30GeV 20000 40GeV 200000 50GeV 500000"),
        ([100, 120],              [130000, 13000],                            "100GeV 130000 120GeV 13000"),
        ([10, 50, 100],           [480000, 48000, 840000],                    "10GeV 480000개, 50GeV 48000개, 100GeV 840000개"),
        ([20, 40, 60, 80],        [15000, 150000, 25000, 250000],             "20gev 15000 40gev 150000개 60gev 25000 80gev 250000개"),
        ([10, 20],                [999000, 99900],                            "10GeV 999000개 20GeV 99900개"),
        ([10, 20, 30],            [80000, 80000, 800000],                     "10,20GeV 80000개, 30GeV 800000개"),
        ([10, 20, 30, 40],        [500000, 500000, 50000, 50000],             "10,20GeV 500000개 30,40GeV 50000개"),

        # ── 비-라운드 5-6자리 (8개) ──────────────────────────────────────
        ([10, 20, 30],            [123000, 234000, 345000],                   "10GeV 123000개 20GeV 234000개 30GeV 345000개"),
        ([20, 40, 60],            [85000, 175000, 465000],                    "20GeV 85000개 40GeV 175000개 60GeV 465000개"),
        ([10, 50, 90],            [120000, 240000, 360000],                   "10gev 120000 50gev 240000 90gev 360000"),
        ([10, 20, 30, 40],        [110000, 220000, 330000, 440000],           "10GeV 110000 20GeV 220000 30GeV 330000 40GeV 440000"),
        ([30, 60, 120],           [75000, 150000, 675000],                    "30GeV 75000개, 60GeV 150000개, 120GeV 675000개"),
        ([10, 40, 80],            [98000, 198000, 298000],                    "10GeV 98000개 40GeV 198000개 80GeV 298000개"),
        ([20, 50, 80, 110],       [135000, 245000, 355000, 465000],           "20GeV 135000개 50GeV 245000개 80GeV 355000개 110GeV 465000개"),
        ([10, 20],                [625000, 62500],                            "10GeV 625000개 20GeV 62500개"),

        # ── 3-4자리 이벤트 (소량 유지 — 커버리지용) (6개) ─────────────────
        ([10, 20, 30],            [500, 1000, 2000],                          "10GeV 500개 20GeV 1000개 30GeV 2000개"),
        ([20, 40, 60],            [100, 200, 300],                            "20gev 100 40gev 200 60gev 300"),
        ([10, 30, 50, 70],        [500, 500, 500, 500],                       "10,30,50,70GeV 각각 500개씩"),
        ([20, 40, 60],            [3000, 5000, 8000],                         "20GeV 3000개 40GeV 5000개 60GeV 8000개"),
        ([10, 20, 30],            [5000, 5000, 5000],                         "10,20,30GeV 모두 5000개씩"),
        ([10, 50, 100],           [700, 7000, 70000],                         "10gev 700 50gev 7000 100gev 70000"),

        # ── 단일 에너지 (2개) ────────────────────────────────────────────
        ([50],                    [50000],                                    "50GeV 50000개"),
        ([100],                   [100000],                                   "100GeV 100000개"),
    ]
    for energy_list, events_list, user_input in test_cases:
        all_ex.extend(generate_workflow_normal(energy_list, events_list, user_input))

    # ── 에너지별 DAQ config 지정 케이스 (기본은 setup, 지정 시 그 이름 사용) ──
    # 각 항목: (energy_list, events_list, config_list, user_input)
    config_cases = [
        ([1, 2, 3],        [10000, 20000, 200000],  [None, None, "setup1"],
         "1GeV 10000 2GeV 20000 3GeV setup1로 200000"),
        ([10, 20, 30],     [100000, 200000, 300000], [None, None, "setup1"],
         "10GeV 100000 20GeV 200000 30GeV setup1로 300000"),
        ([20, 40],         [100000, 200000],         [None, "config1"],
         "20GeV 100000개 40GeV config1로 200000개"),
        ([10, 50, 100],    [80000, 80000, 80000],    ["test", "test", "test"],
         "test로 10,50,100GeV 모두 80000개"),
        ([10, 20],         [50000, 100000],          ["setup1", "setup2"],
         "10GeV setup1로 50000개 20GeV setup2로 100000개"),
        ([30, 60],         [150000, 300000],         ["test", None],
         "30GeV는 test 설정으로 150000개, 60GeV는 300000개"),
        ([10, 20, 30, 40], [100000, 100000, 100000, 500000], [None, None, None, "setup2"],
         "10,20,30GeV 100000개 40GeV setup2로 500000개"),
        ([50, 100],        [200000, 400000],         ["physics", "physics"],
         "physics 설정으로 50GeV 200000개 100GeV 400000개"),
    ]
    for energy_list, events_list, config_list, user_input in config_cases:
        all_ex.extend(generate_workflow_normal(energy_list, events_list, user_input, config_list=config_list))

    mid_cases = [
        # 각각
        ([10, 50, 100, 120],      [50000, 200000, 300000, 500000],   2),
        ([10, 20, 30, 40, 50],    [50000, 100000, 200000, 300000, 400000], 3),
        ([20, 40, 60, 80, 100],   [50000, 100000, 200000, 300000, 500000], 2),
        ([20, 40, 60, 80, 100],   [50000, 100000, 200000, 300000, 500000], 4),
        ([10, 20, 30, 40, 50, 60],[50000, 100000, 150000, 200000, 250000, 300000], 2),
        ([10, 20, 30, 40, 50, 60],[50000, 100000, 150000, 200000, 250000, 300000], 4),
        ([10, 20, 30, 40, 50],    [100000, 200000, 300000, 400000, 500000], 3),
        ([20, 40, 60, 80, 100, 120],[100000, 150000, 200000, 250000, 300000, 350000], 3),
        # 같은
        ([10, 20, 30, 40],        [100000, 100000, 100000, 100000],  2),
        ([20, 50, 100, 120],      [100000, 100000, 100000, 100000],  3),
        ([10, 20, 60, 70],        [80000, 80000, 200000, 200000],    2),
        ([10, 20, 30, 40, 50, 60],[100000, 100000, 100000, 100000, 100000, 100000], 3),
        ([10, 20, 30, 40, 50],    [100000, 100000, 100000, 100000, 100000], 2),
        ([10, 30, 50, 70, 100],   [80000, 100000, 150000, 200000, 300000], 2),
        # 5-6자리 경계 변별 mid cases
        ([10, 20],                [80000, 800000],   1),
        ([10, 20, 30],            [50000, 500000, 100000], 2),
        ([10, 50],                [10000, 100000],   1),
        ([20, 40, 60],            [90000, 900000, 99000], 2),
        ([10, 20, 30, 40],        [20000, 200000, 30000, 300000], 2),
        # 3-4자리 (소량)
        ([10, 20, 30],            [500, 1000, 2000], 1),
        ([20, 40, 60],            [3000, 5000, 8000], 1),
        ([10, 20, 30, 40],        [1000, 2000, 3000, 4000], 2),
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
