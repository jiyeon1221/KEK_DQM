#!/usr/bin/env python3
"""
Training data generator for PositionScanAgent

다른 3개 생성기(EM/calib/hv)와 동일한 방식: agent를 구동하지 않고 프롬프트/컨텍스트를
템플릿으로 직접 재현한다. SYSTEM_PROMPT / _build_state_context / _get_step_hint /
build_full_context 포맷이 PositionScanAgent(agents/position_scan_agent.py)와 완전히 동일해야 한다.

모델이 출력해야 하는 결정은 3종뿐 (측정/cross/보간/이동/종료는 코드 소유):
  1a. position move message
  1b. daq_run_tool
  1c. plot confirmation message
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MSG_PLOT_CONFIRM
from tools.position_calculator_tool import get_calculator

VALID_TOWERS = [f"M{m}T{t}" for m in range(1, 10) for t in range(1, 5)]
ENERGIES = [1, 2, 5, 10, 20, 50, 100, 200]

MESSAGE_MOVE_REQ = "x = {x:.3f} mm, y = {y:.3f} mm 으로 이동해주세요."
MESSAGE_PLOT_CONFIRM = MSG_PLOT_CONFIRM

SYSTEM_PROMPT = """You are Position Scan Agent for test beam experiments.

Follow these steps EXACTLY, repeating for each scan position:

1a. Ask user to move to the current scan position:
  Output: {"message": "x = <x> mm, y = <y> mm 으로 이동해주세요."}
  (Replace <x>, <y> with the position from the step hint.)

After user says "완료":
The SYSTEM marks position confirmed automatically — you do NOT output any state update.

1b. Execute DAQ:
  Output: {"tool": "daq_run_tool", "params": {"events": <target_events>, "pos_h": <x>, "pos_v": <y>, "beam_energy": <energy>}}
  (Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

1c. Request Plot Confirmation (only AFTER the DAQ tool has run):
  Output: {"message": "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."}

After user says "완료" to the plot message:
The SYSTEM advances automatically — you do NOT output any state update.
Proceed as the step hint tells you.

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip or reorder steps.
2. The SYSTEM (not you) owns all bookkeeping and termination. NEVER output update_state — only the step hint tells you the next action.
3. Output JSON format (CHOOSE ONE, NEVER BOTH "tool" and "message"):
   - {"tool": "...", "params": {...}}   (tool execution)
   - {"message": "..."}                  (user message)
4. All "message" field values MUST be written in Korean (한국어) only. Never use Chinese characters (한자).
"""


# ── 그리드 인접 (position_calculator와 무관한 순수 인덱스 매핑) ──
def _tower_to_grid(tower: str):
    m, t = int(tower[1]), int(tower[3])
    return 2 * ((m - 1) % 3) + (t - 1) % 2, 2 * ((m - 1) // 3) + (t - 1) // 2


def _grid_to_tower(col: int, row: int) -> Optional[str]:
    if not (0 <= col <= 5 and 0 <= row <= 5):
        return None
    m = (row // 2) * 3 + (col // 2) + 1
    t = (row % 2) * 2 + (col % 2) + 1
    return f"M{m}T{t}"


def random_events() -> int:
    digits = random.randint(3, 6)
    return random.randint(10 ** (digits - 1), 10 ** digits - 1)


def _interior_towers(direction: str) -> List[str]:
    out = []
    for tw in VALID_TOWERS:
        col, row = _tower_to_grid(tw)
        if direction == "horizontal":
            ok = _grid_to_tower(col - 1, row) and _grid_to_tower(col + 1, row)
        else:
            ok = _grid_to_tower(col, row - 1) and _grid_to_tower(col, row + 1)
        if ok:
            out.append(tw)
    return out


def _sweep_signs(center: str, direction: str):
    """PositionScanAgent._compute_sweep_signs와 동일: sign(neighbor_axis − center_axis)."""
    calc = get_calculator()
    axis = "x" if direction == "horizontal" else "y"
    col, row = _tower_to_grid(center)
    if direction == "horizontal":
        nneg, npos = _grid_to_tower(col - 1, row), _grid_to_tower(col + 1, row)
    else:
        nneg, npos = _grid_to_tower(col, row - 1), _grid_to_tower(col, row + 1)
    c_axis = calc.calculate_tower_position(center)[axis]

    def _s(n):
        if n is None:
            return None
        d = calc.calculate_tower_position(n)[axis] - c_axis
        return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)

    sn, sp = _s(nneg), _s(npos)
    if sn is None:
        sn = -sp if sp else -1.0
    if sp is None:
        sp = -sn if sn else 1.0
    return sn, sp


# ── 컨텍스트 재현 (PositionScanAgent와 문자 단위로 동일) ──

def _build_state_context(state: Dict) -> str:
    pos = state["pos"]
    return "\n".join([
        f"Phase: {state['phase']}",
        f"Beam Energy: {state['beam_energy']} GeV | Target Events: {state['target_events']}",
        f"needs_plot_confirm: {state.get('needs_plot_confirm', False)}",
        f"Current Position: x={pos['x']:.3f}, y={pos['y']:.3f}  [Position confirmed: {state.get('y_confirmed', False)}]",
    ])


def _get_step_hint(state: Dict) -> str:
    pos = state["pos"]
    base = "Phase: scanning"
    if not state.get("y_confirmed"):
        return f"{base} | REQUIRED NEXT: position move message (step 1a, x={pos['x']:.3f}, y={pos['y']:.3f})"
    if not state.get("needs_plot_confirm"):
        return f"{base} | REQUIRED NEXT: daq_run_tool (step 1b, x={pos['x']:.3f}, y={pos['y']:.3f})"
    return (f"{base} | DAQ done (Run {state.get('last_run_number')}) — "
            f"REQUIRED NEXT: plot confirmation message (step 1c). DO NOT call daq_run_tool.")


def _build_history_context(history: List[Dict]) -> str:
    if not history:
        return "(No conversation yet)"
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content']}" for m in history[-10:]
    )


def build_full_context(state: Dict, history: List[Dict], current_input: Optional[str] = None) -> str:
    if current_input is None and history and history[-1]["role"] == "user":
        current_input = history[-1]["content"]
        temp_history = history[:-1]
    else:
        temp_history = history

    parts = ["=== Current State ===", _build_state_context(state), ""]
    parts += ["=== Recent Conversation ===", _build_history_context(temp_history), ""]
    if current_input:
        parts += ["=== Current User Input ===", current_input, ""]
    parts += ["=== Your Task ===", _get_step_hint(state), "", "Output JSON with tool name and parameters."]
    return "\n".join(parts)


def make_example(state, history, decision) -> Dict[str, Any]:
    ctx = build_full_context(state, history)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ctx},
        {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
    ]}


def generate_workflow(center: str, direction: str, channel: str,
                      energy: float, events: int,
                      est_center: Dict[str, float], interval: float) -> List[Dict[str, Any]]:
    sn, sp = _sweep_signs(center, direction)
    axis = "x" if direction == "horizontal" else "y"

    examples: List[Dict[str, Any]] = []
    history: List[Dict] = []
    state = {
        "phase": "scanning", "beam_energy": energy, "target_events": events,
        "needs_plot_confirm": False, "y_confirmed": False,
        "last_run_number": None, "pos": None,
    }
    run = [99000]

    def _pos_at(sign, step):
        coord = est_center[axis] + sign * interval * step
        if direction == "horizontal":
            return {"x": coord, "y": est_center["y"]}
        return {"x": est_center["x"], "y": coord}

    def _emit(sign, step):
        pos = _pos_at(sign, step)
        state["pos"] = pos
        state["y_confirmed"] = False
        state["needs_plot_confirm"] = False

        # 1a: position move message
        dec = {"message": MESSAGE_MOVE_REQ.format(x=pos["x"], y=pos["y"])}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        state["y_confirmed"] = True

        # 1b: DAQ
        dec = {"tool": "daq_run_tool", "params": {
            "events": events, "pos_h": round(pos["x"], 3), "pos_v": round(pos["y"], 3),
            "beam_energy": energy,
        }}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        run[0] += 1
        state["last_run_number"] = run[0]
        state["needs_plot_confirm"] = True

        # 1c: plot confirmation message
        dec = {"message": MESSAGE_PLOT_CONFIRM}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        state["needs_plot_confirm"] = False

    # neg 스윕 → pos 스윕 (각 스윕은 est_center에서 시작해 sign 방향으로 이동)
    for step in range(random.randint(3, 7)):
        _emit(sn, step)
    for step in range(random.randint(3, 7)):
        _emit(sp, step)
    return examples


def main():
    output_file = Path(__file__).parent / "data" / "position_scan_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_ex = []

    for direction in ("horizontal", "vertical"):
        towers = _interior_towers(direction)
        for _ in range(12):
            center = random.choice(towers)
            channel = random.choice(["C", "S"])
            energy = random.choice(ENERGIES)
            events = random_events()
            est = {"x": round(random.uniform(50.0, 160.0), 3),
                   "y": round(random.uniform(850.0, 1050.0), 3)}
            interval = round(random.uniform(2.0, 8.0), 2)
            all_ex.extend(generate_workflow(center, direction, channel, energy, events, est, interval))

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    lengths = [sum(len(m["content"]) for m in ex["messages"]) for ex in all_ex]
    print(f"Generated {len(all_ex)} samples → {output_file}")
    print(f"   char len  max={max(lengths):,}  avg={sum(lengths)/len(lengths):,.0f}")


if __name__ == "__main__":
    main()
