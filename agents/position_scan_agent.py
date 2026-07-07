#!/usr/bin/env python3
"""Position Scan Agent — 센터 타워와 이웃 타워의 peakADC가 교차하는 경계를 찾아
   타워의 horizontal/vertical 센터를 결정한다.

흐름(calib_scan_agent 구조를 그대로 따름):
  센터 타워 / 방향(horizontal·vertical) / 채널(C·S) / 에너지 / 이벤트 /
  estimated center (x,y) / 이동 간격(interval)을 입력받은 뒤,
  estimated center에서 한쪽(neg) 방향으로 interval씩 이동하며 매 위치에서 run을 받고,
  센터 타워와 그 방향 이웃 타워의 peakADC(Mean, DQM valley-cut)가 교차하는 지점을
  선형보간으로 찾는다. 그 다음 반대(pos) 방향으로 같은 작업을 반복해 두 경계를 얻고,
  두 경계의 평균을 센터로 산출해 JSON 파일로 저장한다.

position_calculator는 사용하지 않는다 — 이동 위치는 estimated center + interval로만 계산하고,
이웃 타워의 "이름"만 그리드 인접(M{m}T{t} 인덱스)으로 찾는다.
"""

import json
import sys
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

from tools.daq_tool import DAQRunTool

from .base_agent import BaseAgent
sys.path.append(str(Path(__file__).parent.parent))
from config import AGENT_MODELS, MSG_PLOT_CONFIRM, PROJECT_ROOT


# ── 6x6 그리드 (M{m}T{t} → grid col/row). position_calculator와 무관한 순수 인덱스 매핑. ──
#   grid_col = 2*((m-1)%3) + (t-1)%2
#   grid_row = 2*((m-1)//3) + (t-1)//2
def _tower_to_grid(tower: str) -> Tuple[int, int]:
    m, t = int(tower[1]), int(tower[3])
    col = 2 * ((m - 1) % 3) + (t - 1) % 2
    row = 2 * ((m - 1) // 3) + (t - 1) // 2
    return col, row


def _grid_to_tower(col: int, row: int) -> Optional[str]:
    if not (0 <= col <= 5 and 0 <= row <= 5):
        return None
    m = (row // 2) * 3 + (col // 2) + 1
    t = (row % 2) * 2 + (col % 2) + 1
    return f"M{m}T{t}"


# 스윕 방향(neg=왼쪽/위, pos=오른쪽/아래)별 스테이지 좌표 이동 부호는 하드코딩하지 않고
# Position Calculator의 타워 좌표 차이 sign(neighbor − center)로 도출한다 (_compute_sweep_signs).
# 즉 "표에서 그 이웃 타워로 빔을 옮기려면 스테이지를 어느 방향으로 움직여야 하는가"를
# position calc 규약 그대로 따른다. (스캔 위치 자체는 여전히 est_center + interval로만 계산.)

MESSAGE_MOVE_REQ = "x = {x:.3f} mm, y = {y:.3f} mm 으로 이동해주세요."

RESULTS_DIR = PROJECT_ROOT / "position_scan_results"


class PositionScanAgent(BaseAgent):
    def __init__(
        self,
        center_tower: str,
        direction: str,            # "horizontal" | "vertical"
        channel: str,              # "C" | "S"
        beam_energy: float,
        target_events: int,
        est_center: Dict[str, float],   # {"x":.., "y":..}
        interval: float,
        max_steps: int = 20,
        use_base_model: bool = False,
        io_handler=None,
    ):
        model_config = AGENT_MODELS["position_scan"]
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
            agent_name=f"Position Scan [{center_tower}/{direction}]",
            io_handler=io_handler,
        )

        self.daq_tool = DAQRunTool()

        center_tower = center_tower.upper()
        direction = direction.lower()
        channel = channel.upper()

        col, row = _tower_to_grid(center_tower)
        if direction == "horizontal":
            neighbor_neg = _grid_to_tower(col - 1, row)   # 왼쪽
            neighbor_pos = _grid_to_tower(col + 1, row)   # 오른쪽
        else:
            neighbor_neg = _grid_to_tower(col, row - 1)   # 위쪽
            neighbor_pos = _grid_to_tower(col, row + 1)   # 아래쪽

        # 이동 부호(neg/pos)를 Position Calculator 기준으로 도출
        self._sign_neg, self._sign_pos = self._compute_sweep_signs(
            center_tower, direction, neighbor_neg, neighbor_pos
        )

        self.state = {
            "phase": "scanning",
            "center_tower": center_tower,
            "channel": channel,
            "direction": direction,
            "beam_energy": beam_energy,
            "target_events": target_events,
            "est_center": {"x": float(est_center["x"]), "y": float(est_center["y"])},
            "interval": float(interval),
            "max_steps": int(max_steps),

            "neighbor_neg": neighbor_neg,
            "neighbor_pos": neighbor_pos,

            # 스윕 진행 상태
            "sweep": "neg",            # "neg" → "pos"
            "step_idx": 0,             # 이번 스윕에서 estimated center로부터 이동한 스텝 수
            "samples": [],             # [{sweep, step, coord, center_adc, neighbor_adc, diff, run}]
            "boundary_neg": None,      # 왼쪽 / 위 경계 좌표
            "boundary_pos": None,      # 오른쪽 / 아래 경계 좌표

            # per-step 부킹 (calib과 동일)
            "y_confirmed": False,
            "needs_plot_confirm": False,
            "last_run_number": None,
            "done": False,
        }
        self.log(
            f"Agent 초기화: center={center_tower}, dir={direction}, ch={channel}, "
            f"neighbors(neg={neighbor_neg}[sign{self._sign_neg:+.0f}], "
            f"pos={neighbor_pos}[sign{self._sign_pos:+.0f}]), "
            f"E={beam_energy}GeV, events={target_events}, est={est_center}, interval={interval}"
        )

    @staticmethod
    def _compute_sweep_signs(center_tower, direction, neighbor_neg, neighbor_pos):
        """Position Calculator로 center/neighbor 좌표를 구해 이동 부호를 도출.
        sign = sign(neighbor_axis − center_axis) — 그 이웃 타워로 빔을 옮기는 스테이지 방향.
        position calc는 부호 결정에만 쓰고 스캔 위치 계산엔 쓰지 않는다."""
        from tools.position_calculator_tool import get_calculator
        calc = get_calculator()
        axis = "x" if direction == "horizontal" else "y"
        center_axis = calc.calculate_tower_position(center_tower)[axis]

        def _sign_toward(neigh):
            if neigh is None:
                return None
            d = calc.calculate_tower_position(neigh)[axis] - center_axis
            return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)

        sign_neg = _sign_toward(neighbor_neg)
        sign_pos = _sign_toward(neighbor_pos)
        # edge 타워(이웃 없음) fallback: 반대 부호, 둘 다 없으면 관례값
        if sign_neg is None:
            sign_neg = -sign_pos if sign_pos else -1.0
        if sign_pos is None:
            sign_pos = -sign_neg if sign_neg else +1.0
        return sign_neg, sign_pos

    # ── 좌표/이웃 헬퍼 ───────────────────────────────────────────

    def _axis_key(self) -> str:
        return "x" if self.state["direction"] == "horizontal" else "y"

    def _current_sign(self) -> float:
        return self._sign_neg if self.state["sweep"] == "neg" else self._sign_pos

    def _current_axis_coord(self) -> float:
        base = self.state["est_center"][self._axis_key()]
        return base + self._current_sign() * self.state["interval"] * self.state["step_idx"]

    def _position_for_current_step(self) -> Optional[Dict[str, float]]:
        est = self.state["est_center"]
        coord = self._current_axis_coord()
        if self.state["direction"] == "horizontal":
            return {"x": coord, "y": est["y"]}
        return {"x": est["x"], "y": coord}

    def _current_neighbor(self) -> Optional[str]:
        return self.state["neighbor_neg"] if self.state["sweep"] == "neg" else self.state["neighbor_pos"]

    # ── peakADC 측정 (sim agent가 이 메서드만 override) ──────────

    def _measure_peakadc(self, run_number: int, tower: str, channel: str) -> Optional[float]:
        """DQM JSON valley-cut peakADC Mean (config 구간 기반). hv_equalization과 동일 소스."""
        from tools.hv_equalization_tool import calculate_valley_cut_average
        mean, _ = calculate_valley_cut_average(run_number, channel, tower)
        return mean

    # ── System Prompt ───────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        return """You are Position Scan Agent for test beam experiments.

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

    # ── Step hint ───────────────────────────────────────────────

    def _get_step_hint(self) -> str:
        st = self.state
        pos = self._position_for_current_step()
        base = "Phase: scanning"
        if not st.get("y_confirmed"):
            return f"{base} | REQUIRED NEXT: position move message (step 1a, x={pos['x']:.3f}, y={pos['y']:.3f})"
        if not st.get("needs_plot_confirm"):
            return f"{base} | REQUIRED NEXT: daq_run_tool (step 1b, x={pos['x']:.3f}, y={pos['y']:.3f})"
        return (f"{base} | DAQ done (Run {st.get('last_run_number')}) — "
                f"REQUIRED NEXT: plot confirmation message (step 1c). DO NOT call daq_run_tool.")

    def _sweep_label(self) -> str:
        if self.state["direction"] == "horizontal":
            return "왼쪽" if self.state["sweep"] == "neg" else "오른쪽"
        return "위쪽" if self.state["sweep"] == "neg" else "아래쪽"

    # ── State context ───────────────────────────────────────────

    def _build_state_context(self) -> str:
        # LLM은 다음 행동(이동/daq/plot)만 결정한다 — 측정/경계/스윕 등 코드 소유 정보는 노출하지 않는다.
        st = self.state
        pos = self._position_for_current_step()
        lines = []
        lines.append(f"Phase: {st['phase']}")
        lines.append(f"Beam Energy: {st['beam_energy']} GeV | Target Events: {st['target_events']}")
        lines.append(f"needs_plot_confirm: {st.get('needs_plot_confirm', False)}")
        lines.append(f"Current Position: x={pos['x']:.3f}, y={pos['y']:.3f}  [Position confirmed: {st.get('y_confirmed', False)}]")
        return "\n".join(lines)

    def build_full_context(self, current_input: Optional[str] = None) -> str:
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

    # ── Tool 실행 (daq만) ───────────────────────────────────────

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        if tool_name == "none":
            return "no_tool_executed"

        if tool_name == "daq_run_tool":
            self._apply_daq_params_from_state(
                params,
                events=self.state.get("target_events"),
                beam_energy=self.state.get("beam_energy"),
                program="Position Scan",
                pos=self._position_for_current_step(),
            )
            result = self._run_tool_with_retry(
                lambda: self.daq_tool.execute(params, line_callback=self.io.send_tool_output),
                "daq_run_tool",
            )
            run_number = self._extract_run_number(result)
            if run_number:
                self.state["last_run_number"] = run_number
                self.log(f"DAQ Run {run_number} 완료: {self._position_for_current_step()}, "
                         f"{params.get('events', 0)} events")
            self.state["needs_plot_confirm"] = True
            return result

        return f"Error: Unknown tool {tool_name}"

    # ── Guards ──────────────────────────────────────────────────

    def _guard_tool(self, tool_name: str, decision) -> Optional[str]:
        if self.state.get("needs_plot_confirm"):
            return (f"needs_plot_confirm=True — DO NOT call {tool_name}. "
                    f'Send: {{"message": "{MSG_PLOT_CONFIRM}"}}')
        if tool_name == "daq_run_tool" and not self.state.get("y_confirmed"):
            return f"위치 미확인. 위치 이동 메시지를 먼저 보내세요. {self._get_step_hint()}"
        return None

    def _guard_ai_message(self, message: str) -> Optional[str]:
        if MSG_PLOT_CONFIRM in message and not self.state.get("needs_plot_confirm"):
            return f"needs_plot_confirm=False — DO NOT send plot confirmation. {self._get_step_hint()}"
        return None

    # ── 사용자 입력 처리 (위치 확인 + plot 확인 → 측정/판정/이동) ──

    def _on_user_input(self, user_input: str):
        # 1) 위치 이동 확인
        if not self.state.get("y_confirmed"):
            self.state["y_confirmed"] = True
            self.log(f"위치 확인: {self._position_for_current_step()}")
            return
        # 2) DAQ 후 plot 확인 → 측정 + cross 판정 + 다음 단계
        if self.state.get("needs_plot_confirm"):
            self.state["needs_plot_confirm"] = False
            self._measure_and_advance()

    def _measure_and_advance(self):
        st = self.state
        run = st.get("last_run_number") or 0
        center_tower = st["center_tower"]
        neighbor = self._current_neighbor()
        channel = st["channel"]
        coord = self._current_axis_coord()

        c_adc = self._measure_peakadc(run, center_tower, channel)
        n_adc = self._measure_peakadc(run, neighbor, channel) if neighbor else None
        if c_adc is None:
            c_adc = 0.0
            self.log(f"WARNING: {center_tower}{channel} peakADC 측정 실패 → 0.0 처리")
        if n_adc is None:
            n_adc = 0.0
            self.log(f"WARNING: {neighbor}{channel} peakADC 측정 실패 → 0.0 처리")

        diff = c_adc - n_adc
        sample = {
            "sweep": st["sweep"], "step": st["step_idx"], "coord": round(coord, 4),
            "center_adc": round(c_adc, 2), "neighbor_adc": round(n_adc, 2),
            "diff": round(diff, 2), "run": run,
        }
        st["samples"].append(sample)
        self.io.send_tool_output(
            f"📏 [{self._sweep_label()}] step {st['step_idx']} @ {self._axis_key()}={coord:.3f} | "
            f"{center_tower}{channel}={c_adc:.1f}, {neighbor}{channel}={n_adc:.1f} | diff={diff:.1f}"
        )

        # cross 판정: 직전 diff>0, 현재 diff<=0 → 두 점 선형보간
        prev = self._prev_sample_this_sweep()
        crossed = prev is not None and prev["diff"] > 0 and diff <= 0
        if crossed:
            d0, d1 = prev["diff"], diff
            p0, p1 = prev["coord"], coord
            boundary = p0 + (p1 - p0) * d0 / (d0 - d1) if (d0 - d1) != 0 else p1
            self._record_boundary(boundary)
            return

        # cross 없이 max_steps 도달 → 경계 미발견
        if st["step_idx"] + 1 >= st["max_steps"]:
            self.log(f"WARNING: {self._sweep_label()} 스윕 max_steps({st['max_steps']}) 도달 — 경계 미발견")
            self.io.send_ai_message(
                f"⚠️ {self._sweep_label()} 방향에서 max_steps({st['max_steps']}) 내에 경계를 찾지 못했습니다."
            )
            self._record_boundary(None)
            return

        # 한 칸 더 이동
        st["step_idx"] += 1
        st["y_confirmed"] = False

    def _prev_sample_this_sweep(self) -> Optional[Dict[str, Any]]:
        for s in reversed(self.state["samples"][:-1]):
            if s["sweep"] == self.state["sweep"]:
                return s
        return None

    def _record_boundary(self, boundary: Optional[float]):
        st = self.state
        if st["sweep"] == "neg":
            st["boundary_neg"] = boundary
            if boundary is not None:
                self._plot_crossing(boundary)
                self.io.send_ai_message(
                    f"✅ {self._sweep_label()} 경계 발견: {self._axis_key()}={boundary:.3f} mm"
                )
            # 반대 방향 스윕으로 전환 (estimated center로 복귀)
            st["sweep"] = "pos"
            st["step_idx"] = 0
            st["y_confirmed"] = False
            st["needs_plot_confirm"] = False
            self.io.send_ai_message(
                f"이제 {self._sweep_label()} 방향 경계를 찾기 위해 estimated center로 복귀합니다."
            )
        else:
            st["boundary_pos"] = boundary
            if boundary is not None:
                self._plot_crossing(boundary)
                self.io.send_ai_message(
                    f"✅ {self._sweep_label()} 경계 발견: {self._axis_key()}={boundary:.3f} mm"
                )
            self._finalize()

    # ── crossing 플롯 (경계 발견 시 왜 여기인지 시각화) ───────────

    def _plot_crossing(self, boundary: float):
        """이번 스윕에서 센터/이웃 타워 peakADC가 교차하는 지점을 그린다.
        hv_equalization의 fitting plot처럼 사용자가 경계 근거를 눈으로 확인하게 한다.
        (측정된 sample 점들을 그대로 사용 — real/sim 공통)."""
        st = self.state
        sweep = st["sweep"]
        samples = [s for s in st["samples"] if s["sweep"] == sweep]
        if not samples:
            return
        samples = sorted(samples, key=lambda s: s["coord"])
        axis = self._axis_key()
        center_tower = st["center_tower"]
        neighbor = self._current_neighbor()
        channel = st["channel"]

        try:
            import tempfile
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            coords = [s["coord"] for s in samples]
            c_adc = [s["center_adc"] for s in samples]
            n_adc = [s["neighbor_adc"] for s in samples]

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.suptitle(
                f"Position Scan — {center_tower} vs {neighbor}  "
                f"({st['direction']}, {self._sweep_label()}, peakADC {channel})",
                fontsize=12,
            )
            ax.plot(coords, c_adc, "-o", color="#4c72b0", markersize=6,
                    label=f"{center_tower}{channel} (center)")
            ax.plot(coords, n_adc, "-s", color="#dd8452", markersize=6,
                    label=f"{neighbor}{channel} (neighbor)")

            # 교차하는 마지막 두 점 구간 강조 + 경계 수직선
            cross_seg = None
            for i in range(1, len(samples)):
                if samples[i - 1]["diff"] > 0 and samples[i]["diff"] <= 0:
                    cross_seg = (samples[i - 1], samples[i])
            if cross_seg is not None:
                x0, x1 = cross_seg[0]["coord"], cross_seg[1]["coord"]
                ax.axvspan(min(x0, x1), max(x0, x1), color="#f2d0d0", alpha=0.4,
                           label="crossing region")
            ax.axvline(boundary, color="red", linestyle="--", linewidth=1.6,
                       label=f"boundary {axis}={boundary:.3f} mm")
            ax.annotate(f"{boundary:.3f}", xy=(boundary, min(min(c_adc), min(n_adc))),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color="red")

            ax.set_xlabel(f"{axis} (mm)")
            ax.set_ylabel("peakADC")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            tmp = tempfile.NamedTemporaryFile(
                suffix=f"_PosScan_{center_tower}_{st['direction']}_{sweep}.png",
                delete=False,
            )
            plt.savefig(tmp.name, dpi=120, bbox_inches="tight")
            plt.close(fig)
            self.io.send_plots([tmp.name])
            self.log(f"crossing plot 저장: {tmp.name}")
        except Exception as e:
            self.log(f"WARNING: crossing plot 생성 실패: {e}")

    # ── 결과 산출 + 파일 저장 ────────────────────────────────────

    def _finalize(self):
        st = self.state
        b_neg, b_pos = st["boundary_neg"], st["boundary_pos"]
        axis = self._axis_key()
        est = st["est_center"]

        center_coord = None
        if b_neg is not None and b_pos is not None:
            center_coord = (b_neg + b_pos) / 2.0

        if st["direction"] == "horizontal":
            boundaries = {"left": b_neg, "right": b_pos}
            found_center = {"x": center_coord, "y": est["y"]}
        else:
            boundaries = {"top": b_neg, "bottom": b_pos}
            found_center = {"x": est["x"], "y": center_coord}

        result = {
            "center_tower": st["center_tower"],
            "channel": st["channel"],
            "direction": st["direction"],
            "beam_energy": st["beam_energy"],
            "events": st["target_events"],
            "interval": st["interval"],
            "max_steps": st["max_steps"],
            "estimated_center": est,
            "neighbors": {"neg": st["neighbor_neg"], "pos": st["neighbor_pos"]},
            "boundaries": boundaries,
            "found_center": found_center,
            "samples": st["samples"],
            "timestamp": datetime.now().isoformat(),
        }

        path = self._save_result(result)
        self.io.send_tool_output(f"💾 결과 저장: {path}")
        self.io.send_ai_message(self._format_result_message(result))
        st["done"] = True
        self.log(f"Position Scan 완료: {result['boundaries']} → center={found_center}")

    def _save_result(self, result: Dict[str, Any]) -> str:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{result['center_tower']}_{result['direction']}.json"
        path = RESULTS_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return str(path)

    def _format_result_message(self, result: Dict[str, Any]) -> str:
        b = result["boundaries"]
        fc = result["found_center"]
        if result["direction"] == "horizontal":
            lines = [
                f"📊 Position Scan 완료 — {result['center_tower']} (horizontal, peakADC {result['channel']})",
                f"  왼쪽 경계: {self._fmt(b['left'])} mm",
                f"  오른쪽 경계: {self._fmt(b['right'])} mm",
                f"  → horizontal center: x = {self._fmt(fc['x'])} mm (y = {fc['y']:.3f} 고정)",
            ]
        else:
            lines = [
                f"📊 Position Scan 완료 — {result['center_tower']} (vertical, peakADC {result['channel']})",
                f"  위 경계: {self._fmt(b['top'])} mm",
                f"  아래 경계: {self._fmt(b['bottom'])} mm",
                f"  → vertical center: y = {self._fmt(fc['y'])} mm (x = {fc['x']:.3f} 고정)",
            ]
        return "\n".join(lines)

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        return f"{v:.3f}" if v is not None else "N/A(미발견)"

    # ── State 보호: 이 agent는 LLM의 update_state를 받지 않는다 ──

    def _update_state(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            self.log(f"WARNING: LLM tried to update '{key}' = {value} — rejected (코드 소유)")

    # ── 공용 드라이버 hooks ──────────────────────────────────────

    def _print_banner(self):
        print(f"\n{'='*70}\n⚡ Position Scan Agent — {self.state['center_tower']} ({self.state['direction']})\n{'='*70}")

    def _is_complete(self) -> bool:
        return bool(self.state.get("done"))
