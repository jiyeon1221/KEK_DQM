#!/usr/bin/env python3
"""AgentRunner — FastAPI WebSocket ↔ blocking agent .run() bridge."""

import queue
import threading
import traceback
from typing import Optional

from agents.io_handler import WebSocketIO

# 타워 지그재그 스캔 순서 — 모든 agent에서 이 상수를 참조 (6x6 그리드, 36타워 serpentine)
TOWER_ORDER = [
    # Row 0, L→R
    "M1T1", "M1T2", "M2T1", "M2T2", "M3T1", "M3T2",
    # Row 1, R→L
    "M3T4", "M3T3", "M2T4", "M2T3", "M1T4", "M1T3",
    # Row 2, L→R
    "M4T1", "M4T2", "M5T1", "M5T2", "M6T1", "M6T2",
    # Row 3, R→L
    "M6T4", "M6T3", "M5T4", "M5T3", "M4T4", "M4T3",
    # Row 4, L→R
    "M7T1", "M7T2", "M8T1", "M8T2", "M9T1", "M9T2",
    # Row 5, R→L
    "M9T4", "M9T3", "M8T4", "M8T3", "M7T4", "M7T3",
]


shared_locks = {
    "daq": threading.Lock(),
    "hv": threading.Lock(),
}

shared_state: dict = {}


def update_shared_state(updates: dict):
    """Merge updates into shared_state."""
    shared_state.update(updates)


def set_shared_state_ref(agent_state: dict):
    """agent.state를 shared_state에 반영. runner 주입 키(agent_type, _output_queue)는 보존."""
    global shared_state
    preserved = {k: shared_state[k] for k in ("agent_type", "_output_queue")
                 if k in shared_state}
    shared_state.clear()
    shared_state.update(agent_state)
    shared_state.update(preserved)
    # Keep a back-reference so periodic sync can refresh
    shared_state["_agent_state_ref"] = agent_state


def sync_shared_state():
    ref = shared_state.get("_agent_state_ref")
    if ref is None:
        return
    for key in ("current_energy", "current_tower", "last_run_number",
                "phase", "current_energy_idx", "scan_order"):
        if key in ref:
            shared_state[key] = ref[key]
    if "last_run_number" in ref:
        shared_state["current_run"] = ref["last_run_number"]


def clear_shared_state():
    """Clear shared_state when scenario agent stops."""
    shared_state.clear()


def run_agent_thread(
    agent_name: str,
    params: dict,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    waiting_flag: threading.Event = None,
):
    io = WebSocketIO(input_queue, output_queue, stop_event, waiting_flag=waiting_flag)

    try:
        io.send_status(f"{agent_name} 에이전트 실행 중")
        # _output_queue is consumed by DAQRunTool to push DQM live events
        # back to the browser without taking io_handler as a dependency.
        update_shared_state({"agent_type": agent_name, "_output_queue": output_queue})

        if agent_name == "em_scan":
            from agents.energy_scan_agent import EnergyScanAgent
            _em_kwargs = {}
            if params.get("tower"):
                _em_kwargs["tower"] = params["tower"]
            # dqm_dashboards.yml의 EM_scan cells가 ${current_tower}를 쓰므로
            # (fCanvas_Tower5 하드코딩 제거) DQM live가 올바른 타워를 그리려면
            # calib_scan/hv_equalization과 동일하게 shared_state를 채워야 한다.
            # EnergyScanAgent의 기본 tower("M5T3")와 동일한 fallback을 사용.
            update_shared_state({"current_tower": params.get("tower") or "M5T3"})
            agent = EnergyScanAgent(
                energy_config={},
                use_base_model=params.get("use_base_model", False),
                io_handler=io,
                **_em_kwargs,
            )

        elif agent_name == "calib_scan":
            from agents.calib_scan_agent import CalibScanAgent
            agent = CalibScanAgent(
                tower_order=params.get("tower_order") or TOWER_ORDER,
                use_base_model=params.get("use_base_model", False),
                io_handler=io,
            )

        elif agent_name in ("hv_equalization", "hv_equalization_sim"):
            _HV_TOWER_ORDER = params.get("tower_order") or TOWER_ORDER
            _is_sim = (agent_name == "hv_equalization_sim")

            def _ask_float(prompt):
                while True:
                    io.send_ai_message(prompt)
                    val = io.get_input()
                    if val in ("종료", "exit"):
                        return None
                    try:
                        return float(val.strip())
                    except ValueError:
                        io.send_ai_message("⚠️ 숫자를 입력해주세요.")

            def _ask_int(prompt):
                v = _ask_float(prompt)
                return int(v) if v is not None else None

            beam_energy = _ask_float("빔 에너지 (GeV)를 입력해주세요.")
            if beam_energy is None:
                output_queue.put({"type": "agent_done"}); return
            target_events = _ask_int(f"이벤트 수를 입력해주세요.")
            if target_events is None:
                output_queue.put({"type": "agent_done"}); return
            target_adc = _ask_float("목표 peakADC 값을 입력해주세요.")
            if target_adc is None:
                output_queue.put({"type": "agent_done"}); return

            update_shared_state({
                "current_energy": beam_energy,
            })

            from tools.hv_equalization_tool import hv_equalization_start
            result = hv_equalization_start.invoke({
                "target_c": target_adc, "target_s": target_adc,
                "tower": _HV_TOWER_ORDER[0],
            }) if hasattr(hv_equalization_start, "invoke") else hv_equalization_start(
                target_c=target_adc, target_s=target_adc, tower=_HV_TOWER_ORDER[0]
            )
            io.send_tool_output(result)

            AgentClass = None
            if _is_sim:
                from agents.hv_equalization_sim_agent import HVEqualizationSimAgent
                AgentClass = HVEqualizationSimAgent
            else:
                from agents.hv_equalization_agent import HVEqualizationAgent
                AgentClass = HVEqualizationAgent

            for i, tower in enumerate(_HV_TOWER_ORDER):
                if stop_event.is_set():
                    break
                io.send_status(f"[{i + 1}/{len(_HV_TOWER_ORDER)}] {tower} 타워 시작")
                update_shared_state({"current_tower": tower, "agent_type": agent_name})
                tower_agent = AgentClass(
                    tower=tower,
                    beam_energy=beam_energy,
                    target_events=target_events,
                    target_adc=target_adc,
                    use_base_model=params.get("use_base_model", False),
                    io_handler=io,
                )
                set_shared_state_ref(tower_agent.state)
                _hv_sync_stop = threading.Event()
                def _hv_sync(evt=_hv_sync_stop):
                    while not evt.is_set():
                        sync_shared_state()
                        evt.wait(1.0)
                _hv_sync_t = threading.Thread(target=_hv_sync, daemon=True)
                _hv_sync_t.start()
                with tower_agent:
                    tower_agent.run()
                _hv_sync_stop.set()
                io.send_status(f"✅ {tower} 완료")

            if not stop_event.is_set():
                io.send_ai_message("모든 타워 HV Equalization이 완료되었습니다.")
                try:
                    from hv_equalization_scan import _write_fixed_hv
                    _write_fixed_hv()
                    io.send_status("✅ fixed_hv.txt 업데이트 완료")
                except Exception as _fhv_err:
                    io.send_status(f"⚠️ fixed_hv.txt 업데이트 실패: {_fhv_err}")
            io.send_status("모델 언로드 중...")
            clear_shared_state()
            output_queue.put({"type": "agent_done"})
            return

        elif agent_name in ("position_scan", "position_scan_sim"):
            _is_sim = (agent_name == "position_scan_sim")
            center_tower = params.get("tower") or (params.get("tower_order") or TOWER_ORDER)[0]

            def _ask_float(prompt):
                while True:
                    io.send_ai_message(prompt)
                    val = io.get_input()
                    if val in ("종료", "exit"):
                        return None
                    try:
                        return float(val.strip())
                    except ValueError:
                        io.send_ai_message("⚠️ 숫자를 입력해주세요.")

            def _ask_int(prompt):
                v = _ask_float(prompt)
                return int(v) if v is not None else None

            def _ask_choice(prompt, options):
                opts_lower = {o.lower(): o for o in options}
                while True:
                    io.send_ai_message(prompt)
                    val = io.get_input()
                    if val in ("종료", "exit"):
                        return None
                    key = val.strip().lower()
                    if key in opts_lower:
                        return opts_lower[key]
                    io.send_ai_message(f"⚠️ {' / '.join(options)} 중 하나를 입력해주세요.")

            # 방향/채널은 프론트 버튼(params)에서 옴. 없으면 채팅으로 질문(하위호환).
            direction = (params.get("direction") or "").lower()
            if direction not in ("horizontal", "vertical", "both"):
                direction = _ask_choice(
                    "스캔 방향을 선택해주세요. (horizontal / vertical / both)",
                    ["horizontal", "vertical", "both"])
                if direction is None:
                    output_queue.put({"type": "agent_done"}); return
            channel = (params.get("channel") or "").upper()
            if channel not in ("C", "S"):
                channel = _ask_choice("어떤 peakADC를 볼까요? (C / S)", ["C", "S"])
                if channel is None:
                    output_queue.put({"type": "agent_done"}); return

            # 나머지 수치 입력은 한 번만 받아 두 방향(both)에서 공유한다.
            beam_energy = _ask_float("빔 에너지 (GeV)를 입력해주세요.")
            if beam_energy is None:
                output_queue.put({"type": "agent_done"}); return
            target_events = _ask_int("이벤트 수를 입력해주세요.")
            if target_events is None:
                output_queue.put({"type": "agent_done"}); return
            est_x = _ask_float("estimated center X 좌표 (mm)를 입력해주세요.")
            if est_x is None:
                output_queue.put({"type": "agent_done"}); return
            est_y = _ask_float("estimated center Y 좌표 (mm)를 입력해주세요.")
            if est_y is None:
                output_queue.put({"type": "agent_done"}); return
            interval = _ask_float("이동 간격 (mm)을 입력해주세요.")
            if interval is None:
                output_queue.put({"type": "agent_done"}); return

            update_shared_state({"current_tower": center_tower, "current_energy": beam_energy})

            if _is_sim:
                # Sim-ADC: daq/motor 등은 실제로 돌리고 peakADC 측정만 시뮬레이션.
                # (모든 걸 mock하는 완전 sim은 run_web_sim.py → sim.agent_runner 경로.)
                from agents.position_scan_sim_agent import PositionScanSimADCAgent as _PSAgent
            else:
                from agents.position_scan_agent import PositionScanAgent as _PSAgent

            # both → horizontal 먼저 완료·언로드 후 vertical 재로드 (초기 입력은 재질문 안 함).
            directions = ["horizontal", "vertical"] if direction == "both" else [direction]
            for _di, _dir in enumerate(directions):
                if stop_event.is_set():
                    break
                if len(directions) > 1:
                    io.send_ai_message(
                        f"[{_di + 1}/{len(directions)}] {_dir} 방향 스캔을 시작합니다. 모델을 로드합니다...")
                update_shared_state({"agent_type": agent_name, "current_tower": center_tower})
                agent = _PSAgent(
                    center_tower=center_tower,
                    direction=_dir,
                    channel=channel,
                    beam_energy=beam_energy,
                    target_events=target_events,
                    est_center={"x": est_x, "y": est_y},
                    interval=interval,
                    use_base_model=params.get("use_base_model", False),
                    io_handler=io,
                )
                set_shared_state_ref(agent.state)
                _ps_sync_stop = threading.Event()
                def _ps_sync(evt=_ps_sync_stop):
                    while not evt.is_set():
                        sync_shared_state()
                        evt.wait(1.0)
                _ps_sync_t = threading.Thread(target=_ps_sync, daemon=True)
                _ps_sync_t.start()
                with agent:
                    agent.run()
                _ps_sync_stop.set()
                if len(directions) > 1 and _di < len(directions) - 1 and not stop_event.is_set():
                    io.send_ai_message(
                        f"✅ {_dir} 방향 완료. 모델을 언로드하고 다음 방향을 이어서 진행합니다.")
                    io.send_status("모델 언로드 중...")

            if not stop_event.is_set():
                io.send_ai_message("✅ 모든 Position Scan이 완료되었습니다.")
            io.send_status("모델 언로드 중...")
            clear_shared_state()
            output_queue.put({"type": "agent_done"})
            return

        else:
            output_queue.put({"type": "error", "content": f"Unknown agent: {agent_name}"})
            output_queue.put({"type": "agent_done"})
            return

        set_shared_state_ref(agent.state)
        _sync_stop = threading.Event()
        def _sync_loop():
            while not _sync_stop.is_set():
                sync_shared_state()
                _sync_stop.wait(1.0)
        _sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        _sync_thread.start()
        with agent:
            agent.run()
        _sync_stop.set()

        io.send_status("모델 언로드 중...")
        clear_shared_state()
        output_queue.put({"type": "agent_done"})

    except StopAgentException:
        clear_shared_state()
        output_queue.put({"type": "status", "content": "모델 언로드 중..."})
        output_queue.put({"type": "agent_done"})

    except Exception as e:
        clear_shared_state()
        tb = traceback.format_exc()
        output_queue.put({"type": "error", "content": f"Agent error: {str(e)}\n{tb}"})
        output_queue.put({"type": "agent_done"})


class StopAgentException(Exception):
    """Raised inside the agent thread to break out of the run loop."""
    pass


class AgentRunner:

    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()
        self.stop_event: threading.Event = threading.Event()
        self.waiting_flag: threading.Event = threading.Event()

        self.adhoc_queue: queue.Queue = queue.Queue()
        self.confirm_queue: queue.Queue = queue.Queue()
        self.clarify_queue: queue.Queue = queue.Queue()
        self._brain_agent = None
        self._brain_thread: Optional[threading.Thread] = None
        self._brain_stop = threading.Event()

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def brain_ready(self) -> bool:
        return (self._brain_agent is not None
                and self._brain_agent.model is not None
                and self._brain_thread is not None
                and self._brain_thread.is_alive())

    def start_brain(self, use_base_model: bool = False):
        if self.brain_ready:
            return

        from agents.brain_agent import BrainAgent, run_brain_thread

        self._brain_stop.clear()
        self._brain_agent = BrainAgent(
            shared_state=shared_state,
            shared_locks=shared_locks,
            use_base_model=use_base_model,
        )
        self._brain_agent.load()

        self._brain_thread = threading.Thread(
            target=run_brain_thread,
            args=(self._brain_agent, self.adhoc_queue, self.output_queue,
                  self._brain_stop, self.confirm_queue, self.clarify_queue),
            daemon=True,
        )
        self._brain_thread.start()

    def stop_brain(self):
        self._brain_stop.set()
        if self._brain_thread and self._brain_thread.is_alive():
            self._brain_thread.join(timeout=5)
        if self._brain_agent:
            self._brain_agent.unload()
            self._brain_agent = None

    def send_adhoc(self, text: str):
        self.adhoc_queue.put(text)

    def send_confirm(self, confirmed: bool):
        self.confirm_queue.put(confirmed)

    def send_clarify(self, text: str):
        self.clarify_queue.put(text)

    def start(self, agent_name: str, params: dict):
        if self.is_running:
            raise RuntimeError("An agent is already running.")

        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.output_queue.empty():
            self.output_queue.get_nowait()
        while not self.adhoc_queue.empty():
            self.adhoc_queue.get_nowait()
        self.stop_event.clear()

        self.waiting_flag.clear()
        self.thread = threading.Thread(
            target=run_agent_thread,
            args=(agent_name, params, self.input_queue, self.output_queue,
                  self.stop_event, self.waiting_flag),
            daemon=True,
        )
        self.thread.start()

    def send_input(self, text: str):
        self.input_queue.put(text)

    def stop(self):
        try:
            from tools.daq_tool import KILLME_FILE
            from pathlib import Path
            Path(KILLME_FILE).touch()
        except Exception:
            pass
        self.stop_event.set()
        self.input_queue.put("종료")
