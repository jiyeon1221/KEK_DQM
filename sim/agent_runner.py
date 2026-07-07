#!/usr/bin/env python3
"""
AgentRunner (Simulation)
------------------------
run_web_sim.py 전용 — finetuned agent는 그대로, tool execution만 mock.
"""

import queue
import threading
import traceback

from agents.agent_runner import (
    AgentRunner,
    TOWER_ORDER,
    clear_shared_state,
    set_shared_state_ref,
    shared_locks,
    shared_state,
    sync_shared_state,
    update_shared_state,
)
from agents.brain_agent import run_brain_thread
from agents.io_handler import WebSocketIO
from sim.tool_simulator import get_simulator


def run_agent_thread(
    agent_name: str,
    params: dict,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    waiting_flag: threading.Event = None,
):
    io = WebSocketIO(input_queue, output_queue, stop_event, waiting_flag=waiting_flag)
    sim = get_simulator()

    try:
        io.send_status(f"[SIM] {agent_name} 에이전트 실행 중")
        update_shared_state({"agent_type": agent_name, "_output_queue": output_queue})

        if agent_name == "em_scan":
            from sim.energy_scan_agent import EnergyScanSimAgent
            _em_kwargs = {}
            if params.get("tower"):
                _em_kwargs["tower"] = params["tower"]
            agent = EnergyScanSimAgent(
                energy_config={},
                use_base_model=params.get("use_base_model", False),
                io_handler=io,
                **_em_kwargs,
            )

        elif agent_name == "calib_scan":
            from sim.calib_scan_agent import CalibScanSimAgent
            agent = CalibScanSimAgent(
                tower_order=params.get("tower_order") or TOWER_ORDER,
                use_base_model=params.get("use_base_model", False),
                io_handler=io,
            )

        elif agent_name in ("hv_equalization", "hv_equalization_sim"):
            _HV_TOWER_ORDER = params.get("tower_order") or TOWER_ORDER

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
                output_queue.put({"type": "agent_done"})
                return
            target_events = _ask_int("이벤트 수를 입력해주세요.")
            if target_events is None:
                output_queue.put({"type": "agent_done"})
                return
            target_adc = _ask_float("목표 peakADC 값을 입력해주세요.")
            if target_adc is None:
                output_queue.put({"type": "agent_done"})
                return

            update_shared_state({"current_energy": beam_energy})

            result = sim.hv_equalization_start(
                target_c=target_adc,
                target_s=target_adc,
                tower=_HV_TOWER_ORDER[0],
            )
            io.send_tool_output(result)

            from sim.hv_equalization_agent import HVEqualizationSimAgent

            for i, tower in enumerate(_HV_TOWER_ORDER):
                if stop_event.is_set():
                    break
                io.send_status(f"[SIM] [{i + 1}/{len(_HV_TOWER_ORDER)}] {tower} 타워 시작")
                update_shared_state({"current_tower": tower, "agent_type": agent_name})
                tower_agent = HVEqualizationSimAgent(
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
                io.send_status(f"✅ [SIM] {tower} 완료")

            if not stop_event.is_set():
                io.send_ai_message("모든 타워 HV Equalization이 완료되었습니다. [SIM]")
                io.send_status("✅ [SIM] fixed_hv.txt 업데이트 건너뜀 (sim mode)")
            io.send_status("모델 언로드 중...")
            clear_shared_state()
            output_queue.put({"type": "agent_done"})
            return

        elif agent_name in ("position_scan", "position_scan_sim"):
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

            from sim.position_scan_agent import PositionScanSimAgent as _PSAgent

            # both → horizontal 먼저 완료·언로드 후 vertical 재로드 (초기 입력은 재질문 안 함).
            directions = ["horizontal", "vertical"] if direction == "both" else [direction]
            for _di, _dir in enumerate(directions):
                if stop_event.is_set():
                    break
                if len(directions) > 1:
                    io.send_ai_message(
                        f"[SIM] [{_di + 1}/{len(directions)}] {_dir} 방향 스캔을 시작합니다. 모델을 로드합니다...")
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
                        f"✅ [SIM] {_dir} 방향 완료. 모델을 언로드하고 다음 방향을 이어서 진행합니다.")
                    io.send_status("모델 언로드 중...")

            if not stop_event.is_set():
                io.send_ai_message("✅ 모든 Position Scan이 완료되었습니다. [SIM]")
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

    except Exception as e:
        clear_shared_state()
        tb = traceback.format_exc()
        output_queue.put({"type": "error", "content": f"Agent error: {str(e)}\n{tb}"})
        output_queue.put({"type": "agent_done"})


class AgentRunnerSim(AgentRunner):
    """Web simulation runner — same API as AgentRunner, mock tools only."""

    def start_brain(self, use_base_model: bool = False):
        if self.brain_ready:
            return

        from sim.brain_agent import BrainAgentSim

        self._brain_stop.clear()
        self._brain_agent = BrainAgentSim(
            shared_state=shared_state,
            shared_locks=shared_locks,
            use_base_model=use_base_model,
        )
        self._brain_agent.load()

        self._brain_thread = threading.Thread(
            target=run_brain_thread,
            args=(
                self._brain_agent,
                self.adhoc_queue,
                self.output_queue,
                self._brain_stop,
                self.confirm_queue,
                self.clarify_queue,
            ),
            daemon=True,
        )
        self._brain_thread.start()

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
            args=(
                agent_name,
                params,
                self.input_queue,
                self.output_queue,
                self.stop_event,
                self.waiting_flag,
            ),
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        """Sim mode — DAQ KILLME 없이 agent thread만 중지."""
        self.stop_event.set()
        self.input_queue.put("종료")
