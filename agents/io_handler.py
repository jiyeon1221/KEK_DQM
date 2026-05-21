#!/usr/bin/env python3
"""
IO Handler
----------
Abstraction layer so agent code stays independent of the execution environment.

IOHandler (ABC)
  ├─ TerminalIO   : Local terminal. Uses print/input directly.
  └─ WebSocketIO  : Web UI. Pushes typed dicts to output_queue;
                    reads input from input_queue (0.2 s poll, stop_event aware).
                    Emits awaiting_input when the last message requires a physical
                    action (e.g. "이동해주세요") → frontend shows the 완료 button.
"""

import queue
import threading
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path


class IOHandler(ABC):
    @abstractmethod
    def get_input(self) -> str:
        """Block until user provides input. Returns stripped string."""
        pass

    @abstractmethod
    def send_ai_message(self, message: str):
        """Send AI conversation message → left panel."""
        pass

    @abstractmethod
    def send_tool_output(self, text: str):
        """Send tool stdout → right panel."""
        pass

    @abstractmethod
    def send_plots(self, file_paths: List[str]):
        """Send plot image paths → right panel."""
        pass

    @abstractmethod
    def send_status(self, text: str):
        """Update status bar."""
        pass

    @abstractmethod
    def send_tool_error(self, tool_name: str, error_msg: str, attempts: int):
        """Tool 최종 실패 알림 → 왼쪽 패널 빨간 버블."""
        pass

    @abstractmethod
    def wait_for_retry(self):
        """사용자가 '다시 시도' 버튼을 클릭할 때까지 블로킹."""
        pass


class TerminalIO(IOHandler):
    """Default: same behavior as before (print / input)."""

    def get_input(self) -> str:
        return input("👤 You: ").strip()

    def send_ai_message(self, message: str):
        print(f"\n💬 Agent: {message}")

    def send_tool_output(self, text: str):
        print(text, flush=True)

    def send_plots(self, file_paths: List[str]):
        for f in file_paths:
            print(f"   📊 Plot saved → {Path(f).name}")

    def send_status(self, text: str):
        print(f"[Status] {text}")

    def send_tool_error(self, tool_name: str, error_msg: str, attempts: int):
        print(f"\n❌ [Tool Error] {tool_name} ({attempts}회 시도 모두 실패): {error_msg}")

    def wait_for_retry(self):
        input("다시 시도하려면 Enter를 누르세요...")


class WebSocketIO(IOHandler):
    """WebSocket-backed I/O for the web UI."""

    # Messages containing these keywords require a physical action from the
    # operator → show the 완료 button.
    # Everything else (value requests) just needs typed input → no button.
    _CONFIRM_KEYWORDS = [
        "이동해주세요",       # move stage / tower
        "설정해주세요",       # set beam energy
        "확인해주세요",       # check results
        "다음 DAQ를 시작합니다",  # ready for next DAQ
        "전압이 변경되었습니다",  # HV voltage changed
    ]

    # HV approval prompt → shows 완료 + 수정 buttons (modify allowed)
    _HV_CONFIRM_KEYWORDS = [
        "적용하시겠습니까",
    ]

    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue,
                 stop_event: Optional[threading.Event] = None,
                 waiting_flag: Optional[threading.Event] = None):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.waiting_flag = waiting_flag      # set while blocked on get_input()
        self._last_needs_confirm = False   # set by send_ai_message
        self._last_needs_hv_confirm = False   # HV approval → 완료+수정 buttons

    def send_ai_message(self, message: str):
        self._last_needs_hv_confirm = any(kw in message for kw in self._HV_CONFIRM_KEYWORDS)
        self._last_needs_confirm = (
            not self._last_needs_hv_confirm
            and any(kw in message for kw in self._CONFIRM_KEYWORDS)
        )
        self.output_queue.put({"type": "ai_message", "content": message})

    def get_input(self) -> str:
        from agents.agent_runner import StopAgentException
        # HV approval → show 완료 + 수정 buttons; other physical actions → 완료 only
        if self._last_needs_hv_confirm:
            self.output_queue.put({"type": "awaiting_hv_confirm"})
        elif self._last_needs_confirm:
            self.output_queue.put({"type": "awaiting_input"})
        self._last_needs_confirm = False   # reset for next call
        self._last_needs_hv_confirm = False
        # Signal that the scenario agent is blocked waiting for user input
        if self.waiting_flag is not None:
            self.waiting_flag.set()
        try:
            # Poll so we can react to stop_event even while blocked
            while True:
                if self.stop_event and self.stop_event.is_set():
                    raise StopAgentException()
                try:
                    value = self.input_queue.get(timeout=0.2)
                    return value.strip()
                except queue.Empty:
                    continue
        finally:
            if self.waiting_flag is not None:
                self.waiting_flag.clear()

    def send_tool_output(self, text: str):
        if text and text.strip():
            self.output_queue.put({"type": "tool_output", "content": text})

    def send_plots(self, file_paths: List[str]):
        for path in file_paths:
            # Send just the filename; server serves /plots/<filename>
            self.output_queue.put({"type": "plot", "filename": Path(path).name})

    def send_status(self, text: str):
        self.output_queue.put({"type": "status", "content": text})

    def send_tool_error(self, tool_name: str, error_msg: str, attempts: int):
        self.output_queue.put({
            "type": "tool_error",
            "tool_name": tool_name,
            "error": error_msg,
            "attempts": attempts,
        })

    def wait_for_retry(self) -> str:
        from agents.agent_runner import StopAgentException
        self.output_queue.put({"type": "awaiting_retry"})
        if self.waiting_flag is not None:
            self.waiting_flag.set()
        try:
            while True:
                if self.stop_event and self.stop_event.is_set():
                    raise StopAgentException()
                try:
                    val = self.input_queue.get(timeout=0.2)
                    return val
                except queue.Empty:
                    continue
        finally:
            if self.waiting_flag is not None:
                self.waiting_flag.clear()
