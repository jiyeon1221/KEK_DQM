#!/usr/bin/env python3
"""Calibration Scan Agent — simulation mode (no hardware)."""

from typing import Dict

from agents.calib_scan_agent import CalibScanAgent
from sim.sim_base import SimExecMixin
from sim.tool_simulator import get_simulator


class CalibScanSimAgent(SimExecMixin, CalibScanAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sim = get_simulator()
        self.agent_name = f"{self.agent_name} [SIM]"

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        if tool_name == "none":
            return "no_tool_executed"

        if tool_name == "daq_run_tool":
            tower = self._current_tower_name()
            result, run_number = self._sim_run_daq(
                params,
                events=self.state.get("target_events"),
                beam_energy=self.state.get("beam_energy"),
                program="Calibration",
                pos=self._position_for_current_step(),
            )
            if run_number:
                self.state["last_run_number"] = run_number
                self.state["tower_status"][tower]["runs"].append(run_number)
                self.state["tower_status"][tower]["collected_events"] = params.get("events", 0)
                self.log(f"[SIM] DAQ Run {run_number} 완료: {tower}, {params.get('events', 0)} events")
            return result

        self._sim_emit_tool_call(tool_name, params)
        return f"Error: Unknown tool {tool_name}"
