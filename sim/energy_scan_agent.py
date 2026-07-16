#!/usr/bin/env python3
"""Energy Scan Agent — simulation mode (no hardware)."""

from typing import Dict

from agents.energy_scan_agent import EnergyScanAgent
from sim.sim_base import SimExecMixin
from sim.tool_simulator import get_simulator


class EnergyScanSimAgent(SimExecMixin, EnergyScanAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sim = get_simulator()
        self.agent_name = f"{self.agent_name} [SIM]"

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        if tool_name == "none":
            return "no_tool_executed"

        if tool_name == "daq_run_tool":
            energy_key = self._resolve_daq_energy_key()
            events = None
            if energy_key is not None and energy_key in self.state["energy_config"]:
                events = self.state["energy_config"][energy_key]["target_events"]
            result, run_number = self._sim_run_daq(
                params,
                events=events,
                beam_energy=energy_key,
                program="EM Scan",
                pos=self._position_for_current_step(),
                daq_config=self._daq_config_for(energy_key),
            )
            if run_number:
                self.state["last_run_number"] = run_number
                if energy_key is not None and energy_key in self.state["energy_config"]:
                    self.state["current_energy"] = energy_key
                    self.state["energy_config"][energy_key]["runs"].append(run_number)
                    self.state["energy_config"][energy_key]["collected_events"] = params.get("events", 0)
                    self.log(f"[SIM] DAQ Run {run_number} 완료: {energy_key} GeV, {params.get('events', 0)} events")
            return result

        self._sim_emit_tool_call(tool_name, params)
        return f"Error: Unknown tool {tool_name}"
