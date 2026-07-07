#!/usr/bin/env python3
"""BrainAgent — simulation mode (no hardware / remote services)."""

from agents.brain_agent import BrainAgent
from sim.tool_simulator import get_simulator


class BrainAgentSim(BrainAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sim = get_simulator()
        self.agent_name = "BrainAgent [SIM]"

    def _execute_tool(self, tool_name: str, params: dict, io, max_retries: int = 3):
        sim = self._sim
        io.send_tool_output(sim.format_tool_call(tool_name, params))

        try:
            if tool_name == "daq_run":
                io.send_status("DAQ 실행 중... [SIM]")
                result = sim.daq_run(params, line_callback=io.send_tool_output)
                import re
                for line in result.splitlines():
                    m = re.search(r"Run:\s*(\d+)", line)
                    if m:
                        self._last_daq_run = int(m.group(1))
                        break

            elif tool_name == "dqm_plot":
                io.send_status("DQM 플롯 생성 중... [SIM]")
                result = sim.dqm_plot(params)
                io.send_tool_output(result)
                io.send_ai_message(
                    f"[SIM] Run {params.get('run_number')} DQM — 실제 monit 미실행"
                )

            elif tool_name in ("run_log", "run_log_read"):
                result = sim.run_log(params)
                io.send_tool_output(result)

            elif tool_name in ("hv_read", "hv_status"):
                params["command"] = "status"
                result = sim.hv_status(params.get("channels"))
                io.send_tool_output(result)

            elif tool_name == "hv_write":
                io.send_status("HV 변경 중... [SIM]")
                cmd = params.get("command", "voltage")
                if cmd == "voltage":
                    ch = params.get("channels")
                    voltage = float(params.get("voltage", 0))
                    if isinstance(ch, str):
                        result = sim.hv_on_off("voltage", ch)
                    else:
                        cv = {c: voltage for c in (ch or [])}
                        result = sim.hv_voltage(cv)
                else:
                    result = sim.hv_on_off(cmd, params.get("channels", []))
                io.send_tool_output(result)

            else:
                io.send_ai_message(f"알 수 없는 도구: {tool_name}")

        except Exception as e:
            io.send_ai_message(f"[SIM] 도구 실행 중 오류: {e}")
