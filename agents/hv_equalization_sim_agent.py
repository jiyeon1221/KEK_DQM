#!/usr/bin/env python3
"""
HV Equalization Simulation Agent
hv_execute_tool(status/voltage)는 실제 하드웨어 사용.
hv_equalization_suggest의 ADC 측정만 시뮬레이션.
나머지는 부모(HVEqualizationAgent) 로직 사용.
"""

import json
import math
import random
from typing import Dict

from .hv_equalization_agent import HVEqualizationAgent


class HVEqualizationSimAgent(HVEqualizationAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_name = f"HV Equalization Sim-ADC [{self.tower}]"

        # 지수함수 PMT 파라미터: ADC = A * exp(B * HV)
        # 실제 HV는 하드웨어에서 읽어오므로 ref_hv는 ADC 파라미터 계산용 기준값으로만 사용
        t = self.tower
        B_c = random.uniform(0.0060, 0.0080)
        B_s = random.uniform(0.0060, 0.0080)
        ref_hv_c = random.uniform(750.0, 800.0)
        ref_hv_s = random.uniform(750.0, 800.0)
        ref_target = float(self.state.get("target_adc_c") or 1230)
        frac_c = random.uniform(0.35, 0.70)
        frac_s = random.uniform(0.35, 0.70)
        A_c = frac_c * ref_target / math.exp(B_c * ref_hv_c)
        A_s = frac_s * ref_target / math.exp(B_s * ref_hv_s)
        noise_c = random.uniform(0.015, 0.040)
        noise_s = random.uniform(0.015, 0.040)

        self._sim_params = {
            "C": {"A": A_c, "B": B_c, "noise": noise_c},
            "S": {"A": A_s, "B": B_s, "noise": noise_s},
        }
        self.log(f"Sim-ADC 초기화: {t} (HV는 실제 하드웨어, ADC만 시뮬레이션)")

    def _simulate_adc(self, channel: str, hv: float) -> float:
        p = self._sim_params[channel]
        base = p["A"] * math.exp(p["B"] * hv)
        # iteration이 늘수록 노이즈 감쇠: noise / (1 + 0.5 * iter)
        itr = self.state.get("iterations", 0)
        effective_noise = p["noise"] / (1.0 + 0.5 * itr)
        return max(0.0, base + random.normalvariate(0.0, base * effective_noise))

    def _execute_tool(self, tool_name: str, params: Dict) -> str:

        if tool_name == "hv_execute_tool":
            cmd = params.get("command", "").lower()

            if cmd == "voltage":
                # last_suggested로 override (방향 반전 등 LLM 오류 방지) 후 실제 하드웨어 적용
                if self.state.get("last_suggested_hv_c") is not None:
                    cv = {}
                    if not self.state.get("channel_done_c", False):
                        cv[f"{self.tower}C"] = self.state["last_suggested_hv_c"]
                    if not self.state.get("channel_done_s", False):
                        cv[f"{self.tower}S"] = self.state["last_suggested_hv_s"]
                    if cv:
                        params["channel_values"] = cv

            result = self.hv_control_tool.execute(params)
            self.io.send_tool_output(result)

            if cmd == "status":
                v_c, v_s = self._extract_voltages(result)
                if v_c is not None:
                    self.state["last_hv_c"] = v_c
                    self.state["last_hv_s"] = v_s
                    self.log(f"Real HV Status: C={v_c}V, S={v_s}V")

            elif cmd == "voltage":
                if self.state.get("last_suggested_hv_c") is not None:
                    self.state["last_hv_c"] = self.state["last_suggested_hv_c"]
                    self.state["last_hv_s"] = self.state["last_suggested_hv_s"]
                self.state["last_suggested_hv_c"] = None
                self.state["last_suggested_hv_s"] = None
                self.log(f"Real HV voltage applied: C={self.state['last_hv_c']}V, S={self.state['last_hv_s']}V")

                self.io.send_tool_output(f"🔍 HV 적용 확인 중 ({self.tower})...")
                verify = self.hv_control_tool.execute({
                    "command": "status",
                    "channels": [f"{self.tower}C", f"{self.tower}S"],
                })
                self.io.send_tool_output(verify)
                v_c, v_s = self._extract_voltages(verify)
                if v_c is not None:
                    self.state["last_hv_c"] = v_c
                    self.state["last_hv_s"] = v_s
                    self.log(f"HV Verified: C={v_c}V, S={v_s}V")

            return result

        elif tool_name == "hv_equalization_suggest":
            hv_c = float(self.state.get("last_hv_c") or 775.0)
            hv_s = float(self.state.get("last_hv_s") or 775.0)
            adc_c = self._simulate_adc("C", hv_c)
            adc_s = self._simulate_adc("S", hv_s)
            run_number = params.get("run_number") or self.state.get("last_run_number", 0)

            from tools.hv_equalization_tool import _session_manager
            result_dict = _session_manager.process_suggestion(
                "default", int(run_number),
                float(adc_c), float(adc_s),
                float(hv_c), float(hv_s),
            )

            if result_dict.get("status") == "success":
                cur = result_dict.get("current", {})
                sug = result_dict.get("suggested", {})
                self.state["last_adc_c"] = cur.get("C", {}).get("adc", adc_c)
                self.state["last_adc_s"] = cur.get("S", {}).get("adc", adc_s)
                raw_hv_c = sug.get("C", {}).get("hv")
                raw_hv_s = sug.get("S", {}).get("hv")
                self.state["last_suggested_hv_c"] = int(round(raw_hv_c)) if raw_hv_c is not None else None
                self.state["last_suggested_hv_s"] = int(round(raw_hv_s)) if raw_hv_s is not None else None
                self.state["channel_done_c"] = bool(sug.get("C", {}).get("done", False))
                self.state["channel_done_s"] = bool(sug.get("S", {}).get("done", False))
                self.state["last_hv_c"] = round(hv_c, 1)
                self.state["last_hv_s"] = round(hv_s, 1)
            else:
                # fallback: 직접 계산
                target_c = float(self.state.get("target_adc_c") or 1230)
                target_s = float(self.state.get("target_adc_s") or 1230)
                tol = 0.02
                c_done = abs(adc_c - target_c) / target_c < tol
                s_done = abs(adc_s - target_s) / target_s < tol
                p_c = self._sim_params["C"]
                p_s = self._sim_params["S"]
                next_hv_c = int(round(math.log(target_c / p_c["A"]) / p_c["B"])) if not c_done else int(round(hv_c))
                next_hv_s = int(round(math.log(target_s / p_s["A"]) / p_s["B"])) if not s_done else int(round(hv_s))
                self.state["last_adc_c"] = adc_c
                self.state["last_adc_s"] = adc_s
                self.state["last_suggested_hv_c"] = next_hv_c
                self.state["last_suggested_hv_s"] = next_hv_s
                self.state["channel_done_c"] = c_done
                self.state["channel_done_s"] = s_done
                result_dict = {
                    "status": "success",
                    "current": {"C": {"hv": hv_c, "adc": round(adc_c, 1)},
                                "S": {"hv": hv_s, "adc": round(adc_s, 1)}},
                    "suggested": {"C": {"hv": next_hv_c, "done": c_done},
                                  "S": {"hv": next_hv_s, "done": s_done}},
                }

            summary = (
                f"🔬 [SIM-ADC] HV Suggest — {self.tower} | "
                f"ADC(sim): C={adc_c:.1f}, S={adc_s:.1f} | "
                f"Real HV: C={hv_c:.0f}V→{self.state['last_suggested_hv_c']}V, "
                f"S={hv_s:.0f}V→{self.state['last_suggested_hv_s']}V | "
                f"Done: C={self.state['channel_done_c']}, S={self.state['channel_done_s']}"
            )
            self.io.send_tool_output(summary)
            self.log(summary)

            try:
                from tools.hv_equalization_tool import generate_fitting_summary
                run_number = self.state.get("last_run_number", 0) or 0
                fit_result = generate_fitting_summary(
                    session_id="default", tower=self.tower, run_number=run_number
                )
                self.io.send_tool_output(
                    f"── HV Fitting History ({self.tower}) ──\n{fit_result['table']}\nEq: {fit_result['equation']}"
                )
                if fit_result.get("plot_path"):
                    self.io.send_plots([fit_result["plot_path"]])
            except Exception as e:
                self.log(f"fitting summary 실패: {e}")

            self.state["needs_suggest"] = False  # suggest 완료
            return json.dumps(result_dict, ensure_ascii=False, indent=2)

        return super()._execute_tool(tool_name, params)
