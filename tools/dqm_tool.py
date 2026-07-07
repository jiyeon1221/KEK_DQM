#!/usr/bin/env python3
"""DQM Tool — C++ monit 실행으로 ROOT 파일 생성"""

import os
import re
import subprocess
import shutil
from typing import Dict, Any
from pathlib import Path

from .base_tool import BaseTool
from .config_loader import get_path_config


# ===== DQM 설정 (Config from YAML) =====
DQM_DIR = get_path_config("DqmDir")
MONIT_EXECUTABLE = os.path.join(DQM_DIR, "monit")  # compile.sh로 생성됨
DQM_OUTPUT_DIR = os.path.join(DQM_DIR, "output")

from .config_loader import CONFIG_FILE as CONFIG_YML_PATH


os.makedirs(DQM_OUTPUT_DIR, exist_ok=True)


class DQMPlotTool(BaseTool):
    """DQM Plot 생성 Tool"""
    
    def __init__(self):
        super().__init__(
            name="dqm_plot_tool",
            description="Generate DQM plots for test-beam data analysis"
        )
        
        os.makedirs(DQM_OUTPUT_DIR, exist_ok=True)
    
    @staticmethod
    def _normalize_module(name: str) -> str:
        """M1T1S → M1-T1-S, M3T2C → M3-T2-C. Leaves M1, MCPPMT, M1-T1-S unchanged."""
        return re.sub(r'^(M\d)(T\d)([SC])$', r'\1-\2-\3', name)

    def execute(self, params: Dict[str, Any]) -> str:
        """
        DQM Plot 생성

        Args:
            params:
                - run_number (int): Run 번호
                - method (str, optional): 'IntADC' | 'PeakADC' (기본: IntADC)
                - type (str, optional): 'full'|'heatmap'|'module'|'single' (기본: full)
                - modules (list, optional): single → 채널 목록, heatmap/module → ["MCPPMT"]
                - max_event (int, optional): 처리할 최대 이벤트 수

        Returns:
            실행 결과 문자열
        """
        valid, error = self.validate_params(params, ["run_number"])
        if not valid:
            raise RuntimeError(f"파라미터 오류: {error}")

        run_number = params["run_number"]
        method = params.get("method", "IntADC")
        type_ = params.get("type", "full")
        modules = [self._normalize_module(m) for m in params.get("modules", [])]
        max_event = params.get("max_event", None)

        try:
            run_number = int(run_number)
            if run_number <= 0:
                raise RuntimeError("Run 번호는 양수여야 합니다")
        except (TypeError, ValueError):
            raise RuntimeError(f"잘못된 Run 번호: {run_number}")

        if method not in ('PeakADC', 'IntADC'):
            raise RuntimeError(f"잘못된 method: {method} (허용: IntADC, PeakADC)")
        if type_ not in ('full', 'heatmap', 'module', 'single'):
            raise RuntimeError(f"잘못된 type: {type_} (허용: full, heatmap, module, single)")

        if not os.path.exists(MONIT_EXECUTABLE):
            raise RuntimeError(f"monit 실행파일을 찾을 수 없습니다: {MONIT_EXECUTABLE}")
        if not os.path.exists(CONFIG_YML_PATH):
            raise RuntimeError(f"설정 파일을 찾을 수 없습니다: {CONFIG_YML_PATH}")

        # Resolve module string for heatmap/module (always MCPPMT)
        if type_ in ('heatmap', 'module'):
            module_str = modules[0] if modules else "MCPPMT"
        else:
            module_str = ""

        try:
            output_lines = []
            output_lines.append("📊 DQM Plot 생성 시작")
            output_lines.append(f"🔢 Run Number: {run_number}")
            output_lines.append(f"📈 Method: {method}")
            output_lines.append(f"🗂️  Type: {type_}" + (f" · Module: {module_str}" if module_str else "") + (f" · Channels: {' '.join(modules)}" if type_ == "single" and modules else ""))
            if max_event:
                output_lines.append(f"🔢 Max Event: {max_event}")
            else:
                output_lines.append(f"🔢 Max Event: 전체 (all events)")
            output_lines.append("")

            output_lines.append("⚙️  C++ monit 실행 중...")

            cmd = [
                MONIT_EXECUTABLE,
                "--RunNumber", str(run_number),
                "--Config", str(CONFIG_YML_PATH),
                "--type", type_,
                "--method", method,
            ]
            if type_ in ('heatmap', 'module'):
                cmd.extend(["--module", module_str])
            elif type_ == "single" and modules:
                cmd.extend(["--module"] + modules)

            if max_event is not None:
                try:
                    max_event = int(max_event)
                    if max_event > 0:
                        cmd.extend(["--MaxEvent", str(max_event)])
                except (TypeError, ValueError):
                    output_lines.append(f"⚠️  잘못된 max_event 값: {max_event}, 무시하고 계속 진행")

            result = subprocess.run(
                cmd,
                cwd=DQM_DIR,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                output_lines.append(f"⚙️  monit 종료 (Exit Code: {result.returncode})")
                if result.stderr and result.stderr.strip():
                    output_lines.append(f"⚠️  stderr: {result.stderr.strip()[:400]}")
                elif result.stdout and result.stdout.strip():
                    output_lines.append(f"📄 stdout: {result.stdout.strip()[-300:]}")
            else:
                output_lines.append("✅ monit 정상 종료")

            # Determine expected ROOT filename
            if type_ == "full":
                root_filename = f"Run{run_number}_full_{method}.root"
            elif type_ in ('heatmap', 'module'):
                root_filename = f"Run{run_number}_{type_}_{method}_{module_str}.root"
            else:
                # single: fModule="" in C++, so trailing underscore in prefix
                root_filename = f"Run{run_number}_single_{method}_.root"

            dqm_root_file = os.path.join(DQM_OUTPUT_DIR, root_filename)
            if not os.path.exists(dqm_root_file):
                # Also accept any matching ROOT file (single type may have different suffix)
                root_pattern = f"Run{run_number}_{type_}_{method}*.root"
                matches = list(Path(DQM_OUTPUT_DIR).glob(root_pattern))
                if matches:
                    root_filename = matches[0].name
                else:
                    raise RuntimeError("ROOT 파일이 생성되지 않았습니다")

            output_lines.append(f"📁 ROOT 파일: {root_filename}")
            output_lines.append("")
            output_lines.append("✅ DQM Plot 생성 완료!")

            return "\n".join(output_lines)

        except RuntimeError:
            raise
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Plot 생성 타임아웃 (10분 초과) — Run {run_number}의 데이터가 너무 큽니다."
            )
        except Exception as e:
            raise RuntimeError(f"Plot 생성 중 오류: {str(e)}") from e
