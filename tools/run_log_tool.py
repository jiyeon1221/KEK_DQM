#!/usr/bin/env python3
"""Run Log Tool — Google Spreadsheet에 실험 로그 기록"""

import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .base_tool import BaseTool
from .config_loader import get_path_config
from .hv_control_tool import HVControlTool
from .hodoscope_hv_tool import read_hv_from_log

# ===== 경로 및 설정 (Config from YAML) =====
RUNNUM_FILE = get_path_config("RunNumberFile")
JSON_KEY_FILE = get_path_config("JsonKeyFile")
SPREADSHEET_ID = get_path_config("SpreadsheetId")

class RunLogTool(BaseTool):
    """Google Spreadsheet에 실험 로그를 기록하는 Tool"""
    
    def __init__(self):
        super().__init__(
            name="run_log_tool",
            description="Record experimental logs to Google Spreadsheet"
        )
        self.client = None
        self.sheet = None

    def _authenticate(self):
        """Google Sheets API 인증 및 시트 연결"""
        if self.sheet:
            return True
            
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            creds = Credentials.from_service_account_file(str(JSON_KEY_FILE), scopes=scopes)
            self.client = gspread.authorize(creds)
            # URL 대신 ID로 오픈
            spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
            self.sheet = spreadsheet.get_worksheet(0)  # 첫 번째 시트 선택
            return True
        except Exception as e:
            print(f"❌ Google Spreadsheet 인증 실패: {str(e)}")
            return False

    def execute(self, params: Dict[str, Any]) -> str:
        """
        Args:
            params:
                - command (str): "write" | "update" | "read"
                - run_num, program, notes, evts, start_time, end_time, config
        """
        if not self._authenticate():
            raise RuntimeError("Google Spreadsheet 인증에 실패했습니다.")

        command = params.get("command", "write")

        if command == "write":
            return self._write_row(params)
        elif command == "update":
            return self._update_row(params)
        elif command == "read":
            return self._read_row(params)
        else:
            raise RuntimeError(f"지원하지 않는 명령입니다: {command}")

    # Path to the fixed HV reference file — READ ONLY, never write to this file
    _FIXED_HV_PATH = Path(__file__).parent.parent / "fixed_hv.txt"

    @staticmethod
    def _parse_fixed_hv(path: Path) -> Dict[str, str]:
        """fixed_hv.txt 파싱 → {CHANNEL: vset} dict"""
        result = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    ch, val = line.split(":", 1)
                    result[ch.strip().upper()] = val.strip()
        return result

    def _build_hv_drc_string(self, name_to_vset: Dict[str, str]) -> str:
        """fixed_hv.txt 유무에 따라 DRC HV 문자열 생성"""
        if self._FIXED_HV_PATH.exists():
            fixed = self._parse_fixed_hv(self._FIXED_HV_PATH)
            diff_lines = []
            for i in range(1, 10):
                parts = []
                for suffix in ("S", "C"):
                    name = f"T{i}{suffix}"
                    cur = name_to_vset.get(name, "")
                    if cur and cur != fixed.get(name, ""):
                        parts.append(f"{name}: {cur}")
                if parts:
                    diff_lines.append(", ".join(parts))
            if not diff_lines:
                return "Same as Fixed HV"
            return "Compared to Fixed HV\n" + "\n".join(diff_lines)
        else:
            lines = []
            for i in range(1, 10):
                parts = []
                for suffix in ("S", "C"):
                    name = f"T{i}{suffix}"
                    if name in name_to_vset and name_to_vset[name] != "":
                        parts.append(f"{name}:{name_to_vset[name]}")
                if parts:
                    lines.append(" ".join(parts))
            return "\n".join(lines)

    def _collect_hv_vset_snapshot(self) -> Dict[str, str]:
        """
        HV config에서 채널별 V0Set snapshot 수집.
        - Aux: Trig1, Trig2
        - DRC: fixed_hv.txt 존재 시 비교, 없으면 전체 출력
        """
        hv = HVControlTool()
        try:
            if not hv._ensure_connection():
                return {"hv_drc": "", "hv_aux": ""}

            rows = hv._read_config_rows()
            name_to_vset = {}
            for row in rows:
                name = str(row.get("name", "")).strip()
                if not name or name.lower() == "none":
                    continue
                name_to_vset[name.upper()] = str(row.get("V0Set", "")).strip()

            aux_names = ["TRIG1", "TRIG2"]
            hv_aux = ", ".join(
                f"{name}:{name_to_vset[name]}"
                for name in aux_names
                if name in name_to_vset and name_to_vset[name] != ""
            )
            hv_drc = self._build_hv_drc_string(name_to_vset)
            return {"hv_drc": hv_drc, "hv_aux": hv_aux}
        except Exception:
            return {"hv_drc": "", "hv_aux": ""}
        finally:
            try:
                if hv.ssh_client:
                    hv.ssh_client.close()
            except Exception:
                pass

    def _write_row(self, params: Dict[str, Any]) -> str:
        """새로운 로그 행 추가 (DAQ 호출용)"""
        run_num = params.get("run_num")
        if not run_num:
            try:
                with open(RUNNUM_FILE, "r") as f:
                    # Run이 종료된 후 runnum.txt가 다음 번호로 업데이트되므로, 
                    # 방금 종료된 Run 정보를 위해 -1을 수행함
                    val = f.read().strip()
                    run_num = str(int(val) - 1)
            except Exception:
                run_num = "Unknown"

        program = params.get("program", "")
        evts = params.get("evts", "")
        start_time = params.get("start_time", "")
        end_time = params.get("end_time", "")
        config = params.get("config", "")
        notes = params.get("notes", "")
        
        def safe_round(val, digits=1):
            if val == "" or val is None:
                return ""
            try:
                return round(float(val), digits)
            except (ValueError, TypeError):
                return val

        pos_h = safe_round(params.get("pos_h", ""))
        pos_v = safe_round(params.get("pos_v", ""))
        pos_rot = safe_round(params.get("pos_rot", ""))
        pos_tilt = safe_round(params.get("pos_tilt", ""))
        beam_energy = params.get("beam_energy", "")

        hv_drc = params.get("hv_drc", "")
        hv_aux = params.get("hv_aux", "")
        if hv_drc == "" and hv_aux == "":
            hv_snapshot = self._collect_hv_vset_snapshot()
            hv_drc = hv_snapshot.get("hv_drc", "")
            hv_aux = hv_snapshot.get("hv_aux", "")

        # Append hodoscope HV from the run log (actual per-channel values)
        hodo_hvs = read_hv_from_log(str(run_num))
        if hodo_hvs:
            hodo_str = ", ".join(
                f"Hodo[{ch}]:{hodo_hvs[ch]}" for ch in sorted(hodo_hvs)
            )
            hv_aux = f"{hv_aux}, {hodo_str}" if hv_aux else hodo_str

        # 실제 시트 컬럼 순서 (스크린샷 기준)
        # B(2): Program | C(3): Run # | D(4): evts | E(5): Time(start) | F(6): Time(end)
        # G(7): HV DRC | H(8): HV Aux | I(9): Pos H | J(10): Pos V | K(11): Pos Rot | L(12): Pos Tilt
        # M(13): Trigger Setup | N(14): Beam Type | O(15): Beam Energy | P(16): Beam Rate
        # Q(17): Config | R(18): Notes
        row = [
            program,        # B: Program
            run_num,        # C: Run #
            evts,           # D: evts
            start_time,     # E: Time (start)
            end_time,       # F: Time (end)
            hv_drc,         # G: HV DRC
            hv_aux,         # H: HV Aux
            pos_h,          # I: Position H
            pos_v,          # J: Position V
            pos_rot,        # K: Position Rot
            pos_tilt,       # L: Position Tilt
            "",             # M: Trigger Setup
            "e-",           # N: Beam Type
            beam_energy,    # O: Beam Energy
            "",             # P: Beam Rate
            config,         # Q: Config
            notes           # R: Notes
        ]

        try:
            # 마지막 데이터 행 찾기 (Run #가 있는 C열 기준)
            col_c_values = self.sheet.col_values(3)
            next_row = len(col_c_values) + 1
            
            # 헤더가 5행까지 있으므로, 데이터는 최소 6행부터 시작해야 함
            if next_row < 6:
                next_row = 6
            
            range_label = f"B{next_row}:R{next_row}"
            self.sheet.update(range_label, [row], value_input_option='USER_ENTERED')
            
            return f"✅ 새 Run 로그 추가 완료 (Run: {run_num}, Row: {next_row})"
        except Exception as e:
            raise RuntimeError(f"로그 추가 실패: {str(e)}") from e

    # Header row index (1-based) and column mapping for read
    _HEADER_ROW = 5
    _READ_COLUMNS = {
        2: "Program", 3: "Run #", 4: "Events", 5: "Start", 6: "End",
        7: "HV DRC", 8: "HV Aux", 9: "Pos H", 10: "Pos V",
        11: "Pos Rot", 12: "Pos Tilt", 13: "Trigger", 14: "Beam Type",
        15: "Beam Energy", 16: "Beam Rate", 17: "Config", 18: "Notes",
    }

    def _read_row(self, params: Dict[str, Any]) -> str:
        """Run 번호로 해당 행의 모든 정보를 읽어 반환"""
        run_num = params.get("run_num")
        if not run_num:
            return "⚠️ 조회할 Run 번호를 알려주세요."
        try:
            all_data = self.sheet.get_all_values()
            target_row = None
            for row in all_data:
                if len(row) > 2 and str(row[2]) == str(run_num):
                    target_row = row
                    break
            if target_row is None:
                raise RuntimeError(f"Run {run_num}을 시트에서 찾을 수 없습니다.")

            lines = [f"📋 Run {run_num} 로그:"]
            for col_idx, label in sorted(self._READ_COLUMNS.items()):
                val = target_row[col_idx - 1] if col_idx - 1 < len(target_row) else ""
                if val:
                    lines.append(f"  {label}: {val}")
            return "\n".join(lines)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"로그 조회 실패: {str(e)}") from e

    # Column name → (sheet column index, display label)
    UPDATABLE_COLUMNS = {
        "program":       (2,  "Program"),
        "notes":         (18, "Notes"),
        "config":        (17, "Config"),
        "beam_energy":   (15, "Beam Energy"),
        "beam_type":     (14, "Beam Type"),
        "trigger_setup": (13, "Trigger Setup"),
        "hv_drc":        (7,  "HV DRC"),
        "hv_aux":        (8,  "HV Aux"),
    }

    def _update_row(self, params: Dict[str, Any]) -> str:
        """기존 로그 행 업데이트 (UPDATABLE_COLUMNS 에 있는 모든 열 지원)"""
        run_num = params.get("run_num")

        to_update = {
            col: params[col]
            for col in self.UPDATABLE_COLUMNS
            if params.get(col) is not None
        }

        if not to_update:
            return "⚠️ 업데이트할 정보가 없습니다."

        try:
            all_data = self.sheet.get_all_values()

            if run_num:
                target_row_idx = -1
                for i, row in enumerate(all_data):
                    if len(row) > 2 and str(row[2]) == str(run_num):
                        target_row_idx = i + 1
                        break
            else:
                target_row_idx = len(all_data)
                run_num = all_data[-1][2] if len(all_data[-1]) > 2 else "Last"

            if target_row_idx <= 1:
                raise RuntimeError(f"Run {run_num}을 시트에서 찾을 수 없습니다.")

            updates = []
            for col_key, value in to_update.items():
                col_idx, label = self.UPDATABLE_COLUMNS[col_key]
                self.sheet.update_cell(target_row_idx, col_idx, value)
                updates.append(f"{label}='{value}'")

            return f"✅ Run {run_num} 업데이트 완료: {', '.join(updates)}"

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"로그 업데이트 실패: {str(e)}") from e
