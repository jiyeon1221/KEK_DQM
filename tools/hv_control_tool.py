#!/usr/bin/env python3
"""
HV Control Tool — CAEN HV Supply 제어 (SSH 원격)

Available Commands:
- 'voltage': 전압 설정 (requires: channels, voltage)
- 'current': 전류 설정 (requires: channels, current)
- 'on': HV 전원 켜기 (requires: channels)
- 'off': HV 전원 끄기 (requires: channels)
- 'status': 상태 확인 (optional: channels, default='all')

Channel Specification:
- 'all' 또는 '전체': 모든 채널 (전체 슬롯)
- 'even': 짝수 채널번호 (모든 슬롯)
- 'odd': 홀수 채널번호 (모든 슬롯)
- 'slot:12': 슬롯 12의 모든 채널
- 'slot:12:even' / 'slot:12:odd': 슬롯 + 짝홀 조합
- ['M1T1C', 'M1T2C']: Name으로 지정 → config에서 (slot, ch) 자동 매핑
- [{'slot': 11, 'ch': 3}]: 명시적 (slot, ch) 지정
- '11:3': slot 11, ch 3
"""

import re
import shlex
from typing import Any, Dict, List, Tuple
from datetime import datetime

import paramiko

from .base_tool import BaseTool
from .config_loader import get_hv_config


# ===== HV 설정 (Config from YAML) =====
_hv_config = get_hv_config()
_hv_ssh = _hv_config.get('SSH', {})
_hv_paths = _hv_config.get('Paths', {})

HV_SSH_CONFIG = {
    'host': _hv_ssh.get('Host'),
    'port': _hv_ssh.get('Port'),
    'username': _hv_ssh.get('Username'),
    'password': _hv_ssh.get('Password'),
    'key_path': _hv_ssh.get('KeyPath')
}

HV_WRAPPER_WORKDIR = _hv_paths.get('WrapperWorkDir')
HV_CONFIG_FILENAME = "config.txt"
HV_CONFIG_RELATIVE_PATH = f"../config/{HV_CONFIG_FILENAME}"
HV_CONFIG_FULL_PATH = _hv_paths.get('ConfigFullPath')
HV_ENV_PRE_COMMAND = "export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH"

# (slot, ch) 타입 별칭
SlotCh = Tuple[int, int]


class HVControlTool(BaseTool):
    """HV 제어 Tool — multi-slot 지원"""

    def __init__(self):
        super().__init__(
            name="hv_execute_tool",
            description=(
                "Control CAEN high voltage (HV) system with multi-slot support. "
                "Available commands: "
                "1) 'voltage' - Set voltage (requires: channels, voltage), "
                "2) 'current' - Set current (requires: channels, current), "
                "3) 'on' - Turn on HV channels (requires: channels), "
                "4) 'off' - Turn off HV channels (requires: channels), "
                "5) 'status' - Check HV status (optional: channels). "
                "Channels: 'all', 'even', 'odd', 'slot:12', 'slot:12:even', "
                "names like ['M1T1C','M1T2C'], explicit [{'slot':11,'ch':3}], or '11:3' pairs."
            )
        )
        self.ssh_client = None

    def execute(self, params: Dict[str, Any]) -> str:
        valid, error = self.validate_params(params, ["command"])
        if not valid:
            raise RuntimeError(f"파라미터 오류: {error}")

        command = params["command"].lower()

        try:
            if not self._ensure_connection():
                raise RuntimeError(
                    f"SSH Connection Failed — "
                    f"{HV_SSH_CONFIG['host']}:{HV_SSH_CONFIG['port']} "
                    f"에 연결할 수 없습니다. HV 서버가 실행 중인지 확인하세요."
                )

            if command == "voltage":
                return self._set_voltage(params)
            elif command == "current":
                return self._set_current(params)
            elif command == "svmax":
                return self._set_svmax(params)
            elif command == "rup":
                return self._set_ramp(params, "RampUp")
            elif command == "rdown":
                return self._set_ramp(params, "RampDown")
            elif command == "name":
                return self._set_name(params)
            elif command == "on":
                return self._power_toggle(params, "On")
            elif command == "off":
                return self._power_toggle(params, "Off")
            elif command == "status":
                return self._get_status(params)
            else:
                raise RuntimeError(
                    f"Unsupported command: {command}. "
                    f"Supported: voltage, current, svmax, rup, rdown, name, on, off, status"
                )

        except RuntimeError:
            raise
        except Exception as e:
            import traceback
            raise RuntimeError(
                f"HV Control Error: {str(e)}\n{traceback.format_exc()}"
            ) from e

        finally:
            pass

    # ===== 명령 구현 =====

    def _set_voltage(self, params: Dict[str, Any]) -> str:
        """전압 설정"""
        if "channel_values" in params:
            channel_values = params["channel_values"]
            if not channel_values:
                return "❌ channel_values가 비어있습니다"

            rows = self._read_config_rows()
            row_map, name_map, all_pairs = self._build_lookup(rows)

            resolved: List[SlotCh] = []
            for identifier, voltage in channel_values.items():
                pairs = self._resolve_identifier(str(identifier), name_map, all_pairs)
                if not pairs:
                    return f"❌ '{identifier}'에 해당하는 채널을 찾을 수 없습니다"
                for pair in pairs:
                    row = row_map.get(pair)
                    if not row:
                        return f"❌ Slot{pair[0]} Ch{pair[1]}이(가) config.txt에 없습니다"
                    row['V0Set'] = self._format_numeric(voltage)
                    resolved.append(pair)

            self._write_config_rows(rows)

            changes = []
            for identifier, voltage in channel_values.items():
                for pair in self._resolve_identifier(str(identifier), name_map, all_pairs):
                    changes.append(f"Slot{pair[0]}Ch{pair[1]}→{self._format_numeric(voltage)}V")

            out = ["🔧 HV Voltage Command Executed", f"📋 Request: {', '.join(sorted(changes))}", ""]
            out += self._run_per_slot(sorted(set(resolved)), "--config {cfg} --Pw On".format(cfg=HV_CONFIG_RELATIVE_PATH))
            return "\n".join(out)

        else:
            if "channel" in params and "channels" not in params:
                params["channels"] = [params["channel"]]
            if "value" in params and "voltage" not in params:
                params["voltage"] = params["value"]
            if "channels" not in params or "voltage" not in params:
                return "❌ channels와 voltage 파라미터가 필요합니다"

            try:
                pairs = self._parse_channels(params["channels"])
            except ValueError as e:
                return str(e)
            if not pairs:
                return "❌ 유효한 채널을 찾을 수 없습니다"

            voltage = float(params["voltage"])
            rows = self._read_config_rows()
            row_map, _, _ = self._build_lookup(rows)

            for pair in pairs:
                row = row_map.get(pair)
                if not row:
                    return f"❌ Slot{pair[0]} Ch{pair[1]}이(가) config.txt에 없습니다"
                row['V0Set'] = self._format_numeric(voltage)

            self._write_config_rows(rows)

            out = ["🔧 HV Voltage Command Executed"]
            out.append(self._fmt_request(pairs) + f" → {self._format_numeric(voltage)}V")
            out.append("")
            out += self._run_per_slot(pairs, f"--config {HV_CONFIG_RELATIVE_PATH} --Pw On")
            return "\n".join(out)

    def _set_current(self, params: Dict[str, Any]) -> str:
        """전류 설정"""
        if "channel_values" in params:
            channel_values = params["channel_values"]
            if not channel_values:
                return "❌ channel_values가 비어있습니다"

            rows = self._read_config_rows()
            row_map, name_map, all_pairs = self._build_lookup(rows)

            resolved: List[SlotCh] = []
            for identifier, current in channel_values.items():
                pairs = self._resolve_identifier(str(identifier), name_map, all_pairs)
                if not pairs:
                    return f"❌ '{identifier}'에 해당하는 채널을 찾을 수 없습니다"
                for pair in pairs:
                    row = row_map.get(pair)
                    if not row:
                        return f"❌ Slot{pair[0]} Ch{pair[1]}이(가) config.txt에 없습니다"
                    row['I0Set'] = self._format_numeric(current)
                    resolved.append(pair)

            self._write_config_rows(rows)

            changes = []
            for identifier, current in channel_values.items():
                for pair in self._resolve_identifier(str(identifier), name_map, all_pairs):
                    changes.append(f"Slot{pair[0]}Ch{pair[1]}→{self._format_numeric(current)}μA")

            out = ["🔧 HV Current Command Executed", f"📋 Request: {', '.join(sorted(changes))}", ""]
            out += self._run_per_slot(sorted(set(resolved)), f"--config {HV_CONFIG_RELATIVE_PATH} --Pw On")
            return "\n".join(out)

        else:
            if "channel" in params and "channels" not in params:
                params["channels"] = [params["channel"]]
            if "value" in params and "current" not in params:
                params["current"] = params["value"]
            if "channels" not in params or "current" not in params:
                return "❌ channels와 current 파라미터가 필요합니다"

            try:
                pairs = self._parse_channels(params["channels"])
            except ValueError as e:
                return str(e)
            if not pairs:
                return "❌ 유효한 채널을 찾을 수 없습니다"

            current = float(params["current"])
            rows = self._read_config_rows()
            row_map, _, _ = self._build_lookup(rows)

            for pair in pairs:
                row = row_map.get(pair)
                if not row:
                    return f"❌ Slot{pair[0]} Ch{pair[1]}이(가) config.txt에 없습니다"
                row['I0Set'] = self._format_numeric(current)

            self._write_config_rows(rows)

            out = ["🔧 HV Current Command Executed"]
            out.append(self._fmt_request(pairs) + f" → {self._format_numeric(current)}μA")
            out.append("")
            out += self._run_per_slot(pairs, f"--config {HV_CONFIG_RELATIVE_PATH} --Pw On")
            return "\n".join(out)

    def _set_svmax(self, params: Dict[str, Any]) -> str:
        """SVMax 설정"""
        if "channel" in params and "channels" not in params:
            params["channels"] = [params["channel"]]
        if "value" in params and "svmax" not in params:
            params["svmax"] = params["value"]
        if "channels" not in params or "svmax" not in params:
            return "❌ channels와 svmax 파라미터가 필요합니다"

        try:
            pairs = self._parse_channels(params["channels"])
        except ValueError as e:
            return str(e)
        if not pairs:
            return "❌ 유효한 채널을 찾을 수 없습니다"

        svmax = float(params["svmax"])
        rows = self._read_config_rows()
        row_map, _, _ = self._build_lookup(rows)

        for pair in pairs:
            row = row_map.get(pair)
            if not row:
                return f"❌ Slot{pair[0]} Ch{pair[1]}이(가) config.txt에 없습니다"
            row['SVMax'] = self._format_numeric(svmax)

        self._write_config_rows(rows)

        out = ["🔧 HV SVMax Command Executed"]
        out.append(self._fmt_request(pairs) + f" SVMax → {self._format_numeric(svmax)}")
        out.append("")
        out += self._run_per_slot(pairs, f"--config {HV_CONFIG_RELATIVE_PATH} --Pw On")
        return "\n".join(out)

    def _set_name(self, params: Dict[str, Any]) -> str:
        """채널 이름 변경"""
        if "channel" in params and "channels" not in params:
            params["channels"] = [params["channel"]]
        if "channels" not in params or "name" not in params:
            return "❌ channels와 name 파라미터가 필요합니다"

        try:
            pairs = self._parse_channels(params["channels"])
        except ValueError as e:
            return str(e)
        if not pairs:
            return "❌ 유효한 채널을 찾을 수 없습니다"

        new_name = str(params["name"]).strip()
        if not new_name:
            return "❌ name이 비어있습니다"

        rows = self._read_config_rows()
        row_map, _, _ = self._build_lookup(rows)

        changes = []
        for pair in pairs:
            row = row_map.get(pair)
            if not row:
                return f"❌ Slot{pair[0]} Ch{pair[1]}이(가) config.txt에 없습니다"
            old_name = row['name']
            row['name'] = new_name
            changes.append(f"Slot{pair[0]} Ch{pair[1]}: {old_name} → {new_name}")

        self._write_config_rows(rows)
        return "\n".join(["🔧 HV Name Changed"] + changes)

    def _set_ramp(self, params: Dict[str, Any], field: str) -> str:
        """RampUp / RampDown 설정"""
        key = "rup" if field == "RampUp" else "rdown"
        if "channel" in params and "channels" not in params:
            params["channels"] = [params["channel"]]
        if "value" in params and key not in params:
            params[key] = params["value"]
        if "channels" not in params or key not in params:
            return f"❌ channels와 {key} 파라미터가 필요합니다"

        try:
            pairs = self._parse_channels(params["channels"])
        except ValueError as e:
            return str(e)
        if not pairs:
            return "❌ 유효한 채널을 찾을 수 없습니다"

        value = float(params[key])
        rows = self._read_config_rows()
        row_map, _, _ = self._build_lookup(rows)

        for pair in pairs:
            row = row_map.get(pair)
            if not row:
                return f"❌ Slot{pair[0]} Ch{pair[1]}이(가) config.txt에 없습니다"
            row[field] = self._format_numeric(value)

        self._write_config_rows(rows)

        out = [f"🔧 HV {field} Command Executed"]
        out.append(self._fmt_request(pairs) + f" {field} → {self._format_numeric(value)}")
        out.append("")
        out += self._run_per_slot(pairs, f"--config {HV_CONFIG_RELATIVE_PATH} --Pw On")
        return "\n".join(out)

    def _power_toggle(self, params: Dict[str, Any], state: str) -> str:
        """전원 On/Off"""
        if "channel" in params and "channels" not in params:
            params["channels"] = [params["channel"]]
        if "channels" not in params:
            return "❌ channels 파라미터가 필요합니다"

        try:
            pairs = self._parse_channels(params["channels"])
        except ValueError as e:
            return str(e)
        if not pairs:
            return "❌ 유효한 채널을 찾을 수 없습니다"

        out = [f"🔧 HV Power {state} Command Executed"]
        out.append(self._fmt_request(pairs) + f" → {state}")
        out.append("")
        out += self._run_per_slot(pairs, f"--Pw {state}")
        return "\n".join(out)

    def _get_status(self, params: Dict[str, Any]) -> str:
        """상태 확인"""
        if "channel" in params and "channels" not in params:
            params["channels"] = [params["channel"]]

        channels_param = params.get("channels", "all")

        if channels_param in ("all", "전체"):
            rows = self._read_config_rows()
            pairs = [(row['slot'], row['ch']) for row in rows]
        else:
            try:
                pairs = self._parse_channels(channels_param)
            except ValueError as e:
                return str(e)
            if not pairs:
                return "❌ 유효한 채널을 찾을 수 없습니다"

        out = [
            "📊 HV Status Query",
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        out += self._run_per_slot(pairs, "--Status --VMon --IMon --V0Set --I0Set")
        return "\n".join(out)

    # ===== 공통 헬퍼 =====

    def _run_per_slot(self, pairs: List[SlotCh], extra_args: str) -> List[str]:
        """(slot, ch) 리스트를 슬롯별로 그룹화하여 명령 실행, 결과 라인 리스트 반환"""
        out = []
        for slot, chs in self._group_by_slot(pairs).items():
            ch_arg = " ".join(map(str, chs))
            cmd = f"./HVWrappdemo --slot {slot} --ch {ch_arg} {extra_args}"
            stdout, stderr = self._run_remote_command(cmd)
            out.append(f"💻 Slot {slot}: {cmd}")
            if stdout and stdout.strip():
                out.append("📄 Output:")
                out.extend(stdout.strip().split('\n'))
            if stderr and stderr.strip():
                out.append("⚠️ Stderr:")
                out.extend(stderr.strip().split('\n'))
            out.append("")
        return out

    def _group_by_slot(self, pairs: List[SlotCh]) -> Dict[int, List[int]]:
        """(slot, ch) 리스트 → {slot: [ch, ...]} (슬롯·채널 정렬)"""
        groups: Dict[int, List[int]] = {}
        for slot, ch in pairs:
            groups.setdefault(slot, []).append(ch)
        return {slot: sorted(chs) for slot, chs in sorted(groups.items())}

    def _fmt_request(self, pairs: List[SlotCh]) -> str:
        """📋 Request 접두사 포함 채널 목록 문자열"""
        if len(pairs) == 1:
            return f"📋 Request: Slot{pairs[0][0]} Ch{pairs[0][1]}"
        items = ", ".join(f"Slot{s}Ch{c}" for s, c in pairs)
        return f"📋 Request: {items}"

    # ===== SSH 관리 =====

    def _ensure_connection(self) -> bool:
        try:
            if (self.ssh_client and
                    self.ssh_client.get_transport() and
                    self.ssh_client.get_transport().is_active()):
                return True
        except Exception:
            pass
        return self._connect_ssh()

    def _connect_ssh(self) -> bool:
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if HV_SSH_CONFIG['key_path']:
                key = paramiko.RSAKey.from_private_key_file(HV_SSH_CONFIG['key_path'])
                self.ssh_client.connect(
                    hostname=HV_SSH_CONFIG['host'],
                    port=HV_SSH_CONFIG['port'],
                    username=HV_SSH_CONFIG['username'],
                    pkey=key,
                    timeout=10
                )
            else:
                self.ssh_client.connect(
                    hostname=HV_SSH_CONFIG['host'],
                    port=HV_SSH_CONFIG['port'],
                    username=HV_SSH_CONFIG['username'],
                    password=HV_SSH_CONFIG['password'],
                    timeout=10
                )
            return True
        except Exception:
            return False

    def _run_remote_command(self, command: str) -> Tuple[str, str]:
        command_segments = [
            f"cd {shlex.quote(HV_WRAPPER_WORKDIR)}",
            HV_ENV_PRE_COMMAND,
            command
        ]
        remote_cmd = " && ".join(command_segments)
        wrapped = f'bash -c {shlex.quote(remote_cmd)}'

        stdin, stdout, stderr = self.ssh_client.exec_command(wrapped, timeout=30)
        output = stdout.read().decode('utf-8', errors='ignore').strip()
        error = stderr.read().decode('utf-8', errors='ignore').strip()

        return output, error

    # ===== Config 읽기/쓰기 =====

    def _read_config_rows(self) -> List[Dict[str, Any]]:
        """config.txt 읽기 — 첫 열: slot, 두 번째 열: ch"""
        sftp = self.ssh_client.open_sftp()
        try:
            with sftp.open(HV_CONFIG_FULL_PATH, 'r') as f:
                content = f.read().decode('utf-8', errors='ignore')
        finally:
            sftp.close()

        rows = []
        in_data = False

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if re.match(r'^slot\b', stripped, re.IGNORECASE):
                in_data = True
                continue
            if not in_data:
                continue

            parts = stripped.split()
            if len(parts) < 6:
                continue

            try:
                slot = int(parts[0])
                ch_str = parts[1]
                match = re.match(r'^(\d+)', ch_str)
                ch = int(match.group(1)) if match else int(ch_str)
            except (ValueError, AttributeError):
                continue

            rows.append({
                'slot':     slot,
                'ch':       ch,
                'ch_str':   ch_str,
                'name':     parts[2],
                'V0Set':    parts[3],
                'I0Set':    parts[4],
                'SVMax':    parts[5],
                'RampUp':   parts[6] if len(parts) > 6 else None,
                'RampDown': parts[7] if len(parts) > 7 else None,
            })

        return rows

    def _write_config_rows(self, rows: List[Dict[str, Any]]):
        """config.txt 쓰기 — 첫 열: slot"""
        sftp = self.ssh_client.open_sftp()
        try:
            with sftp.open(HV_CONFIG_FULL_PATH, 'r') as f:
                content = f.read().decode('utf-8', errors='ignore')

            original_lines = content.splitlines()
            header_lines = []
            channel_header_idx = -1

            for idx, line in enumerate(original_lines):
                if re.match(r'^slot\b', line.strip(), re.IGNORECASE):
                    channel_header_idx = idx
                    break
                header_lines.append(line)

            new_lines = header_lines
            if channel_header_idx >= 0:
                new_lines.append(original_lines[channel_header_idx])
            else:
                new_lines.append("slot ch name V0Set I0Set SVMax RampUp RampDown")

            for row in sorted(rows, key=lambda r: (r['slot'], r['ch'])):
                ch_id = row.get('ch_str', str(row['ch']))
                line = f"{row['slot']} {ch_id} {row['name']} {row['V0Set']} {row['I0Set']} {row['SVMax']}"
                if row.get('RampUp') is not None:
                    line += f" {row['RampUp']}"
                if row.get('RampDown') is not None:
                    line += f" {row['RampDown']}"
                new_lines.append(line)

            temp_path = f"{HV_CONFIG_FULL_PATH}.tmp"
            with sftp.open(temp_path, 'w') as f:
                f.write("\n".join(new_lines) + "\n")

            try:
                sftp.remove(HV_CONFIG_FULL_PATH)
            except FileNotFoundError:
                pass
            sftp.rename(temp_path, HV_CONFIG_FULL_PATH)

        finally:
            sftp.close()

    # ===== 채널 파싱 / 조회 =====

    def _build_lookup(self, rows: List[Dict[str, Any]]):
        """rows에서 (row_map, name_map, all_pairs) 반환"""
        row_map: Dict[SlotCh, Dict] = {(r['slot'], r['ch']): r for r in rows}
        name_map: Dict[str, SlotCh] = self._get_name_map(rows)
        all_pairs: List[SlotCh] = [(r['slot'], r['ch']) for r in rows]
        return row_map, name_map, all_pairs

    def _get_name_map(self, rows: List[Dict[str, Any]]) -> Dict[str, SlotCh]:
        """Name → (slot, ch) 매핑. None 채널 제외, 중복 시 ValueError."""
        bucket: Dict[str, List[SlotCh]] = {}
        for row in rows:
            name = row.get('name', '').strip()
            if not name or name.lower() == 'none':
                continue
            bucket.setdefault(name.upper(), []).append((row['slot'], row['ch']))

        duplicates = {n: ps for n, ps in bucket.items() if len(ps) > 1}
        if duplicates:
            dup_info = ', '.join(
                n + "(" + ", ".join(f"Slot{s}Ch{c}" for s, c in ps) + ")"
                for n, ps in duplicates.items()
            )
            raise ValueError(f"❌ 중복된 Name이 있습니다: {dup_info}")

        return {n: ps[0] for n, ps in bucket.items()}

    def _resolve_identifier(self, identifier: str,
                             name_map: Dict[str, SlotCh],
                             all_pairs: List[SlotCh]) -> List[SlotCh]:
        """단일 식별자 → (slot, ch) 리스트.

        지원 형식:
        - "M1T1C"    → name lookup
        - "11:3"     → 명시적 slot:ch
        - "3"        → ch==3 인 모든 (slot, ch) [슬롯 무관]
        """
        identifier = str(identifier).strip()

        # "slot:ch" 명시 형식
        m = re.match(r'^(\d+):(\d+)$', identifier)
        if m:
            return [(int(m.group(1)), int(m.group(2)))]

        # 순수 숫자 → 슬롯 정보 없이 ch만 지정된 경우 → 에러로 명시 요청
        if re.match(r'^\d+$', identifier):
            ch = int(identifier)
            slots = sorted({s for s, c in all_pairs if c == ch})
            if not slots:
                raise ValueError(f"❌ Ch{ch}에 해당하는 채널이 config.txt에 없습니다")
            slot_list = ", ".join(map(str, slots))
            raise ValueError(
                f"❌ Ch{ch}만으로는 슬롯을 특정할 수 없습니다 "
                f"(가능한 슬롯: {slot_list}). "
                f"'slot:{slot_list.split(',')[0].strip()}:{ch}' 형식으로 지정해주세요."
            )

        name_key = identifier.upper()

        # C / S side 선택: 타워 채널 이름은 "T1C..T9C"/"T1S..T9S" 또는 "M1T1C" 형식 모두 지원.
        # (TRIG1/TRIG2, MCP-C/MCP-S 같은 특수 채널은 제외 — 순수 타워 채널만)
        if name_key in ("C", "S"):
            return sorted(
                pair for name, pair in name_map.items()
                if re.match(rf'^(?:M\d+[-_ ]?)?T\d+[-_ ]?{name_key}$', name)
            )

        # Tower 선택: "T1"~"T4" → 모든 모듈의 해당 타워 C/S 채널
        tower_m = re.match(r'^T(\d+)$', name_key)
        if tower_m:
            tn = tower_m.group(1)
            return sorted(
                pair for name, pair in name_map.items()
                if re.match(rf'^(?:M\d+[-_ ]?)?T{tn}[-_ ]?[CS]$', name)
            )

        # Name lookup
        if name_key in name_map:
            return [name_map[name_key]]

        # Module shorthand: "M5" → M5T1C, M5T1S, M5T2C, M5T2S, M5T3C, M5T3S, M5T4C, M5T4S
        mod_m = re.match(r'^M([1-9])$', name_key)
        if mod_m:
            mod_num = mod_m.group(1)
            result = []
            for t in range(1, 5):
                for s in ('C', 'S'):
                    ch = f"M{mod_num}T{t}{s}"
                    if ch in name_map:
                        result.append(name_map[ch])
            return result

        return []

    def _parse_channels(self, channels: Any) -> List[SlotCh]:
        """채널 표현을 (slot, ch) 리스트로 변환 (정렬·중복제거)"""
        rows = self._read_config_rows()
        _, name_map, all_pairs = self._build_lookup(rows)

        # 리스트 형태
        if isinstance(channels, list):
            result: List[SlotCh] = []
            for item in channels:
                if isinstance(item, dict):
                    s = item.get('slot')
                    c = item.get('ch')
                    if s is not None and c is not None:
                        result.append((int(s), int(c)))
                else:
                    key = str(item).strip().lower()
                    if key in ('all', '전체'):
                        result.extend(all_pairs)
                    elif key == 'even':
                        result.extend((s, c) for s, c in all_pairs if c % 2 == 0)
                    elif key == 'odd':
                        result.extend((s, c) for s, c in all_pairs if c % 2 == 1)
                    else:
                        m = re.match(r'^slot[:\s]?(\d+)[:\s](even|odd)$', key)
                        if m:
                            tgt = int(m.group(1))
                            par = 0 if m.group(2) == 'even' else 1
                            result.extend((s, c) for s, c in all_pairs if s == tgt and c % 2 == par)
                        else:
                            result.extend(self._resolve_identifier(str(item), name_map, all_pairs))
            return sorted(set(result))

        expr = str(channels).strip()

        # all / 전체
        if expr.lower() in ('all', '전체'):
            return sorted(set(all_pairs))

        # even
        if expr.lower() == 'even':
            return sorted({(s, c) for s, c in all_pairs if c % 2 == 0})

        # odd
        if expr.lower() == 'odd':
            return sorted({(s, c) for s, c in all_pairs if c % 2 == 1})

        # "slot:12:even" / "slot:12:odd"
        m = re.match(r'^slot[:\s]?(\d+)[:\s](even|odd)$', expr, re.IGNORECASE)
        if m:
            target = int(m.group(1))
            parity = 0 if m.group(2).lower() == 'even' else 1
            return sorted({(s, c) for s, c in all_pairs if s == target and c % 2 == parity})

        # "slot:12" / "slot12" / "slot 12"
        m = re.match(r'^slot[:\s]?(\d+)$', expr, re.IGNORECASE)
        if m:
            target = int(m.group(1))
            return sorted({(s, c) for s, c in all_pairs if s == target})

        # "11:3" 명시적 pair
        m = re.match(r'^(\d+):(\d+)$', expr)
        if m:
            return [(int(m.group(1)), int(m.group(2)))]

        # "N-M" ch 번호 범위 (전 슬롯)
        m = re.match(r'^(\d+)-(\d+)$', expr)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return sorted({(s, c) for s, c in all_pairs if lo <= c <= hi})

        # 쉼표 구분 목록: 순수 숫자들이면 ch 번호 목록, 아니면 이름/slot:ch
        parts = [p.strip() for p in expr.split(',') if p.strip()]
        if parts and all(re.match(r'^\d+$', p) for p in parts):
            ch_set = {int(p) for p in parts}
            return sorted({(s, c) for s, c in all_pairs if c in ch_set})

        result = []
        for part in parts:
            result.extend(self._resolve_identifier(part, name_map, all_pairs))
        return sorted(set(result))

    # ===== 유틸 =====

    def _format_numeric(self, value: Any) -> str:
        try:
            num = float(value)
            if abs(num - int(num)) < 1e-6:
                return str(int(num))
            return str(num)
        except (TypeError, ValueError):
            return str(value)

    def __del__(self):
        if self.ssh_client:
            self.ssh_client.close()
