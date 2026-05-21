#!/usr/bin/env python3
"""
Hodoscope HV Tool
-----------------
Set file  (single value)  → read via NFS/Thunderbolt mount, write via SSH to Mac Studio.
Run log   (4 actual values) → parsed separately by run_log_tool after each DAQ run.

Set file format (one relevant line):
  hv <value>        e.g.  hv 4.5   or   hv 0.0
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .base_tool import BaseTool
from .config_loader import get_path_config, load_config


def _get_setfile_path() -> str:
    return get_path_config("HodoscopeSetFile")


def _get_logdir_path() -> str:
    return get_path_config("DaqLogDir")


def _get_studio_ssh() -> dict:
    cfg = load_config()
    s = cfg.get("StudioSSH", {})
    return {"host": s.get("Host"), "user": s.get("User"), "key": s.get("KeyFile")}


# ── Public helpers (used by run_log_tool as well) ─────────────────────────────

def read_hv_from_setfile() -> Optional[float]:
    """Return the single 'hv <value>' from the set file, or None on error."""
    try:
        for line in Path(_get_setfile_path()).read_text().splitlines():
            m = re.match(r'^\s*hv\s+([+-]?\d+(?:\.\d+)?)', line)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None


def read_hv_from_log(run_num: str) -> Optional[Dict[int, float]]:
    """
    Parse log_set_{run_num}.log → {1: v1, 2: v2, 3: v3, 4: v4}.
    Returns None if file not found or no HV lines.
    """
    try:
        log_path = Path(_get_logdir_path()) / f"log_set_{run_num}.log"
        result = {}
        for line in log_path.read_text().splitlines():
            m = re.search(r'High voltage\[(\d)\]\s*=\s*([+-]?\d+(?:\.\d+)?)', line)
            if m:
                ch = int(m.group(1))
                val = float(m.group(2))
                result[ch] = val
        return result if result else None
    except Exception:
        return None


def write_hv_via_ssh(value: float) -> str:
    """Replace 'hv <old>' with 'hv <new>' in the set file on Mac Studio via SSH."""
    ssh = _get_studio_ssh()
    # /Volumes/yhep mount on MacBook → /Users/yhep on Mac Studio
    remote_path = get_path_config("HodoscopeSetFile").replace(
        "/Volumes/yhep", "/Users/yhep"
    )

    value_str = f"{value:.6g}"   # compact: 4.5, 0.0, 4.123456

    sed_cmd = f"sed -i '' 's/^hv [^ ]*/hv {value_str}/' {remote_path}"
    ssh_cmd = [
        "ssh", "-i", ssh["key"],
        "-o", "IdentitiesOnly=yes",
        "-o", "IdentityAgent=none",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{ssh['user']}@{ssh['host']}",
        sed_cmd,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH sed failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return f"✅ Hodoscope HV → {value_str} V (set file updated)"


# ── Tool class ────────────────────────────────────────────────────────────────

class HodoscopeHVTool(BaseTool):
    """Read / write hodoscope HV via the DAQ set file."""

    def __init__(self):
        super().__init__(
            name="hodoscope_hv_tool",
            description="Read or modify hodoscope HV in the DAQ set file",
        )

    def execute(self, params: Dict[str, Any]) -> str:
        command = params.get("command", "read")

        if command == "read":
            hv = read_hv_from_setfile()
            if hv is None:
                return "⚠️ Set file을 읽을 수 없습니다. 경로를 확인하세요."
            if hv == 0.0:
                return "📡 Hodoscope HV: 0.0 V (off)"
            return f"📡 Hodoscope HV: {hv} V"

        if command == "write":
            value = params.get("value")
            if value is None:
                raise RuntimeError("'value' 파라미터가 필요합니다.")
            value = float(value)
            if value < 0:
                raise RuntimeError("HV 값은 0 이상이어야 합니다.")
            return write_hv_via_ssh(value)

        raise RuntimeError(f"지원하지 않는 command: {command}  (read | write)")
