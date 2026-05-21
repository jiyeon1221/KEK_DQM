#!/usr/bin/env python3
"""Thin helper: build and run azd_kren commands. All settings from config_general.yml Motor section."""

import subprocess
from tools.config_loader import load_config


def _cfg() -> dict:
    cfg = load_config()
    m = cfg.get("Motor")
    if not m:
        raise RuntimeError("config_general.yml에 'Motor' 섹션이 없습니다.")
    return m


def _run_cmd(cmd: list) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return True, r.stdout.strip() or "완료"
        return False, r.stderr.strip() or "unknown error"
    except subprocess.TimeoutExpired:
        return False, "Motor move timed out (120 s)"
    except FileNotFoundError:
        return False, f"azd_kren not found: {cmd[0]}"


def _base_cmd(m: dict) -> list:
    return [
        m["EZSBinary"],
        "--ip",    m["IP"],
        "--port",  str(m["Port"]),
        "--speed", str(m["Speed"]),
        "--acc",   str(m["Acc"]),
        "--dec",   str(m["Dec"]),
    ]


def move_x(x_scan: float) -> tuple[bool, str]:
    """절대 이동: azd_kren --moveto <mm>"""
    m = _cfg()
    target_mm = x_scan * m["UnitScale"]
    cmd = _base_cmd(m) + ["--moveto", f"{target_mm:.3f}"]
    ok, msg = _run_cmd(cmd)
    return (True, f"X축 이동 완료: {target_mm:.3f} mm") if ok else (False, msg)


def move_relative(delta_mm: float) -> tuple[bool, str]:
    """상대 이동: azd_kren --move <mm> (양수=오른쪽, 음수=왼쪽)"""
    m = _cfg()
    cmd = _base_cmd(m) + ["--move", f"{delta_mm:.3f}"]
    ok, msg = _run_cmd(cmd)
    direction = "오른쪽" if delta_mm >= 0 else "왼쪽"
    return (True, f"X축 상대 이동 완료: {direction} {abs(delta_mm):.3f} mm") if ok else (False, msg)
