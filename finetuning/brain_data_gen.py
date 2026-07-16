#!/usr/bin/env python3
"""
Training data generator for BrainAgent
- Single-turn: state + user request  →  tool call JSON
- Covers all tool types × expression variety × state combinations
- Target: daq_run 150, dqm_plot 600+250+170, run_log 150+70, hv_read 190, hv_write 240, hodoscope_write 45, none 150
- hodoscope_hv_read merged into hv_read (hv_read now shows CAEN HV + Hodoscope combined)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


SYSTEM_PROMPT = """You are the Brain Agent for a test beam experiment (KEK/CERN).
Your job is to interpret the operator's ad-hoc request and call the right tool.

Available tools:
- daq_run: Run DAQ data collection. params: {"events": int, "config": str (optional)}
  - config is the DAQ config name passed to the run script. It can be ANY name the user says
    (e.g. "test", "setup", "setup1", "config1", "physics", ...). Default is "setup" —
    OMIT config unless the user explicitly names one
    (e.g. "setup1으로 10000개", "config1 설정으로", "test로 돌려줘").
    Copy the name EXACTLY as the user wrote it — never invent or normalize it.
- dqm_plot: Generate DQM plots for a run and display in the DQM panel.
  params: {"run_number": int, "method": "IntADC"|"PeakADC", "type": "full"|"heatmap"|"single", "modules": [list]}
  - type defaults to "full" (all towers + heatmap). No modules needed for full.
  - "heatmap": modules must be ["MCPPMT"]. method: IntADC or PeakADC.
  - "single": modules is a list of channel names, e.g. ["M1-T1-C"], ["M1-T1-S","M1-T1-C"], or ["M1"] (M1 auto-expands).
  - method has NO default. If the user does not say IntADC/적분 or PeakADC/피크, you MUST ask (tool:none) — never assume IntADC.
  - run_number has NO default. Never invent one. If not given and no relative reference, ask (tool:none).
- run_log: Google Sheets run log.
  Read:   params: {"command": "read", "run_num": int}
  Update: params: {"command": "update", "run_num": int, "<column>": "<value>"}
  Updatable columns: program, notes, config, beam_energy, beam_type, trigger_setup, hv_drc, hv_aux
- hv_read: Read current CAEN HV status.
  params: {"command": "status", "channels": <ch_spec>}
  "channels" is OPTIONAL and accepts the SAME <ch_spec> forms as hv_write (see below).
  - Omit "channels" (reads ALL channels) ONLY when there is no channel hint or the user says 전체/모든/all.
  - If the user restricts to a subset (짝수/홀수/C채널/S채널/타워/특정 모듈·채널/슬롯 등),
    put it in "channels" EXACTLY as you would for hv_write
    (짝수→"even", 홀수→"odd", S채널/S만→"S", C채널/C만→"C", 타워 T1→"T1",
     M3만→"M3", 특정 채널→["M3T2C"], 슬롯→"slot:12").
- hv_write: Change CAEN HV voltage or turn channels on/off. User confirmation required.
  Voltage: {"command": "voltage", "channels": <ch_spec>, "voltage": <V as float>}
  On/off:  {"command": "on"|"off", "channels": <ch_spec>}
  Valid channel names (ONLY these): M{1-9}T{1-4}{C,S} (e.g. M1T1C, M1T1S, M1T2C ... M9T4S), TRIG1, TRIG2, MCP-S, MCP-C
  Channel spec (<ch_spec>) options:
    "all"          — 전체 채널 (ONLY when user says 전체/모든/all)
    "S" / "C"      — 모든 S(신틸)/C(체렌코프) 채널
    "T1"~"T4"      — 타워 단위: 모든 모듈의 해당 타워 채널 (예: "T1")
    "M5"           — 모듈 단위: M5의 모든 채널 (M5T1C/S~M5T4C/S, 8채널). M{1-9} 형식.
    ["M3","M5"]    — 모듈 목록: 복수 모듈 지정 (각 모듈 8채널)
    "even"         — 짝수 번호 채널 전체
    "odd"          — 홀수 번호 채널 전체
    "N-M"          — ch 번호 N~M 범위 (예: "0-8", "2-12")
    "N,M,K"        — ch 번호 목록 (예: "1,2,5,6")
    ["M1T1C","M1T2C"]  — 이름 목록
    "slot:S"       — 슬롯 S 전체
    "slot:S:even/odd" — 슬롯 S 짝/홀수

Current experiment state is provided so you can resolve relative references
like "방금", "이번 런", "지금" to concrete run numbers or energies.

Respond with a single JSON object:
{"tool": "<tool_name>", "params": {<params>}, "reason": "<short explanation>"}

If the request is unclear or you cannot determine a tool, respond:
{"tool": "none", "message": "<ask the user for clarification>"}

RULES:
0. All "message" and "reason" field values MUST be written in Korean (한국어) only. Never use Chinese characters (한자).
1. Output ONLY valid JSON. No markdown, no explanation outside JSON.
2. Always resolve relative references using the provided state.
3. For run_log updates, extract column and value from the user's message.
4. run_log supports both READ and WRITE:
   - VIEW/CHECK a log (확인, 보여줘, 읽어줘) WITHOUT a value → {"command": "read", "run_num": ...}
   - WRITE with column+value (e.g. "프로그램에 EM 추가") → {"command": "update", "run_num": ..., "<column>": "<value>"}
5. hv_read for ANY HV status. "HV 확인", "HV 상태" → all use hv_read.
   If the request names a channel subset (짝수/홀수/C/S/타워/모듈/채널명/슬롯), pass it in "channels" just like hv_write;
   otherwise omit "channels" to read ALL channels.
6. DAQ requires an event count. If the user says "DAQ 돌려줘" without a number, ask how many events.
   If the user names a DAQ config (test, setup1, config1 등 — 어떤 이름이든), put it in "config";
   otherwise OMIT config (default "setup").
7. Channel names like M1T1C, M1T1S, M2T3C, ... are HV channels (format: M{1-9}T{1-4}{C,S}) — NOT log columns.
   A SINGLE channel name + voltage → channels: [that single channel].
   ONLY use channels: "all" when the input explicitly says 전체/모든/전 채널/all channels.
8. For daq_run and hv_write, the system asks the user to confirm before execution.
9. "플롯", "그려줘", "그래프" → dqm_plot. Default type: full.
   Method: ONLY set it when the user explicitly says IntADC/intADC/적분 (→ "IntADC") or PeakADC/peakADC/피크 (→ "PeakADC").
   If method is not mentioned, respond with tool:none asking "IntADC로 그릴까요, PeakADC로 그릴까요?"
10. Specific tower/channel (M1, M1-T1-C, M1-T1-S, M5 etc.) → type: single, modules: [name].
11. "heatmap" or "MCPPMT" mentioned → type: heatmap, modules: ["MCPPMT"].
12. dqm_plot requires run_number. NEVER invent or guess a run number.
    Only resolve it from state when the user uses a relative reference (방금/이번/지금/현재/마지막/최근/last).
    If the user gives NO explicit run number AND NO relative reference,
    respond tool:none asking "어떤 런 번호의 DQM 플롯을 그릴까요?".
"""


# ── State builder (mirrors brain_agent.py) ───────────────────────────────────

def _build_state_context(state: dict) -> str:
    if not state:
        return "(No scenario agent running)"
    lines = []
    if state.get("agent_type"):
        lines.append(f"Running agent: {state['agent_type']}")
    if state.get("current_run"):
        lines.append(f"Current run number: {state['current_run']}")
    last_run = state.get("last_run") or state.get("last_run_number")
    if last_run:
        lines.append(f"Last completed run: {last_run}")
    if state.get("current_tower"):
        lines.append(f"Current tower: {state['current_tower']}")
    if state.get("current_energy"):
        lines.append(f"Current energy: {state['current_energy']} GeV")
    if state.get("phase"):
        lines.append(f"Phase: {state['phase']}")
    return "\n".join(lines) if lines else "(No scenario agent running)"


def build_full_context(state: dict, user_input: str) -> str:
    parts = []
    parts.append("=== Current State ===")
    parts.append(_build_state_context(state))
    parts.append("")
    parts.append("=== Recent Conversation ===")
    parts.append("(No conversation yet)")
    parts.append("")
    parts.append("=== Current User Input ===")
    parts.append(user_input)
    parts.append("")
    parts.append("=== Your Task ===")
    parts.append("Based on the current state and conversation, decide the next action.")
    parts.append("Output JSON with tool name and parameters.")
    return "\n".join(parts)


def make_example(state: dict, user_input: str, decision: dict) -> dict:
    ctx = build_full_context(state, user_input)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ]
    }


# ── Random state generators ─────────────────────────────────────────────────

def _random_run():
    # 더 넓은 런 번호 범위 (실제 실험 런 다양성 반영)
    return random.randint(10000, 15000)

def _random_energy():
    return random.choice([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120])

def _random_tower():
    m = random.randint(1, 9)
    t = random.randint(1, 4)
    return f"M{m}T{t}"

def _random_events():
    # 1000 ~ 500000 범위, 실험에서 쓰는 현실적인 값
    return random.choice([
        1000, 2000, 3000, 5000, 7000,
        10000, 20000, 30000, 50000, 70000,
        100000, 150000, 200000, 300000, 500000,
        random.randint(1000, 9999),
        random.randint(10000, 99999),
        random.randint(100000, 500000),
    ])


def _make_state(with_agent=True):
    if not with_agent:
        return {}
    run = _random_run()
    return {
        "agent_type": random.choice(["energy_scan", "calib_scan", "hv_equalization"]),
        "current_run": run,
        "last_run": run - 1,
        "current_tower": _random_tower(),
        "current_energy": _random_energy(),
        "phase": random.choice(["scanning", "daq_running", "waiting_confirm", "hv_adjusting"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DAQ RUN — 100 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_daq_run() -> List[dict]:
    examples = []
    templates = [
        # 기본
        "{n}개 돌려줘", "{n}개 이벤트 받아줘", "이벤트 {n}개 수집해줘",
        "DAQ {n}개 돌려", "데이터 {n}개만 받자", "{n}개만 더 받아줘",
        "{n}개 추가로 받아줘", "DAQ 한번 돌려줘 {n}개", "{n}개 빨리 돌려",
        "{n}개 이벤트 수집", "데이터 수집 {n}개", "이벤트 {n}개 돌려줘",
        "{n}개 데이터 수집해줘", "{n}개 받자",
        # 영어 혼용
        "run {n} events", "DAQ run {n}", "{n} events 돌려줘",
        "{n} events please", "take {n} events", "collect {n} events",
        "{n} events 받아줘", "daq {n}", "{n} evt",
        "start daq with {n} events", "run daq {n}", "daq run {n} events",
        "{n}k events 돌려줘", "run {n} evt please", "get {n} events",
        "fire {n} events", "acquire {n} events", "run {n} evts",
        # 반말 / 채팅체
        "{n}개 받아", "{n}개 돌려", "데이터 {n}개", "{n}개 ㄱㄱ",
        "이벤트 {n}개 좀", "{n}개 ㄱ", "{n}개 달려", "{n} 돌려",
        # 수량 변형
        "{n}개만 받아줘", "{n}개만 돌려", "{n}개 정도 돌려줘",
        "{n}개쯤 받자", "한 {n}개 돌려볼까", "약 {n}개 받아줘",
        "{n}개만 좀 받아줘", "{n}개 정도만", "딱 {n}개만",
        # 목적/맥락
        "테스트로 {n}개 돌려줘", "확인용으로 {n}개만", "pedestal {n}개 받아줘",
        "빔 데이터 {n}개 받자", "노이즈 체크 {n}개", "퀵 체크 {n}개 돌려",
        "캘리브레이션 {n}개 돌려줘", "추가 데이터 {n}개 받아줘",
        "통계 더 쌓으려고 {n}개 돌려줘", "다시 {n}개 돌려줘",
        "한번 더 {n}개 받아줘", "{n}개 더 돌려",
        "cosmic 데이터 {n}개", "LED 데이터 {n}개 받아줘",
        "빔 테스트 {n}개", "잠깐 {n}개만 돌려",
        # 추가 표현
        "지금 {n}개 돌려", "바로 {n}개 받아줘", "시작해줘 {n}개",
        "{n}개 해줘", "수집 {n}개", "{n}개 수집 시작",
        "이번에 {n}개 받자", "일단 {n}개만", "{n}개 돌려볼게",
        "한판 {n}개", "{n}개짜리 돌려줘", "지금 바로 {n}개",
        # 영어 단독
        "run daq 10000".replace("10000", "{n}"), "daq {n}개 start",
        "start data taking {n} events", "{n} events now",
        "please collect {n} events", "take data {n} events",
    ]
    # config 명시 케이스 — 사용자가 config 이름을 말하면 params에 "config" 포함
    config_templates = [
        "{cfg}로 {n}개 돌려줘", "{cfg}으로 {n}개 받아줘", "config {cfg}로 {n}개",
        "{cfg} 설정으로 {n}개 수집해줘", "{cfg} config으로 {n} events",
        "run {n} events with {cfg}", "{cfg}으로 {n} events 돌려",
        "컨피그 {cfg}로 {n}개", "{cfg} 셋업으로 {n}개 받자",
        "config {cfg} {n} events", "daq {cfg} {n}", "{n}개를 {cfg}로 받아줘",
        "{cfg} 설정 {n}개 돌려", "{cfg}로 DAQ {n}개", "config은 {cfg}, {n}개 돌려줘",
    ]
    # config 이름은 자유형 — 다양한 형태를 섞어 학습시켜 임의 이름을 그대로 복사하게 한다
    config_names = [
        "test", "setup", "setup1", "setup2", "setup3",
        "config1", "config2", "config3",
    ]

    for _ in range(150):
        events = _random_events()
        state = _make_state(random.random() > 0.3)
        if random.random() < 0.3:
            cfg = random.choice(config_names)
            template = random.choice(config_templates)
            user_input = template.format(n=events, cfg=cfg)
            decision = {
                "tool": "daq_run",
                "params": {"events": events, "config": cfg},
                "reason": f"DAQ {events} 이벤트 수집 (config {cfg})",
            }
        else:
            template = random.choice(templates)
            user_input = template.format(n=events)
            decision = {
                "tool": "daq_run",
                "params": {"events": events},
                "reason": f"DAQ {events} 이벤트 수집",
            }
        examples.append(make_example(state, user_input, decision))
    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  DQM PLOT — 600 samples
#  full (explicit 200 + relative 150), heatmap (100), single (150)
# ══════════════════════════════════════════════════════════════════════════════

def _random_channel():
    m = random.randint(1, 9)
    t = random.randint(1, 4)
    side = random.choice(["-C", "-S"])
    return f"M{m}-T{t}{side}"


def _nondash(ch: str) -> str:
    """'M1-T1-S' → 'M1T1S'. Leaves 'M1' or 'MCPPMT' unchanged."""
    return ch.replace("-", "")

def _random_tower_name():
    return f"M{random.randint(1, 9)}"

def _random_single_modules():
    """Return a realistic modules list for single type."""
    choice = random.random()
    if choice < 0.3:
        # one specific channel: T1-C
        return [_random_channel()]
    elif choice < 0.6:
        # tower name (auto-expands): T1
        return [_random_tower_name()]
    elif choice < 0.8:
        # scintillator + Cherenkov of same tower
        m = random.randint(1, 9)
        t = random.randint(1, 4)
        return [f"M{m}-T{t}-S", f"M{m}-T{t}-C"]
    else:
        # two different channels
        return [_random_channel(), _random_channel()]


def gen_dqm_plot() -> List[dict]:
    examples = []

    # ── A. full type — explicit run, explicit method ───────────────────────────
    # Only templates with an explicit method keyword (intADC/적분/integral or peakADC/피크/peak)
    full_intadc_tmpl = [
        # intADC explicit
        "{r} intADC 그려줘", "run {r} intADC 보여줘",
        "{r}번 런 intADC 그려줘", "run {r} 적분 그려줘",
        "{r} 적분 ADC 보여줘", "run {r} 적분ADC 그려",
        "{r}번 intADC 플랏", "run {r} int ADC 그려줘",
        "{r} integral 그려줘", "run {r} 적분만 그려줘",
        "{r}번 런 적분 그래프", "run {r} int adc 보여줘",
        # all-tower intADC explicit
        "run {r} 모든 타워 intADC 그려줘", "{r}번 전체 타워 intADC 보여줘",
        "run {r} 모든 채널 intADC", "run {r} all tower intADC 그려",
        "{r}번 런 intADC DQM", "run {r} DQM intADC 그려줘",
        "{r} intADC DQM 보여줘", "run {r} 전체 intADC",
    ]
    # Templates with NO method keyword — model should ask IntADC/PeakADC
    full_nomethod_tmpl = [
        "run {r} 그려줘", "run {r} 플랏 보여줘", "run {r} 전부 그려줘",
        "run {r} 그래프 다 그려줘", "{r}번 런 플랏", "{r} 플랏 보여줘",
        "run {r} 그래프 보여줘", "{r}번 그려줘", "run {r} 플랏 그려",
        "{r} 데이터 그려줘", "run {r} 전체 플랏", "{r}번 런 전부 그려",
        "run {r} plot", "{r} 그래프 전부", "run {r} 다 그려",
        "run {r} DQM 그려줘", "{r}번 DQM 보여줘", "run {r} DQM plot",
        "run {r} 타워 전체 그려줘", "{r} 모든 타워 그래프",
        "{r}번 전타워 플랏", "{r} 전체 타워 그려줘",
        "run {r} 타워 다 그려", "{r}번 런 모든 타워 플랏",
    ]
    full_peakadc_tmpl = [
        "run {r} peakADC 그려줘", "run {r} peak 그려",
        "{r} peakADC 보여줘", "{r}번 peak ADC 그려줘",
        "run {r} 피크 그려", "{r} peak 플랏",
        "run {r} peakADC plot", "{r}번 런 피크ADC",
        "run {r} 피크 ADC 보여줘", "{r} peakADC",
        "run {r} peak ADC 그래프", "{r}번 peakADC 보여줘",
        "run {r} 피크만 그려줘", "{r} peakADC 그래프",
        "run {r} peak만 보여줘", "{r}번 peak 보여줘",
        "run {r} 피크ADC 그래프 보여줘", "{r} 피크 adc",
        "run {r} HV peakADC 그려줘", "{r}번 peakADC DQM",
        # all-tower phrasing
        "run {r} 모든 타워 peakADC 그려줘", "{r}번 전체 타워 peak 보여줘",
        "run {r} 타워 전체 peakADC", "{r} 모든 타워 피크 ADC",
        "run {r} all tower peakADC", "{r}번 전타워 peakADC 플랏",
    ]

    for _ in range(150):
        run = _random_run()
        tmpl = random.choice(full_intadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "IntADC", "type": "full"},
             "reason": f"Run {run} full IntADC DQM 플랏 생성"},
        ))

    for _ in range(50):
        run = _random_run()
        tmpl = random.choice(full_peakadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "PeakADC", "type": "full"},
             "reason": f"Run {run} full PeakADC DQM 플랏 생성"},
        ))

    # nomethod explicit-run: 100 samples → tool:none asking method
    for _ in range(100):
        run = _random_run()
        tmpl = random.choice(full_nomethod_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "none", "message": "IntADC로 그릴까요, PeakADC로 그릴까요?"},
        ))

    # ── B. full type — relative reference ────────────────────────────────────
    # Explicit-method relative templates only
    relative_intadc = [
        "현재 런 intADC 그려줘", "이번 거 intADC 보여줘",
        "방금 런 적분 그려줘", "이번 데이터 적분 ADC",
        "방금 거 integral 그려줘", "이번 런 intADC 그래프",
        "이번 런 전체 타워 intADC", "방금 거 intADC DQM",
        "이번 결과 intADC 그려줘", "방금 런 intADC 보여줘",
        "지금 런 intADC 그려", "현재 데이터 적분 ADC 그려줘",
        "방금 거 적분 그려줘", "이번 런 적분ADC 보여줘",
    ]
    # No method in relative reference → ask method
    relative_nomethod = [
        "방금 데이터 플랏 그려줘", "이번 런 그려줘", "마지막 런 플랏 보여줘",
        "방금 받은 거 그래프", "이번 거 전부 그려줘", "지금 런 플랏 그려줘",
        "방금 거 그려줘", "이번 데이터 플랏", "마지막 데이터 그려",
        "현재 런 그래프 보여줘", "방금 돌린 거 플랏", "지금 데이터 전부 그려줘",
        "이번 런 전체 플랏 보여줘", "방금 런 plot", "최근 런 그려줘",
        "방금 데이터 그래프 보여줘", "이번 런 데이터 플랏", "지금 거 그려줘",
        "현재 데이터 그래프", "이번 런 플랏 보여줘", "마지막 거 그려줘",
        "방금 DAQ 결과 그려줘", "이번 결과 플랏 보여줘",
        "방금 거 전부 그려", "이번 런 다 그려줘", "방금 돌린 데이터 그래프",
        "현재 런 플랏 그려", "지금 런 데이터 그려줘",
        "방금 거 DQM 그려줘", "이번 런 DQM 보여줘", "방금 결과 DQM",
        "방금 거 모든 타워 그려줘", "방금 런 모든 채널 그려",
        "이번 결과 전 타워 보여줘", "방금 거 타워 다 그려", "현재 런 모든 타워 플랏",
    ]
    relative_peakadc = [
        "방금 거 peakADC 그려", "이번 런 peak 그려줘", "방금 데이터 피크 보여줘",
        "마지막 런 peakADC", "지금 런 peak ADC 그려줘", "이번 거 피크 플랏",
        "방금 런 peakADC 보여", "현재 런 피크 그려줘", "방금 돌린 거 peak",
        "이번 데이터 peakADC 그려", "방금 거 피크 ADC 보여줘",
        "이번 런 피크 보여줘", "마지막 데이터 peak ADC",
        "방금 수집한 거 peakADC 그려줘", "지금 거 peak 보여줘",
        "이번 런 peakADC 그래프", "방금 DAQ 결과 피크", "현재 데이터 peak 그려",
        "방금 런 피크 그려줘", "이번 결과 peakADC 보여줘",
        "방금 거 모든 타워 peakADC", "이번 런 전체 타워 peak",
    ]

    for _ in range(70):
        tmpl = random.choice(relative_intadc)
        state = _make_state(with_agent=True)
        run = state["current_run"]
        examples.append(make_example(state, tmpl, {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC", "type": "full"},
            "reason": f"현재 run {run} full IntADC DQM 플랏 생성",
        }))

    for _ in range(80):
        tmpl = random.choice(relative_nomethod)
        state = _make_state(with_agent=True)
        examples.append(make_example(state, tmpl, {
            "tool": "none",
            "message": "IntADC로 그릴까요, PeakADC로 그릴까요?",
        }))

    for _ in range(40):
        tmpl = random.choice(relative_peakadc)
        state = _make_state(with_agent=True)
        run = state["current_run"]
        examples.append(make_example(state, tmpl, {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "PeakADC", "type": "full"},
            "reason": f"현재 run {run} full PeakADC DQM 플랏 생성",
        }))

    # ── C. heatmap type (100) ─────────────────────────────────────────────────
    heatmap_intadc_tmpl = [
        "run {r} MCPPMT intADC heatmap 그려줘",
        "run {r} heatmap IntADC 보여줘",
        "{r}번 heatmap intADC 그려",
        "run {r} MCPPMT heatmap 그려줘",
        "run {r} heatmap 그려줘",
        "{r} MCPPMT heatmap intADC",
        "run {r} heatmap intADC plot",
        "{r}번 런 MCPPMT heatmap",
        "run {r} heatmap 플랏 보여줘",
        "{r} heatmap ADC 그려",
        "run {r} MCPPMT IntADC heatmap",
        "{r}번 heatmap 플랏 IntADC",
        "run {r} heatmap 분포 그려줘",
        "{r} MCPPMT heatmap 보여줘",
        "run {r} heatmap 이미지 그려줘",
        "{r}번 런 heatmap intADC 그려줘",
        "run {r} MCPPMT 히트맵 그려줘",
        "{r} 히트맵 IntADC",
        "run {r} heatmap 분포 IntADC",
        "run {r} MCPPMT 히트맵 IntADC 보여줘",
        "run {r} MCPPMT intADC 히트맵",
        "{r}번 런 히트맵 그려줘",
        "run {r} heatmap 확인해줘",
        "{r} MCPPMT intADC heatmap 그려",
        "run {r} 히트맵 플랏",
    ]
    heatmap_peakadc_tmpl = [
        "run {r} MCPPMT peakADC heatmap 그려줘",
        "run {r} heatmap peakADC 보여줘",
        "{r}번 heatmap peakADC 그려",
        "run {r} MCPPMT heatmap peakADC",
        "{r} heatmap peak ADC",
        "run {r} heatmap peakADC plot",
        "{r}번 런 MCPPMT heatmap peakADC",
        "run {r} MCPPMT 피크 heatmap",
        "{r} heatmap 피크 ADC 그려",
        "run {r} MCPPMT peakADC 히트맵",
        "{r}번 히트맵 peakADC",
        "run {r} heatmap 피크 그려줘",
        "{r}번 heatmap peak 그려줘",
        "run {r} MCPPMT 히트맵 peakADC 보여줘",
        "run {r} heatmap PeakADC",
    ]

    for _ in range(65):
        run = _random_run()
        tmpl = random.choice(heatmap_intadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "IntADC",
                        "type": "heatmap", "modules": ["MCPPMT"]},
             "reason": f"Run {run} heatmap IntADC (MCPPMT) DQM 플랏 생성"},
        ))

    for _ in range(35):
        run = _random_run()
        tmpl = random.choice(heatmap_peakadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "PeakADC",
                        "type": "heatmap", "modules": ["MCPPMT"]},
             "reason": f"Run {run} heatmap PeakADC (MCPPMT) DQM 플랏 생성"},
        ))

    # ── D. single type — specific tower or channel (150) ─────────────────────
    single_intadc_tmpl = [
        # channel name patterns
        "run {r} {ch} intADC 그려줘",
        "run {r} {ch} IntADC 보여줘",
        "{r}번 {ch} intADC 그려",
        "run {r} {ch} intADC plot",
        "{r} {ch} intADC 플랏",
        "run {r} {ch} 적분 ADC 그려줘",
        "{r}번 런 {ch} intADC",
        "run {r} {ch} integral ADC 그려줘",
        "{r} {ch} intADC 보여줘",
        "run {r} {ch} 파형 적분 그려",
        "run {r} {ch} intADC 그래프",
        "{r}번 {ch} 적분 그려줘",
        "run {r} {ch} channel intADC",
        "{r} {ch} 채널 intADC 그려",
        "run {r} {ch}만 intADC 그려줘",
        "run {r} {ch} intADC 확인해줘",
        "{r}번 런 {ch} intADC 확인",
        "run {r} {ch} 그래프 보여줘",
        "{r} {ch} 그려줘",
        "run {r} {ch} 플랏",
    ]
    single_peakadc_tmpl = [
        "run {r} {ch} peakADC 그려줘",
        "run {r} {ch} PeakADC 보여줘",
        "{r}번 {ch} peakADC 그려",
        "run {r} {ch} peak ADC 그려줘",
        "{r} {ch} peakADC 플랏",
        "run {r} {ch} 피크 ADC 그려줘",
        "{r}번 런 {ch} peakADC",
        "run {r} {ch} peakADC plot",
        "{r} {ch} peak 보여줘",
        "run {r} {ch} peakADC 그래프",
        "{r}번 {ch} 피크 그려줘",
        "run {r} {ch}만 peakADC 그려줘",
        "run {r} {ch} peak 확인해줘",
        "{r}번 {ch} PeakADC 확인",
        "run {r} {ch} 피크 플랏",
    ]

    for _ in range(100):
        run = _random_run()
        modules = _random_single_modules()
        ch_str = " ".join(modules)   # e.g. "M1-T1-C" or "M1" or "M1-T3-S M1-T3-C"
        tmpl = random.choice(single_intadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run, ch=ch_str),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "IntADC",
                        "type": "single", "modules": modules},
             "reason": f"Run {run} single IntADC ({ch_str}) DQM 플랏 생성"},
        ))

    for _ in range(50):
        run = _random_run()
        modules = _random_single_modules()
        ch_str = " ".join(modules)
        tmpl = random.choice(single_peakadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run, ch=ch_str),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "PeakADC",
                        "type": "single", "modules": modules},
             "reason": f"Run {run} single PeakADC ({ch_str}) DQM 플랏 생성"},
        ))

    # ── E. NO run number AND NO relative reference → ask run number (100) ──────
    # 런 번호도 없고 방금/이번/지금 같은 상대 참조도 없으면 임의 번호를 지어내지 말고 물어봐야 함.
    # state에 current_run이 있어도, 상대 참조 없이는 절대 state의 런을 끌어오지 않는다.
    ask_run_nomethod_tmpl = [
        "그려줘", "플랏 보여줘", "그래프 그려줘", "DQM 그려줘", "플롯",
        "플랏 그려줘", "그래프 보여줘", "DQM 보여줘", "플랏 뽑아줘",
        "그림 그려줘", "plot 그려줘", "draw", "plot", "DQM plot",
        "히스토그램 그려줘", "그려", "플랏좀", "그래프 좀 그려줘",
        "DQM 플랏 그려줘", "플랏 하나 그려줘",
    ]
    for _ in range(60):
        tmpl = random.choice(ask_run_nomethod_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl,
            {"tool": "none", "message": "어떤 런 번호의 DQM 플롯을 그릴까요?"},
        ))

    # method는 주어졌지만 여전히 run 번호/상대 참조가 없으면 → 먼저 run 번호를 물어봄
    ask_run_withmethod_tmpl = [
        "intADC 그려줘", "적분 ADC 그려줘", "intADC 보여줘", "적분 그려줘",
        "int adc 플랏", "integral 그려줘", "intADC plot", "draw intadc",
        "plot intadc", "적분 플랏 그려줘",
        "peakADC 그려줘", "피크 그려줘", "peakADC 보여줘", "peak ADC 그려줘",
        "피크 ADC 플랏", "peakADC plot", "피크만 그려줘", "draw peakadc",
        "heatmap 그려줘", "히트맵 보여줘", "MCPPMT heatmap 그려줘", "히트맵 그려줘",
    ]
    for _ in range(40):
        tmpl = random.choice(ask_run_withmethod_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl,
            {"tool": "none", "message": "어떤 런 번호의 DQM 플롯을 그릴까요?"},
        ))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  RUN LOG — 100 samples (70 explicit + 30 relative)
# ══════════════════════════════════════════════════════════════════════════════

def gen_run_log() -> List[dict]:
    examples = []

    columns_values = {
        "program": ["EM", "Calib", "HV_EQ", "Pedestal", "LED", "test", "cosmic",
                     "beam_test", "noise", "threshold_scan"],
        "notes": ["pedestal 불안정", "beam unstable", "좋은 데이터", "HV trip 발생",
                   "trigger rate 낮음", "재측정 필요", "테스트 런", "빔 불안정",
                   "DAQ error 발생", "좋은 품질", "detector noise 높음",
                   "energy scan 완료", "position 확인 필요", "HV 안정적",
                   "빔 꺼짐", "spill 불안정", "dark count 높음", "좋음",
                   "beam 정상", "타겟 변경", "trigger 조정 필요"],
        "config": ["standard", "high_gain", "low_threshold", "test_config",
                    "noise_run", "default", "calibration", "debug"],
        "beam_energy": ["1", "2", "3", "5", "10", "20", "30", "40", "60", "80", "100", "120"],
        "beam_type": ["electron", "pion", "muon", "proton", "positron"],
        "trigger_setup": ["standard", "prescale_10", "random", "external",
                          "self_trigger", "cosmic_trigger", "LED_trigger"],
        "hv_drc": ["ON", "OFF", "1400V", "1500V", "1550V", "1600V", "1650V", "1700V"],
        "hv_aux": ["ON", "OFF", "750V", "800V", "850V", "900V", "950V", "1000V"],
    }

    col_korean = {
        "program": ["프로그램", "program", "프로그램을", "프로그램에", "프로그램 란에"],
        "notes": ["노트", "메모", "비고", "노트에", "메모에", "비고에", "메모란에", "notes"],
        "config": ["설정", "config", "설정을", "설정에", "설정란에"],
        "beam_energy": ["빔 에너지", "에너지", "빔에너지를", "에너지를", "beam energy를"],
        "beam_type": ["빔 타입", "타입", "빔타입을", "타입을", "beam type을"],
        "trigger_setup": ["트리거", "trigger", "트리거를", "트리거 설정", "트리거에"],
        "hv_drc": ["HV DRC", "drc", "DRC를", "DRC에", "HV DRC를"],
        "hv_aux": ["HV Aux", "aux", "Aux를", "Aux에", "HV Aux를"],
    }

    explicit_templates = [
        # 추가/쓰기
        "run {r} {col_kr} {val} 추가해줘",  "run {r} {col_kr} {val} 써줘",
        "run {r} {col_kr} {val} 기록해줘",  "run {r} {col_kr} {val} 넣어줘",
        "run {r} {col_kr} {val} 입력해줘",  "run {r} {col_kr} {val} 적어줘",
        "run {r} 로그 {col_kr} {val} 추가", "{r}번 런 로그 {col_kr} {val}",
        "{r} {col_kr} {val} 추가",          "{r}번 {col_kr} {val} 써줘",
        # 수정/변경
        "run {r} {col_kr} {val}로 수정해줘",   "run {r} {col_kr} {val}로 바꿔줘",
        "run {r} {col_kr} {val}로 업데이트",    "run {r} {col_kr} {val}로 변경해줘",
        "run {r} {col_kr} {val}로 고쳐줘",      "{r}번 런 {col_kr} {val}로 바꿔줘",
        "{r} {col_kr} {val}로 수정",            "{r}번 {col_kr} {val}로 변경",
        "{r} 로그 {col_kr} {val}로 업데이트해줘",
        # 축약
        "{r} {col_kr} {val}",  "run {r} {col_kr} {val}",  "{r}번 {col_kr} {val}",
        "{r} 로그 {col_kr} {val}",
        # 존댓말
        "run {r} {col_kr} {val}로 수정해주세요",   "{r}번 {col_kr} {val}로 바꿔주세요",
        "run {r} {col_kr} {val} 기록해주세요",
        # 기타
        "run {r} {col_kr} {val}로 해줘",  "{r} {col_kr} {val}이야",
        "run {r} {col_kr} {val}임",
    ]

    for _ in range(100):
        run = _random_run()
        col = random.choice(list(columns_values.keys()))
        val = random.choice(columns_values[col])
        col_kr = random.choice(col_korean[col])
        template = random.choice(explicit_templates)
        user_input = template.format(r=run, col_kr=col_kr, val=val)
        state = _make_state(random.random() > 0.3)
        decision = {
            "tool": "run_log",
            "params": {"command": "update", "run_num": run, col: val},
            "reason": f"Run {run} {col} 업데이트",
        }
        examples.append(make_example(state, user_input, decision))

    # Relative (50)
    relative_templates = [
        "방금 런 {col_kr} {val} 추가해줘",       "이번 런 {col_kr} {val}로 수정",
        "지금 런 {col_kr} {val} 기록해줘",        "방금 거 {col_kr} {val}로 바꿔줘",
        "이번 데이터 {col_kr} {val}",             "현재 런 {col_kr} {val} 써줘",
        "마지막 런 {col_kr} {val} 추가",          "방금 돌린 런 {col_kr} {val}로 업데이트",
        "이번 거 로그 {col_kr} {val}",            "방금 거 {col_kr} {val} 넣어줘",
        "이번 런 {col_kr} {val}로 변경",          "방금 런 {col_kr} {val}로 해줘",
        "현재 런 {col_kr} {val}로 수정해줘",      "지금 거 {col_kr} {val} 추가",
        "방금 거 {col_kr} {val}",
        # 추가 패턴
        "이번 결과 {col_kr} {val}로 저장",        "방금 돌린 거 {col_kr} {val}로 바꿔",
        "최근 런 {col_kr} {val}",                  "이거 {col_kr} {val}로 해줘",
        "방금 수집한 거 {col_kr} {val} 기록",
    ]
    for _ in range(50):
        col = random.choice(list(columns_values.keys()))
        val = random.choice(columns_values[col])
        col_kr = random.choice(col_korean[col])
        template = random.choice(relative_templates)
        user_input = template.format(col_kr=col_kr, val=val)
        state = _make_state(with_agent=True)
        decision = {
            "tool": "run_log",
            "params": {"command": "update", "run_num": state["current_run"], col: val},
            "reason": f"현재 run {state['current_run']} {col} 업데이트",
        }
        examples.append(make_example(state, user_input, decision))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  RUN LOG READ — 50 samples (run_log의 100에 포함)
# ══════════════════════════════════════════════════════════════════════════════

def gen_run_log_read() -> List[dict]:
    """Log read requests — user wants to VIEW log, not modify."""
    examples = []

    # Explicit run number
    explicit = [
        "run {r} 로그 확인해줘", "run {r} 로그 보여줘", "{r}번 로그 확인",
        "{r} 로그 열어줘", "run {r} 기록 보여줘", "{r}번 런 로그 확인",
        "run {r} 로그 조회", "{r} 로그 읽어줘", "run {r} 로그 내용",
        "{r}번 런 기록 확인해줘", "run {r} log 보여줘", "run {r} 로그 알려줘",
        "{r}번 런 로그 보여줘", "run {r} 정보 확인", "{r} 로그 정보",
        "run {r} 런 정보 보여줘", "{r}번 런 확인해줘", "run {r} 로그 내용 보여줘",
        "{r} 기록 확인", "run {r} 기록 알려줘",
    ]
    for _ in range(40):
        r = _random_run()
        template = random.choice(explicit)
        user_input = template.format(r=r)
        state = _make_state(random.random() > 0.3)
        decision = {
            "tool": "run_log",
            "params": {"command": "read", "run_num": r},
            "reason": f"Run {r} 로그 조회",
        }
        examples.append(make_example(state, user_input, decision))

    # Relative
    relative = [
        "방금 런 로그 확인해줘", "이번 런 로그 보여줘", "지금 런 로그 확인",
        "마지막 런 로그 보여줘", "방금 런 기록 확인", "이번 런 기록 보여줘",
        "현재 런 로그 알려줘", "방금 거 로그 확인", "이번 거 로그 보여줘",
        "방금 돌린 런 로그", "마지막 런 정보 확인", "이번 런 정보 보여줘",
        "방금 런 로그 알려줘", "현재 런 기록 확인해줘",
        # 추가
        "이번 결과 로그 보여줘", "방금 거 기록 확인", "지금 런 정보",
        "최근 런 로그 조회", "이번 런 내용 확인",
    ]
    for _ in range(30):
        template = random.choice(relative)
        state = _make_state(with_agent=True)
        decision = {
            "tool": "run_log",
            "params": {"command": "read", "run_num": state["current_run"]},
            "reason": f"현재 run {state['current_run']} 로그 조회",
        }
        examples.append(make_example(state, template, decision))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  HV READ — 100 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_hv_read() -> List[dict]:
    examples = []

    # ── 전체 채널 조회 템플릿 (channels: "all") ──
    all_ch_templates = [
        # 기본
        "HV 값 읽어줘", "지금 HV 얼마야", "현재 고전압 확인해줘",
        "HV config 보여줘", "전압 값 확인", "HV 상태 알려줘",
        "HV 읽어", "HV 전압 읽어줘",
        # 반말
        "HV 얼마야", "전압 얼마", "HV 확인", "고전압 읽어",
        "HV 좀 봐줘", "전압 확인해봐", "HV 좀 알려줘", "전압 좀",
        "HV 봐봐",
        # 영어 단독 / 혼용
        "read HV config", "HV status", "check HV values", "show HV config",
        "HV read", "read HV", "get HV values", "HV check", "show HV",
        "HV config read", "hv status check", "check hv", "current hv?",
        "what's the hv", "hv 체크", "HV 상태", "show hv status",
        "HV status please", "get hv status", "hv now",
        # 구체적
        "지금 타워 HV 얼마야", "현재 HV 전압 상태", "HV 전압 다 보여줘",
        "CAEN HV 값 읽어줘", "HV 세팅 보여줘",
        "전체 HV 값 확인해줘", "DRC HV 얼마야", "Aux HV 값 알려줘",
        "HV 모듈 전압 확인", "HV 값 전부 보여줘",
        # 질문형
        "HV가 지금 얼마로 되어있어?", "전압이 몇이야?", "현재 HV 설정은?",
        "HV 값이 뭐야?", "전압 세팅 뭐로 되어있어?",
        "지금 전압 세팅 알려줘", "HV configuration 보여줘",
        "전압 config 읽어줘",
        "HV 뭐로 세팅되어있어?", "전압값 몇이지?", "고전압이 얼마야?",
        "HV 전압 몇 볼트야?", "지금 HV config가 뭐야?",
        # 추가
        "채널별 HV 알려줘", "DRC 채널 HV 확인", "지금 세팅된 전압 다 보여줘",
        "HV 모두 확인", "전체 채널 전압 상태", "HV 지금 어떻게 돼있어",
    ]

    for _ in range(120):
        template = random.choice(all_ch_templates)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, template, {
            "tool": "hv_read",
            "params": {"command": "status"},
            "reason": "현재 HV 상태 읽기",
        }))

    # ── 특정 채널 지정 조회 (TRIG, MCP, DRC 포함) ──
    named_ch_templates = [
        "{ch} HV 확인해줘", "{ch} 전압 얼마야", "{ch} HV 읽어줘",
        "{ch} 상태 확인", "{ch} HV 보여줘", "{ch} 전압 확인",
        "{ch} HV 얼마로 설정되어있어",
        "{ch} HV 지금 얼마야", "{ch} voltage?", "{ch} hv?",
        "{ch} HV 체크", "check {ch} HV", "{ch} 전압 좀 봐줘",
        "{ch} HV status", "show {ch} HV", "{ch} hv 얼마야",
        "{ch} 지금 전압 얼마야", "{ch} HV 값 알려줘",
    ]
    named_channels = [
        "TRIG1", "TRIG2", "MCP-S", "MCP-C",
        "M1T1C", "M1T1S", "M2T2C", "M3T3S",
        "M4T1C", "M5T2S", "M6T3C", "M7T4S", "M8T1C", "M9T4S",
    ]
    for _ in range(30):
        ch = random.choice(named_channels)
        tmpl = random.choice(named_ch_templates)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch), {
            "tool": "hv_read",
            "params": {"command": "status", "channels": [ch]},
            "reason": f"{ch} 채널 HV 상태 읽기",
        }))

    # ── 부분 채널 상태 조회: 짝수/홀수/S채널/C채널 (80 samples) ──
    # hv_write와 동일한 <ch_spec> 사용 (even/odd, S만→ALL_S 목록, C만→ALL_C 목록)
    _all_s = "S"
    _all_c = "C"
    subset_read_tmpl = [
        "{grp} HV 확인해줘", "{grp} 전압 얼마야", "{grp} HV 상태",
        "{grp} 전압 확인", "{grp} HV 읽어줘", "{grp} hv status",
        "{grp} 상태 확인", "{grp} HV 보여줘", "{grp} 전압 상태 알려줘",
        "{grp} 지금 전압 얼마야", "{grp} hv 확인", "{grp} 전압 읽어줘",
    ]
    subset_read_specs = [
        (["짝수 채널", "짝수 ch", "even 채널", "짝수만", "even channel", "짝수 번호 채널"], "even", "짝수 채널"),
        (["홀수 채널", "홀수 ch", "odd 채널", "홀수만", "odd channel", "홀수 번호 채널"], "odd", "홀수 채널"),
        (["S채널", "S만", "S쪽", "S side", "scintillator 채널", "모든 S채널", "S 채널"], _all_s, "S채널"),
        (["C채널", "C만", "C쪽", "C side", "cherenkov 채널", "체렌코프 채널", "모든 C채널", "C 채널"], _all_c, "C채널"),
    ]
    for _ in range(80):
        names, ch_spec, reason_prefix = random.choice(subset_read_specs)
        grp = random.choice(names)
        tmpl = random.choice(subset_read_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(grp=grp), {
            "tool": "hv_read",
            "params": {"command": "status", "channels": ch_spec},
            "reason": f"{reason_prefix} HV 상태 읽기",
        }))

    # ── 모듈 단위 상태 조회 (M3만 등, 40 samples) ──
    module_read_tmpl = [
        "{m} HV 확인해줘", "{m} 전압 얼마야", "{m} HV 상태", "{m} HV 읽어줘",
        "{m}만 HV 확인", "{m} 채널 전압 확인", "{m} hv status", "{m} 전압 상태 보여줘",
        "{m} 모듈 HV 확인", "{m}만 상태 확인", "{m} hv 보여줘", "{m} 전압 확인해줘",
    ]
    for _ in range(40):
        m = f"M{random.randint(1, 9)}"
        tmpl = random.choice(module_read_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(m=m), {
            "tool": "hv_read",
            "params": {"command": "status", "channels": m},
            "reason": f"{m} 모듈 HV 상태 읽기",
        }))

    # ── 슬롯 단위 상태 조회 (20 samples) ──
    slot_read_tmpl = [
        "슬롯 {s} HV 확인해줘", "슬롯 {s} 전압 얼마야", "slot {s} HV 상태",
        "슬롯 {s} HV 읽어줘", "슬롯 {s} 채널 상태 확인", "slot {s} 전압 확인",
    ]
    for _ in range(20):
        s = random.choice([11, 12])
        tmpl = random.choice(slot_read_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(s=s), {
            "tool": "hv_read",
            "params": {"command": "status", "channels": f"slot:{s}"},
            "reason": f"슬롯 {s} HV 상태 읽기",
        }))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  HV WRITE — 전압 변경, ON/OFF (BrainAgent가 확인 후 실행)
#  ※ BrainAgent의 handle_request가 실행 전 팝업에서 사용자 확인을 받음
# ══════════════════════════════════════════════════════════════════════════════

def _parse_voltage(v_str: str) -> float:
    """Strip 'V' suffix and convert to float."""
    return float(v_str.rstrip("Vv").strip())


def gen_hv_write() -> List[dict]:
    """HV write requests — voltage change, on/off. BrainAgent will confirm before executing."""
    examples = []

    channels = [f"M{m}T{t}{s}" for m in range(1, 10) for t in range(1, 5) for s in ("C", "S")]
    # 800~1800V 범위 포함, V 있는 것과 없는 것 혼용
    voltages_with_v  = ["100V", "200V", "500V", "800V", "850V", "900V", "1000V",
                        "1200V", "1400V", "1450V", "1500V", "1550V", "1600V",
                        "1650V", "1700V", "1750V", "1800V"]
    voltages_no_v    = ["100", "200", "500", "800", "850", "900", "1000",
                        "1200", "1400", "1450", "1500", "1550", "1600",
                        "1650", "1700", "1750", "1800"]

    def _rand_volt():
        """Return (display_string, numeric_value) with or without V suffix."""
        if random.random() < 0.5:
            s = random.choice(voltages_with_v)
        else:
            s = random.choice(voltages_no_v)
        return s, _parse_voltage(s)

    # ── 단일 채널 전압 변경 (150 samples) ──
    # Critically: include "전압" keyword between channel and value
    templates_ch_voltage = [
        # 동사 있음
        "{ch} {v}로 수정", "{ch} {v}로 바꿔줘", "{ch} 전압 {v}로 변경",
        "{ch}를 {v}로 설정", "{ch} {v}로 올려줘", "{ch} {v}로 내려줘",
        "HV {ch} {v}로 수정", "HV {ch} {v}로 바꿔",
        "{ch} 전압을 {v}로 해줘", "{ch} {v} 설정해줘", "{ch} {v}로 맞춰줘",
        "HV {ch} {v} 세팅", "{ch} 전압 {v}로 조정", "{ch} {v} 적용해줘",
        "{ch} 전압 {v}로 바꿔줘", "{ch} 전압을 {v}로 변경해줘",
        "{ch} 전압 {v}로 설정해줘", "{ch} 전압 {v}로 맞춰줘",
        "{ch} HV {v}로 올려줘", "{ch} 고전압 {v}로 설정",
        "{ch} 전압 {v}로 올려줘", "{ch} 전압 {v}로 내려줘",
        "{ch} {v}로 인가해줘", "{ch} {v}로 해줘",
        # 동사 없음 (bare) — 가장 오해 잦은 패턴
        "{ch} 전압 {v}으로", "{ch} 전압 {v}로",
        "{ch} 전압 {v}", "{ch} {v}",
        "{ch} 전압 {v}으로 해", "{ch} {v}으로 해",
        # 영어 혼용 (실험실 오퍼레이터 스타일)
        "{ch} set to {v}", "{ch} voltage {v}",
        "set {ch} to {v}", "{ch} {v} please",
        "{ch} to {v}", "set {ch} {v}", "{ch} hv {v}",
        "{ch} voltage to {v}", "hv {ch} to {v}",
        "{ch} HV up to {v}", "change {ch} to {v}",
        "update {ch} hv to {v}", "{ch} at {v}",
        # 한영 혼용 (실험실 실무 표현)
        "{ch}를 {v}로", "{ch} {v}로 올려", "{ch} {v}로 내려",
        "T1C를 {v}로".replace("T1C", "{ch}"), "{ch} HV {v}로",
        "{ch} hv {v}로 바꿔줘", "{ch} 전압 올려줘 {v}로",
    ]
    for _ in range(150):
        ch = random.choice(channels)
        v_str, v_num = _rand_volt()
        template = random.choice(templates_ch_voltage)
        user_input = template.format(ch=ch, v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": [ch], "voltage": v_num},
            "reason": f"{ch} 채널 전압을 {v_num}V로 변경",
        }))

    # ── 여러 채널 동시 변경 (30 samples) ──
    templates_multi = [
        "{ch1}, {ch2} {v}로 수정", "{ch1}와 {ch2} 전압 {v}로",
        "{ch1} {ch2} {v}로 설정", "{ch1},{ch2} {v}로 바꿔줘",
        "{ch1} {ch2} 전압 {v}로 변경", "{ch1}, {ch2} 전압 {v}",
        "{ch1},{ch2} 전압 {v}로 맞춰줘",
        "{ch1} 및 {ch2} {v}로 수정", "{ch1}와 {ch2} {v}로 바꿔",
        "{ch1} {ch2} 전압 {v}으로",
    ]
    for _ in range(30):
        ch1, ch2 = random.sample(channels, 2)
        v_str, v_num = _rand_volt()
        template = random.choice(templates_multi)
        user_input = template.format(ch1=ch1, ch2=ch2, v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": [ch1, ch2], "voltage": v_num},
            "reason": f"{ch1}, {ch2} 채널 전압을 {v_num}V로 변경",
        }))

    # ── 전체 채널 변경 — ALL 키워드 필수 (30 samples) ──
    # Must include "전체", "모든", "전 채널", "HV" with explicit all-scope keyword
    templates_all = [
        "HV 전체 {v}로 수정", "모든 채널 {v}로 변경", "전체 HV {v}로 설정",
        "전 채널 {v}로 바꿔줘", "전체 전압 {v}로 맞춰줘",
        "모든 HV {v}로", "전체 채널 전압 {v}로 수정",
        "HV 전체 {v}로 바꿔줘", "전 채널 전압 {v}로",
        "모든 채널 전압 {v}로 변경해줘", "HV 전부 {v}로 설정",
        "전체 HV 전압 {v}로 변경", "모든 타워 HV {v}로 수정",
        "전채널 {v}로 바꿔", "전체 고전압 {v}로",
    ]
    for _ in range(30):
        v_str, v_num = _rand_volt()
        template = random.choice(templates_all)
        user_input = template.format(v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": "all", "voltage": v_num},
            "reason": f"모든 채널 전압을 {v_num}V로 변경",
        }))

    # ── hv_read: 특정 채널 지정 읽기 (30 samples) ──
    named_ch_read_tmpl = [
        "{ch} HV 확인해줘", "{ch} 전압 얼마야", "{ch} HV 읽어줘",
        "{ch} 상태 확인", "{ch} HV 보여줘", "{ch} 전압 확인",
        "{ch} HV 얼마로 설정되어있어", "{ch} 고전압 얼마야",
    ]
    named_channels_for_read = ["TRIG1", "TRIG2", "MCP-S", "MCP-C",
                                "M1T1C", "M1T1S", "M5T2C", "M5T2S", "M9T4C", "M9T4S"]
    for _ in range(30):
        ch = random.choice(named_channels_for_read)
        tmpl = random.choice(named_ch_read_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch), {
            "tool": "hv_read",
            "params": {"command": "status", "channels": [ch]},
            "reason": f"{ch} 채널 HV 상태 읽기",
        }))

    # ── ON/OFF 전체 채널 (30 samples) ──
    templates_on = [
        "HV 켜줘", "HV 전원 켜", "HV on", "HV turn on", "전체 HV 켜줘",
        "HV 다시 켜줘", "모든 HV 켜", "HV 전체 켜", "HV 전원 올려줘", "HV 올려",
        "turn on HV", "HV all on", "전체 HV 전원 켜줘",
        "HV 전체 on 해줘", "모든 채널 HV 켜줘",
    ]
    for _ in range(15):
        template = random.choice(templates_on)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, template, {
            "tool": "hv_write",
            "params": {"command": "on", "channels": "all"},
            "reason": "모든 HV 채널 켜기",
        }))

    templates_off = [
        "HV 꺼줘", "HV 전원 꺼", "HV off", "HV turn off", "전체 HV 꺼줘",
        "HV 다 꺼줘", "모든 HV 꺼", "HV 전체 꺼", "HV 전원 내려줘", "HV 내려",
        "turn off HV", "HV all off", "전체 HV 전원 꺼줘",
        "HV 전체 off 해줘", "모든 채널 HV 꺼줘",
        # "전압 꺼줘" 형태 — command: off (NOT voltage: 0)
        "전압 꺼줘", "전체 전압 꺼줘", "모든 채널 전압 꺼", "전압 다 꺼줘",
        "전압 내려줘", "전체 전압 내려줘", "전압 전체 꺼줘", "전압 off",
        "전압 끄기", "모든 전압 꺼줘", "전체 전압 off", "전압 전체 off",
    ]
    for _ in range(25):
        template = random.choice(templates_off)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, template, {
            "tool": "hv_write",
            "params": {"command": "off", "channels": "all"},
            "reason": "모든 HV 채널 끄기",
        }))

    # ── 단일 채널 ON/OFF (30 samples) ──
    named_on_tmpl = [
        "{ch} 켜줘", "{ch} HV on", "{ch} 전원 켜줘", "{ch} 켜",
        "{ch} on 해줘", "{ch} HV 켜줘", "turn on {ch}",
        "{ch} 켜주세요", "{ch} HV turn on",
    ]
    named_off_tmpl = [
        "{ch} 꺼줘", "{ch} HV off", "{ch} 전원 꺼줘", "{ch} 꺼",
        "{ch} off 해줘", "{ch} HV 꺼줘", "turn off {ch}",
        "{ch} 꺼주세요", "{ch} HV turn off",
    ]
    single_onoff_channels = [
        "TRIG1", "TRIG2", "MCP-S", "MCP-C",
        "M1T1C", "M2T2S", "M5T3C", "M7T4S",
    ]
    for _ in range(15):
        ch = random.choice(single_onoff_channels)
        tmpl = random.choice(named_on_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch), {
            "tool": "hv_write",
            "params": {"command": "on", "channels": [ch]},
            "reason": f"{ch} 켜기",
        }))
    for _ in range(15):
        ch = random.choice(single_onoff_channels)
        tmpl = random.choice(named_off_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch), {
            "tool": "hv_write",
            "params": {"command": "off", "channels": [ch]},
            "reason": f"{ch} 끄기",
        }))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  HV WRITE ADVANCED — S만/C만/짝수/홀수/ch범위/Trig/MCP 채널 그룹 (240 samples)
# ══════════════════════════════════════════════════════════════════════════════

ALL_S = "S"
ALL_C = "C"
EVEN_CH = "even"
ODD_CH  = "odd"


def gen_hv_write_advanced() -> List[dict]:
    """S만/C만/짝수/홀수/ch범위/TRIG/MCP 채널 그룹 제어 시나리오."""
    examples = []

    def _rand_volt():
        v = random.choice([800, 900, 1000, 1200, 1400, 1500, 1550, 1600, 1650, 1700])
        return v, float(v)

    # ── A. S채널 전압 변경 (60 samples) ──────────────────────────────────────
    s_only_tmpl = [
        "S채널만 {v}로 설정해줘", "S채널 전압 {v}로 바꿔줘", "S만 {v}로 수정", "S 쪽만 {v}로 변경", "S채널들 전압 {v}로",
        "타워 S채널 전압 {v}로 바꿔줘", "S만 {v}V로 올려줘", "S채널만 {v}V 설정",
        "모든 S채널 {v}로 수정", "S 쪽 전압 {v}로 설정해줘", "S 채널 전체 {v}로", "S채널 모두 {v}로 해줘",
        "scintillator 채널 {v}로 설정", "S 전압 {v}로 올려줘",
        "{v} S side", "S채널 {v}V로", "S만 {v} 설정",
        # "만" 없는 형태 — 모델 혼동 방지
        "S채널 {v}", "S채널 hv {v}", "S채널 {v}V", "S채널 {v}로",
        "S 채널 {v}로 바꿔줘", "S채널 전압 {v}", "S채널 {v}볼트",
        "S채널 hv {v}로 설정", "S side hv {v}",
    ]
    for _ in range(60):
        v_str, v_num = _rand_volt()
        tmpl = random.choice(s_only_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": ALL_S, "voltage": v_num},
            "reason": f"S채널 전체 전압을 {v_num}V로 변경",
        }))

    # ── B. C채널만 전압 변경 (50 samples) ─────────────────────────────────────
    c_only_tmpl = [
        "C채널만 {v}로 설정해줘", "C채널 전압 {v}로 바꿔줘", "C만 {v}로 수정", "C 쪽만 {v}로 변경", "C채널들 전압 {v}로",
        "타워 C채널 전압 {v}로 바꿔줘", "C만 {v}V로 올려줘", "C채널만 {v}V 설정",
        "모든 C채널 {v}로 수정", "C 쪽 전압 {v}로 설정해줘", "C 채널 전체 {v}로", "C채널 모두 {v}로 해줘",
        "cherenkov 채널 {v}로 설정", "체렌코프만 {v}로 바꿔줘",
        "C채널 {v}V로", "C만 {v} 설정", "C-side 전압 {v}로 변경해줘",
    ]
    for _ in range(50):
        v_str, v_num = _rand_volt()
        tmpl = random.choice(c_only_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": ALL_C, "voltage": v_num},
            "reason": f"C채널 전체 전압을 {v_num}V로 변경",
        }))

    # ── C. TRIG1/TRIG2 (40 samples) ───────────────────────────────────────────
    trig_both_tmpl = [
        "트리거 PMT {v}로 설정해줘", "TRIG 전압 {v}로 바꿔줘", "트리거 {v}로",
        "TRIG1 TRIG2 {v}로 수정", "트리거 채널 전압 {v}로", "TRIG {v}로 설정",
        "트리거 둘 다 {v}로 바꿔줘", "TRIG1이랑 TRIG2 {v}로",
        "trigger PMT {v}로 설정해줘", "트리거 HV {v}로 변경",
    ]
    trig_single_tmpl = [
        ("{ch} {v}로 설정해줘", 1), ("{ch} 전압 {v}로 바꿔", 1),
        ("{ch} HV {v}로 수정", 1), ("{ch} {v}V로 올려줘", 1),
        ("{ch} {v}로 맞춰줘", 1),
    ]
    for _ in range(20):
        v_str, v_num = _rand_volt()
        tmpl = random.choice(trig_both_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": ["TRIG1", "TRIG2"], "voltage": v_num},
            "reason": f"TRIG1/TRIG2 전압을 {v_num}V로 변경",
        }))
    for _ in range(20):
        v_str, v_num = _rand_volt()
        ch = random.choice(["TRIG1", "TRIG2"])
        tmpl, _ = random.choice(trig_single_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch, v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": [ch], "voltage": v_num},
            "reason": f"{ch} 전압을 {v_num}V로 변경",
        }))

    # ── D. MCP-S / MCP-C (40 samples) ─────────────────────────────────────────
    mcp_both_tmpl = [
        "MCP {v}로 설정해줘", "MCP 전압 {v}로 바꿔줘", "MCP PMT {v}로",
        "MCP-S MCP-C {v}로 수정", "MCP 둘 다 {v}로 바꿔줘",
        "MCP 채널 {v}로 설정", "MCP HV {v}로 변경해줘",
        "MCP S랑 C {v}로", "MCP 전압 {v}",
    ]
    mcp_single_tmpl = [
        "{ch} {v}로 설정해줘", "{ch} 전압 {v}로 바꿔",
        "{ch} HV {v}로 수정", "{ch} {v}V로 올려줘",
        "{ch} {v}로 맞춰줘",
    ]
    for _ in range(20):
        v_str, v_num = _rand_volt()
        tmpl = random.choice(mcp_both_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": ["MCP-S", "MCP-C"], "voltage": v_num},
            "reason": f"MCP-S/MCP-C 전압을 {v_num}V로 변경",
        }))
    for _ in range(20):
        v_str, v_num = _rand_volt()
        ch = random.choice(["MCP-S", "MCP-C"])
        tmpl = random.choice(mcp_single_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch, v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": [ch], "voltage": v_num},
            "reason": f"{ch} 전압을 {v_num}V로 변경",
        }))

    # ── E. 짝수 채널 (35 samples) ──────────────────────────────────────────────
    even_tmpl = [
        "짝수 채널 {v}로 설정해줘", "짝수 ch {v}로 바꿔줘", "짝수 채널만 {v}로",
        "even 채널 {v}로 수정", "짝수 채널 전압 {v}로 변경", "짝수만 {v}로 설정",
        "짝수 ch 전압 {v}로", "even channel {v}로 설정해줘",
        "채널 짝수 {v}로 바꿔줘", "짝수 번호 채널 {v}로",
        # "만" 없는 형태
        "짝수 채널 {v}", "짝수 채널 hv {v}", "짝수 ch {v}",
        "even channel hv {v}", "짝수 채널 {v}V", "짝수 채널 {v}로",
    ]
    for _ in range(35):
        v_str, v_num = _rand_volt()
        tmpl = random.choice(even_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": EVEN_CH, "voltage": v_num},
            "reason": f"짝수 채널 전압을 {v_num}V로 변경",
        }))

    # ── F. 홀수 채널 (25 samples) ──────────────────────────────────────────────
    odd_tmpl = [
        "홀수 채널 {v}로 설정해줘", "홀수 ch {v}로 바꿔줘", "홀수 채널만 {v}로",
        "odd 채널 {v}로 수정", "홀수 채널 전압 {v}로 변경", "홀수만 {v}로 설정",
        "홀수 ch 전압 {v}로", "odd channel {v}로 설정해줘",
        "채널 홀수 {v}로 바꿔줘", "홀수 번호 채널 {v}로",
    ]
    for _ in range(25):
        v_str, v_num = _rand_volt()
        tmpl = random.choice(odd_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": ODD_CH, "voltage": v_num},
            "reason": f"홀수 채널 전압을 {v_num}V로 변경",
        }))

    # ── G-0. ch 번호 범위 "N-M" (40 samples) ─────────────────────────────────
    # "ch 0-8", "채널 2-10", "ch 0에서 8까지" → channels: "N-M"
    ch_range_tmpl = [
        "ch {lo}-{hi} {v}로 설정해줘", "채널 {lo}-{hi} 전압 {v}로 바꿔줘",
        "ch {lo}에서 {hi}까지 {v}로", "ch{lo}-{hi} {v}V로 올려줘",
        "채널 {lo}~{hi} {v}로 설정", "ch {lo}-{hi} hv {v}로",
        "채널 {lo}번부터 {hi}번까지 {v}로 바꿔줘", "ch {lo}-{hi} {v}V",
        "ch {lo}-{hi} 전압 {v}", "{lo}번 ch부터 {hi}번 ch까지 {v}로",
        "ch {lo}-{hi} 전압 {v}로 변경", "채널 {lo} to {hi} {v}로",
    ]
    ch_range_off_tmpl = [
        "ch {lo}-{hi} 꺼줘", "채널 {lo}-{hi} off해줘", "ch {lo}에서 {hi} 끄기",
        "ch{lo}-{hi} HV off", "채널 {lo}-{hi} 전원 꺼줘",
    ]
    _possible_ranges = [(0, 8), (0, 10), (1, 9), (2, 12), (0, 4), (4, 8), (6, 12),
                        (0, 16), (1, 5), (8, 16), (0, 6), (10, 20)]
    for _ in range(30):
        lo, hi = random.choice(_possible_ranges)
        v_str, v_num = _rand_volt()
        tmpl = random.choice(ch_range_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(lo=lo, hi=hi, v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": f"{lo}-{hi}", "voltage": v_num},
            "reason": f"ch {lo}-{hi} 전압을 {v_num}V로 변경",
        }))
    for _ in range(10):
        lo, hi = random.choice(_possible_ranges)
        tmpl = random.choice(ch_range_off_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(lo=lo, hi=hi), {
            "tool": "hv_write",
            "params": {"command": "off", "channels": f"{lo}-{hi}"},
            "reason": f"ch {lo}-{hi} 끄기",
        }))

    # ── G-1. ch 번호 목록 "N,M,K" (40 samples) ───────────────────────────────
    # "ch 1,2,5,6", "채널 0,2,4" → channels: "N,M,K"
    ch_list_tmpl = [
        "ch {chs} {v}로 설정해줘", "채널 {chs} 전압 {v}로 바꿔줘",
        "ch {chs} {v}V로 올려줘", "ch {chs} hv {v}로",
        "채널 {chs}번 {v}로 설정", "ch {chs} 전압 {v}",
        "ch {chs} {v}로 변경", "채널 {chs} {v}V 설정해줘",
        "{chs}번 채널 전압 {v}로", "ch {chs} {v}로 맞춰줘",
    ]
    ch_list_off_tmpl = [
        "ch {chs} 꺼줘", "채널 {chs} off", "ch {chs} HV 끄기",
        "채널 {chs}번 끄기", "ch {chs} 전원 꺼줘",
    ]
    def _rand_ch_list():
        pool = list(range(0, 23))
        k = random.choice([2, 3, 4, 5])
        chs = sorted(random.sample(pool, k))
        return ",".join(map(str, chs))
    for _ in range(30):
        chs = _rand_ch_list()
        v_str, v_num = _rand_volt()
        tmpl = random.choice(ch_list_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(chs=chs.replace(",", ", "), v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": chs, "voltage": v_num},
            "reason": f"ch {chs} 전압을 {v_num}V로 변경",
        }))
    for _ in range(10):
        chs = _rand_ch_list()
        tmpl = random.choice(ch_list_off_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(chs=chs.replace(",", ", ")), {
            "tool": "hv_write",
            "params": {"command": "off", "channels": chs},
            "reason": f"ch {chs} 끄기",
        }))

    # ── G. 슬롯 지정 (30 samples) ─────────────────────────────────────────────
    # ch 번호만(슬롯 없이) 지정하는 표현은 ambiguous → "slot:N" 형식 사용
    slot_specs = [
        ("slot:11",      "슬롯 11"),
        ("slot:12",      "슬롯 12"),
        ("slot:11:even", "슬롯 11 짝수 채널"),
        ("slot:11:odd",  "슬롯 11 홀수 채널"),
        ("slot:12:even", "슬롯 12 짝수 채널"),
        ("slot:12:odd",  "슬롯 12 홀수 채널"),
    ]
    slot_tmpl = [
        "{label} {v}로 설정해줘", "{label} 전압 {v}로 바꿔줘",
        "{label} {v}로 수정", "{label} 전압 {v}로 변경해줘",
        "{label} HV {v}로", "{label} {v}V로 설정",
        "{label} {v}로 맞춰줘", "{label} {v}로 올려줘",
    ]
    for _ in range(30):
        ch_spec, label = random.choice(slot_specs)
        v_str, v_num = _rand_volt()
        tmpl = random.choice(slot_tmpl)
        user_input = tmpl.format(label=label, v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": ch_spec, "voltage": v_num},
            "reason": f"{label} 전압을 {v_num}V로 변경",
        }))

    # ── H. hv_read: 그룹 채널 상태 확인 (30 samples) ──────────────────────────
    group_read_tmpl = [
        "{grp_name} HV 확인해줘", "{grp_name} 전압 얼마야", "{grp_name} HV 상태",
        "{grp_name} 전압 확인", "{grp_name} HV 읽어줘",
    ]
    group_specs = [
        ("S채널", ALL_S),
        ("C채널", ALL_C),
        ("TRIG", ["TRIG1", "TRIG2"]),
        ("MCP", ["MCP-S", "MCP-C"]),
        ("TRIG1", ["TRIG1"]),
        ("TRIG2", ["TRIG2"]),
        ("MCP-S", ["MCP-S"]),
        ("MCP-C", ["MCP-C"]),
    ]
    for _ in range(30):
        grp_name, ch_list = random.choice(group_specs)
        tmpl = random.choice(group_read_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(grp_name=grp_name), {
            "tool": "hv_read",
            "params": {"command": "status", "channels": ch_list},
            "reason": f"{grp_name} 채널 HV 상태 읽기",
        }))

    # ── I. ON/OFF — 특정 채널 그룹 (20 samples) ───────────────────────────────
    named_on_tmpl = [
        "{ch} 켜줘", "{ch} HV on", "{ch} 전원 켜줘", "{ch} 켜",
        "{ch} on 해줘", "{ch} HV 켜줘",
    ]
    named_off_tmpl = [
        "{ch} 꺼줘", "{ch} HV off", "{ch} 전원 꺼줘", "{ch} 꺼",
        "{ch} off 해줘", "{ch} HV 꺼줘",
    ]
    named_onoff_channels = ["TRIG1", "TRIG2", "MCP-S", "MCP-C"]
    for _ in range(10):
        ch = random.choice(named_onoff_channels)
        tmpl = random.choice(named_on_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch), {
            "tool": "hv_write",
            "params": {"command": "on", "channels": [ch]},
            "reason": f"{ch} 켜기",
        }))
    for _ in range(10):
        ch = random.choice(named_onoff_channels)
        tmpl = random.choice(named_off_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(ch=ch), {
            "tool": "hv_write",
            "params": {"command": "off", "channels": [ch]},
            "reason": f"{ch} 끄기",
        }))

    # ── J. 그룹별 ON/OFF 종합 (각 그룹 × on/off, 총 ~140 samples) ────────────
    # 켜줘/꺼줘/전압끄기 모두 동일 그룹에서 → 꺼줘는 반드시 command: "off"
    on_tmpl = [
        "{grp} 켜줘", "{grp} 전원 켜줘", "{grp} HV on", "{grp} on 해줘",
        "{grp} 켜주세요", "turn on {grp}", "{grp} 전원 올려줘",
    ]
    off_tmpl = [
        "{grp} 꺼줘", "{grp} 전원 꺼줘", "{grp} HV off", "{grp} off 해줘",
        "{grp} 꺼주세요", "turn off {grp}", "{grp} 전압 꺼줘",
        "{grp} 끄기", "{grp} 전원 내려줘", "{grp} 전압 내려",
    ]

    group_onoff_specs = [
        # (채널 그룹 파라미터, 사용자 표현 목록, 이유 prefix)
        ("even",            ["짝수 채널", "짝수 ch", "even 채널", "짝수만"],         "짝수 채널"),
        ("odd",             ["홀수 채널", "홀수 ch", "odd 채널", "홀수만"],          "홀수 채널"),
        (ALL_S,             ["S채널", "S만", "S쪽", "스캔티 채널", "S side"],       "S채널"),
        (ALL_C,             ["C채널", "C만", "C쪽", "체렌코프 채널", "C side"],     "C채널"),
        (["MCP-S","MCP-C"], ["MCP", "MCP 채널", "MCP PMT"],                        "MCP"),
        (["TRIG1","TRIG2"], ["트리거", "TRIG", "트리거 채널", "trigger"],           "TRIG"),
    ]

    for ch_param, grp_names, reason_prefix in group_onoff_specs:
        for _ in range(10):
            grp = random.choice(grp_names)
            tmpl = random.choice(on_tmpl)
            state = _make_state(random.random() > 0.3)
            examples.append(make_example(state, tmpl.format(grp=grp), {
                "tool": "hv_write",
                "params": {"command": "on", "channels": ch_param},
                "reason": f"{reason_prefix} 켜기",
            }))
        for _ in range(12):
            grp = random.choice(grp_names)
            tmpl = random.choice(off_tmpl)
            state = _make_state(random.random() > 0.3)
            examples.append(make_example(state, tmpl.format(grp=grp), {
                "tool": "hv_write",
                "params": {"command": "off", "channels": ch_param},
                "reason": f"{reason_prefix} 끄기",
            }))

    # ch 범위 ON 추가 (G-0에 off만 있었음)
    ch_range_on_tmpl = [
        "ch {lo}-{hi} 켜줘", "채널 {lo}-{hi} on", "ch {lo}에서 {hi} 전원 켜줘",
        "ch{lo}-{hi} HV on", "채널 {lo}-{hi} 켜주세요",
    ]
    for _ in range(10):
        lo, hi = random.choice(_possible_ranges)
        tmpl = random.choice(ch_range_on_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(lo=lo, hi=hi), {
            "tool": "hv_write",
            "params": {"command": "on", "channels": f"{lo}-{hi}"},
            "reason": f"ch {lo}-{hi} 켜기",
        }))

    # ch 목록 ON 추가 (G-1에 off만 있었음)
    ch_list_on_tmpl = [
        "ch {chs} 켜줘", "채널 {chs} on", "ch {chs} HV on",
        "채널 {chs}번 켜주세요", "ch {chs} 전원 켜줘",
    ]
    for _ in range(10):
        chs = _rand_ch_list()
        tmpl = random.choice(ch_list_on_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(chs=chs.replace(",", ", ")), {
            "tool": "hv_write",
            "params": {"command": "on", "channels": chs},
            "reason": f"ch {chs} 켜기",
        }))

    # ── K. 타워 단위 T1~T4 (40 samples) ──────────────────────────────────────
    tower_v_tmpl = [
        "{t} {v}로 설정해줘", "{t} 전압 {v}로 바꿔줘", "타워 {t} {v}로",
        "{t} 채널 {v}로 변경", "{t} HV {v}로", "{t} {v}V로 올려줘",
        "모든 모듈 {t} {v}로", "{t} 타워 전압 {v}로 설정",
    ]
    tower_read_tmpl = [
        "{t} HV 확인해줘", "{t} 전압 얼마야", "{t} HV 상태", "{t} 타워 전압 확인",
    ]
    for _ in range(24):
        t = f"T{random.randint(1, 4)}"
        v_str, v_num = _rand_volt()
        tmpl = random.choice(tower_v_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(t=t, v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": t, "voltage": v_num},
            "reason": f"{t} 타워 전체 전압을 {v_num}V로 변경",
        }))
    for _ in range(16):
        t = f"T{random.randint(1, 4)}"
        tmpl = random.choice(tower_read_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(t=t), {
            "tool": "hv_read",
            "params": {"command": "status", "channels": t},
            "reason": f"{t} 타워 HV 상태 읽기",
        }))

    return examples



# ══════════════════════════════════════════════════════════════════════════════
#  HV WRITE MODULE — 모듈 단위 제어 (80 samples)
#  "M5 1500V", "M3 M7 꺼줘" 등
# ══════════════════════════════════════════════════════════════════════════════

def gen_hv_write_module() -> List[dict]:
    """모듈 단위(M{1-9}) HV 제어 — 전압 변경 및 ON/OFF."""
    examples = []

    def _rand_volt():
        v = random.choice([800, 900, 1000, 1200, 1400, 1500, 1550, 1600, 1650, 1700])
        return v, float(v)

    # ── A. 단일 모듈 전압 변경 (40 samples) ──────────────────────────────────
    single_mod_tmpl = [
        "{m} {v}로 설정해줘", "{m} HV {v}로 바꿔줘", "{m} 전압 {v}로 변경",
        "{m} {v}V로 올려줘", "{m} {v}로 수정해줘", "{m} 모듈 {v}로",
        "{m} 전압 {v}로 해줘", "{m} HV {v} 세팅", "{m} {v}V 설정",
        "{m} 고전압 {v}로 설정", "{m} 전압 {v}", "{m} {v}로",
        "모듈 {m} {v}V로 바꿔줘", "module {m} hv {v}",
        "{m} set to {v}", "{m} voltage {v}",
        "{m} {v}로 올려", "{m} {v} please",
        "{m} HV {v}로 올려줘", "{m} 전압을 {v}로",
    ]
    for _ in range(40):
        m = f"M{random.randint(1, 9)}"
        v_str, v_num = _rand_volt()
        tmpl = random.choice(single_mod_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(m=m, v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": m, "voltage": v_num},
            "reason": f"{m} 모듈 전체 채널 전압을 {v_num}V로 변경",
        }))

    # ── B. 복수 모듈 전압 변경 (20 samples) ──────────────────────────────────
    multi_mod_tmpl = [
        "{m1} {m2} {v}로 설정해줘", "{m1}랑 {m2} {v}로 바꿔줘",
        "{m1}, {m2} 전압 {v}로", "{m1} {m2} HV {v}로 수정",
        "{m1}하고 {m2} {v}V로 올려줘", "{m1} and {m2} {v}로 설정",
        "{m1}, {m2} {v}V로 변경해줘", "{m1} {m2} 모듈 {v}로",
    ]
    for _ in range(20):
        mods = random.sample([f"M{i}" for i in range(1, 10)], 2)
        v_str, v_num = _rand_volt()
        tmpl = random.choice(multi_mod_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(m1=mods[0], m2=mods[1], v=v_str), {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": mods, "voltage": v_num},
            "reason": f"{mods[0]}, {mods[1]} 모듈 전압을 {v_num}V로 변경",
        }))

    # ── C. 모듈 ON/OFF (20 samples) ──────────────────────────────────────────
    mod_on_tmpl = [
        "{m} 켜줘", "{m} HV on", "{m} 전원 켜줘", "{m} on 해줘",
        "모듈 {m} 켜줘", "{m} HV 켜줘", "turn on {m}",
    ]
    mod_off_tmpl = [
        "{m} 꺼줘", "{m} HV off", "{m} 전원 꺼줘", "{m} off 해줘",
        "모듈 {m} 꺼줘", "{m} HV 꺼줘", "turn off {m}",
    ]
    for _ in range(10):
        m = f"M{random.randint(1, 9)}"
        tmpl = random.choice(mod_on_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(m=m), {
            "tool": "hv_write",
            "params": {"command": "on", "channels": m},
            "reason": f"{m} 모듈 켜기",
        }))
    for _ in range(10):
        m = f"M{random.randint(1, 9)}"
        tmpl = random.choice(mod_off_tmpl)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(m=m), {
            "tool": "hv_write",
            "params": {"command": "off", "channels": m},
            "reason": f"{m} 모듈 끄기",
        }))

    return examples



# ══════════════════════════════════════════════════════════════════════════════
#  NONE (unclear / log-view / clarification) — 100 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_unclear() -> List[dict]:
    examples = []

    templates = [
        # ── 로그 조회 — 런 번호 없음 ──
        ("로그 확인", "어떤 런 번호의 로그를 확인할까요?"),
        ("로그 보여줘", "어떤 런 번호의 로그를 볼까요?"),
        ("로그 열어줘", "어떤 런 번호의 로그를 확인할까요?"),
        ("로그 알려줘", "어떤 런 번호의 로그를 확인할까요?"),
        ("기록 확인", "어떤 런 번호의 기록을 확인할까요?"),
        ("run log 확인", "어떤 런 번호의 로그를 확인할까요?"),
        ("show log", "어떤 런 번호의 로그를 확인할까요?"),

        # ── 로그 수정 불명확 (열+값 부족) ──
        ("로그 수정해줘", "어떤 런 번호의 어느 열을 수정할까요?"),
        ("로그 추가해줘", "어떤 런 번호의 어느 열에 무엇을 추가할까요?"),
        ("로그 업데이트", "어떤 런 번호의 어느 열을 수정할까요?"),
        ("기록해줘", "어떤 런 번호의 어느 열에 기록할까요?"),
        ("로그에 써줘", "어떤 런 번호의 어느 열에 무엇을 쓸까요?"),
        ("log update", "어떤 런 번호의 어느 열을 수정할까요?"),
        ("run {r} 프로그램 수정해줘", "Run {r} 프로그램을 무엇으로 바꿀까요?"),
        ("run {r} 메모 추가해줘", "Run {r} 메모에 무엇을 쓸까요?"),
        ("run {r} 노트 바꿔줘", "Run {r} 노트를 무엇으로 바꿀까요?"),
        ("{r} 에너지 수정", "Run {r} 빔 에너지를 얼마로 수정할까요?"),
        ("run {r} 설정 바꿔줘", "Run {r} 설정을 무엇으로 바꿀까요?"),
        ("방금 런 프로그램 수정", "프로그램을 무엇으로 바꿀까요?"),

        # ── DQM 플랏 불명확 (런 번호 누락) ──
        ("그려줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("플랏 보여줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("그래프", "어떤 런 번호의 그래프를 그릴까요?"),
        ("plot", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("플랏", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("그래프 보여줘", "어떤 런 번호의 그래프를 그릴까요?"),
        ("그림 그려줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("DQM 그려줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("peakADC 보여줘", "어떤 런 번호의 DQM peakADC를 그릴까요?"),
        ("intADC 보여줘", "어떤 런 번호의 DQM intADC를 그릴까요?"),
        ("DQM plot", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("show plot", "어떤 런 번호의 DQM 플랏을 그릴까요?"),

        # ── DQM single type — 채널 누락 ──
        ("run {r} single 그려줘", "어떤 채널을 그릴까요? (예: M1, M1-T1-C, M1-T1-S)"),
        ("run {r} 채널 그려줘", "어떤 채널을 그릴까요? (예: M1, M1-T1-C, M1-T1-S)"),
        ("{r} single 플랏", "어떤 채널을 그릴까요? (예: M1, M1-T1-C, M1-T1-S)"),
        ("run {r} 싱글 그려줘", "어떤 채널을 그릴까요? (예: M1, M1-T1-C, M1-T1-S)"),
        ("{r} 번 single DQM", "어떤 채널을 그릴까요? (예: M1, M1-T1-C, M1-T1-S)"),

        # ── DAQ 불명확 (이벤트 수 누락) ──
        ("DAQ 돌려줘", "몇 개의 이벤트를 수집할까요?"),
        ("데이터 받아줘", "몇 개의 이벤트를 수집할까요?"),
        ("이벤트 받아줘", "몇 개의 이벤트를 수집할까요?"),
        ("돌려줘", "DAQ를 몇 개 이벤트로 돌릴까요?"),
        ("DAQ run", "몇 개의 이벤트를 수집할까요?"),
        ("데이터 수집해줘", "몇 개의 이벤트를 수집할까요?"),
        ("DAQ 한번 돌려줘", "몇 개의 이벤트를 수집할까요?"),
        ("데이터 좀 받자", "몇 개의 이벤트를 수집할까요?"),
        ("이벤트 수집", "몇 개의 이벤트를 수집할까요?"),
        ("run daq", "몇 개의 이벤트를 수집할까요?"),
        ("start daq", "몇 개의 이벤트를 수집할까요?"),

        # ── 런 번호만 ──
        ("12905", "Run 12905에 대해 무엇을 할까요? (플랏, 로그 수정 등)"),
        ("run 12905", "Run 12905에 대해 무엇을 할까요? (플랏, 로그 수정 등)"),
        ("12905번", "Run 12905에 대해 무엇을 할까요? (플랏, 로그 수정 등)"),

        # ── 일반 불명확 ──
        ("확인해줘", "무엇을 확인할까요? (플랏, HV, 로그 등)"),
        ("도와줘", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등을 도와드릴 수 있습니다."),
        ("데이터", "데이터를 수집할까요, 아니면 기존 데이터의 플랏을 그릴까요?"),
        ("뭐 할 수 있어?", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),
        ("help", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),
        ("도움말", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),
        ("뭐 할 수 있어", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),
        ("명령어 목록", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),
        ("what can you do?", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),
        ("what can you do", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),
        ("commands", "DAQ 실행, 플랏 생성, 로그 수정, HV 읽기 등이 가능합니다."),

        # ── 의미 불분명 ──
        ("이상한데", "무엇이 이상한가요? 구체적으로 알려주시면 도움을 드릴 수 있습니다."),
        ("잘 안 돼", "무엇이 안 되나요? 구체적으로 알려주시면 도움을 드릴 수 있습니다."),
        ("다시 해줘", "무엇을 다시 할까요? (DAQ, 플랏, 로그 등)"),
        ("다시", "무엇을 다시 할까요? (DAQ, 플랏, 로그 등)"),
        ("한번 더", "무엇을 한번 더 할까요? (DAQ, 플랏, 로그 등)"),
        ("이거 뭐야", "무엇에 대해 알고 싶으신가요?"),
        ("상태", "무엇의 상태를 확인할까요? (DAQ, HV, 모터 등)"),
        ("뭔가 이상해", "무엇이 이상한가요? 구체적으로 알려주시면 도움을 드릴 수 있습니다."),
        ("잘 모르겠는데", "무엇이 궁금하신가요? 구체적으로 말씀해 주세요."),
        ("문제가 있어", "어떤 문제인가요? 구체적으로 알려주시면 도움을 드릴 수 있습니다."),

        # ── 잡담/인사 ──
        ("ㅎㅎ", "무엇을 도와드릴까요?"),
        ("ㅋㅋ", "무엇을 도와드릴까요?"),
        ("감사", "무엇을 더 도와드릴까요?"),
        ("고마워", "무엇을 더 도와드릴까요?"),
        ("ok", "무엇을 더 도와드릴까요?"),
        ("ㅇㅇ", "무엇을 도와드릴까요?"),
        ("안녕", "무엇을 도와드릴까요? DAQ 실행, 플랏 생성, 로그 수정 등이 가능합니다."),
        ("수고", "무엇을 더 도와드릴까요?"),
        ("테스트", "실험 관련 명령을 말씀해 주세요."),
        ("hi", "무엇을 도와드릴까요?"),
        ("hello", "무엇을 도와드릴까요? DAQ 실행, 플랏 생성, 로그 수정 등이 가능합니다."),

        # ── 실험과 무관한 입력 → 실험 명령 요청 메시지 ──
        ("날씨 어때", "실험 관련 명령을 말씀해 주세요."),
        ("날씨", "실험 관련 명령을 말씀해 주세요."),
        ("점심 뭐 먹지", "실험 관련 명령을 말씀해 주세요."),
        ("배고파", "실험 관련 명령을 말씀해 주세요."),
        ("피곤하다", "실험 관련 명령을 말씀해 주세요."),
        ("coffee 마시고 싶다", "실험 관련 명령을 말씀해 주세요."),
        ("몇 시야", "실험 관련 명령을 말씀해 주세요."),
        ("지금 몇 시", "실험 관련 명령을 말씀해 주세요."),
        ("뭐해?", "실험 관련 명령을 말씀해 주세요."),
        ("잘 됩니까", "실험 관련 명령을 말씀해 주세요."),
        ("아무거나", "실험 관련 명령을 말씀해 주세요."),
        ("???", "실험 관련 명령을 말씀해 주세요."),
        ("...", "실험 관련 명령을 말씀해 주세요."),
        ("1234", "실험 관련 명령을 말씀해 주세요."),
        ("asdf", "실험 관련 명령을 말씀해 주세요."),
    ]

    for _ in range(200):
        user_input, message = random.choice(templates)
        r = _random_run()
        if "{r}" in user_input:
            user_input = user_input.replace("{r}", str(r))
            message = message.replace("{r}", str(r))
        if "12905" in user_input:
            user_input = user_input.replace("12905", str(r))
            message = message.replace("12905", str(r))
        state = _make_state(random.random() > 0.5)
        decision = {"tool": "none", "message": message}
        examples.append(make_example(state, user_input, decision))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  DQM PLOT — overlay + relative + no-dash channel names
#
#  Covers three gaps the existing gen_dqm_plot() misses:
#  1. Relative reference ("방금"/"이번 런") + specific channel  → must use current_run
#  2. "overlay" keyword  → type: single (users say "M1T1S overlay" for single-channel plot)
#  3. No-dash input ("M1T1S", "M2T2C")  → model must output ["M1-T1-S"], ["M2-T2-C"] with hyphens
# ══════════════════════════════════════════════════════════════════════════════

def gen_dqm_overlay_relative_single() -> List[dict]:
    examples = []

    # ── A. Relative reference + single channel (120) ──────────────────────────
    #    User says "방금 run M1T1S overlay" or "이번 런 M1-T1-S 그려줘"
    #    → must resolve to current_run, type: single, modules with hyphens
    rel_single_tmpl = [
        # hyphenated channel
        "방금 {ch} 그려줘",           "이번 런 {ch} 그려줘",
        "방금 {ch} intADC 그려줘",    "이번 런 {ch} intADC 보여줘",
        "현재 런 {ch} 그려줘",        "지금 {ch} 그려줘",
        "방금 {ch} 채널 그려줘",      "이번 거 {ch} 플랏",
        "방금 run {ch} 그려줘",       "이번 run {ch} intADC",
        "방금 {ch} 확인해줘",         "현재 {ch} 플랏 보여줘",
        # no-dash channel (user abbreviates, model must output hyphenated)
        "방금 {nch} 그려줘",          "이번 런 {nch} 그려줘",
        "방금 {nch} intADC 그려줘",   "현재 런 {nch} 그려줘",
        "방금 run {nch} 그려줘",      "이번 run {nch} intADC",
        "지금 {nch} 그려줘",          "방금 {nch} 채널 그려줘",
        # overlay + relative (the exact failing pattern)
        "방금 {ch} overlay 그려줘",       "이번 런 {ch} overlay 보여줘",
        "방금 {nch} overlay 그려줘",      "이번 런 {nch} overlay 보여줘",
        "방금 run {ch} overlay 그려줘",   "이번 run {ch} overlay",
        "방금 run {nch} overlay 그려줘",  "이번 run {nch} overlay",
        "현재 런 {ch} overlay 그려줘",    "지금 {nch} overlay 그려줘",
        "방금 {ch} overlay intADC",       "이번 {nch} overlay intADC",
    ]

    for _ in range(120):
        modules = [_random_channel()]   # single channel only for clarity
        ch = modules[0]                  # e.g. "M3-T2-S"
        nch = _nondash(ch)               # e.g. "M3T2S"
        tmpl = random.choice(rel_single_tmpl)
        state = _make_state(with_agent=True)
        run = state["current_run"]
        examples.append(make_example(state, tmpl.format(ch=ch, nch=nch), {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC",
                       "type": "single", "modules": modules},
            "reason": f"현재 run {run} single IntADC ({ch}) 플랏 생성",
        }))

    # ── B. Explicit run number + overlay keyword (80) ─────────────────────────
    explicit_overlay_tmpl = [
        # hyphenated
        "run {r} {ch} overlay 그려줘",   "run {r} {ch} overlay 보여줘",
        "{r}번 {ch} overlay 그려",        "run {r} {ch} overlay intADC",
        "run {r} {ch} overlay 플랏",      "{r} {ch} overlay 그려줘",
        "{r}번 런 {ch} overlay 그려줘",   "{r} {ch} overlay 확인해줘",
        # no-dash
        "run {r} {nch} overlay 그려줘",   "{r}번 {nch} overlay 보여줘",
        "{r} {nch} overlay 그려줘",       "run {r} {nch} overlay",
        "run {r} {nch} overlay intADC",   "{r}번 {nch} overlay 플랏",
        "{r}번 런 {nch} overlay 그려줘",  "{r} {nch} overlay 확인",
    ]

    for _ in range(80):
        run = _random_run()
        modules = [_random_channel()]
        ch = modules[0]
        nch = _nondash(ch)
        tmpl = random.choice(explicit_overlay_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run, ch=ch, nch=nch),
            {
                "tool": "dqm_plot",
                "params": {"run_number": run, "method": "IntADC",
                           "type": "single", "modules": modules},
                "reason": f"Run {run} single IntADC ({ch}) overlay 플랏 생성",
            }
        ))

    # ── C. No-dash channel names in plain single requests (50) ────────────────
    #    Teaches model to normalize T1S → T1-S even without "overlay"
    no_dash_single_tmpl = [
        "run {r} {nch} intADC 그려줘",  "run {r} {nch} IntADC 보여줘",
        "{r}번 {nch} intADC 그려",      "{r} {nch} intADC 플랏",
        "run {r} {nch} 그려줘",         "{r}번 {nch} 그려줘",
        "run {r} {nch} 그래프 보여줘",  "{r} {nch} 채널 그려줘",
        "run {r} {nch} 플랏 보여줘",    "{r}번 런 {nch} 그려줘",
    ]

    for _ in range(50):
        run = _random_run()
        modules = [_random_channel()]
        ch = modules[0]
        nch = _nondash(ch)
        tmpl = random.choice(no_dash_single_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run, nch=nch),
            {
                "tool": "dqm_plot",
                "params": {"run_number": run, "method": "IntADC",
                           "type": "single", "modules": modules},
                "reason": f"Run {run} single IntADC ({ch}) 플랏 생성",
            }
        ))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  DQM PLOT — relative reference using last_run
#
#  Covers two gaps:
#  1. No scenario running, only "Last completed run" in state (brain standalone DAQ)
#     → "방금/이전/직전" must resolve to last_run
#  2. Scenario running, user explicitly says "이전/직전/전 런"
#     → must resolve to last_run (NOT current_run)
# ══════════════════════════════════════════════════════════════════════════════

def _make_brain_daq_state():
    """Simulates state after brain-initiated standalone DAQ: no scenario, only last_run."""
    return {"last_run": _random_run()}


def gen_dqm_relative_lastrun() -> List[dict]:
    examples = []

    # ── A. No scenario running, only last_run in state (120) ─────────────────
    #    State shows only "Last completed run: XXXXX"
    #    All relative expressions → last_run

    no_agent_full_tmpl = [
        "방금 런 그려줘",             "이전 런 그려줘",
        "직전 런 플랏 보여줘",        "전 런 그래프 그려줘",
        "방금 데이터 그려줘",         "이전 데이터 플랏 보여줘",
        "방금 거 전부 그려",          "직전 데이터 전부 그려줘",
        "방금 DAQ 결과 그려줘",       "이전 run 플랏 그려줘",
        "방금 run 그려줘",            "직전 run 그래프",
        "이전 거 DQM 그려줘",         "방금 거 DQM 보여줘",
        "방금 거 intADC 그려줘",      "이전 런 intADC 보여줘",
        "직전 런 intADC 플랏",        "방금 intADC 그려줘",
        "이전 데이터 intADC 그래프",  "방금 돌린 거 그려줘",
        "직전 돌린 거 그래프",        "이전 결과 그려줘",
        "방금 거 그래프 보여줘",      "이전 런 결과 플랏",
        "방금 거 플랏 그려줘",        "직전 데이터 DQM",
        "방금 돌린 데이터 보여줘",    "이전 run intADC 그려줘",
        "직전 run full 그려줘",       "방금 full 그려줘",
    ]

    for _ in range(80):
        state = _make_brain_daq_state()
        run = state["last_run"]
        tmpl = random.choice(no_agent_full_tmpl)
        examples.append(make_example(state, tmpl, {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC", "type": "full"},
            "reason": f"직전 run {run} full IntADC DQM 플랏 생성",
        }))

    no_agent_single_tmpl = [
        "방금 {ch} 그려줘",           "이전 런 {ch} 그려줘",
        "직전 런 {ch} intADC 그려줘", "방금 {nch} 그려줘",
        "이전 {nch} 플랏 보여줘",     "방금 run {ch} 그려줘",
        "직전 run {nch} 그려줘",      "방금 {ch} intADC 보여줘",
        "이전 {ch} intADC 그려줘",    "직전 {nch} intADC 그래프",
        "방금 {ch} 채널 그려줘",      "이전 런 {nch} 채널 보여줘",
        "방금 {nch} intADC 그래프",   "직전 {ch} 플랏",
        "이전 run {ch} intADC",       "방금 {nch} 보여줘",
    ]

    for _ in range(30):
        state = _make_brain_daq_state()
        run = state["last_run"]
        modules = [_random_channel()]
        ch = modules[0]
        nch = _nondash(ch)
        tmpl = random.choice(no_agent_single_tmpl)
        examples.append(make_example(state, tmpl.format(ch=ch, nch=nch), {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC",
                       "type": "single", "modules": modules},
            "reason": f"직전 run {run} single IntADC ({ch}) 플랏 생성",
        }))

    no_agent_heatmap_tmpl = [
        "방금 heatmap 그려줘",         "이전 런 heatmap 그려줘",
        "직전 run heatmap intADC",     "방금 MCPPMT heatmap 그려줘",
        "이전 거 히트맵 그려줘",       "방금 히트맵 보여줘",
        "직전 런 MCPPMT heatmap",      "이전 heatmap intADC 그려줘",
        "방금 run heatmap 그려줘",     "이전 데이터 heatmap",
    ]

    for _ in range(10):
        state = _make_brain_daq_state()
        run = state["last_run"]
        tmpl = random.choice(no_agent_heatmap_tmpl)
        examples.append(make_example(state, tmpl, {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC",
                       "type": "heatmap", "modules": ["MCPPMT"]},
            "reason": f"직전 run {run} heatmap IntADC (MCPPMT) 플랏 생성",
        }))

    # ── B. Scenario running, "이전/직전/전 런" → last_run (50) ───────────────
    #    State has current_run AND last_run.
    #    "이전/직전/전" always means last_run, not current_run.

    prev_full_tmpl = [
        "이전 런 그려줘",              "직전 런 플랏 보여줘",
        "전 런 그래프 그려줘",         "이전 run 그려줘",
        "직전 run 플랏",               "방금 전 런 그려줘",
        "직전 거 intADC 그려줘",       "이전 데이터 전부 그려줘",
        "이전 런 intADC 그려줘",       "직전 런 intADC 보여줘",
        "이전 거 전부 그려줘",         "전 런 전체 플랏",
        "이전 런 DQM 그려줘",          "직전 런 결과 플랏",
        "이전 run full 그려줘",        "직전 데이터 DQM 보여줘",
        "이전 런 그래프",              "방금 전 run 그려줘",
        "직전 거 그려줘",              "전 런 intADC 보여줘",
    ]

    for _ in range(40):
        state = _make_state(with_agent=True)
        run = state["last_run"]
        tmpl = random.choice(prev_full_tmpl)
        examples.append(make_example(state, tmpl, {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC", "type": "full"},
            "reason": f"직전 run {run} full IntADC DQM 플랏 생성",
        }))

    prev_single_tmpl = [
        "이전 런 {ch} 그려줘",         "직전 {ch} intADC 보여줘",
        "전 런 {nch} 그려줘",          "이전 run {ch} 그려줘",
        "직전 런 {nch} intADC",        "이전 {nch} 그래프",
        "방금 전 런 {ch} 그려줘",      "이전 거 {ch} 플랏",
        "직전 {ch} 채널 그려줘",       "전 런 {nch} 채널 보여줘",
    ]

    for _ in range(10):
        state = _make_state(with_agent=True)
        run = state["last_run"]
        modules = [_random_channel()]
        ch = modules[0]
        nch = _nondash(ch)
        tmpl = random.choice(prev_single_tmpl)
        examples.append(make_example(state, tmpl.format(ch=ch, nch=nch), {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC",
                       "type": "single", "modules": modules},
            "reason": f"직전 run {run} single IntADC ({ch}) 플랏 생성",
        }))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  HODOSCOPE HV READ — 40 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_hodoscope_hv_read() -> List[dict]:
    """Hodoscope HV 확인 → hv_read (combined CAEN + Hodoscope status)."""
    examples = []
    templates = [
        # 호도스코프 명시 — 통합 hv_read로 처리
        "호도스코프 HV 확인해줘", "호도 HV 얼마야", "hodoscope HV 읽어줘",
        "호도스코프 고전압 확인", "호도 HV 확인", "hodoscope HV 얼마야",
        "호도스코프 HV 세팅 보여줘", "호도 전압 얼마야", "호도 HV 알려줘",
        "hodoscope HV 보여줘", "호도스코프 전압 읽어줘",
        # 존댓말
        "호도스코프 HV 확인해주세요", "호도 HV 얼마인지 알려주세요",
        "hodoscope HV 좀 보여주세요", "호도 전압 확인 부탁드립니다",
        # 영어
        "read hodoscope HV", "hodoscope HV check", "show hodoscope HV",
        "hodoscope voltage", "get hodoscope HV",
        # 반말
        "호도 HV 봐봐", "호도스코프 HV 좀", "호도 전압 확인",
        "hodoscope HV", "호도 hv",
        # 구체적
        "지금 호도스코프 HV 얼마로 설정되어 있어?",
        "현재 hodoscope HV 세팅은?",
        "호도 HV 지금 어떻게 되어있어",
        "hodoscope hv 지금 켜져있어?",
        "hodo HV 읽어줘", "호도 고전압 얼마야",
        "hodoscope 전압 확인해줘", "호도스코프 HV 현재값 알려줘",
        "hodo hv check", "호도 HV 값",
        "호도스코프 HV 지금 몇 볼트야", "hodoscope 고전압 확인",
    ]
    for _ in range(40):
        template = random.choice(templates)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, template, {
            "tool": "hv_read",
            "params": {"command": "status"},
            "reason": "HV 상태 확인 (CAEN + Hodoscope 통합)",
        }))
    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  HODOSCOPE HV WRITE — 45 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_hodoscope_hv_write() -> List[dict]:
    examples = []

    hv_values_with_v = ["2V", "3V", "3.5V", "4V", "4.5V", "5V", "5.5V", "6V"]
    hv_values_no_v   = ["2", "3", "3.5", "4", "4.5", "5", "5.5", "6"]

    def _rand_hv():
        if random.random() < 0.5:
            s = random.choice(hv_values_with_v)
        else:
            s = random.choice(hv_values_no_v)
        return s, float(s.rstrip("Vv").strip())

    # ── 전압 설정 (35 samples) ──
    templates_set = [
        # 동사 있음
        "호도스코프 HV {v}로 설정해줘", "호도 HV {v}로 바꿔줘",
        "hodoscope HV {v}로 수정해줘", "호도 HV {v}로 올려줘",
        "호도스코프 전압 {v}로 변경해줘", "호도 HV {v} 설정",
        "hodoscope HV {v}로 맞춰줘", "호도스코프 HV {v}로 해줘",
        "호도 전압 {v}로 바꿔줘", "hodoscope 전압 {v}로 설정",
        "호도스코프 고전압 {v}로 수정", "호도 HV {v}로 조정해줘",
        "hodoscope HV {v}로 인가해줘", "호도 HV {v}",
        "호도스코프 HV {v}", "hodoscope hv {v}로",
        # 존댓말
        "호도스코프 HV {v}로 설정해주세요", "호도 HV {v}로 바꿔주세요",
        "hodoscope HV {v}로 변경 부탁드립니다",
        # 영어
        "set hodoscope HV to {v}", "hodoscope HV {v}",
        "hodoscope voltage {v}", "set hodo HV {v}",
        # 반말
        "호도 HV {v}로", "호도스코프 HV {v}로 해",
        "hodo hv {v}로 바꿔", "호도 {v}로",
        # 구체적 맥락
        "빔 켜기 전에 호도 HV {v}로 올려줘",
        "호도스코프 HV {v}로 세팅하고 싶어",
        "호도 HV {v}볼트로 바꿔줘",
        "hodoscope hv {v}V로 수정",
        "호도 set file hv {v}로 바꿔줘",
    ]
    for _ in range(35):
        v_str, v_num = _rand_hv()
        template = random.choice(templates_set)
        user_input = template.format(v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hodoscope_hv_write",
            "params": {"command": "write", "value": v_num},
            "reason": f"호도스코프 HV를 {v_num}V로 변경",
        }))

    # ── 끄기 / off / 0 (10 samples) ──
    templates_off = [
        "호도스코프 HV 꺼줘", "호도 HV 꺼줘", "hodoscope HV off",
        "호도 HV 0으로 내려줘", "호도스코프 HV 0으로 설정",
        "hodoscope HV 끄기", "호도 전압 0으로", "호도 HV 끄기",
        "hodoscope hv 꺼줘", "호도스코프 HV 0볼트로",
    ]
    for _ in range(10):
        template = random.choice(templates_off)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, template, {
            "tool": "hodoscope_hv_write",
            "params": {"command": "write", "value": 0.0},
            "reason": "호도스코프 HV 끄기 (0V)",
        }))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  DQM PLOT — METHOD 미지정 → 물어보기 (120 samples)
#  "run NNNN 그려줘" / "방금 그려줘" 등 method 없는 요청
#  → {"tool": "none", "message": "IntADC로 그릴까요, PeakADC로 그릴까요?"}
# ══════════════════════════════════════════════════════════════════════════════

def gen_dqm_ask_method() -> List[dict]:
    examples = []
    ask_msg = "IntADC로 그릴까요, PeakADC로 그릴까요?"

    # A. explicit run number, no method (70 samples)
    no_method_explicit = [
        "run {r} 그려줘", "run {r} 플랏 보여줘", "{r}번 런 그려줘",
        "{r} 그래프 그려줘", "run {r} 플랏", "{r}번 그려줘",
        "run {r} 그래프 보여줘", "{r} 데이터 그려줘",
        "run {r} DQM 보여줘", "{r}번 런 DQM 그려줘",
        "run {r} 전체 플랏 보여줘", "{r} 전체 그려줘",
        "run {r} plot 그려줘", "{r} DQM plot",
        "run {r} 데이터 플랏", "{r}번 그래프 보여줘",
        "run {r} 그려", "{r} 그려", "run {r} 플랏 그려",
        "{r}번 런 플랏 그려줘", "run {r} 결과 보여줘",
        "{r} 결과 그려줘", "run {r} 그래프", "{r}번 그래프",
        "run {r} DQM 그려", "{r} 모든 타워 그려줘",
        "run {r} 타워 전체 보여줘", "{r}번 전체 그려",
        "run {r} 확인해줘", "{r} 확인 그래프",
    ]
    for _ in range(70):
        run = _random_run()
        tmpl = random.choice(no_method_explicit)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, tmpl.format(r=run), {
            "tool": "none",
            "message": ask_msg,
        }))

    # B. relative reference ("방금"/"이번"), no method (50 samples)
    no_method_relative = [
        "방금 그려줘", "방금 거 그려줘", "방금 런 그려줘",
        "방금 데이터 그려줘", "방금 플랏 보여줘", "방금 거 플랏",
        "이번 런 그려줘", "이번 거 그려줘", "이번 데이터 그려줘",
        "이번 플랏 보여줘", "이번 결과 그려줘",
        "지금 런 그려줘", "지금 거 그려줘", "지금 데이터 그려줘",
        "방금 돌린 거 그려줘", "방금 받은 거 그려줘",
        "방금 DAQ 결과 그려줘", "방금 런 플랏",
        "마지막 런 그려줘", "최근 런 그려줘",
        "이번 런 플랏 보여줘", "방금 거 전부 그려줘",
        "방금 런 결과 보여줘", "현재 런 그려줘",
        "이번 데이터 플랏 보여줘",
    ]
    for _ in range(50):
        tmpl = random.choice(no_method_relative)
        state = _make_state(with_agent=True)
        examples.append(make_example(state, tmpl, {
            "tool": "none",
            "message": ask_msg,
        }))

    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    output_file = Path(__file__).parent / "data" / "brain_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_examples = []
    all_examples.extend(gen_daq_run())                      # 150
    all_examples.extend(gen_dqm_plot())                     # 600  (full 350 + heatmap 100 + single 150)
    all_examples.extend(gen_dqm_overlay_relative_single())  # 250  (relative+single 120 + overlay 80 + no-dash 50)
    all_examples.extend(gen_dqm_relative_lastrun())         # 170  (no-agent 120 + prev-with-scenario 50)
    all_examples.extend(gen_dqm_ask_method())               # 120  (method 미지정 → 물어보기)
    all_examples.extend(gen_run_log())                      # 150  (write)
    all_examples.extend(gen_run_log_read())                 #  70  (read)
    all_examples.extend(gen_hv_read())                      # 150  (all_ch 120 + named_ch 30)
    all_examples.extend(gen_hv_write())                     # 330  (single 150 + multi 30 + all 30 + on/off 60)
    all_examples.extend(gen_hv_write_advanced())            # 340  (S 60 + C만 50 + TRIG 40 + MCP 40 + 짝수 35 + 홀수 25 + 범위 30 + read 30 + on/off 20)
    all_examples.extend(gen_hv_write_module())              #  80  (단일모듈 40 + 복수모듈 20 + on/off 20)
    all_examples.extend(gen_unclear())                      # 200  (clarify + off-topic)

    random.shuffle(all_examples)

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    lengths = [sum(len(m["content"]) for m in ex["messages"]) for ex in all_examples]
    max_chars = max(lengths)
    avg_chars = sum(lengths) / len(lengths)
    print(f"Generated {len(all_examples)} samples -> {output_file}")
    print(f"   char len  max={max_chars:,}  avg={avg_chars:,.0f}  "
          f"(≈token max={max_chars // 2:,}  avg={avg_chars // 2:,.0f})")


if __name__ == "__main__":
    main()
