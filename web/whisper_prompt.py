#!/usr/bin/env python3
"""Whisper 음성인식 — 모델 로드, 프롬프트, 후처리"""

import re
import threading

# ── Model (lazy-loaded) ───────────────────────────────────────────────────────
_whisper_model = None
_whisper_lock = threading.Lock()


def get_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                _whisper_model = WhisperModel(
                    "medium", device="cpu", compute_type="int8"
                )
    return _whisper_model


# ── Prompt (자연어 문장으로 도메인 용어를 Whisper에 노출) ─────────────────────
PROMPT = (
    "빔 에너지를 1 GeV, 3 GeV, 5 GeV로 설정합니다. "
    "HV 고전압을 조정하고 ADC 값을 확인합니다. "
    "DAQ로 1000개 이벤트를 수집합니다. "
    "이벤트를 1만개 받겠습니다. waveform, peakADC, intADC 플롯을 확인합니다. "
    "M1T1, M5T3, M9T4 타워를 캘리브레이션합니다. "
    "완료."
)


# ── Post-processing: 물리 용어 오인식 교정 ────────────────────────────────────
_CORRECTIONS = [
    (re.compile(r'\bG\s*[12e]?\s*V\b', re.IGNORECASE), 'GeV'),
    (re.compile(r'\b기가\s*전자\s*볼트\b'),              'GeV'),
    (re.compile(r'\b지이브이\b'),                        'GeV'),
    (re.compile(r'웨이브\s*폼'),                         'waveform'),
    (re.compile(r'\b에이디씨\b'),                        'ADC'),
    (re.compile(r'\b다큐\b'),                            'DAQ'),
    (re.compile(r'\b에이치브이\b'),                      'HV'),
    (re.compile(r'\b케이에이치\b'),                      'KEK'),
    (re.compile(r'(\d),(\d{3})'),                        r'\1\2'),
]

_NUM_COMMA = re.compile(r'(\d),(\d{3})\b')


def fix_physics(text: str) -> str:
    """물리 용어 오인식 및 숫자 포맷 교정"""
    for pattern, replacement in _CORRECTIONS:
        text = pattern.sub(replacement, text)
    while _NUM_COMMA.search(text):
        text = _NUM_COMMA.sub(r'\1\2', text)
    return text
