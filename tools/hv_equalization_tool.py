#!/usr/bin/env python3
"""
HV Equalization Tool for Test Beam Automation
PMT 간 신호 균일화를 위한 HV 조정 도구 (peakADC 계산 통합)
"""

import os
import re
import json
import glob
import numpy as np
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Dummy decorator for testing without langchain
    def tool(func):
        return func
    LANGCHAIN_AVAILABLE = False

try:
    from scipy.optimize import curve_fit
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ======================= Data Paths =======================
from .config_loader import get_data_directory, get_mapping_root_path, get_path_config

# Data paths (from config_general.yml)
DATA_DIR = get_data_directory()
MAPPING_PATH = get_mapping_root_path()
RUNNUM_PATH = get_path_config("RunNumberFile")
DQM_OUTPUT_DIR = Path(get_path_config("DqmDir")) / "output"
CS_LIST = ['C', 'S']


# ======================= Mapping (ROOT) =======================

def _load_mapping_from_root(root_path: str) -> List[dict]:
    """
    mapping_KEK.root의 mapping_DAQ 트리에서 mid/ch/name을 읽어 반환.
    반환 형식: [{"mid": int, "ch": int, "name": str}, ...]
    """
    try:
        import uproot
        with uproot.open(root_path) as f:
            tree = f["mapping_DAQ"]
            arrays = tree.arrays(["mid", "ch", "name"], library="np")
        return [
            {"mid": int(m), "ch": int(c), "name": str(n)}
            for m, c, n in zip(arrays["mid"], arrays["ch"], arrays["name"])
        ]
    except Exception:
        return []


_mapping_cache: Optional[List[dict]] = None


def _get_mapping() -> List[dict]:
    global _mapping_cache
    if _mapping_cache is None:
        _mapping_cache = _load_mapping_from_root(MAPPING_PATH)
    return _mapping_cache

# HV Control Tool import (autoTB 내부)
try:
    from tools.hv_control_tool import HVControlTool
    HV_CONTROL_AVAILABLE = True
except ImportError:
    HV_CONTROL_AVAILABLE = False


# ======================= Run Number 관리 =======================

def get_current_run_number() -> Optional[int]:
    """runnum.txt에서 현재(방금 종료된) run number 가져오기"""
    try:
        with open(RUNNUM_PATH, 'r') as f:
            # Run이 종료된 후 runnum.txt가 다음 번호로 업데이트되므로, 
            # 방금 종료된 Run 정보를 위해 -1을 수행함
            content = f.read().strip()
            return int(content) - 1
    except Exception:
        return None


# ======================= PeakADC 계산 =======================

def count_events_in_file(path: str) -> int:
    """Count total events in a .dat file"""
    filesize = os.path.getsize(path)
    event_size = 64 + 32736*2
    return filesize // event_size


def load_dat_event(path: str, target_ch: int, event_idx: int) -> Optional[np.ndarray]:
    """Load a single event waveform from .dat file"""
    event_size = 64 + 32736*2
    try:
        with open(path, "rb") as f:
            f.seek(event_idx * event_size)
            header = f.read(64)
            data = f.read(32736*2)
        adc = np.frombuffer(data, dtype="<i2")
        reshaped = adc.reshape((1023, 32))
        wf = reshaped[:, target_ch][1:1000]
        return wf
    except Exception:
        return None


def compute_peakADC_100bin(wf: np.ndarray) -> Optional[float]:
    """Compute peakADC using first 100 bins for baseline"""
    if len(wf) < 100:
        return None
    
    baseline = np.mean(wf[:100])
    peakADC = np.max(baseline - wf)
    return float(peakADC)


def collect_peakADC_for_run(run_num: int, cs_type: str, center: Optional[str] = None) -> list:
    """
    Collect all peakADC values for a specific run and CS type.

    Args:
        run_num: Run 번호
        cs_type: 'C' 또는 'S'
        center: 타워 위치 (예: "M5T3"). 필수 (agent에서 제공)
    """
    if center is None:
        return []

    target_name = f"{center}-{cs_type}"
    mapping = _get_mapping()

    peakADC_values = []
    for entry in mapping:
        if entry["name"].strip() != target_name:
            continue
        mid = entry["mid"]
        ch = entry["ch"] - 1  # 0-based indexing

        if not (0 <= ch < 32):
            continue

        pattern = os.path.join(
            DATA_DIR,
            f"Run_{run_num}/Run_{run_num}_Wave/Run_{run_num}_Wave_MID_{mid}"
            f"/Run_{run_num}_Wave_MID_{mid}_FILE_*.dat"
        )
        for target_file in sorted(glob.glob(pattern)):
            n_events = count_events_in_file(target_file)
            for event_idx in range(n_events):
                wf = load_dat_event(target_file, ch, event_idx)
                if wf is not None:
                    peakADC = compute_peakADC_100bin(wf)
                    if peakADC is not None:
                        peakADC_values.append(peakADC)

    return peakADC_values


def smart_valley_cut(values: list, min_valley_height_ratio: float = 0.1, 
                     cut_at_zero: bool = True, bins: int = 200) -> np.ndarray:
    """
    Smart valley cut to remove noise peaks while preserving real signals
    
    Logic:
    - Deep valley (< threshold) → Noise peak → Cut at valley position
    - Shallow valley (≥ threshold) → Double peak (real signals) → Don't cut
    - Cut at first zero count bin from right
    """
    if len(values) == 0:
        return np.array([])
    
    values = np.array(values)
    
    # Create histogram for analysis
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth histogram for peak detection
    if SCIPY_AVAILABLE:
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
    else:
        # Simple moving average fallback
        window = 5
        hist_smooth = np.convolve(hist.astype(float), np.ones(window)/window, mode='same')
    
    # Find peaks (local maxima with significant height)
    peaks = []
    min_peak_height = 0.05 * np.max(hist_smooth)
    
    for i in range(1, len(hist_smooth) - 1):
        if (hist_smooth[i] > hist_smooth[i-1] and 
            hist_smooth[i] > hist_smooth[i+1] and 
            hist_smooth[i] > min_peak_height):
            peaks.append(i)
    
    if len(peaks) < 2:
        filtered_values = values
    else:
        # Find main peak (rightmost significant peak)
        main_peak_idx = peaks[-1]
        main_peak_height = hist_smooth[main_peak_idx]
        
        # Look for valleys to the left of main peak
        best_cut_position = None
        
        for peak_idx in reversed(peaks[:-1]):
            # Find valley between this peak and the main peak
            start_idx = peak_idx
            end_idx = main_peak_idx
            
            if start_idx < end_idx:
                valley_region = hist_smooth[start_idx:end_idx]
                valley_idx_relative = np.argmin(valley_region)
                valley_idx_absolute = start_idx + valley_idx_relative
                valley_height = hist_smooth[valley_idx_absolute]
                
                # Calculate valley height ratio
                valley_height_ratio = valley_height / main_peak_height
                
                # DEEP valley indicates noise peak -> CUT
                if valley_height_ratio < min_valley_height_ratio:
                    best_cut_position = bin_centers[valley_idx_absolute]
                    break
        
        if best_cut_position is None:
            filtered_values = values
        else:
            # Apply vertical cut at the valley position
            filtered_values = values[values >= best_cut_position]
    
    # Apply zero-count cut if requested (right cut from main peak)
    if cut_at_zero and len(filtered_values) > 0:
        hist_for_zero, bin_edges_for_zero = np.histogram(filtered_values, bins=bins)
        bin_centers_for_zero = (bin_edges_for_zero[:-1] + bin_edges_for_zero[1:]) / 2
        
        # Find main peak position in the histogram
        main_peak_bin_idx = np.argmax(hist_for_zero)
        
        # From main peak, go right and find first zero count bin
        zero_cut_position = None
        for i in range(main_peak_bin_idx + 1, len(hist_for_zero)):
            if hist_for_zero[i] == 0:
                zero_cut_position = bin_edges_for_zero[i]
                break
        
        if zero_cut_position is not None:
            filtered_values = filtered_values[filtered_values < zero_cut_position]
    
    return filtered_values


def _read_peakADC_from_dqm_json(run_num: int, center: str, cs_type: str) -> Tuple[Optional[float], int]:
    """
    DQM live가 생성한 JSON 파일에서 TH1D peakADC mean 추출 (valley cut 적용).
    DQM monit과 동일한 데이터 소스 사용.
    """
    hist_name = f"{center}-{cs_type}"

    # DQM은 per-tower 캔버스를 "이름 정렬 순번"(fCanvas_Tower{1..N})으로 저장한다
    # (TBplotengine init_Generic). 이 순번은 타워명에서 계산할 수 없으므로
    # (예: 36타워 "M5-T1" → Tower17), 파일명을 유추하지 말고 Tower*.json을 훑어
    # {center}-{cs_type} 히스토를 담은 파일을 찾는다. 9타워("T1"→Tower1)·36타워
    # ("M1-T1"→Tower1) 모두 네이밍/인덱싱 규칙과 무관하게 동작한다.
    pattern = str(DQM_OUTPUT_DIR / f"Run{run_num}_full_PeakADC_fCanvas_Tower*.json")
    for json_path in sorted(glob.glob(pattern)):
        try:
            with open(json_path) as f:
                canvas = json.load(f)
        except Exception:
            continue

        for pad in canvas.get('fPrimitives', {}).get('arr', []):
            for prim in pad.get('fPrimitives', {}).get('arr', []):
                if prim.get('_typename') != 'TH1D' or prim.get('fName') != hist_name:
                    continue

                # hist 찾음 — 이후 실패는 명시적으로 (None, 0)
                try:
                    ax = prim.get('fXaxis', {})
                    nbins = ax.get('fNbins', 0)
                    xmin = ax.get('fXmin', 0.0)
                    xmax = ax.get('fXmax', float(nbins))
                    if nbins <= 0:
                        return None, 0

                    bin_width = (xmax - xmin) / nbins
                    farray = prim.get('fArray', [])
                    # fArray[0] = underflow, [1..nbins] = bins, [nbins+1] = overflow
                    counts = np.array(farray[1:nbins + 1], dtype=float)
                    centers = np.array([xmin + (i + 0.5) * bin_width for i in range(nbins)])

                    if counts.sum() <= 0:
                        return None, 0

                    # Valley cut on histogram bins (mirrors smart_valley_cut logic)
                    if SCIPY_AVAILABLE:
                        smooth = gaussian_filter1d(counts, sigma=2.0)
                    else:
                        w = 5
                        smooth = np.convolve(counts, np.ones(w) / w, mode='same')

                    min_peak_h = 0.05 * smooth.max()
                    peaks = [i for i in range(1, len(smooth) - 1)
                             if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]
                             and smooth[i] > min_peak_h]

                    cut_idx = 0
                    if len(peaks) >= 2:
                        main_peak = peaks[-1]
                        for p in reversed(peaks[:-1]):
                            valley_region = smooth[p:main_peak]
                            vi = p + int(np.argmin(valley_region))
                            if smooth[vi] / smooth[main_peak] < 0.1:
                                cut_idx = vi
                                break

                    zero_cut_idx = nbins
                    if peaks:
                        for i in range(peaks[-1] + 1, nbins):
                            if counts[i] == 0:
                                zero_cut_idx = i
                                break

                    sel_counts = counts[cut_idx:zero_cut_idx]
                    sel_centers = centers[cut_idx:zero_cut_idx]
                    w_sum = sel_counts.sum()
                    if w_sum <= 0:
                        return None, 0

                    mean = float(np.dot(sel_counts, sel_centers) / w_sum)
                    return mean, int(w_sum)
                except Exception:
                    return None, 0

    return None, 0


def calculate_valley_cut_average(run_num: int, cs_type: str, center: Optional[str] = None) -> Tuple[Optional[float], int]:
    """
    DQM JSON 출력에서 peakADC mean 추출 (DQM monit과 동일한 방식).
    JSON 없을 경우 raw .dat 파일 직접 읽기로 fallback.
    """
    if center is None:
        return None, 0

    # 타워 이름 정규화: HV/position scan은 무대시(M5T1)로 넘기지만, DQM 히스토·mapping은
    # 항상 대시(M5-T1) 포맷을 쓴다. 조회 경계에서 M{m}T{t} → M{m}-T{t}로 변환한다.
    # (KEK 9타워 "T1"·이미 대시인 "M5-T1"은 패턴 불일치라 그대로 유지 → 멱등.)
    center = re.sub(r'^M(\d+)T(\d+)$', r'M\1-T\2', center)

    # Primary: DQM JSON (monit과 동일한 데이터)
    mean, count = _read_peakADC_from_dqm_json(run_num, center, cs_type)
    if mean is not None:
        return mean, count

    # Fallback: raw .dat 직접 읽기
    peakADC_data = collect_peakADC_for_run(run_num, cs_type, center)
    if len(peakADC_data) == 0:
        return None, 0

    filtered_data = smart_valley_cut(peakADC_data, min_valley_height_ratio=0.3, cut_at_zero=True)
    if len(filtered_data) == 0:
        return None, 0

    return float(np.mean(filtered_data)), len(filtered_data)


# ======================= Exponential Fitting =======================

class ExponentialHVPredictor:
    """물리학 기반 Exponential HV-ADC 관계 예측기"""
    
    def __init__(self):
        self.data_points = {"C": [], "S": []}
        self.coefficients = {"C": None, "S": None}
        self.is_fitted = {"C": False, "S": False}
        self.channel_status = {"C": "active", "S": "active"}
        
    def reset_data(self):
        """모든 데이터 초기화"""
        self.data_points = {"C": [], "S": []}
        self.coefficients = {"C": None, "S": None}
        self.is_fitted = {"C": False, "S": False}
        self.channel_status = {"C": "active", "S": "active"}
    
    def mark_channel_done(self, channel: str):
        """채널 완료 표시 - 해당 채널은 더 이상 suggestion 안함"""
        self.channel_status[channel] = "done"
    
    def is_channel_active(self, channel: str) -> bool:
        """채널이 아직 active 상태인지 확인"""
        return self.channel_status[channel] == "active"
            
    def exponential_func(self, hv: np.ndarray, A: float, B: float) -> np.ndarray:
        """Exponential function: ADC = A * exp(B * HV)"""
        return A * np.exp(B * hv)
    
    def _fit_exponential(self, channel: str) -> Tuple[bool, Optional[str]]:
        """채널별 exponential fitting"""
        points = self.data_points[channel]
        if len(points) < 2:
            return False, None
        
        hvs = [p[0] for p in points]
        adcs = [p[1] for p in points]
        
        try:
            fitting_message = None
            
            if SCIPY_AVAILABLE and len(points) >= 3:
                # scipy를 이용한 curve fitting
                popt, _ = curve_fit(self.exponential_func, hvs, adcs, 
                                  p0=[100, 0.001],
                                  maxfev=1000)
                self.coefficients[channel] = popt
                self.is_fitted[channel] = True
                fitting_message = f"{channel} 채널 fitting: ADC = {popt[0]:.2f} * exp({popt[1]:.5f} * HV)"
            elif len(points) >= 2:
                # 간단한 2점 기반 fitting
                hv1, adc1 = points[0]
                hv2, adc2 = points[-1]
                
                if hv1 != hv2 and adc1 > 0 and adc2 > 0:
                    # ADC = A * exp(B * HV)에서 B = ln(adc2/adc1) / (hv2-hv1)
                    B = math.log(adc2 / adc1) / (hv2 - hv1)
                    A = adc1 / math.exp(B * hv1)
                    
                    self.coefficients[channel] = (A, B)
                    self.is_fitted[channel] = True
                    fitting_message = f"{channel} 채널 fitting: ADC = {A:.2f} * exp({B:.5f} * HV)"
            
            return self.is_fitted[channel], fitting_message
            
        except Exception as e:
            error_msg = f"❌ {channel} 채널 fitting 실패: {e}"
            return False, error_msg
    
    def predict_hv_for_target(self, channel: str, target_adc: float) -> Optional[float]:
        """Target ADC에 필요한 HV 예측"""
        if not self.is_fitted[channel] or self.coefficients[channel] is None:
            return None
        
        A, B = self.coefficients[channel]
        try:
            # ADC = A * exp(B * HV) => HV = ln(ADC/A) / B
            if A > 0 and B != 0:
                required_hv = math.log(target_adc / A) / B
                return max(0, required_hv)  # HV는 양수
        except (ValueError, ZeroDivisionError):
            pass
        
        return None
    
    def initial_exploration(self, channel: str, current_hv: float, 
                          current_adc: float, target_adc: float) -> Dict[str, Any]:
        """초기 ±10V 탐색으로 방향성 파악"""
        
        # 방향성 판단: target보다 크면 아래로, 작으면 위로
        if current_adc > target_adc:
            explore_hv = current_hv - 10
            direction = "down"
        else:
            explore_hv = current_hv + 10
            direction = "up"
        
        # 현재점을 데이터에 추가
        self.data_points[channel].append((current_hv, current_adc))
        
        return {
            "suggested_hv": explore_hv,
            "direction": direction,
            "reason": f"ADC {current_adc:.1f} {'>' if current_adc > target_adc else '<'} target {target_adc}, explore {direction}"
        }
    
    def add_data_point(self, channel: str, hv: float, adc: float) -> Optional[str]:
        """새로운 (HV, ADC) 점 추가 및 exponential re-fitting"""
        self.data_points[channel].append((hv, adc))
        
        # 점이 2개 이상이면 fitting 시도
        fitting_message = None
        if len(self.data_points[channel]) >= 2:
            fitted, fitting_message = self._fit_exponential(channel)
            
        return fitting_message
    
    def predict_hv_adjustment(self, channel: str, current_hv: float,
                            current_adc: float, target_adc: float,
                            tolerance: float = 0.05) -> Dict[str, Any]:
        """단일 채널 HV 조정 예측. tolerance: ADC 허용 오차 비율 (기본 1%)"""

        # Done 상태면 skip — 단, 현재 ADC가 tolerance를 벗어났으면 재활성화
        if not self.is_channel_active(channel):
            if target_adc > 0 and abs(current_adc - target_adc) / target_adc <= tolerance:
                return {
                    "status": "done",
                    "message": f"{channel} 채널 equalization 완료됨 - suggestion 없음",
                    "current_adc": current_adc,
                    "target_adc": target_adc,
                    "adc_error": abs(target_adc - current_adc)
                }
            else:
                # ADC가 tolerance 밖으로 벗어남 (HV가 잘못 변경됨) → 재활성화
                self.channel_status[channel] = "active"

        # ADC가 이미 target ±tolerance% 이내이면 자동 done
        if target_adc > 0 and abs(current_adc - target_adc) / target_adc <= tolerance:
            self.mark_channel_done(channel)
            return {
                "status": "done",
                "message": (f"{channel} 채널 수렴 완료: ADC={current_adc:.1f} "
                            f"(target={target_adc}, error={abs(current_adc-target_adc)/target_adc*100:.1f}%)"),
                "current_adc": current_adc,
                "target_adc": target_adc,
                "adc_error": abs(target_adc - current_adc)
            }

        # Fitting이 되어 있으면 정확한 HV 예측
        if self.is_fitted[channel]:
            required_hv = self.predict_hv_for_target(channel, target_adc)
            if required_hv is not None:
                hv_change = int(round(required_hv - current_hv))
                hv_change = max(-500, min(500, hv_change))  # 안전 제한

                return {
                    "hv_change": hv_change,
                    "current_hv": current_hv,
                    "target_hv": current_hv + hv_change,
                    "current_adc": current_adc,
                    "target_adc": target_adc,
                    "adc_error": abs(target_adc - current_adc),
                    "method": "exponential_fit"
                }
            else:
                return {"error": "Exponential fitting 계산 실패"}
        else:
            # 초기 탐색 제안
            exploration = self.initial_exploration(channel, current_hv, current_adc, target_adc)
            hv_change = int(exploration["suggested_hv"] - current_hv)
            
            return {
                "hv_change": hv_change,
                "current_hv": current_hv,
                "target_hv": exploration["suggested_hv"],
                "current_adc": current_adc,
                "target_adc": target_adc,
                "adc_error": abs(target_adc - current_adc),
                "method": "initial_exploration",
                "reason": exploration["reason"]
            }


# ======================= Session Management =======================

class HVEqualizationSession:
    """HV Equalization 세션 관리 클래스"""
    
    def __init__(self):
        self.sessions = {}  # session_id -> session_data
        self.exp_predictor = ExponentialHVPredictor()
        self.exp_initialized = False
    
    def reset_for_new_tower(self, session_id: str = "default"):
        """새로운 타워를 위한 데이터 초기화 (target 유지)"""
        if session_id in self.sessions:
            self.exp_predictor.reset_data()
            
            session = self.sessions[session_id]
            target_c = session.get("target_c")
            target_s = session.get("target_s")
            
            self.sessions[session_id] = {
                "target_c": target_c,
                "target_s": target_s,
                "history": [],
                "active": True,
                "current_hv_c": 800,
                "current_hv_s": 800,
                "iteration": 0,
                "current_tower": None
            }
            
            return f"""새로운 타워 equalization 준비 완료!

Target 유지: C={target_c} ADC, S={target_s} ADC
데이터 초기화: 이전 exponential fitting 데이터 모두 지워짐

사용법: get_peak_adc_averages(run_number)로 현재 ADC 확인 후
hv_equalization_suggest로 HV 조정 제안 받으세요."""
    
    def start_session(self, session_id: str, target_c: float, target_s: float,
                     tower: str = "M5T3") -> str:
        """새로운 HV equalization 세션 시작"""

        # 항상 리셋: 중단된 테스트의 잔류 데이터가 다음 세션에 영향을 주는 것을 방지
        self.exp_predictor.reset_data()
        self.exp_initialized = True

        self.sessions[session_id] = {
            "target_c": target_c,
            "target_s": target_s,
            "history": [],
            "active": True,
            "current_hv_c": 800,
            "current_hv_s": 800,
            "iteration": 0,
            "current_tower": tower
        }

        return f"✅ HV Equalization 세션 시작: Target C={target_c}, S={target_s} | Tower={tower}"
    
    def process_suggestion(self, session_id: str, run_num: int, 
                          adc_c: float, adc_s: float,
                          hv_c: float, hv_s: float) -> Dict[str, Any]:
        """
        HV 조정 제안 생성 (HV Control Tool 형식으로 반환)
        
        Returns:
            {
                "status": "success",
                "tower": "M5T3",
                "run": 12345,
                "current": {"C": {"hv": 800, "adc": 1200}, "S": {"hv": 800, "adc": 1100}},
                "target": {"C": 1500, "S": 1500},
                "suggested": {"C": {"hv": 810, "change": 10}, "S": {"hv": 810, "change": 10}},
                "hv_control_params": {"command": "voltage", "channel_values": {"8": 810, "9": 810}},
                "message": "상세 메시지..."
            }
        """
        
        if session_id not in self.sessions:
            return {
                "status": "error",
                "message": "세션이 시작되지 않았습니다. 먼저 hv_equalization_start를 호출하세요."
            }
        
        session = self.sessions[session_id]
        session["iteration"] += 1
        
        # 이전 제안이 적용되었는지 확인하고 자동 학습
        fitting_updates = []
        if "last_suggestion" in session:
            auto_learn_messages = self._auto_learn_from_previous(session, hv_c, hv_s, adc_c, adc_s)
            fitting_updates.extend(auto_learn_messages)
        
        session["current_hv_c"] = hv_c
        session["current_hv_s"] = hv_s
        
        target_c = session["target_c"]
        target_s = session["target_s"]
        
        # C, S 채널 각각 예측
        result_c = self.exp_predictor.predict_hv_adjustment("C", hv_c, adc_c, target_c)
        result_s = self.exp_predictor.predict_hv_adjustment("S", hv_s, adc_s, target_s)
        
        if "error" in result_c:
            return {"status": "error", "message": result_c['error']}
        if "error" in result_s:
            return {"status": "error", "message": result_s['error']}
        
        # 제안된 HV 계산
        if result_c.get("status") == "done":
            change_c = 0
            next_hv_c = hv_c
            c_done = True
        else:
            change_c = result_c["hv_change"]
            next_hv_c = hv_c + change_c
            c_done = False
            
        if result_s.get("status") == "done":
            change_s = 0
            next_hv_s = hv_s
            s_done = True
        else:
            change_s = result_s["hv_change"]
            next_hv_s = hv_s + change_s
            s_done = False
        
        # 히스토리 기록
        session["history"].append({
            "run": run_num,
            "hv_c": hv_c, "hv_s": hv_s,
            "adc_c": adc_c, "adc_s": adc_s,
            "change_c": change_c, "change_s": change_s
        })
        
        # 마지막 제안 저장 (자동 학습용)
        error_c = abs(adc_c - target_c)
        error_s = abs(adc_s - target_s)
        session["last_suggestion"] = {
            "run_num": run_num,
            "suggested_hv_c": next_hv_c,
            "suggested_hv_s": next_hv_s,
            "suggested_change_c": change_c, 
            "suggested_change_s": change_s,
            "from_hv_c": hv_c,
            "from_hv_s": hv_s,
            "from_adc_c": adc_c,
            "from_adc_s": adc_s,
            "target_c": target_c,
            "target_s": target_s
        }
        
        # 절대 HV 상한 검사 (950V 초과 금지)
        HV_MAX = 2800
        over_limit = []
        if not c_done and next_hv_c >= HV_MAX:
            over_limit.append(f"C채널: {next_hv_c:.0f}V")
        if not s_done and next_hv_s >= HV_MAX:
            over_limit.append(f"S채널: {next_hv_s:.0f}V")
        if over_limit:
            return {
                "status": "error",
                "message": (
                    f"⚠️  HV 상한 초과 오류: 제안된 HV가 최대 허용값({HV_MAX}V)을 넘었습니다. "
                    f"({', '.join(over_limit)}) — HV를 변경하지 않았습니다. "
                    f"target_adc 값이 올바른지 확인하세요."
                )
            }

        # HV Control Tool 파라미터 생성 — 채널명은 현재 타워 기준 ({tower}C/{tower}S)
        tower = session.get("current_tower", "M5T3")
        ch_c = f"{tower}C"
        ch_s = f"{tower}S"
        hv_control_params = None

        if not c_done or not s_done:
            channel_values = {}
            if not c_done:
                channel_values[ch_c] = next_hv_c
            if not s_done:
                channel_values[ch_s] = next_hv_s
            
            hv_control_params = {
                "command": "voltage",
                "channel_values": channel_values
            }
        
        # 결과 메시지 생성
        message_lines = [f"Run {run_num} Analysis:", ""]
        message_lines.append("Current ADC:")
        message_lines.append(f"  C: {adc_c:.1f} (target: {target_c}, error: ±{error_c:.1f})")
        message_lines.append(f"  S: {adc_s:.1f} (target: {target_s}, error: ±{error_s:.1f})")
        message_lines.append("")
        message_lines.append("HV Adjustment:")
        
        if c_done:
            message_lines.append(f"  C: {hv_c:.1f}V → DONE (equalization 완료)")
        else:
            message_lines.append(f"  C: {hv_c:.1f}V → {change_c:+.0f}V → {next_hv_c:.1f}V")
            
        if s_done:
            message_lines.append(f"  S: {hv_s:.1f}V → DONE (equalization 완료)")
        else:
            message_lines.append(f"  S: {hv_s:.1f}V → {change_s:+.0f}V → {next_hv_s:.1f}V")
        
        if fitting_updates:
            message_lines.append("")
            message_lines.append("Exponential Fitting Updates:")
            for update in fitting_updates:
                message_lines.append(f"  {update}")
        
        # 결과 딕셔너리 구성
        result_dict = {
            "status": "success",
            "tower": tower,
            "run": run_num,
            "current": {
                "C": {"hv": hv_c, "adc": adc_c, "done": c_done},
                "S": {"hv": hv_s, "adc": adc_s, "done": s_done}
            },
            "target": {
                "C": target_c,
                "S": target_s
            },
            "suggested": {
                "C": {"hv": next_hv_c, "change": change_c, "done": c_done},
                "S": {"hv": next_hv_s, "change": change_s, "done": s_done}
            },
            "errors": {
                "C": error_c,
                "S": error_s
            },
            "message": "\n".join(message_lines)
        }
        
        # HV Control 파라미터 추가
        if hv_control_params:
            result_dict["hv_control_params"] = hv_control_params
        
        return result_dict
    
    def _auto_learn_from_previous(self, session: dict, current_hv_c: float, 
                                 current_hv_s: float, current_adc_c: float, 
                                 current_adc_s: float) -> list:
        """이전 AI 제안이 적용되었는지 확인하고 자동 학습"""
        
        messages = []
        
        last_suggestion = session["last_suggestion"]
        suggested_hv_c = last_suggestion["suggested_hv_c"] 
        suggested_hv_s = last_suggestion["suggested_hv_s"]
        
        hv_tolerance = 2
        
        # C 채널 학습
        if abs(current_hv_c - suggested_hv_c) <= hv_tolerance:
            try:
                fitting_msg = self.exp_predictor.add_data_point(
                    "C", current_hv_c, current_adc_c
                )
                if fitting_msg:
                    messages.append(fitting_msg)
            except Exception:
                pass
        
        # S 채널 학습
        if abs(current_hv_s - suggested_hv_s) <= hv_tolerance:
            try:
                fitting_msg = self.exp_predictor.add_data_point(
                    "S", current_hv_s, current_adc_s
                )
                if fitting_msg:
                    messages.append(fitting_msg)
            except Exception:
                pass
        
        session.pop("last_suggestion", None)
        
        return messages


# ======================= Global Session Manager =======================
_session_manager = HVEqualizationSession()


# ======================= Fitting Visualization =======================

def generate_fitting_summary(session_id: str = "default", tower: str = "T?",
                             run_number: int = 0) -> dict:
    """
    현재 세션의 HV Equalization 진행 상황을 텍스트 테이블로 요약.

    Returns:
        {"table": str, "plot_path": str | None, "equation": str}
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    session = _session_manager.sessions.get(session_id)
    predictor = _session_manager.exp_predictor
    history = session.get("history", []) if session else []
    target_c = session.get("target_c", 1500) if session else 1500
    target_s = session.get("target_s", 1500) if session else 1500

    # ── 텍스트 테이블 ─────────────────────────────────────────────
    lines = [f"{'Iter':>4} | {'HV_C':>6} | {'ADC_C':>8} | {'HV_S':>6} | {'ADC_S':>8} | Target"]
    lines.append("-" * 56)
    for idx, h in enumerate(history, 1):
        lines.append(
            f"{idx:>4} | {h['hv_c']:>5.0f}V | {h['adc_c']:>8.1f} | "
            f"{h['hv_s']:>5.0f}V | {h['adc_s']:>8.1f} | {target_c}"
        )
    table = "\n".join(lines) if history else "(아직 데이터 없음)"

    # ── 피팅 수식 문자열 ──────────────────────────────────────────
    eq_parts = []
    for ch in ("C", "S"):
        if predictor.is_fitted.get(ch) and predictor.coefficients.get(ch) is not None:
            A, B = predictor.coefficients[ch]
            eq_parts.append(f"{ch}: ADC = {A:.2f} * exp({B:.5f} * HV)")
        else:
            eq_parts.append(f"{ch}: 탐색 중 (데이터 부족)")
    equation = "  |  ".join(eq_parts)

    # ── matplotlib 그래프 (임시 파일로 생성) ─────────────────────────
    plot_path = None
    try:
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f"Tower {tower}  |  Run {run_number}  -  HV Equalization Fitting", fontsize=12)

        colors = {"C": "#4c72b0", "S": "#dd8452"}
        targets = {"C": target_c, "S": target_s}

        for ax, ch in zip(axes, ("C", "S")):
            pts = predictor.data_points.get(ch, [])
            hvs  = [p[0] for p in pts]
            adcs = [p[1] for p in pts]

            ax.set_title(f"{ch} Channel", fontsize=11)
            ax.set_xlabel("HV (V)")
            ax.set_ylabel("peakADC")

            if hvs:
                ax.scatter(hvs, adcs, color=colors[ch], zorder=5, s=60, label="Measured")
                for i, (hv, adc) in enumerate(zip(hvs, adcs), 1):
                    ax.annotate(str(i), (hv, adc), textcoords="offset points",
                                xytext=(6, 4), fontsize=8, color=colors[ch])

            if predictor.is_fitted.get(ch) and predictor.coefficients.get(ch) is not None:
                A, B = predictor.coefficients[ch]
                hv_min = min(hvs) - 20 if hvs else 750
                hv_max = max(hvs) + 40 if hvs else 950
                hv_range = np.linspace(hv_min, hv_max, 200)
                ax.plot(hv_range, A * np.exp(B * hv_range),
                        color=colors[ch], alpha=0.6, linewidth=1.8,
                        label=f"A={A:.1f}, B={B:.5f}")
                pred_hv = predictor.predict_hv_for_target(ch, targets[ch])
                if pred_hv is not None:
                    ax.axvline(pred_hv, color=colors[ch], linestyle="--", alpha=0.5,
                               label=f"Pred HV={pred_hv:.0f}V")

            ax.axhline(targets[ch], color="red", linestyle=":", linewidth=1.4, label=f"Target={targets[ch]}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_HV_fitting_{tower}_run{run_number}.png", delete=False
        )
        plt.savefig(tmp.name, dpi=120, bbox_inches="tight")
        plt.close(fig)
        plot_path = tmp.name

    except Exception as e:
        plot_path = None
        equation += f"  [plot error: {e}]"

    return {"table": table, "equation": equation, "plot_path": plot_path}



# ======================= LangChain Tools =======================

@tool
def hv_equalization_start(target_c: float, target_s: float, tower: str = "M5T3") -> str:
    """
    HV Equalization 세션을 시작합니다.
    
    Args:
        target_c: C 채널 목표 ADC 값
        target_s: S 채널 목표 ADC 값
        tower: 현재 타워 ID (기본값: T5)
    
    Returns:
        세션 시작 확인 메시지
    """
    try:
        session_id = "default"
        result = _session_manager.start_session(session_id, target_c, target_s, tower)
        return result
    except Exception as e:
        return f"ERROR: HV Equalization 세션 시작 실패: {str(e)}"


@tool
def hv_equalization_suggest(run_number: Optional[int] = None, hv_c: Optional[float] = None, 
                           hv_s: Optional[float] = None, tower: Optional[str] = None) -> str:
    """
    현재 run의 ADC 값을 분석하고 exponential fitting을 사용하여 HV 조정을 제안합니다.
    
    Args:
        run_number: 분석할 run 번호 (None이면 runnum.txt에서 자동 읽기)
        hv_c: 현재 C 채널 HV 값 (None이면 세션에서 가져옴)
        hv_s: 현재 S 채널 HV 값 (None이면 세션에서 가져옴)
        tower: 타워 위치 (예: "M5T3", None이면 세션에서 가져옴)
    
    Returns:
        HV 조정 제안 (JSON 형태, HV Control Tool로 넘길 수 있는 형식)
    """
    try:
        # Run number 결정
        if run_number is None:
            run_number = get_current_run_number()
            if run_number is None:
                return '{"status": "error", "message": "Run number를 가져올 수 없습니다."}'
        
        # 세션 확인
        session_id = "default"
        session = _session_manager.sessions.get(session_id)
        
        if session is None:
            return '{"status": "error", "message": "세션이 시작되지 않았습니다. 먼저 hv_equalization_start를 호출하세요."}'
        
        # Tower 결정
        if tower is None:
            tower = session.get("current_tower", "M5T3")
        
        # HV 값 결정
        if hv_c is None:
            hv_c = session.get("current_hv_c", 800)
        if hv_s is None:
            hv_s = session.get("current_hv_s", 800)
        
        # peakADC 계산
        avg_c, _ = calculate_valley_cut_average(run_number, 'C', tower)
        avg_s, _ = calculate_valley_cut_average(run_number, 'S', tower)
        
        if avg_c is None or avg_s is None:
            return f'{{"status": "error", "message": "Run {run_number}에서 ADC 데이터를 가져올 수 없습니다."}}'
        
        # HV 조정 제안
        result = _session_manager.process_suggestion(
            session_id, run_number, avg_c, avg_s, hv_c, hv_s
        )
        
        # Dict를 JSON 문자열로 변환
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"ERROR: HV 조정 제안 실패: {str(e)}"


@tool
def hv_equalization_done_channel(channels: str = "all") -> str:
    """
    채널별 HV equalization 완료를 표시합니다.
    
    Args:
        channels: 완료된 채널 ("C", "S", "all", 기본값: "all")
    
    Returns:
        채널 완료 상태 업데이트 결과
    """
    try:
        channels = channels.upper()
        
        if channels == "ALL" or channels == "":
            channels_to_complete = ["C", "S"]
            completion_type = "모든 채널"
        elif "C" in channels and "S" in channels:
            channels_to_complete = ["C", "S"]
            completion_type = "모든 채널"
        elif "C" in channels:
            channels_to_complete = ["C"]
            completion_type = "C 채널"
        elif "S" in channels:
            channels_to_complete = ["S"]
            completion_type = "S 채널"
        else:
            return "ERROR: 올바른 채널을 지정해주세요 (C, S, all)"
        
        for channel in channels_to_complete:
            _session_manager.exp_predictor.mark_channel_done(channel)
        
        c_status = "DONE" if not _session_manager.exp_predictor.is_channel_active("C") else "Active"
        s_status = "DONE" if not _session_manager.exp_predictor.is_channel_active("S") else "Active"
        
        if len(channels_to_complete) == 1:
            result = f"✅ {channels_to_complete[0]} 채널 equalization 완료!"
        else:
            result = "✅ 모든 채널 equalization 완료!"
        
        result += f"""

채널 상태:
  C: {c_status}
  S: {s_status}

완료된 채널은 더 이상 HV 조정 suggestion을 제공하지 않습니다."""
        
        # 모든 채널이 완료되면 자동으로 새 타워 준비
        if not _session_manager.exp_predictor.is_channel_active("C") and \
           not _session_manager.exp_predictor.is_channel_active("S"):
            reset_message = _session_manager.reset_for_new_tower("default")
            result += f"\n\n{reset_message}"
            
        return result
        
    except Exception as e:
        return f"ERROR: Done 처리 실패: {str(e)}"

