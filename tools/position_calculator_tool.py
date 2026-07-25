#!/usr/bin/env python3
"""Position Calculator — M5T3 기준 타워 위치 계산"""

import math
from typing import Dict, Any, Optional
from .config_loader import load_config


# ======================= Tower Layout =======================
r"""
타워 레이아웃 (6x6 그리드, 9모듈 × 4타워 = 36타워):

    Row\Col  0     1     2     3     4     5
      0    M1T1  M1T2  M2T1  M2T2  M3T1  M3T2
      1    M1T3  M1T4  M2T3  M2T4  M3T3  M3T4
      2    M4T1  M4T2  M5T1  M5T2  M6T1  M6T2
      3    M4T3  M4T4  M5T3  M5T4  M6T3  M6T4   ← 기준행 row=3
      4    M7T1  M7T2  M8T1  M8T2  M9T1  M9T2
      5    M7T3  M7T4  M8T3  M8T4  M9T3  M9T4
                       ↑ 기준열 col=2 (M5T3)

M5T3가 기준점(dx=0, dy=0)이며, 오프셋은 xlsx(Position Calculator)와 동일하게
"Module 위치 + Tower 위치" 2단으로 분해해 계산한다 (Uniformity 사분면은 미사용):
  module_x = ((m-1) % 3  - 1) * 2 * TowerWidth      # M1/4/7=-, M2/5/8=0, M3/6/9=+
  module_y = ((m-1) // 3 - 1) * 2 * TowerHeight     # M1/2/3=-, M4/5/6=0, M7/8/9=+
  tower_x  = ((t-1) % 2)      * TowerWidth           # T1/T3=0, T2/T4=+
  tower_y  = (((t-1) // 2) - 1) * TowerHeight        # T1/T2=-, T3/T4=0
  dx = module_x + tower_x,   dy = module_y + tower_y
"""

VALID_TOWERS = [f"M{m}T{t}" for m in range(1, 10) for t in range(1, 5)]


# ======================= DQM Canvas Numbering =======================
# DQM(monit/TBplotengine::init_Generic, DQM/src/TBplotengine.cc)은 타워별 캔버스를
# "fCanvas_Tower{N}"으로 저장하는데, N은 타워명에서 바로 계산되는 게 아니라 매핑 CSV에서
# row>0·col>0·이름이 "-C"로 끝나고 isCeren==1인 이름들을 뽑아 사전식(lexicographic)
# 정렬한 뒤 매긴 1-based 순번이다 (예: "M5-T1" → 17). 하드코딩 공식((module-1)*4+sub)
# 대신 실제 매핑 파일을 그대로 읽어 C++ 알고리즘을 재현한다 — 모듈/타워 구성이 바뀌어도
# TBplotengine과 항상 일치하게 하기 위함.
_dqm_tower_rank_cache: Optional[Dict[str, int]] = None


def _load_dqm_tower_ranks() -> Dict[str, int]:
    global _dqm_tower_rank_cache
    if _dqm_tower_rank_cache is not None:
        return _dqm_tower_rank_cache

    from .config_loader import get_dqm_mapping_csv_path

    ranks: Dict[str, int] = {}
    try:
        names = []
        with open(get_dqm_mapping_csv_path()) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                name = parts[0]
                try:
                    is_ceren, row, col = int(parts[1]), int(parts[2]), int(parts[3])
                except ValueError:
                    continue
                if row <= 0 or col <= 0:
                    continue
                if is_ceren == 1 and name.endswith("-C"):
                    names.append(name)
        for i, name in enumerate(sorted(names), start=1):
            root = name[:-2].replace("-", "")  # "M5-T1-C" -> "M5T1"
            ranks[root] = i
    except Exception:
        ranks = {}

    _dqm_tower_rank_cache = ranks
    return ranks


def tower_to_dqm_canvas_number(tower: Any) -> str:
    """타워 이름("M5T1")을 DQM 라이브 캔버스 순번 문자열("17")로 변환.
    매핑에서 순번을 못 찾으면(파일 없음/이름 불일치 등) 입력을 그대로 반환한다 —
    기존에 값을 그대로 셀 이름에 꽂던 폴백 동작을 보존."""
    if not isinstance(tower, str):
        return str(tower)
    rank = _load_dqm_tower_ranks().get(tower.strip().upper())
    return str(rank) if rank is not None else tower


# ======================= Position Calculator =======================

class PositionCalculator_Sym:
    """타워 위치 계산기 — 대칭 모듈용 (xlsx `Position Calculator` 수식 그대로 반영)

    CERN 버전: Tilting은 모든 시나리오에서 0으로 고정되어 계산에 반영하지 않는다.
    xlsx와 동일하게 Y는 회전과 무관하게 `Y = OffsetY + dy`, X만 Rotation(수직축 회전)
    으로 보정된다. 호출부 호환을 위해 tilting 파라미터는 시그니처에 남겨두지만 무시한다.
    """

    def __init__(self):
        config = load_config()

        pos_scan = config.get("PositionScan") or {}
        pos_consts = config.get("PositionConstants") or {}

        for key in ["OffsetX", "OffsetY", "TowerWidth", "TowerHeight"]:
            if pos_scan.get(key) is None:
                raise RuntimeError(
                    f"config_general.yml PositionScan.{key} 가 정의되지 않았습니다."
                )

        for key in ["RotationAxisAngle", "RotationAxisDist"]:
            if pos_consts.get(key) is None:
                raise RuntimeError(
                    f"config_general.yml PositionConstants.{key} 가 정의되지 않았습니다."
                )

        self.offset_x = float(pos_scan["OffsetX"])
        self.offset_y = float(pos_scan["OffsetY"])
        self.tower_width = float(pos_scan["TowerWidth"])
        self.tower_height = float(pos_scan["TowerHeight"])

        self.rotation_axis_angle = float(pos_consts["RotationAxisAngle"])
        self.rotation_axis_dist  = float(pos_consts["RotationAxisDist"])

        self.rotation = 0.0
        self.tilting = 0.0

    def _tower_offset_x(self, tower: str) -> float:
        """타워별 X 오프셋 dx (M5T3 기준) = Module 위치 + Tower 위치"""
        m, t = int(tower[1]), int(tower[3])
        module_x = ((m - 1) % 3 - 1) * 2 * self.tower_width
        tower_x = ((t - 1) % 2) * self.tower_width
        return module_x + tower_x

    def _tower_offset_y(self, tower: str) -> float:
        """타워별 Y 오프셋 dy (M5T3 기준) = Module 위치 + Tower 위치"""
        m, t = int(tower[1]), int(tower[3])
        module_y = ((m - 1) // 3 - 1) * 2 * self.tower_height
        tower_y = (((t - 1) // 2) - 1) * self.tower_height
        return module_y + tower_y

    def calculate_tower_position(self, tower: str,
                                 rotation: Optional[float] = None,
                                 tilting: Optional[float] = None) -> Dict[str, float]:
        """
        특정 타워의 중심 위치 계산 (Rotation 적용, Tilting 미사용)

        xlsx 수식 (B46=X, C46=Y):
        - dx, dy = Module 위치 + Tower 위치  (M5T3 기준)
        - x = OffsetX + dx*cos(rot) - ( l*sin(axis) - l*sin(axis - rot) )
        - y = OffsetY + dy
              (rot=Rotation, axis=RotationAxisAngle, l=RotationAxisDist)
        """
        tower = tower.upper()
        if tower not in VALID_TOWERS:
            raise ValueError(f"유효하지 않은 타워: {tower}. {VALID_TOWERS} 중 하나여야 합니다.")

        if rotation is None:
            rotation = self.rotation
        # tilting 파라미터는 CERN 버전에서 0으로 고정 — 계산에 사용하지 않음

        dx = self._tower_offset_x(tower)
        dy = self._tower_offset_y(tower)

        # X: Rotation(수직축 회전) 보정
        axis_rad = math.radians(self.rotation_axis_angle)
        rot_rad = math.radians(rotation)
        rotation_term = (self.rotation_axis_dist * math.sin(axis_rad)
                         - self.rotation_axis_dist * math.sin(axis_rad - rot_rad))
        x = self.offset_x + dx * math.cos(rot_rad) - rotation_term

        # Y: Tilting 미사용 → dy 그대로
        y = self.offset_y + dy

        return {"x": x, "y": y}

    def calculate_all_positions(self, rotation: Optional[float] = None,
                                tilting: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """모든 타워의 위치 계산"""
        return {
            tower: self.calculate_tower_position(tower, rotation, tilting)
            for tower in VALID_TOWERS
        }

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 확인"""
        return {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "tower_spacing": {
                "x": self.tower_width,
                "y": self.tower_height,
            },
            "constants": {
                "rotation_axis_angle": self.rotation_axis_angle,
                "rotation_axis_dist":  self.rotation_axis_dist,
            },
            "all_positions": self.calculate_all_positions(),
        }


# ======================= Global Calculator =======================

_position_calculator = PositionCalculator_Sym()


# ======================= Direct Access Functions =======================

def calculate_position(tower: str) -> Dict[str, float]:
    """직접 접근용 함수 (tool decorator 없이)"""
    return _position_calculator.calculate_tower_position(tower)


def get_calculator() -> "PositionCalculator_Sym":
    """Calculator 객체 직접 접근"""
    return _position_calculator
