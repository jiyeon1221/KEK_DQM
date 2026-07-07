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

M5T3가 기준점(dx=0, dy=0)이며, 수식으로 오프셋 계산됨.
  grid_col = 2 * ((m-1) % 3) + (t-1) % 2
  grid_row = 2 * ((m-1) // 3) + (t-1) // 2
"""

VALID_TOWERS = [f"M{m}T{t}" for m in range(1, 10) for t in range(1, 5)]


# ======================= Position Calculator =======================

class PositionCalculator_Sym:
    """타워 위치 계산기 — 대칭 모듈용 (스프레드시트 수식 구조 반영)"""

    def __init__(self):
        config = load_config()

        pos_scan = config.get("PositionScan") or {}
        pos_consts = config.get("PositionConstants") or {}

        for key in ["OffsetX", "OffsetY", "TowerWidth", "TowerHeight"]:
            if pos_scan.get(key) is None:
                raise RuntimeError(
                    f"config_general.yml PositionScan.{key} 가 정의되지 않았습니다."
                )

        for key in ["RotationAxisAngle", "RotationAxisDist", "CenterToBottom", "AxisToModule"]:
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
        self.center_to_bottom    = float(pos_consts["CenterToBottom"])
        self.axis_to_module      = float(pos_consts["AxisToModule"])

        self.rotation = 0.0
        self.tilting = 0.0

    def _tower_offset_x(self, tower: str) -> float:
        """타워별 X 방향 오프셋 (M5T3 기준, ref col=2)"""
        m, t = int(tower[1]), int(tower[3])
        grid_col = 2 * ((m - 1) % 3) + (t - 1) % 2
        return -(grid_col - 2) * self.tower_width  # ref: M5T3 col=2

    def _tower_offset_y(self, tower: str) -> float:
        """타워별 Y 방향 오프셋 (M5T3 기준, ref row=3)"""
        m, t = int(tower[1]), int(tower[3])
        grid_row = 2 * ((m - 1) // 3) + (t - 1) // 2
        return (grid_row - 3) * self.tower_height  # ref: M5T3 row=3

    def calculate_tower_position(self, tower: str,
                                 rotation: Optional[float] = None,
                                 tilting: Optional[float] = None) -> Dict[str, float]:
        """
        특정 타워의 중심 위치 계산 (Rotation/Tilting 적용)

        계산 구조:
        - B43 = 타워별 X 오프셋 (SWITCH 함수)
        - C43 = 타워별 Y 오프셋 (SWITCH 함수, ±TowerHeight)
        - x = B43 + offset_x + rotation_term
        - y = offset_y + C43 * cos(θ)
              - ( center_to_bottom * cos(θ) + axis_to_module * sin(θ) - center_to_bottom )
        """
        tower = tower.upper()
        if tower not in VALID_TOWERS:
            raise ValueError(f"유효하지 않은 타워: {tower}. {VALID_TOWERS} 중 하나여야 합니다.")

        if rotation is None:
            rotation = self.rotation
        if tilting is None:
            tilting = self.tilting

        b45 = self._tower_offset_x(tower)  # B45 = B43
        c45 = self._tower_offset_y(tower)  # C45 = C43

        # Rotation 보정 (B46)
        if rotation != 0.0:
            base_rad = math.radians(self.rotation_axis_angle)
            rot_rad = math.radians(rotation)
            rotation_term = (self.rotation_axis_dist * math.sin(base_rad + rot_rad)
                             - self.rotation_axis_dist * math.sin(base_rad))
            x = b45 + self.offset_x + rotation_term
        else:
            x = b45 + self.offset_x

        # Tilting 보정 (C46)
        tilt_rad = math.radians(tilting)
        tilting_correction = (
            self.center_to_bottom * math.cos(tilt_rad)
            + self.axis_to_module * math.sin(tilt_rad)
            - self.center_to_bottom
        )
        y = self.offset_y + c45 * math.cos(tilt_rad) - tilting_correction

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
                "center_to_bottom":    self.center_to_bottom,
                "axis_to_module":      self.axis_to_module,
            },
            "all_positions": self.calculate_all_positions(),
        }


# ======================= Unsymmetric Position Calculator =======================

class PositionCalculator:
    """
    타워 위치 계산기 — 비대칭 모듈용

    모듈 크기가 모두 다를 경우 사용.
    TowerWidth/TowerHeight 기반 균일 간격 대신,
    기준 타워(M5T3)로부터 각 타워까지의 x,y 거리를 아래 TOWER_OFFSETS에 직접 기재한다.

    Rotation/Tilting 보정 방식은 PositionCalculator_Sym과 동일.
    """

    # ── 타워별 M5T3 기준 상대 거리 (mm) ── 직접 수정하세요 ──────────────────
    # Approximate values based on uniform 46.75mm × 49.0mm grid; refine with actual measurements
    #
    # grid_col = 2*((m-1)%3) + (t-1)%2
    # grid_row = 2*((m-1)//3) + (t-1)//2
    # dx = -(grid_col - 2) * 46.75,  dy = (grid_row - 3) * 49.0
    TOWER_OFFSETS: Dict[str, Dict[str, float]] = {
        "M1T1": {"dx":  93.5,  "dy": -147.0},
        "M1T2": {"dx":  46.75, "dy": -147.0},
        "M1T3": {"dx":  93.5,  "dy":  -98.0},
        "M1T4": {"dx":  46.75, "dy":  -98.0},
        "M2T1": {"dx":   0.0,  "dy": -147.0},
        "M2T2": {"dx": -46.75, "dy": -147.0},
        "M2T3": {"dx":   0.0,  "dy":  -98.0},
        "M2T4": {"dx": -46.75, "dy":  -98.0},
        "M3T1": {"dx": -93.5,  "dy": -147.0},
        "M3T2": {"dx": -140.25,"dy": -147.0},
        "M3T3": {"dx": -93.5,  "dy":  -98.0},
        "M3T4": {"dx": -140.25,"dy":  -98.0},
        "M4T1": {"dx":  93.5,  "dy":  -49.0},
        "M4T2": {"dx":  46.75, "dy":  -49.0},
        "M4T3": {"dx":  93.5,  "dy":    0.0},
        "M4T4": {"dx":  46.75, "dy":    0.0},
        "M5T1": {"dx":   0.0,  "dy":  -49.0},
        "M5T2": {"dx": -46.75, "dy":  -49.0},
        "M5T3": {"dx":   0.0,  "dy":    0.0},
        "M5T4": {"dx": -46.75, "dy":    0.0},
        "M6T1": {"dx": -93.5,  "dy":  -49.0},
        "M6T2": {"dx": -140.25,"dy":  -49.0},
        "M6T3": {"dx": -93.5,  "dy":    0.0},
        "M6T4": {"dx": -140.25,"dy":    0.0},
        "M7T1": {"dx":  93.5,  "dy":   49.0},
        "M7T2": {"dx":  46.75, "dy":   49.0},
        "M7T3": {"dx":  93.5,  "dy":   98.0},
        "M7T4": {"dx":  46.75, "dy":   98.0},
        "M8T1": {"dx":   0.0,  "dy":   49.0},
        "M8T2": {"dx": -46.75, "dy":   49.0},
        "M8T3": {"dx":   0.0,  "dy":   98.0},
        "M8T4": {"dx": -46.75, "dy":   98.0},
        "M9T1": {"dx": -93.5,  "dy":   49.0},
        "M9T2": {"dx": -140.25,"dy":   49.0},
        "M9T3": {"dx": -93.5,  "dy":   98.0},
        "M9T4": {"dx": -140.25,"dy":   98.0},
    }
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self):
        config = load_config()

        pos_scan   = config.get("PositionScan")    or {}
        pos_consts = config.get("PositionConstants") or {}

        for key in ["OffsetX", "OffsetY"]:
            if pos_scan.get(key) is None:
                raise RuntimeError(
                    f"config_general.yml PositionScan.{key} 가 정의되지 않았습니다."
                )

        for key in ["RotationAxisAngle", "RotationAxisDist", "CenterToBottom", "AxisToModule"]:
            if pos_consts.get(key) is None:
                raise RuntimeError(
                    f"config_general.yml PositionConstants.{key} 가 정의되지 않았습니다."
                )

        self.offset_x = float(pos_scan["OffsetX"])
        self.offset_y = float(pos_scan["OffsetY"])

        self.rotation_axis_angle = float(pos_consts["RotationAxisAngle"])
        self.rotation_axis_dist  = float(pos_consts["RotationAxisDist"])
        self.center_to_bottom    = float(pos_consts["CenterToBottom"])
        self.axis_to_module      = float(pos_consts["AxisToModule"])

        self.rotation = 0.0
        self.tilting  = 0.0

    def calculate_tower_position(self, tower: str,
                                  rotation: Optional[float] = None,
                                  tilting: Optional[float] = None) -> Dict[str, float]:
        """
        특정 타워의 중심 위치 계산 (Rotation/Tilting 적용)

        계산 구조:
        - dx, dy = TOWER_OFFSETS[tower]  (M5T3 기준 상대 거리)
        - x = offset_x + dx + rotation_term
        - y = offset_y + dy * cos(θ)
              - ( center_to_bottom * cos(θ) + axis_to_module * sin(θ) - center_to_bottom )
        """
        tower = tower.upper()
        if tower not in VALID_TOWERS:
            raise ValueError(f"유효하지 않은 타워: {tower}. {VALID_TOWERS} 중 하나여야 합니다.")

        if rotation is None:
            rotation = self.rotation
        if tilting is None:
            tilting = self.tilting

        dx = self.TOWER_OFFSETS[tower]["dx"]
        dy = self.TOWER_OFFSETS[tower]["dy"]

        # Rotation 보정
        if rotation != 0.0:
            base_rad = math.radians(self.rotation_axis_angle)
            rot_rad  = math.radians(rotation)
            rotation_term = (self.rotation_axis_dist * math.sin(base_rad + rot_rad)
                             - self.rotation_axis_dist * math.sin(base_rad))
            x = dx + self.offset_x + rotation_term
        else:
            x = dx + self.offset_x

        # Tilting 보정
        tilt_rad = math.radians(tilting)
        tilting_correction = (
            self.center_to_bottom * math.cos(tilt_rad)
            + self.axis_to_module * math.sin(tilt_rad)
            - self.center_to_bottom
        )
        y = self.offset_y + dy * math.cos(tilt_rad) - tilting_correction

        return {"x": x, "y": y}

    def calculate_all_positions(self, rotation: Optional[float] = None,
                                 tilting: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """모든 타워의 위치 계산"""
        return {
            tower: self.calculate_tower_position(tower, rotation, tilting)
            for tower in self.TOWER_OFFSETS
        }

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 확인"""
        return {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "tower_offsets": self.TOWER_OFFSETS,
            "constants": {
                "rotation_axis_angle": self.rotation_axis_angle,
                "rotation_axis_dist":  self.rotation_axis_dist,
                "center_to_bottom":    self.center_to_bottom,
                "axis_to_module":      self.axis_to_module,
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
