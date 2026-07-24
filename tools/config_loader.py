#!/usr/bin/env python3
"""Configuration Loader — config_general.yml 파싱 유틸리티"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config_general.yml"

_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """config_general.yml 파일을 읽어서 딕셔너리로 반환"""
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {CONFIG_FILE}\n"
            f"config_general.yml 파일이 존재하는지 확인하세요."
        )
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f) or {}
        return _config_cache
    except Exception as e:
        raise RuntimeError(f"설정 파일 읽기 실패: {e}")


def get_data_directory() -> str:
    """데이터 디렉토리 경로 반환 (BaseDirectory 사용)"""
    config = load_config()
    base_dir = config.get("BaseDirectory")
    if not base_dir:
        raise ValueError(
            "설정 파일에 'BaseDirectory'가 없습니다.\n"
            f"config_general.yml 파일에 다음을 추가하세요:\n"
            f"  BaseDirectory: \"/path/to/data\""
        )
    return str(base_dir)


def get_mapping_root_path() -> str:
    """매핑 ROOT 파일 경로 반환"""
    config = load_config()
    mapping = config.get("Mapping")
    if not mapping:
        raise ValueError(
            "설정 파일에 'Mapping'이 없습니다.\n"
            f"config_general.yml 파일에 다음을 추가하세요:\n"
            f"  Mapping: \"/path/to/mapping_KEK.root\""
        )

    if not os.path.isabs(mapping):
        dqm_dir = PROJECT_ROOT / "DQM"
        mapping_path = (dqm_dir / mapping).resolve()
    else:
        mapping_path = Path(mapping)

    return str(mapping_path)


def get_mapping_csv_path() -> str:
    """매핑 CSV 파일 경로 반환 (Mapping 경로에서 .root를 .csv로 변환)"""
    root_path = get_mapping_root_path()
    mapping_path = Path(root_path)
    mapping_csv_path = mapping_path.parent / (mapping_path.stem + ".csv")
    return str(mapping_csv_path)



def get_path_config(key: str) -> str:
    """설정 파일의 Paths 섹션에서 경로를 가져옴"""
    config = load_config()
    paths = config.get("Paths", {})
    val = paths.get(key)
    if not val:
        raise ValueError(f"설정 파일의 Paths 섹션에 '{key}'가 정의되지 않았습니다.")
    
    # SpreadsheetId는 경로가 아니므로 변환 제외
    if key == "SpreadsheetId":
        return str(val)
        
    # 상대 경로인 경우 프로젝트 루트와 결합하여 절대 경로로 변환
    if not os.path.isabs(str(val)):
        return str((PROJECT_ROOT / str(val)).resolve())
        
    return str(val)


def get_hv_config() -> Dict[str, Any]:
    """HV 설정 반환"""
    config = load_config()
    return config.get("HV", {})


def get_run_log_beam_type() -> str:
    """새 Run 로그 작성 시 사용할 기본 Beam Type 반환 (RunLog.BeamType)"""
    config = load_config()
    return str(config.get("RunLog", {}).get("BeamType", "e-"))




