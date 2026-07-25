#!/usr/bin/env python3
"""
Energy Scan Agent 실행
"""

import json
from agents import EnergyScanAgent
from agents.agent_runner import _parse_energy_config


def main():
    print("\n" + "="*70)
    print("⚡ Energy Scan Agent")
    print("="*70)
    print("\n 에너지 설정을 입력하세요.")

    energy_config = {}
    while not energy_config:
        energy_input = input("\n입력: ").strip()
        energy_config = _parse_energy_config(energy_input)
        if not energy_config:
            print("⚠️  형식을 인식하지 못했습니다. 다시 입력해주세요.")

    print(f"\n✅ 파싱 완료: {energy_config}")

    agent = EnergyScanAgent(
        energy_config=energy_config,
        use_base_model=False,
    )

    # Synthetic STEP-0 history — 모델이 에너지 설정을 다시 묻지 않도록
    energy_config_snapshot = {
        str(k): {
            "target_events": v, "collected_events": 0,
            "runs": [], "completed": False, "completed_at": None,
        }
        for k, v in energy_config.items()
    }
    agent.add_to_history("user", energy_input)
    agent.add_to_history("assistant", json.dumps({
        "tool": "none",
        "update_state": {
            "energy_config": energy_config_snapshot,
            "scan_order": sorted(energy_config.keys()),
            "phase": "idle",
        },
    }, ensure_ascii=False))

    with agent:
        agent.run()


if __name__ == "__main__":
    main()
