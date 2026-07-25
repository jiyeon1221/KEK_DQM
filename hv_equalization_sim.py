#!/usr/bin/env python3
"""
HV Equalization Simulation Controller
HV 읽기/쓰기(status/voltage)는 실제 하드웨어 사용.
ADC 측정(hv_equalization_suggest)만 시뮬레이션.
"""

from tools.hv_equalization_tool import hv_equalization_start
from agents.hv_equalization_sim_agent import HVEqualizationSimAgent
from agents.agent_runner import TOWER_ORDER


def main():
    print("\n" + "=" * 60)
    print("⚡ HV Equalization Scan (Simulation Mode)")
    print("=" * 60)

    # Config 수집 (한 번만)
    beam_energy = float(input("빔 에너지 (GeV): ").strip())
    target_events = int(input("이벤트 수: ").strip())
    target_adc = float(input("목표 peakADC: ").strip())

    # HV Equalization 세션 초기화
    result = hv_equalization_start.invoke({
        "target_c": target_adc,
        "target_s": target_adc,
        "tower": TOWER_ORDER[0],
    }) if hasattr(hv_equalization_start, "invoke") else hv_equalization_start(
        target_c=target_adc, target_s=target_adc, tower=TOWER_ORDER[0]
    )
    print(result)

    # 타워 루프
    for i, tower in enumerate(TOWER_ORDER):
        print(f"\n[{i + 1}/{len(TOWER_ORDER)}] {tower} 타워 시작 (HV 실제 / ADC 시뮬레이션)")
        agent = HVEqualizationSimAgent(
            tower=tower,
            beam_energy=beam_energy,
            target_events=target_events,
            target_adc=target_adc,
            use_base_model=False,
        )
        with agent:
            agent.run()
        print(f"✅ {tower} 완료")

    print("\n🎉 모든 타워 HV Equalization 완료!")


if __name__ == "__main__":
    main()
