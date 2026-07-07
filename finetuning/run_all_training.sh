#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_all_training.sh
# 데이터 생성 → 학습을 순차적으로 실행 (메모리 부족으로 병렬 불가)
# 실행: bash finetuning/run_all_training.sh [agent1 agent2 ...]
#
# 예시:
#   bash finetuning/run_all_training.sh               # 전체 파이프라인
#   bash finetuning/run_all_training.sh brain         # brain만
#   bash finetuning/run_all_training.sh calibration energy_scan
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SUMMARY_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

# 가상환경 활성화 (경로가 다르면 수정)
if [ -f "$PROJECT_DIR/ai/bin/activate" ]; then
    source "$PROJECT_DIR/ai/bin/activate"
fi

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

run_step() {
    local step_name="$1"; shift
    local step_log="$LOG_DIR/${step_name}_${TIMESTAMP}.log"

    log "────────────────────────────────────────────────────"
    log "▶ START : $step_name"
    log "  Log   : $step_log"

    local start_ts
    start_ts=$(date +%s)

    if python "$@" 2>&1 | tee "$step_log"; then
        local elapsed=$(( $(date +%s) - start_ts ))
        log "✅ DONE  : $step_name  (${elapsed}s)"
    else
        log "❌ FAILED: $step_name — 파이프라인 중단"
        exit 1
    fi
}

# ──────────────────────────────────────────────────────────────

log "════════════════════════════════════════════════════════"
log "  AutoTB Training Pipeline  ($TIMESTAMP)"
log "  Project : $PROJECT_DIR"
log "  Python  : $(python --version 2>&1)"
log "════════════════════════════════════════════════════════"

PIPELINE_START=$(date +%s)

# 학습할 agent 목록 결정 (인수 없으면 전체)
if [ $# -gt 0 ]; then
    AGENTS=("$@")
else
    AGENTS=(calibration energy_scan hv_equalization position_scan brain)
fi

# ── Data generation ───────────────────────────────────────────
get_data_gen_script() {
    case "$1" in
        calibration)     echo "$SCRIPT_DIR/calib_data_gen.py" ;;
        energy_scan)     echo "$SCRIPT_DIR/EM_data_gen.py" ;;
        hv_equalization) echo "$SCRIPT_DIR/hv_equalization_data_gen.py" ;;
        position_scan)   echo "$SCRIPT_DIR/position_scan_data_gen.py" ;;
        brain)           echo "$SCRIPT_DIR/brain_data_gen.py" ;;
        *)               echo "" ;;
    esac
}

for agent in "${AGENTS[@]}"; do
    script="$(get_data_gen_script "$agent")"
    if [ -n "$script" ] && [ -f "$script" ]; then
        run_step "${agent}_data_gen" "$script"
    else
        log "⚠ data_gen 스크립트 없음 (건너뜀): $agent"
    fi
done

# ── Training ──────────────────────────────────────────────────
for agent in "${AGENTS[@]}"; do
    run_step "${agent}_train" \
        "$SCRIPT_DIR/train.py" "$agent"
done

# ──────────────────────────────────────────────────────────────

TOTAL=$(( $(date +%s) - PIPELINE_START ))
HOURS=$(( TOTAL / 3600 ))
MINS=$(( (TOTAL % 3600) / 60 ))
SECS=$(( TOTAL % 60 ))

log "════════════════════════════════════════════════════════"
log "  전체 파이프라인 완료!"
log "  총 소요 시간: ${HOURS}h ${MINS}m ${SECS}s"
log "  요약 로그: $SUMMARY_LOG"
log "════════════════════════════════════════════════════════"
