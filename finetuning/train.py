#!/usr/bin/env python3
"""
AutoTB 통합 Fine-tuning 스크립트
────────────────────────────────
사용법:
  python finetuning/train.py <agent>  [옵션]
  python finetuning/train.py all      [옵션]   # 모든 agent 순차 학습

agent 목록:
  brain | calibration | energy_scan | hv_equalization | position_scan

옵션 (모두 선택 사항):
  --epochs   INT    학습 epoch 수
  --lr       FLOAT  learning rate
  --batch    INT    per_device_train_batch_size
  --grad_acc INT    gradient_accumulation_steps
  --lora_r   INT    LoRA rank
  --alpha    INT    LoRA alpha
  --data     PATH   데이터 파일 경로 (기본값: finetuning/data/<agent_data>.json)
  --out      PATH   모델 출력 디렉터리

예시:
  python finetuning/train.py brain
  python finetuning/train.py brain --epochs 5 --lr 5e-5
  python finetuning/train.py calibration --epochs 10
  python finetuning/train.py all
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import re
import sys
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── 경로 설정: 어느 디렉터리에서 실행하든 동작 ──────────────
FINETUNING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT   = FINETUNING_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))   # config.py 접근
sys.path.insert(0, str(FINETUNING_DIR)) # training_utils.py 직접 접근

from config import AGENT_MODELS
from training_utils import (
    EpochLossCallback,
    MemoryCleanupCallback,
    plot_loss_curve,
    setup_clean_logging,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_LENGTH = 2048   # 시퀀스 최대 토큰 길이 (truncation 기준)

# eval/save 간격: 0.25 epoch마다 (epoch당 4회). eval_steps는 데이터셋 크기에서
# 자동 계산되므로 agent/데이터가 바뀌어도 "0.25 epoch마다" 빈도가 유지된다.
EVAL_EVERY_EPOCH_FRAC = 0.25


# ──────────────────────────────────────────────────────────
# Agent별 기본 하이퍼파라미터 + 데이터 파일
# ──────────────────────────────────────────────────────────

AGENT_DEFAULTS: dict[str, dict] = {
    "calibration": {
        "data_file": "calib_scan_data.json",
        "epochs":    3,
        "lora_r":    16,
        "lora_alpha":32,
        "lr":        2e-4,
        "batch":     4,
        "grad_acc":  8,
    },
    "energy_scan": {
        "data_file": "EM_scan_data.json",
        "epochs":    3,
        "lora_r":    16,
        "lora_alpha":32,
        "lr":        2e-4,
        "batch":     4,
        "grad_acc":  8,
    },
    "hv_equalization": {
        "data_file": "hv_equalization_data.json",
        "epochs":    3,
        "lora_r":    16,
        "lora_alpha":32,
        "lr":        2e-4,
        "batch":     4,
        "grad_acc":  8,
    },
    "position_scan": {
        "data_file": "position_scan_data.json",
        "epochs":    3,
        "lora_r":    16,
        "lora_alpha":32,
        "lr":        2e-4,
        "batch":     4,
        "grad_acc":  8,
    },
    "brain": {
        "data_file": "brain_data.json",
        "epochs":    3,
        "lora_r":    32,
        "lora_alpha":64,
        "lr":        5e-5,
        "batch":     4,
        "grad_acc":  8,
        "decision_weight": 3.0,
    },
}


# ──────────────────────────────────────────────────────────
# WeightedLossTrainer
# ──────────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    각 샘플의 precomputed weight_mask를 사용해 loss를 계산한다.
    weight_mask는 전처리 단계에서 미리 계산되어 배치에 포함된다.
    """

    def __init__(self, decision_weight: float = 5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_weight = decision_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # weight_mask: (batch, seq_len) — 배치에서 꺼내 모델에 전달하지 않음
        weight_mask = inputs.pop("weight_mask", None)

        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss_fct     = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        per_token    = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        if weight_mask is not None:
            # weight_mask를 shift_labels에 맞게 1칸 오른쪽 슬라이스
            weights = weight_mask[:, 1:].to(dtype=per_token.dtype, device=per_token.device)
        else:
            weights = torch.ones_like(per_token)

        loss = (per_token * weights).mean()
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────
# Dynamic-padding Collator
# ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class WeightedCollator:
    """
    배치 내 최대 길이로만 패딩 (dynamic padding).
    weight_mask도 함께 패딩한다.
    """
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        weight_masks = [f.pop("weight_mask") for f in features]

        seq_lens = [len(f["input_ids"]) for f in features]
        raw_max  = max(seq_lens)
        if self.pad_to_multiple_of > 1:
            batch_max = (
                (raw_max + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        else:
            batch_max = raw_max

        pad_id = self.tokenizer.pad_token_id

        input_ids_batch      = []
        attention_mask_batch = []
        labels_batch         = []

        for f in features:
            n       = len(f["input_ids"])
            pad_len = batch_max - n
            input_ids_batch.append(f["input_ids"] + [pad_id] * pad_len)
            attn = f.get("attention_mask", [1] * n)
            attention_mask_batch.append(attn + [0] * pad_len)
            labels_batch.append(f["labels"] + [-100] * pad_len)

        # weight_mask 패딩 (패딩 위치는 labels=-100이므로 loss=0, 값 무관)
        wm_tensor = torch.ones(len(features), batch_max, dtype=torch.float32)
        for i, wm in enumerate(weight_masks):
            l = min(len(wm), batch_max)
            wm_tensor[i, :l] = torch.tensor(wm[:l], dtype=torch.float32)

        return {
            "input_ids":      torch.tensor(input_ids_batch,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels":         torch.tensor(labels_batch,         dtype=torch.long),
            "weight_mask":    wm_tensor,
        }


# ──────────────────────────────────────────────────────────
# 전처리 유틸
# ──────────────────────────────────────────────────────────

def _compute_weight_mask(
    input_ids: list[int],
    tokenizer,
    decision_weight: float = 5.0,
) -> list[float]:
    """
    assistant / user 블록 토큰에 decision_weight, 나머지는 1.0을 부여한다.
    전처리 단계에서 한 번만 계산되어 데이터셋에 저장된다.
    """
    weights = [1.0] * len(input_ids)
    try:
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        for pattern in (
            r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
        ):
            for match in re.finditer(pattern, full_text, re.DOTALL):
                part_tokens = tokenizer.encode(
                    full_text[match.start(): match.end()],
                    add_special_tokens=False,
                )
                for i in range(len(input_ids) - len(part_tokens) + 1):
                    if input_ids[i: i + len(part_tokens)] == part_tokens:
                        for pos in range(i, i + len(part_tokens)):
                            if pos < len(weights):
                                weights[pos] = decision_weight
                        break
    except Exception:
        pass
    return weights


def preprocess_function(
    examples: dict,
    tokenizer,
    decision_weight: float = 5.0,
) -> dict:
    """
    chat template 적용 → 토크나이징 (패딩 없음) → weight_mask 사전 계산.
    패딩은 WeightedCollator가 배치 단위로 처리한다.
    """
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    model_inputs = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,          # 패딩 없음 — dynamic padding
        return_attention_mask=True,
    )
    model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
    model_inputs["weight_mask"] = [
        _compute_weight_mask(ids, tokenizer, decision_weight)
        for ids in model_inputs["input_ids"]
    ]
    return model_inputs


def _cache_path(agent_key: str, data_path: str, split: str) -> str:
    """데이터 파일 내용 기반 캐시 키 생성 (data 변경 시 자동 무효화)."""
    data_file = Path(data_path)
    mtime     = int(data_file.stat().st_mtime) if data_file.exists() else 0
    key       = hashlib.md5(
        f"{agent_key}_{data_path}_{mtime}_{MAX_LENGTH}".encode()
    ).hexdigest()[:10]
    cache_dir = FINETUNING_DIR / "data" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{agent_key}_{split}_{key}.arrow")


# ──────────────────────────────────────────────────────────
# 핵심 학습 함수
# ──────────────────────────────────────────────────────────

def finetune(agent_key: str, cfg: dict) -> None:
    setup_clean_logging()

    label      = AGENT_MODELS[agent_key]["description"]
    base_model = AGENT_MODELS[agent_key]["base_model"]
    data_path  = cfg["data_path"]
    output_dir = cfg["output_dir"]
    log_dir    = Path(cfg["log_dir"])

    logger.info("=" * 60)
    logger.info(f"  {label}")
    logger.info("=" * 60)
    logger.info(f"  Base model : {base_model}")
    logger.info(f"  Data       : {data_path}")
    logger.info(f"  Output     : {output_dir}")
    logger.info(
        f"  epochs={cfg['epochs']}  lr={cfg['lr']}  "
        f"batch={cfg['batch']}  grad_acc={cfg['grad_acc']}  "
        f"lora_r={cfg['lora_r']}  lora_alpha={cfg['lora_alpha']}"
    )

    # 1. Dataset
    raw = load_dataset("json", data_files=data_path)["train"]
    if cfg.get("max_samples"):
        raw = raw.shuffle(seed=42).select(range(min(cfg["max_samples"], len(raw))))
        logger.info(f"  max_samples={cfg['max_samples']} (테스트 모드)")
    dataset = raw.train_test_split(test_size=0.1, seed=42)
    logger.info(f"  Train: {len(dataset['train'])}  /  Val: {len(dataset['test'])}")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 전처리 (dynamic padding — 배치 단위 패딩은 WeightedCollator에서)
    #    데이터 파일 mtime 기반 캐시 → 데이터 변경 시 자동 재계산
    tokenized = dataset.map(
        lambda x: preprocess_function(x, tokenizer, cfg["decision_weight"]),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        cache_file_names={
            "train": _cache_path(agent_key, data_path, "train"),
            "test":  _cache_path(agent_key, data_path, "test"),
        },
    )
    logger.info(f"  Max length cap: {MAX_LENGTH} tokens  (truncation threshold)")

    # 4. Device / dtype
    if torch.backends.mps.is_available():
        device, train_dtype = "mps",  torch.bfloat16
        logger.info("  Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device, train_dtype = "cuda", torch.bfloat16
        logger.info("  Device: CUDA")
    else:
        device, train_dtype = "cpu",  torch.float32
        logger.info("  Device: CPU")

    # 5. Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=train_dtype, trust_remote_code=True
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.enable_input_require_grads()
    model = model.to(device)
    if device == "mps":
        torch.mps.empty_cache()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"  LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 6. eval/save 간격을 0.25 epoch(= epoch당 4회)로 계산.
    #    optimizer step 수 = ceil(#샘플 / batch) // grad_acc  (HF 계산 방식과 동일)
    batches_per_epoch = math.ceil(len(tokenized["train"]) / cfg["batch"])
    steps_per_epoch   = max(1, batches_per_epoch // cfg["grad_acc"])
    total_steps       = steps_per_epoch * cfg["epochs"]
    #    eval/save는 epoch마다. eval_steps는 train loss logging 간격 계산에만 사용.
    eval_steps        = max(1, round(steps_per_epoch * EVAL_EVERY_EPOCH_FRAC))
    logger.info(
        f"  Eval/save 간격: epoch마다 (총 {cfg['epochs']}회 / {total_steps} steps)"
    )

    # 7. Training args
    #    eval/save는 step 기반 — eval_steps를 데이터셋 크기에서 계산하므로
    #    데이터가 작아도 eval이 반드시 여러 번 수행된다.
    #    load_best_model_at_end=True는 save/eval strategy가 같아야 하고
    #    save_steps가 eval_steps의 배수여야 함 → save_steps=eval_steps로 맞춤.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=4,   # eval은 backward가 없어 train보다 크게 잡아 속도 개선
        gradient_accumulation_steps=cfg["grad_acc"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=max(1, eval_steps // 2),  # eval보다 촘촘하게 train loss 기록
        eval_strategy="epoch",        # epoch마다 eval
        save_strategy="epoch",        # epoch마다 저장
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=(device == "cpu"),
        gradient_checkpointing=True,
        optim="adamw_torch",
        group_by_length=True,
        dataloader_num_workers=2,
        bf16=(device != "cpu"),       # CPU는 bf16 미지원
    )

    # 8. Callback + Collator
    loss_callback = EpochLossCallback(agent_name=agent_key, log_dir=log_dir)
    collator      = WeightedCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    # 9. Trainer
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = WeightedLossTrainer(
        decision_weight=cfg["decision_weight"],
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[loss_callback, MemoryCleanupCallback()],
    )

    # 10. Train
    trainer.train()

    # 11. Loss curve 이미지 저장
    plot_loss_curve(
        log_history=trainer.state.log_history,
        output_path=log_dir / f"{agent_key}_{ts}_loss.png",
        title=f"{label} — Loss Curve",
    )

    # 12. 모델 저장 (LoRA adapter + merged full model)
    lora_out  = Path(output_dir) / "final_lora"
    final_out = Path(output_dir) / "final"
    model.save_pretrained(str(lora_out))
    tokenizer.save_pretrained(str(lora_out))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(final_out))
    tokenizer.save_pretrained(str(final_out))
    logger.info(f"  완료 → {final_out}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def build_cfg(agent_key: str, args: argparse.Namespace) -> dict:
    d = AGENT_DEFAULTS[agent_key]
    data_path = (
        args.data if args.data
        else str(FINETUNING_DIR / "data" / d["data_file"])
    )
    output_dir = (
        args.out if args.out
        else str(PROJECT_ROOT / "models" / f"{agent_key}_agent")
    )
    return {
        "data_path":       data_path,
        "output_dir":      output_dir,
        "log_dir":         str(FINETUNING_DIR / "logs"),
        "epochs":          args.epochs   if args.epochs   is not None else d["epochs"],
        "lora_r":          args.lora_r   if args.lora_r   is not None else d["lora_r"],
        "lora_alpha":      args.alpha    if args.alpha    is not None else d["lora_alpha"],
        "lr":              args.lr       if args.lr       is not None else d["lr"],
        "batch":           args.batch    if args.batch    is not None else d["batch"],
        "grad_acc":        args.grad_acc if args.grad_acc is not None else d["grad_acc"],
        "decision_weight": d.get("decision_weight", 5.0),
        "max_samples":     getattr(args, "max_samples", None),
    }


def main():
    parser = argparse.ArgumentParser(
        description="AutoTB 통합 Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "agent",
        choices=list(AGENT_DEFAULTS.keys()) + ["all"],
        help="학습할 agent (또는 all)",
    )
    parser.add_argument("--epochs",   type=int,   default=None, help="학습 epoch 수")
    parser.add_argument("--lr",       type=float, default=None, help="learning rate")
    parser.add_argument("--batch",    type=int,   default=None, help="train batch size")
    parser.add_argument("--grad_acc", type=int,   default=None, help="gradient accumulation steps")
    parser.add_argument("--lora_r",   type=int,   default=None, help="LoRA rank")
    parser.add_argument("--alpha",    type=int,   default=None, help="LoRA alpha")
    parser.add_argument("--data",        type=str,   default=None, help="데이터 파일 경로")
    parser.add_argument("--out",         type=str,   default=None, help="모델 출력 경로")
    parser.add_argument("--max-samples", type=int,   default=None, help="데이터셋 최대 샘플 수 (테스트용)")
    args = parser.parse_args()

    agents = list(AGENT_DEFAULTS.keys()) if args.agent == "all" else [args.agent]
    (FINETUNING_DIR / "logs").mkdir(parents=True, exist_ok=True)

    for agent_key in agents:
        finetune(agent_key, build_cfg(agent_key, args))


if __name__ == "__main__":
    main()
