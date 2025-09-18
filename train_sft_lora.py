# train_sft_lora.py
import os, json, argparse, random
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


# ------------------------
# 1) CLI 인자
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--train_file", type=str, default="data/sft.train.jsonl")
    ap.add_argument("--val_file", type=str, default="data/sft.val.jsonl")
    ap.add_argument("--out_dir", type=str, default="outputs/docent-tinyllama-lora")
    ap.add_argument("--seed", type=int, default=42)

    # 학습 세팅(6GB VRAM 대응)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)     # per-device
    ap.add_argument("--grad_accum", type=int, default=12)    # 메모리 절약
    ap.add_argument("--max_seq_len", type=int, default=640)  # 6GB 권장: 640~768
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_strategy", type=str, default="epoch")  # steps보다 메모리/시간 절약
    ap.add_argument("--eval_steps", type=int, default=500)

    # dtype / 양자화
    ap.add_argument("--bf16", action="store_true", help="가능하면 사용(에러시 빼기)")
    ap.add_argument("--use_4bit", action="store_true", help="QLoRA; Windows bnb 문제시 끄기")

    return ap.parse_args()


# ------------------------
# 2) 프롬프트 → 텍스트 포맷팅
#    (우리는 내부 JSON 습관을 학습시키는 목적)
# ------------------------
INSTRUCTION_SYSTEM = (
    "너는 미술관 도슨트다. 사용자의 질문을 보고 반드시 아래 JSON 스키마에 맞춰 한국어로 답하라. "
    "과장은 피하고, 사실은 정확하게 말하라.\n"
    '{ "intro": string, "facts": string[], "interpretation": string, '
    '"tips": string[], "followup": string, "sources": string[] }\n'
    "JSON만 출력하라."
)

def format_example(example: Dict[str, Any], tokenizer) -> str:
    prompt = example["prompt"]
    target_json = json.dumps(example["completion"], ensure_ascii=False)

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION_SYSTEM},
                {"role": "user", "content": f"질문: {prompt}\nJSON만 출력."},
                {"role": "assistant", "content": target_json},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = (
            f"<s>[SYSTEM]\n{INSTRUCTION_SYSTEM}\n"
            f"[USER]\n질문: {prompt}\nJSON만 출력.\n"
            f"[ASSISTANT]\n{target_json}"
        )
    return text


# ------------------------
# 3) TRL 0.23 방식: formatting_func
#    (dataset에서 "text" 컬럼을 읽어서 그대로 사용)
# ------------------------
def formatting_func(batch):
    # map에서 만든 "text" 리스트를 그대로 리턴
    return batch["text"]


# ------------------------
# 4) 메인
# ------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # 데이터 로드
    train_ds = load_dataset("json", data_files=args.train_file)["train"]
    val_ds   = load_dataset("json", data_files=args.val_file)["train"]

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len  # 길이 제한을 토크나이저에 반영

    # 텍스트 컬럼 생성
    def map_fn(ex):
        return {"text": format_example(ex, tokenizer)}
    train_text = train_ds.map(map_fn, remove_columns=train_ds.column_names, batched=False)
    val_text   = val_ds.map(map_fn, remove_columns=val_ds.column_names, batched=False)

    # 모델 로드 (6GB VRAM 대응)
    qconf = None
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    if args.use_4bit:
        # QLoRA: 4bit 양자화 로드 (Windows에서 bnb 이슈면 이 옵션 빼기)
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=dtype if not args.use_4bit else None,  # 4bit면 dtype은 bnb가 관리
        quantization_config=qconf,
        attn_implementation="eager",  # Windows/VRAM 대비 안전
    )

    # 메모리 절약
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        # 구버전 호환
        model.gradient_checkpointing_enable()

    # LoRA (가볍게)
    lora = LoraConfig(
        r=8,                 # 16 -> 8
        lora_alpha=16,       # 32 -> 16
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # q,k,v,o -> q,v만
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # 학습 인자 (TRL/Transformers 최신 시그니처)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,  # "epoch" 권장 (메모리/시간 절약)
        eval_steps=args.eval_steps,
        save_strategy="epoch",             # 에폭 끝에만 저장 (경량)
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        report_to="none",
        optim="adamw_torch",               # bnb OK면 "paged_adamw_8bit"로 교체 가능
    )

    # SFTTrainer (trl 0.23): tokenizer/data_collator/max_seq_length/packing 인자 사용 ❌
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_text,
        eval_dataset=val_text,
        formatting_func=formatting_func,
    )

    # 학습
    trainer.train()

    # 저장 (LoRA 어댑터 + 토크나이저)
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"[info] saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
