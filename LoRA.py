import os
from dotenv import load_dotenv
import torch

# ── TF32 / matmul precision (free speed on Ampere+) ──────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# 1. 환경 변수 로드 및 Huggingface 로그인
load_dotenv()
huggingface_token = os.getenv("HF_TOKEN_read")
from huggingface_hub import login
login(token=huggingface_token)

# ── Attention backend selection ─────────────────────────────────────────
# 공식 FlashAttention-2 지원 아키텍처에 Ampere/Hopper(=A100/H100/H200, sm_80/sm_90)가 포함된다.
# 전제: flash-attn 설치 필요 (예: pip install flash-attn --no-build-isolation)
attn_impl = "flash_attention_2"

# ── padding_free는 packing=True와 함께 써야 동작합니다 ────────────────────────
# packing 없이 True로 설정하면 FlashAttention varlen 커널에서 cu_seqlens_q 오류 발생
use_padding_free = False
print(f"[attn] Selected backend: {attn_impl}")
print(f"[attn] use_padding_free={use_padding_free}")

# 2. bf16 LoRA 설정 (4-bit 양자화 비활성화)

# 3. PEFT LoRA 설정
from peft import LoraConfig
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# 4. 모델 및 토크나이저 로드
#    - attn_implementation    → 위에서 선택한 backend 적용
#    - use_cache=False        → 학습 시 더 안전
#    - device_map 미지정      → 단일 GPU에 자동 배치
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation=attn_impl,
)
model.config.use_cache = False  # 학습 중 KV 캐시 비활성화

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
model.config.pad_token_id = tokenizer.pad_token_id          # 예: 128004

# 5. 데이터셋 로드 및 전처리
from datasets import load_dataset, concatenate_datasets

# ── EQA 데이터셋 ──────────────────────────────────────────────────────────────
EQA_dataset = load_dataset("BaekSeungJu/Ophthalmology-EQA-v3", split="train")
EQA_dataset = EQA_dataset.shuffle(seed=42)

def EQA_format_chat_template(row):
    system_instruction = (
        "You are an expert ophthalmologist. Please provide accurate and "
        "medically sound answers to the user's ophthalmology-related question."
    )
    row_json = [
        {"role": "system",    "content": system_instruction},
        {"role": "user",      "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

mapped_EQA_dataset = EQA_dataset.map(EQA_format_chat_template, num_proc=4)
split_EQA_dataset   = mapped_EQA_dataset.train_test_split(test_size=0.01, seed=42)
train_EQA_dataset   = split_EQA_dataset["train"]
test_EQA_dataset    = split_EQA_dataset["test"]
print(f"Train EQA dataset size: {len(train_EQA_dataset)}")
print(f"Test  EQA dataset size: {len(test_EQA_dataset)}")

# ── MCQA 데이터셋 ─────────────────────────────────────────────────────────────
MCQA_dataset  = load_dataset("BaekSeungJu/Ophthalmology-MCQA-v3", split="train")
MCQA_dataset  = MCQA_dataset.shuffle(seed=42)

def MCQA_format_chat_template(row):
    system_instruction = (
        "You are an expert ophthalmologist. Please provide accurate and "
        "medically sound answers to the user's ophthalmology-related question."
    )

    def _select_letter(answer, a, b, c, d, e):
        for letter, opt in zip("ABCDE", [a, b, c, d, e]):
            if answer == opt:
                return letter
        return ""

    def _format_question(question, a, b, c, d, e):
        if c == "":
            return f"Question:\n{question}\n\nOptions:\nA) {a}\nB) {b}"
        elif e == "":
            return f"Question:\n{question}\n\nOptions:\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
        return f"Question:\n{question}\n\nOptions:\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nE) {e}"

    def _format_response(a, b, c, d, e, explanation, answer):
        letter = _select_letter(answer, a, b, c, d, e)
        return f"Explanation:\n{explanation}\n\nAnswer:\n{letter}) {answer}"

    row_json = [
        {"role": "system", "content": system_instruction},
        {"role": "user",   "content": _format_question(
            row["question"], row["option_a"], row["option_b"],
            row["option_c"], row["option_d"], row["option_e"])},
        {"role": "assistant", "content": _format_response(
            row["option_a"], row["option_b"], row["option_c"],
            row["option_d"], row["option_e"],
            row["explanation"], row["answer"])},
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

mapped_MCQA_dataset  = MCQA_dataset.map(MCQA_format_chat_template, num_proc=4)
split_MCQA_dataset   = mapped_MCQA_dataset.train_test_split(test_size=0.01, seed=42)
train_MCQA_dataset   = split_MCQA_dataset["train"]
test_MCQA_dataset    = split_MCQA_dataset["test"]
print(f"Train MCQA dataset size:  {len(train_MCQA_dataset)}")
print(f"Test  MCQA dataset size:  {len(test_MCQA_dataset)}")

# ── 데이터셋 결합 ─────────────────────────────────────────────────────────────
train_dataset = concatenate_datasets([train_EQA_dataset, train_MCQA_dataset])
test_dataset  = concatenate_datasets([test_EQA_dataset,  test_MCQA_dataset])
train_dataset = train_dataset.shuffle(seed=42)
print(f"Combined Train dataset size: {len(train_dataset)}")
print(f"Combined Test  dataset size: {len(test_dataset)}")
print(train_dataset[1]["text"])

# 6. Data Collator (completion-only loss)
from trl import DataCollatorForCompletionOnlyLM
response_template = "<|start_header_id|>assistant<|end_header_id|>"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(
    response_template_ids,
    tokenizer=tokenizer,
    mlm=False,
    padding_free=use_padding_free,           # packing 없이는 False (varlen 오류 방지)
)
print(f"[v3.1] DataCollator 활성화 (completion-only loss, response_template_ids={response_template_ids})")

# 7. 출력 디렉토리
local_output_dir = "./Ophtimus_LoRA_accelerate/Ophtimus_8B_Instruct_checkpoint_v3"
os.makedirs(local_output_dir, exist_ok=True)

# 8. 학습 인자 설정 (단일 GPU, DeepSpeed 없음)
from trl import SFTTrainer, SFTConfig

training_arguments = SFTConfig(
    output_dir=local_output_dir,
    report_to="tensorboard",
    bf16=True,

    # ── 처리량 핵심 ───────────────────────────────────────────────────────────
    # global batch size = per_device_train_batch_size × gradient_accumulation_steps = 12×4 = 48
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=4,
    max_seq_length=512,                  # 데이터 분석 결과: max=451 토큰 → 512로 충분

    # ── 메모리 ────────────────────────────────────────────────────────────────
    # gradient_checkpointing=True  → VRAM 절약, 속도 ~20~30% 손실
    # gradient_checkpointing=False → 속도 최대, VRAM 더 사용 (A100/H100/H200에서 주로 유리)
    gradient_checkpointing=False,

    # ── 옵티마이저 ────────────────────────────────────────────────────────────
    optim="adamw_torch_fused",           # 단일 GPU에서 fused 사용 (더 빠름)
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=10,
    lr_scheduler_type="constant_with_warmup",

    # ── 학습 스케줄 ───────────────────────────────────────────────────────────
    num_train_epochs=5,
    seed=42,

    # ── 로깅 / 평가 / 저장 ─────────────────────────────────────────────────────
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=3,

    # ── DataLoader 최적화 ─────────────────────────────────────────────────────
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=4,
)

# 9. Trainer 구성
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    data_collator=data_collator,
    args=training_arguments,
)

# 10. 학습 수행
train_stats = trainer.train()

# 11. LoRA 어댑터 최종 저장
trainer.save_model(f"{local_output_dir}/final")
print(f"[v3.1] LoRA 어댑터 저장: {local_output_dir}/final")

# 12. 학습 로그 저장 (CSV)
import pandas as pd
lossPD = pd.DataFrame(trainer.state.log_history)
lossPD.to_csv(f"{local_output_dir}/loss.csv", index=False)
print(f"[v3.1] 학습 완료. 로그 저장: {local_output_dir}/loss.csv")
