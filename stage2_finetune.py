"""
Stage 2: SFT of qwen3-4b on full RBI corpus QA pairs.

Training format (grounded):
  System: Full RBI domain expert
  User:   [Context: <chunk>]\nQuestion: <question>
  Asst:   <answer>

Uses unsloth LoRA for memory efficiency, then merges to full model.
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset

_nvlib = os.path.expanduser("~/envs/rbi-finetune/lib/python3.12/site-packages/nvidia/cu13/lib")
if os.path.isdir(_nvlib):
    os.environ["LD_LIBRARY_PATH"] = _nvlib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template

# ── config ────────────────────────────────────────────────────────────────────
BASE_MODEL    = "/home/ng6355/models/qwen3-4b"
OUTPUT_DIR    = "/home/ng6355/santhosh/qwen-rbi-full/models/qwen-rbi-full"
CHECKPOINT_DIR= "/home/ng6355/santhosh/qwen-rbi-full/checkpoints"
QA_JSONL      = "/home/ng6355/santhosh/qwen-rbi-full/data/qa_pairs.jsonl"

MAX_SEQ_LENGTH = 4096
LORA_R         = 16
LORA_ALPHA     = 32
BATCH_SIZE     = 4
GRAD_ACCUM     = 4       # effective batch = 16
EPOCHS         = 3       # small dataset — 3 epochs to bake it in
LR             = 2e-4
SEED           = 42

SYSTEM_PROMPT = (
    "You are an expert on Reserve Bank of India (RBI) regulations, master directions, circulars, "
    "and monetary policy. You have deep knowledge across all RBI-regulated entities including "
    "commercial banks, NBFCs, cooperative banks, small finance banks, payment banks, and other "
    "financial institutions. Answer accurately and cite relevant RBI provisions."
)
# ─────────────────────────────────────────────────────────────────────────────


def load_qa_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def format_conversation(pair: dict) -> str:
    """Format as Qwen3 chat template string with context in user turn."""
    context  = pair.get("context", "").strip()
    question = pair["question"].strip()
    answer   = pair["answer"].strip()

    user_content = f"Context:\n{context}\n\nQuestion: {question}" if context else question

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    return messages


def main():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"Loading {BASE_MODEL} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ── 2. LoRA ───────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # ── 3. Dataset ───────────────────────────────────────────────────────────
    print(f"Loading QA pairs from {QA_JSONL} ...")
    pairs = load_qa_pairs(QA_JSONL)
    print(f"  {len(pairs)} pairs loaded")

    def apply_template(sample):
        messages = format_conversation(sample)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    raw_ds = Dataset.from_list(pairs)
    train_ds = raw_ds.map(apply_template, remove_columns=raw_ds.column_names)
    print(f"  Sample formatted:\n{train_ds[0]['text'][:400]}\n...")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            output_dir=CHECKPOINT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            seed=SEED,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            report_to="none",
            optim="adamw_8bit",
        ),
    )

    print("Training ...")
    stats = trainer.train()
    print(f"Done. loss={stats.training_loss:.4f}  steps={stats.global_step}")

    # ── 5. Merge & save ──────────────────────────────────────────────────────
    print(f"Merging LoRA → {OUTPUT_DIR} ...")
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
