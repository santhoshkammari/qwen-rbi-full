"""
Stage 1: Synthetic QA generation for full RBI corpus.

Flow:
  All 14K docs (all entity types, all sources) → chunk → qwen3-27b → (question, answer) pairs → JSONL
  Skips docs < 200 chars. Resumes from existing JSONL (checks done doc_ids).
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ── config ────────────────────────────────────────────────────────────────────
PARQUET_PATH  = "/home/ng6355/santhosh/qwen-rbi-full/rbi_corpus.parquet"
OUTPUT_PATH   = "data/qa_pairs.jsonl"
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200
QA_PER_CHUNK  = 5
MIN_CHARS     = 200   # skip tiny docs
ENTITY_FILTER = None  # None = all entities

VLLM_BASE_URL = "http://localhost:8077/v1"   # running locally on GPU4
MODEL_ID       = "/home/ng6355/models/qwen3-6-27b"
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(base_url=VLLM_BASE_URL, api_key="x")

SYSTEM_PROMPT = """You are an expert on RBI (Reserve Bank of India) regulations, master directions, circulars, and monetary policy.
Generate high-quality question-answer pairs from the given regulatory text.
Rules:
- Questions must be answerable ONLY from the given text
- Answers must be factual, complete, and directly sourced from the text
- Cover diverse aspects: definitions, requirements, limits, procedures, penalties, timelines
- Do NOT generate generic or vague questions
- Output ONLY valid JSON array, no explanation"""

QA_PROMPT_TEMPLATE = """Generate exactly {n} question-answer pairs from this RBI regulatory text.

TEXT:
{chunk}

Output a JSON array of objects with keys "question" and "answer":
[
  {{"question": "...", "answer": "..."}},
  ...
]"""


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def clean_html(text: str) -> str:
    # strip HTML table tags but keep content readable
    text = re.sub(r'<tr>', '\n', text)
    text = re.sub(r'<td>', ' | ', text)
    text = re.sub(r'</td>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def generate_qa_for_chunk(chunk: str, doc_id: str, chunk_idx: int, entity: str, source: str) -> list[dict]:
    prompt = QA_PROMPT_TEMPLATE.format(n=QA_PER_CHUNK, chunk=chunk)
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content.strip()

        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            print(f"  [WARN] no JSON array found for doc={doc_id} chunk={chunk_idx}")
            return []
        pairs = json.loads(match.group())
        for p in pairs:
            p["doc_id"]    = doc_id
            p["chunk_idx"] = chunk_idx
            p["entity"]    = entity
            p["source"]    = source
            p["context"]   = chunk
        return pairs

    except Exception as e:
        print(f"  [ERROR] doc={doc_id} chunk={chunk_idx}: {e}")
        return []


def main():
    Path("data").mkdir(exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total docs: {len(df)} | Total chars: {df['char_count'].sum():,}", flush=True)

    # resume — skip already-processed doc_ids
    done_ids = set()
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} doc_ids already done", flush=True)

    total = 0
    with open(OUTPUT_PATH, "a") as out_f:
        for i, row in df.iterrows():
            doc_id = row["doc_id"]
            if doc_id in done_ids:
                continue
            entity = str(row.get("entity_type", "") or "")
            source = str(row.get("source", "") or "")
            text   = clean_html(row["text"])
            chunks = chunk_text(text)
            print(f"\nDoc {i+1}/{len(df)}  id={doc_id[:12]}  entity={entity[:20]}  chunks={len(chunks)}", flush=True)

            for ci, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:   # only skip truly empty chunks
                    continue
                print(f"  chunk {ci+1}/{len(chunks)} ({len(chunk)} chars) ...", end=" ", flush=True)
                pairs = generate_qa_for_chunk(chunk, doc_id, ci, entity, source)
                print(f"→ {len(pairs)} pairs  [total so far: {total + len(pairs)}]", flush=True)
                for p in pairs:
                    out_f.write(json.dumps(p) + "\n")
                out_f.flush()
                total += len(pairs)
                time.sleep(0.1)

    print(f"\nDone. {total} QA pairs → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
