"""
Stage 1: Synthetic QA generation for full RBI corpus.

Flow:
  All 14K docs → chunk → global queue → 32 concurrent vLLM requests → JSONL
  Resume-safe: skips already-done doc_ids on startup.
"""

import asyncio
import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# ── config ────────────────────────────────────────────────────────────────────
PARQUET_PATH        = "/home/ng6355/santhosh/qwen-rbi-full/rbi_corpus.parquet"
OUTPUT_PATH         = "data/qa_pairs.jsonl"
CHUNK_SIZE          = 2000
CHUNK_OVERLAP       = 200
QA_PER_CHUNK        = 5
CONCURRENT_REQUESTS = 32

VLLM_BASE_URL = "http://localhost:8077/v1"
MODEL_ID       = "/home/ng6355/models/qwen3-6-27b"
# ─────────────────────────────────────────────────────────────────────────────

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


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def clean_html(text: str) -> str:
    text = re.sub(r'<tr>', '\n', text)
    text = re.sub(r'<td>', ' | ', text)
    text = re.sub(r'</td>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def process_chunk(client, sem, chunk, doc_id, chunk_idx, entity, source):
    prompt = QA_PROMPT_TEMPLATE.format(n=QA_PER_CHUNK, chunk=chunk)
    async with sem:
        try:
            resp = await client.chat.completions.create(
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
                return []
            pairs = json.loads(match.group())
            for p in pairs:
                p.update({"doc_id": doc_id, "chunk_idx": chunk_idx,
                           "entity": entity, "source": source, "context": chunk})
            return pairs
        except Exception as e:
            print(f"[ERROR] doc={doc_id} chunk={chunk_idx}: {e}", flush=True)
            return []


async def main():
    Path("data").mkdir(exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total docs: {len(df)}", flush=True)

    done_ids = set()
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} docs already done", flush=True)

    # build global chunk list across all pending docs
    all_chunks = []
    for _, row in df.iterrows():
        if row["doc_id"] in done_ids:
            continue
        text   = clean_html(str(row["text"]))
        chunks = [c for c in chunk_text(text) if len(c.strip()) >= 10]
        for ci, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk": chunk, "doc_id": row["doc_id"], "chunk_idx": ci,
                "entity": str(row.get("entity_type", "") or ""),
                "source": str(row.get("source", "") or ""),
            })

    total_chunks = len(all_chunks)
    print(f"Total chunks to process: {total_chunks}", flush=True)

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="x")
    sem    = asyncio.Semaphore(CONCURRENT_REQUESTS)

    completed = 0
    total_pairs = 0
    t0 = time.time()
    lock = asyncio.Lock()

    async def run_and_write(item, out_f):
        nonlocal completed, total_pairs
        pairs = await process_chunk(
            client, sem,
            item["chunk"], item["doc_id"], item["chunk_idx"],
            item["entity"], item["source"],
        )
        async with lock:
            for p in pairs:
                out_f.write(json.dumps(p) + "\n")
            out_f.flush()
            completed += 1
            total_pairs += len(pairs)
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / (elapsed / 3600)
                eta = (total_chunks - completed) / rate if rate > 0 else 0
                print(f"[{completed}/{total_chunks}] pairs={total_pairs}  "
                      f"rate={rate:.0f} chunks/hr  ETA={eta:.1f}hr", flush=True)

    with open(OUTPUT_PATH, "a") as out_f:
        tasks = [run_and_write(item, out_f) for item in all_chunks]
        await asyncio.gather(*tasks)

    print(f"\nDone. {total_pairs} QA pairs → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
