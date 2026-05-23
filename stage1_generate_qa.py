"""
Stage 1: Synthetic QA generation for full RBI corpus.

Flow:
  All 14K docs (all entity types, all sources) → chunk → qwen3-27b → (question, answer) pairs → JSONL
  Skips docs < 200 chars. Resumes from existing JSONL (checks done doc_ids).

Async: sends CONCURRENT_REQUESTS chunks in parallel to saturate vLLM batch capacity.
"""

import asyncio
import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# ── config ────────────────────────────────────────────────────────────────────
PARQUET_PATH       = "/home/ng6355/santhosh/qwen-rbi-full/rbi_corpus.parquet"
OUTPUT_PATH        = "data/qa_pairs.jsonl"
CHUNK_SIZE         = 2000
CHUNK_OVERLAP      = 200
QA_PER_CHUNK       = 5
MIN_CHARS          = 200
ENTITY_FILTER      = None
CONCURRENT_REQUESTS = 32   # saturate vLLM — tune up/down based on GPU util

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


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def clean_html(text: str) -> str:
    text = re.sub(r'<tr>', '\n', text)
    text = re.sub(r'<td>', ' | ', text)
    text = re.sub(r'</td>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def generate_qa_async(client: AsyncOpenAI, sem: asyncio.Semaphore,
                             chunk: str, doc_id: str, chunk_idx: int,
                             entity: str, source: str) -> list[dict]:
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
                print(f"  [WARN] no JSON for doc={doc_id} chunk={chunk_idx}", flush=True)
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
            print(f"  [ERROR] doc={doc_id} chunk={chunk_idx}: {e}", flush=True)
            return []


async def main():
    Path("data").mkdir(exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total docs: {len(df)} | Total chars: {df['char_count'].sum():,}", flush=True)

    done_ids = set()
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} doc_ids already done", flush=True)

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="x")
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    total = 0
    t0 = time.time()

    with open(OUTPUT_PATH, "a") as out_f:
        # build all pending (doc, chunk) tasks in batches of docs
        pending_docs = [
            row for _, row in df.iterrows()
            if row["doc_id"] not in done_ids
        ]
        print(f"Docs to process: {len(pending_docs)}", flush=True)

        for di, row in enumerate(pending_docs):
            doc_id = row["doc_id"]
            entity = str(row.get("entity_type", "") or "")
            source = str(row.get("source", "") or "")
            text   = clean_html(str(row["text"]))
            chunks = [c for c in chunk_text(text) if len(c.strip()) >= 10]

            print(f"\nDoc {di+1}/{len(pending_docs)}  id={doc_id[:12]}  "
                  f"entity={entity[:20]}  chunks={len(chunks)}", flush=True)

            # fire all chunks of this doc concurrently
            tasks = [
                generate_qa_async(client, sem, chunk, doc_id, ci, entity, source)
                for ci, chunk in enumerate(chunks)
            ]
            results = await asyncio.gather(*tasks)

            doc_pairs = 0
            for pairs in results:
                for p in pairs:
                    out_f.write(json.dumps(p) + "\n")
                doc_pairs += len(pairs)
            out_f.flush()

            total += doc_pairs
            elapsed = time.time() - t0
            rate = (di + 1) / (elapsed / 3600)
            remaining = len(pending_docs) - (di + 1)
            eta_hrs = remaining / rate if rate > 0 else 0
            print(f"  → {doc_pairs} pairs  [total={total}  "
                  f"rate={rate:.0f} docs/hr  ETA={eta_hrs:.1f}hr]", flush=True)

    print(f"\nDone. {total} QA pairs → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
