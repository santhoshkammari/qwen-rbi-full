# CONTEXT — qwen-rbi-full project

## What we are building

Fine-tuning `Qwen3-4B` on the **full RBI corpus** (14,192 docs — all entity types, all sources) to create `qwen-rbi-full`: a domain expert model that knows RBI regulations, master directions, circulars, monetary policy across all regulated entities.

This is the second model. The first was `qwen-rbi-commercial-banks-4b` (only 12 CB docs, scored 7.12/10) — that was a prototype to validate the approach. This is the full-scale production version.

---

## Pipeline (3 stages)

```
Stage 1  rbi_corpus.parquet (14,192 docs)
            → chunk (2000 chars, 200 overlap)
            → qwen3-27b generates 5 QA pairs per chunk
            → data/qa_pairs.jsonl
         Expected: ~300,000+ QA pairs, runs ~20-30 hrs

Stage 2  qa_pairs.jsonl
            → unsloth SFT on qwen3-4b (LoRA r=16, 3 epochs)
            → merge LoRA → merged 16-bit model
         Expected: ~2-4 hrs on RTX PRO 6000

Stage 3  Eval on 10 broad RBI questions across all entity types
            → eval_results.json
```

### Current status (as of 2026-05-23)
- **Stage 1: RUNNING** — ~1,700 pairs done out of ~300K expected, doc 14/14,192
- **Stage 2: NOT STARTED** — runs after stage 1 completes
- **Stage 3: NOT STARTED**

---

## Data

| File | Location | Description |
|---|---|---|
| Raw corpus | `/home/ng6355/santhosh/qwen-rbi-full/rbi_corpus.parquet` | 14,192 docs, 94MB, all RBI entity types |
| QA pairs | `/home/ng6355/santhosh/qwen-rbi-full/data/qa_pairs.jsonl` | Generated incrementally, resume-safe |

### Corpus breakdown
- `rbi_circular`: 13,970 docs (bulk, blank entity_type)
- `master_directions`: 222 docs (labeled by entity type — NBFCs, CBs, SFBs, UCBs, etc.)
- Total chars: ~1.5B | Approx tokens: ~375M

### QA pair format (each line in JSONL)
```json
{
  "question": "...",
  "answer": "...",
  "context": "<source chunk text>",
  "doc_id": "...",
  "chunk_idx": 0,
  "entity": "commercial_banks_33",
  "source": "master_directions"
}
```

---

## GPU4 — where everything runs

- **Host:** `192.168.170.49`
- **SSH:** `ssh ng6355@192.168.170.49`
- **Requires:** NT VPN (connect via `bash -i -c 'snt'` or `python3 ~/.vpn_credentials/connect_vpn.py`)
- **GPU:** NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM
- **CUDA:** 13.2

### Key paths on GPU4
```
/home/ng6355/santhosh/qwen-rbi-full/     ← project root
    stage1_generate_qa.py                ← QA generation script
    stage2_finetune.py                   ← SFT training script
    stage3_eval.py                       ← eval script
    stage1.log                           ← stage1 live log
    stage2.log                           ← stage2 training log (after stage 1)
    data/qa_pairs.jsonl                  ← generated QA pairs
    models/qwen-rbi-full/                ← final merged model (after stage 2)
    checkpoints/                         ← LoRA checkpoints during training

/home/ng6355/models/qwen3-4b             ← base model for SFT
/home/ng6355/models/qwen3-6-27b          ← teacher model for QA generation
/home/ng6355/vllm_server.log             ← vLLM server log
```

### Python envs on GPU4
- `~/envs/py3_12_env` — has vLLM, openai client (use for stage1 QA gen + inference)
- `~/envs/llm_env` — has unsloth, trl (use for stage2 SFT training)

### vLLM aliases on GPU4 (from ~/.bashrc)
```bash
svllm      # start 27B qwen3 on port 8077 (for stage1 QA generation)
kvllm      # kill vLLM
lvllm      # tail vLLM log
```

### Check stage1 progress
```bash
ssh ng6355@192.168.170.49 "wc -l /home/ng6355/santhosh/qwen-rbi-full/data/qa_pairs.jsonl && tail -3 /home/ng6355/santhosh/qwen-rbi-full/stage1.log"
```

### Check if stage1 process is alive
```bash
ssh ng6355@192.168.170.49 "pgrep -f stage1_generate_qa && echo running || echo DEAD"
```

---

## Stage 1 resume behavior

Script is **resume-safe** — if it dies, restart and it will skip already-processed `doc_id`s:
```bash
# On GPU4:
cd /home/ng6355/santhosh/qwen-rbi-full
source ~/envs/py3_12_env/bin/activate
nohup python stage1_generate_qa.py > stage1.log 2>&1 & disown && echo done
```

**NOTE:** vLLM (27B model) must be running on port 8077 before stage1 starts. If not:
```bash
# On GPU4:
bash -i -c 'svllm'   # starts 27B on port 8077, takes ~2-3 min to load
```

---

## Stage 2 — run after stage 1 finishes

```bash
# On GPU4:
# IMPORTANT: kill vLLM first (needs full VRAM for training)
pkill -f "vllm serve"
sleep 5

cd /home/ng6355/santhosh/qwen-rbi-full
source ~/envs/llm_env/bin/activate
nohup python stage2_finetune.py > stage2.log 2>&1 & disown && echo done
```

Training config: LoRA r=16 alpha=32, batch=4, grad_accum=4, 3 epochs, lr=2e-4, cosine schedule.

---

## Related project — commercial banks prototype

- **Repo:** https://github.com/santhoshkammari/qwen-rbi-commercial-banks-4b
- **Local:** `/home/ntlpt24/Master/Projects/qwen-rbi-commercial-banks-4b`
- **Model on GPU4:** `/home/ng6355/santhosh/qwen-rbi-commercial-banks-4b/models/qwen-rbi-commercial-banks-4b`
- **Result:** 7.12/10 eval score, trained on 12 docs → 1,466 QA pairs, 25 min training

---

## GitHub repos

| Repo | URL |
|---|---|
| Full RBI model | https://github.com/santhoshkammari/qwen-rbi-full |
| CB prototype | https://github.com/santhoshkammari/qwen-rbi-commercial-banks-4b |

---

## NT VPN — required to reach GPU4

- VPN name: `nt`
- Connect: `bash -i -c 'snt'` (alias in ~/.bashrc) or `python3 ~/.vpn_credentials/connect_vpn.py`
- Disconnect: `bash -i -c 'knt'`
- Status: `forticlient vpn status`
- **Note:** NT VPN server is sometimes unreachable/down. If SSH to 192.168.170.49 times out, VPN is down — wait and retry.
- **Note:** Candor VPN (office) does NOT reach GPU4. Must use NT VPN.
