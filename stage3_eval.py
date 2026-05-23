"""
Stage 3: Eval — fine-tuned qwen-rbi-full on broad RBI questions across all entity types.
Runs on GPU4, uses vLLM for inference.
"""

import json
from openai import OpenAI

FINETUNED_MODEL = "/home/ng6355/santhosh/qwen-rbi-full/models/qwen-rbi-full"
BASE_MODEL      = "/home/ng6355/models/qwen3-4b"
VLLM_URL        = "http://localhost:8077/v1"

SYSTEM_PROMPT = (
    "You are an expert on Reserve Bank of India (RBI) regulations, master directions, circulars, "
    "and monetary policy. Answer accurately and concisely based on official RBI documents."
)

EVAL_QUESTIONS = [
    # Commercial banks
    "What are Basel III capital adequacy requirements as per RBI directions for commercial banks?",
    "What is CRR and what are the RBI directions for commercial banks regarding Cash Reserve Ratio?",
    "What are the RBI guidelines on priority sector lending for commercial banks?",
    # NBFCs
    "What are the RBI regulations for Non-Banking Financial Companies (NBFCs) regarding registration?",
    "What are the capital requirements for NBFCs as per RBI master directions?",
    # Small Finance Banks
    "What are the eligibility criteria for setting up a Small Finance Bank as per RBI guidelines?",
    # Payments Banks
    "What are the scope of activities permitted for Payments Banks under RBI regulations?",
    # Cooperative banks
    "What are the RBI directions for Urban Co-operative Banks on capital adequacy?",
    # General RBI
    "What does RBI mean by 'Too Big To Fail' (TBTF) and what policy measures address it?",
    "What is the role of the Reserve Bank of India in regulating the credit system in India?",
]

def ask(client, model_id, question):
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        temperature=0,
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content.strip()

def score_answer(answer: str, question: str) -> dict:
    """Simple heuristic scoring — checks for RBI-specific content."""
    rbi_terms = [
        "RBI", "Reserve Bank", "master direction", "circular", "regulation",
        "commercial bank", "scheduled bank", "Basel", "CRR", "SLR", "TBTF",
        "capital", "liquidity", "priority sector", "licensing", "Narasimhan",
        "2013", "continuous authorisation", "prudential"
    ]
    hits = sum(1 for t in rbi_terms if t.lower() in answer.lower())
    length_ok = len(answer) > 100
    return {
        "rbi_term_hits": hits,
        "length_ok": length_ok,
        "score": min(10, hits * 1.5 + (3 if length_ok else 0)),
    }

def main():
    client = OpenAI(base_url=VLLM_URL, api_key="x")

    results = []
    print(f"\n{'='*70}")
    print("EVAL: Fine-tuned qwen-rbi-full")
    print(f"{'='*70}\n")

    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"Q{i+1}: {q}")
        print("-" * 60)

        ft_ans  = ask(client, FINETUNED_MODEL, q)
        ft_score = score_answer(ft_ans, q)

        print(f"FINE-TUNED (score={ft_score['score']:.1f}/10, rbi_terms={ft_score['rbi_term_hits']}):")
        print(ft_ans[:400])
        print()

        results.append({
            "question": q,
            "finetuned_answer": ft_ans,
            "finetuned_score": ft_score,
        })

    avg_ft = sum(r["finetuned_score"]["score"] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"  Fine-tuned avg score : {avg_ft:.2f}/10")
    print(f"{'='*70}\n")

    with open("eval_results.json", "w") as f:
        json.dump({
            "avg_finetuned_score": avg_ft,
            "results": results,
        }, f, indent=2)
    print("Saved → eval_results.json")

if __name__ == "__main__":
    main()
