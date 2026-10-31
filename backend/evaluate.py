"""Run a small live evaluation and print retrieval inspection and outcome tables."""
import time

from datasets import load_dataset
from config import DATASET_ID
from support_rag import SupportRAG

QUESTIONS = [
    ("How can I request a refund?", "supported"),
    ("How do I check my invoice or receipt?", "supported"),
    ("What should I do if my payment was charged twice?", "supported"),
    ("What is the return policy?", "supported"),
    ("I need my money back for a purchase. What steps should I follow?", "paraphrase"),
    ("Where is my order right now?", "live_data"),
    ("Has my package been delivered?", "live_data"),
    ("What is the weather today?", "out_of_scope"),
    ("Can you explain home insurance coverage?", "out_of_scope"),
]

if __name__ == "__main__":
    rag = SupportRAG.from_environment()
    # Select real rows for the four core support topics so reference answers are inspectable.
    data = load_dataset(DATASET_ID, split="train")
    real_examples = []
    topic_terms = [("refund", ("refund", "reimburse")), ("invoice", ("invoice", "receipt", "bill")),
                   ("payment", ("payment", "charge", "transaction")), ("return", ("return", "exchange"))]
    for label, terms in topic_terms:
        row = next((r for r in data if any(t in str(r.get("intent", "")).lower() for t in terms)
                    and str(r.get("instruction", "")).strip() and str(r.get("response", "")).strip()), None)
        if row:
            real_examples.append((str(row["instruction"]).strip(), "dataset_example", str(row["response"]).strip(), label))
    cases = [(q, kind, "", "") for q, kind in QUESTIONS]
    cases.extend(real_examples)
    print("| Question | Type | Dataset reference | Retrieved source/score | Outcome | Answer | Retries | Latency ms | Review |")
    print("|---|---|---|---|---|---|---:|---:|---|")
    for question, kind, reference, topic in cases:
        started = time.perf_counter()
        result = rag.ask(question)
        latency = (time.perf_counter() - started) * 1000
        refs = "; ".join(f"{s['record_id']} ({s['score']:.3f})" for s in result["sources"])
        expected = "fallback" if kind in {"live_data", "out_of_scope"} else None
        passed = result["outcome"] == expected if expected else result["outcome"] in {"supported", "partial"}
        short_answer = result["answer"].replace("|", "/").replace("\n", " ")
        short_reference = reference.replace("|", "/").replace("\n", " ")
        print(f"| {question} | {kind}{(' / ' + topic) if topic else ''} | {short_reference or '—'} | {refs or 'none'} | {result['outcome']} | {short_answer} | {result['retry_count']} | {latency:.1f} | {'PASS' if passed else 'REVIEW'} |")
        for source in result["sources"]:
            print(f"  - Evidence: {source['passage'].replace('|', '/')} ")
    print("PASS here means the outcome matched the basic expected category. Manually verify passages and answers; this is not an accuracy score.")
