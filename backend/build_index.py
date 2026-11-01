"""Build a compact, stratified Chroma index from the Bitext support dataset.

Run from backend/:
    python build_index.py

By default, builds a 1,500-record index in chroma_db_candidate/.
The output directory must be new or empty; this script never deletes an index.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, COLLECTION_NAME, DATASET_ID, EMBEDDING_MODEL

MAX_RECORDS = 1500
RANDOM_SEED = 42
BATCH_SIZE = 128
SOURCE_NAME = "Bitext Customer Support Dataset"

# Retrieval-inspection questions. Review the returned passages manually.
INSPECTION_QUESTIONS = [
    "How can I request a refund?",
    "What is the return policy?",
    "How do I check my invoice or receipt?",
    "What should I do if my payment was charged twice?",
    "How can I reset my password?",
    "How do I change or cancel an order?",
    "How can I track a shipment?",
    "What delivery options are available?",
    "How do I create or manage my account?",
    "How can I contact a human support agent?",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("build_index")


def clean_text(value: Any) -> str:
    """Normalize whitespace while preserving the question and answer wording."""
    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split())


def stable_id(question: str, answer: str) -> str:
    """Generate a deterministic Chroma ID from a cleaned question-answer pair."""
    pair = f"{question}\0{answer}".encode("utf-8")
    return hashlib.sha256(pair).hexdigest()


def distribute_extra(
    extra: int,
    capacities: dict[str, int],
    rng: random.Random,
) -> dict[str, int]:
    """Allocate extra slots proportionally without exceeding group capacities."""
    allocated = {key: 0 for key in capacities}

    while extra > 0:
        active = [
            key for key, capacity in capacities.items()
            if allocated[key] < capacity
        ]
        if not active:
            raise ValueError("Not enough records to satisfy the requested sample size.")

        total_weight = sum(capacities[key] - allocated[key] for key in active)
        exact = {
            key: extra * (capacities[key] - allocated[key]) / total_weight
            for key in active
        }

        added = 0
        for key in active:
            amount = min(
                capacities[key] - allocated[key],
                int(exact[key]),
            )
            allocated[key] += amount
            added += amount

        extra -= added
        if extra <= 0:
            break

        # Assign remaining fractional slots by largest remainder; seeded tie
        # ordering makes the allocation deterministic.
        active = [
            key for key, capacity in capacities.items()
            if allocated[key] < capacity
        ]
        if not active:
            break
        tie_order = {key: rng.random() for key in active}
        active.sort(
            key=lambda key: (exact.get(key, 0) % 1, tie_order[key]),
            reverse=True,
        )
        for key in active:
            if extra <= 0:
                break
            if allocated[key] < capacities[key]:
                allocated[key] += 1
                extra -= 1

    return allocated


def prepare_clean_rows(dataset) -> tuple[list[dict[str, Any]], dict[str, int]]:
    required = {"instruction", "response"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    removed_empty = 0
    removed_duplicates = 0

    for source_row_id, raw in enumerate(dataset):
        question = clean_text(raw.get("instruction"))
        answer = clean_text(raw.get("response"))

        if not question or not answer:
            removed_empty += 1
            continue

        pair = (question, answer)
        if pair in seen_pairs:
            removed_duplicates += 1
            continue
        seen_pairs.add(pair)

        intent = clean_text(raw.get("intent")) or "unknown_intent"
        category = clean_text(raw.get("category")) or "unknown_category"

        rows.append({
            "question": question,
            "answer": answer,
            "intent": intent,
            "category": category,
            "record_id": str(source_row_id),
        })

    if not rows:
        raise ValueError("No usable question-answer records remain after cleaning.")

    return rows, {
        "original_rows": len(dataset),
        "removed_empty_pairs": removed_empty,
        "removed_duplicate_pairs": removed_duplicates,
        "cleaned_rows": len(rows),
    }


def stratified_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample across intents, then across categories within each intent."""
    if sample_size <= 0:
        raise ValueError("Sample size must be greater than zero.")
    if sample_size > len(rows):
        raise ValueError(
            f"Requested {sample_size} records, but only {len(rows)} cleaned rows exist."
        )

    rng = random.Random(seed)
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[row["intent"]].append(row)

    intents = sorted(by_intent)
    if len(intents) > sample_size:
        raise ValueError(
            f"Cannot represent all {len(intents)} intents with a sample of "
            f"{sample_size} records."
        )

    # Give every intent one slot, then distribute remaining slots in proportion
    # to each intent's remaining capacity.
    intent_quotas = {intent: 1 for intent in intents}
    remaining = sample_size - len(intents)
    extra_by_intent = distribute_extra(
        remaining,
        {intent: len(by_intent[intent]) - 1 for intent in intents},
        rng,
    )
    for intent, extra in extra_by_intent.items():
        intent_quotas[intent] += extra

    selected: list[dict[str, Any]] = []

    for intent in intents:
        intent_rows = by_intent[intent]
        quota = intent_quotas[intent]

        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in intent_rows:
            by_category[row["category"]].append(row)

        categories = sorted(by_category)
        if quota >= len(categories):
            chosen_categories = categories
            category_quotas = {category: 1 for category in categories}
            extra = quota - len(categories)
            extras = distribute_extra(
                extra,
                {
                    category: len(by_category[category]) - 1
                    for category in categories
                },
                rng,
            )
            for category, count in extras.items():
                category_quotas[category] += count
        else:
            # If an intent has more categories than allocated slots, favor the
            # least represented categories first; seeded tie-breaks are stable.
            categories_by_rarity = sorted(
                categories,
                key=lambda category: (
                    len(by_category[category]),
                    rng.random(),
                    category,
                ),
            )
            chosen_categories = categories_by_rarity[:quota]
            category_quotas = {category: 1 for category in chosen_categories}

        for category in chosen_categories:
            group = by_category[category]
            count = category_quotas[category]
            selected.extend(rng.sample(group, count))

    if len(selected) != sample_size:
        raise RuntimeError(
            f"Sampling produced {len(selected)} records, expected {sample_size}."
        )

    return selected


def directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def require_empty_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Output path must be new or empty: {path}\n"
            "Choose a different output directory. The script will not overwrite "
            "or delete an existing Chroma index."
        )
    path.mkdir(parents=True, exist_ok=True)


def build_index(
    output_dir: Path,
    sample_size: int = MAX_RECORDS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    require_empty_output_dir(output_dir)

    log.info("Loading original dataset: %s", DATASET_ID)
    dataset = load_dataset(DATASET_ID, split="train")

    rows, cleaning_stats = prepare_clean_rows(dataset)
    log.info(
        "Dataset rows: original=%s cleaned=%s removed_empty=%s "
        "removed_duplicate_pairs=%s",
        cleaning_stats["original_rows"],
        cleaning_stats["cleaned_rows"],
        cleaning_stats["removed_empty_pairs"],
        cleaning_stats["removed_duplicate_pairs"],
    )

    selected = stratified_sample(rows, sample_size=sample_size, seed=seed)
    if len(selected) == len(rows) and len(rows) > sample_size:
        raise RuntimeError("Sampling unexpectedly selected the full cleaned dataset.")

    documents: list[Document] = []
    ids: list[str] = []

    for row in selected:
        question = row["question"]
        answer = row["answer"]
        documents.append(
            Document(
                page_content=(
                    f"Customer question: {question}\n"
                    f"Support answer: {answer}"
                ),
                metadata={
                    "source": SOURCE_NAME,
                    "category": row["category"],
                    "intent": row["intent"],
                    "record_id": row["record_id"],
                },
            )
        )
        ids.append(stable_id(question, answer))

    if any(not doc.page_content.strip() for doc in documents):
        raise RuntimeError("Selected records include an empty document.")

    log.info(
        "Selected %s records across %s intents and %s categories",
        len(selected),
        len({row["intent"] for row in selected}),
        len({row["category"] for row in selected}),
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # L2 matches the existing Chroma default used by the current backend.
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(output_dir),
        collection_metadata={"hnsw:space": "l2"},
    )

    for start in range(0, len(documents), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(documents))
        store.add_documents(
            documents=documents[start:stop],
            ids=ids[start:stop],
        )
        log.info("Indexed %s / %s records", stop, len(documents))

    collection_count = store._collection.count()
    if collection_count != sample_size:
        raise RuntimeError(
            f"Collection count mismatch: expected {sample_size}, "
            f"got {collection_count}."
        )

    log.info("Retrieval inspection: top results for representative questions")
    for question in INSPECTION_QUESTIONS:
        pairs = store.similarity_search_with_relevance_scores(question, k=3)
        print(f"\nQUESTION: {question}")
        for rank, (doc, score) in enumerate(pairs, start=1):
            print(
                f"{rank}. score={score:.4f} "
                f"intent={doc.metadata.get('intent')} "
                f"category={doc.metadata.get('category')} "
                f"record_id={doc.metadata.get('record_id')}"
            )
            print(doc.page_content)

    size_bytes = directory_size_bytes(output_dir)
    result = {
        **cleaning_stats,
        "selected_records": len(selected),
        "collection_count": collection_count,
        "selected_intents": len({row["intent"] for row in selected}),
        "selected_categories": len({row["category"] for row in selected}),
        "sample_size_limit": sample_size,
        "random_seed": seed,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": 384,
        "collection_name": COLLECTION_NAME,
        "chroma_path": str(output_dir),
        "index_size_bytes": size_bytes,
        "index_size_mb": size_bytes / 1_000_000,
    }

    print("\nINDEX BUILD SUMMARY")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(
        "\nReview the printed retrieval passages manually. Similarity scores "
        "alone do not establish relevance."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS,
        help=f"Maximum sampled records (default: {MAX_RECORDS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Sampling seed for reproducible selection (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHROMA_DIR.parent / "chroma_db_candidate",
        help="New or empty index output directory.",
    )
    args = parser.parse_args()
    build_index(
        output_dir=args.output_dir,
        sample_size=args.max_records,
        seed=args.seed,
    )