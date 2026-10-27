"""Build the persistent local Chroma index from the Bitext FAQ dataset."""
import argparse
import logging
import re

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, COLLECTION_NAME, DATASET_ID, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_index")


def clean_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u00a0", " ")).strip()


def build_index(rebuild: bool = False) -> dict:
    data = load_dataset(DATASET_ID, split="train")
    required = {"instruction", "response"}
    missing = required - set(data.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    rows, seen = [], set()
    empty = duplicates = 0
    for record_id, row in enumerate(data):
        question, answer = clean_text(row.get("instruction")), clean_text(row.get("response"))
        if not question or not answer:
            empty += 1
            continue
        pair = (question, answer)
        if pair in seen:
            duplicates += 1
            continue
        seen.add(pair)
        rows.append((record_id, question, answer, row))

    docs = [Document(
        page_content=f"Customer question: {q}\nSupport answer: {a}",
        metadata={"source": "Bitext Customer Support Dataset", "category": clean_text(row.get("category")),
                  "intent": clean_text(row.get("intent")), "flags": clean_text(row.get("flags")),
                  "record_id": str(record_id)},
    ) for record_id, q, a, row in rows]
    if not docs:
        raise ValueError("No valid FAQ records remain after cleaning")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    if rebuild:
        store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                       persist_directory=str(CHROMA_DIR))
        store.delete_collection()
    Chroma.from_documents(documents=docs, embedding=embeddings,
        collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR))
    report = {"original_rows": len(data), "removed_empty": empty,
              "removed_duplicate_pairs": duplicates, "indexed_records": len(docs),
              "chroma_path": str(CHROMA_DIR)}
    log.info("Cleaning/index report: %s", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Delete the existing collection before indexing")
    args = parser.parse_args()
    print(build_index(rebuild=args.rebuild))
