"""Environment-backed configuration for the local customer-support RAG app."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "bitext_customer_support"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT / "chroma_db")))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
