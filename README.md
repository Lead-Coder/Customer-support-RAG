# Customer Support RAG — Experiment 6 extended for Experiment 7

Local academic project: sentence-transformer embeddings and Chroma retrieval run on this computer; Groq handles text generation. FastAPI owns the reusable RAG graph, and Streamlit calls its `/ask` endpoint. No deployment, Docker, Ollama, local generation model, or external database is used.

## Request flow

`Streamlit → FastAPI validation/timing/logging → LangGraph intent classification → top‑4 Chroma retrieval → heuristic evidence check → grounded Groq draft/review → optional one expanded-query retry → response`

Each dataset row remains a complete question-and-answer FAQ document. Index construction validates `instruction` and `response`, normalizes whitespace, removes empty rows and exact duplicate pairs, keeps category/intent/flags metadata, and reports cleaning counts. Normalized `all-MiniLM-L6-v2` embeddings are calculated locally. Evidence rules reuse the notebook’s relevance/term overlap approach; these thresholds are heuristics, not calibrated confidence or factuality guarantees. Manually inspect answers and sources.

Live order, shipment, payment, and account state is unavailable in the static dataset. Those requests receive a limitation response. The backend logs request status, outcome, retries, latency, and exception type without logging question text or credentials.

## Requirements

- Windows PowerShell and Python 3.11 (recommended; use a version supported by the installed PyTorch/sentence-transformers wheels).
- A Groq API key for answer generation. Do not commit it.
- Internet access on first setup for packages and model/dataset downloads.

## Setup (PowerShell)

Run in the directory containing this README:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `GROQ_API_KEY` in `.env` to your key. Keep `.env` private. `GROQ_MODEL` defaults to `openai/gpt-oss-120b`, which is listed as a Groq production model; select a model currently enabled for your account if needed. [Groq's supported models](https://console.groq.com/docs/models) and [deprecation notices](https://console.groq.com/docs/deprecations) can change over time. Paths and timeout can also be set in `.env`.

Build the persistent local index once (add `--rebuild` to replace the collection):

```powershell
python build_index.py
# python build_index.py --rebuild
```

First index construction downloads the Bitext dataset and the embedding model and may take a while. Index data is stored in `chroma_db/` by default and excluded from Git.

## Run

Terminal 1, from this directory with the virtual environment active:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open the Streamlit URL shown in the terminal. For direct API exploration, open `http://127.0.0.1:8000/docs`. `GET /health` reports readiness without credentials; `POST /ask` accepts `{"question":"How can I request a refund?"}` and returns `answer`, `outcome` (`supported`, `partial`, or `fallback`), source passages and metadata/scores, retry count, and end-to-end `latency_ms`.

## Tests and evaluation

Run automated API and deterministic helper tests:

```powershell
pytest -q
```

Run the live evaluation after the index, API configuration, and Groq access are ready:

```powershell
python evaluate.py
```

The evaluation prints question, retrieved record IDs/scores, evidence text, answer/outcome, retries, measured latency, and a basic expected-category review. `PASS` checks only broad expected behavior; it is not an accuracy result. Manually compare each returned passage and answer. The evaluation questions are fixed requirements; because each API/index build may retrieve differently, no outcomes are asserted as already passed. The dataset-derived reference questions and answers can be sampled/inspected directly from the indexed FAQ records; no fabricated dataset rows are used.

The API tests cover health, validation, latency field, unavailable initialization, and safe rate-limit handling. Provider timeouts/authentication failures and index initialization failure are handled in the service code but require provider or environment simulation for live integration verification.

## Safe errors and limitations

- Missing key or index: API starts in `not_ready` state; `/ask` returns a generic 503. Check server logs and `.env`, then build the index.
- Groq authentication: safe 502 response; verify the key and model access.
- Rate limit: 429; wait and retry.
- Timeout/network error: 504/502; check connectivity and configured timeout.
- Model download or dependency failure: retry setup with compatible Python 3.11 wheels; ensure free disk space and internet access.
- These heuristic evidence and review checks can make mistakes. Retrieval scores are not probabilities. Review answers against the displayed passages.
- This is a local academic example, not a public deployment security review. The app cannot fetch live customer records. Keep the API bound to localhost and `.env` private.

## Experiment 7 checklist

- [x] FastAPI `/health` and `/ask`, Pydantic validation, safe client errors, and latency measurement.
- [x] Structured operational logging without question text or secrets.
- [x] Explicit persistent Chroma build/rebuild command and one-time API resource initialization.
- [x] Local normalized sentence-transformer vectors; Groq-only generation.
- [x] LangGraph intent, retrieval, evidence assessment, answer draft/review, one retry, and fallback.
- [x] Static live-data limitation, top-four sources, Streamlit chat UI via HTTP, and evaluation script.
- [ ] Run the commands above in the target Windows environment and manually inspect actual outputs before reporting a successful end-to-end run.
