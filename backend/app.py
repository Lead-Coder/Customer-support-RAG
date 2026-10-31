"""FastAPI service for the local customer-support RAG workflow."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from groq import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from schemas import AskRequest, AskResponse, HealthResponse
from support_rag import SupportRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("customer_support_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = None
    app.state.init_error = None
    try:
        app.state.rag = SupportRAG.from_environment()
        log.info("RAG resources initialized")
    except Exception as exc:
        # Keep the service available for /health and safe diagnostics; do not expose details to clients.
        app.state.init_error = exc.__class__.__name__
        log.exception("RAG initialization failed (%s)", exc.__class__.__name__)
    yield


app = FastAPI(title="Customer Support RAG API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    ready = app.state.rag is not None
    return HealthResponse(status="ok" if ready else "not_ready", rag_initialized=ready)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    rag = app.state.rag
    if rag is None:
        raise HTTPException(status_code=503, detail="Customer-support service is not ready. Check server configuration and index setup.")
    start = time.perf_counter()
    try:
        result = rag.ask(request.question)
        latency = (time.perf_counter() - start) * 1000
        log.info("request_completed outcome=%s retries=%s latency_ms=%.2f", result["outcome"], result["retry_count"], latency)
        return AskResponse(**result, latency_ms=round(latency, 2))
    except AuthenticationError:
        log.error("provider_authentication_failed")
        raise HTTPException(status_code=502, detail="The language model provider rejected its credentials. Check GROQ_API_KEY.")
    except RateLimitError:
        log.warning("provider_rate_limited")
        raise HTTPException(status_code=429, detail="The language model is temporarily rate limited. Please try again shortly.")
    except (APITimeoutError, TimeoutError):
        log.warning("provider_timeout")
        raise HTTPException(status_code=504, detail="The language model request timed out. Please try again.")
    except APIConnectionError:
        log.warning("provider_connection_error")
        raise HTTPException(status_code=502, detail="The language model could not be reached. Please try again shortly.")
    except Exception as exc:
        log.exception("request_failed error_type=%s", exc.__class__.__name__)
        raise HTTPException(status_code=500, detail="The request could not be completed. Please try again later.")
