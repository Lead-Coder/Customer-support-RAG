from fastapi.testclient import TestClient

import app as api


class FakeRAG:
    def ask(self, question):
        if question == "raise-rate-limit":
            from groq import RateLimitError
            raise RateLimitError("private", response=None, body=None)
        return {"answer": "Please use the account help page to request support.", "outcome": "supported",
                "sources": [], "retry_count": 0}


def client_with(rag):
    # Avoid loading local models/index during endpoint unit tests.
    api.app.state.rag = rag
    client = TestClient(api.app)
    return client


def test_health_ready():
    with client_with(FakeRAG()) as client:
        api.app.state.rag = FakeRAG()
        result = client.get("/health")
    assert result.status_code == 200 and result.json()["rag_initialized"] is True


def test_ask_response_and_latency():
    with client_with(FakeRAG()) as client:
        api.app.state.rag = FakeRAG()
        result = client.post("/ask", json={"question": "How can I get help?"})
    assert result.status_code == 200
    assert result.json()["outcome"] == "supported"
    assert result.json()["latency_ms"] >= 0


def test_invalid_questions_rejected():
    with client_with(FakeRAG()) as client:
        api.app.state.rag = FakeRAG()
        for payload in ({}, {"question": ""}, {"question": "  "}, {"question": "x" * 1001}):
            assert client.post("/ask", json=payload).status_code == 422


def test_not_ready_is_safe():
    with client_with(None) as client:
        api.app.state.rag = None
        response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == 503
    assert "Traceback" not in response.text


def test_rate_limit_is_safe():
    with client_with(FakeRAG()) as client:
        api.app.state.rag = FakeRAG()
        response = client.post("/ask", json={"question": "raise-rate-limit"})
    assert response.status_code == 429
    assert "private" not in response.text
