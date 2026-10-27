"""Reusable LangGraph customer-support RAG workflow (local retrieval, Groq generation)."""
from __future__ import annotations

import logging
import re
from typing import TypedDict

from groq import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from config import (CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GROQ_API_KEY,
                    GROQ_MODEL, GROQ_TIMEOUT_SECONDS)

log = logging.getLogger("support_rag")
FALLBACK = ("I couldn't find enough reliable information in the customer-support FAQ to answer that confidently. "
            "Please contact support through the official Help section. Never share passwords, one-time codes, "
            "or full payment-card details.")
LIVE_FALLBACK = ("I can answer general questions from the FAQ, but I can't access live order, shipment, payment, "
                 "or account records. Please check the official account portal or contact support for this information.")


def classify_intent_label(question: str) -> str:
    q = re.sub(r"\s+", " ", question.lower()).strip()
    rules = [
        ("recover_password", ["forgot password", "reset password", "password reset", "can't log in", "cannot log in", "sign in problem"]),
        ("get_refund", ["refund", "money back", "reimburse", "return my money"]),
        ("check_return_policy", ["return policy", "return an item", "return a product", "exchange an item", "eligible to return"]),
        ("check_invoice", ["invoice", "bill", "receipt"]),
        ("contact_human_agent", ["human", "agent", "representative", "real person", "speak to someone", "customer service"]),
        ("change_order", ["cancel order", "change my order", "modify order", "edit my order"]),
        ("track_order", ["track my order", "tracking number", "shipment tracking", "where is my order", "status of order", "track order", "where is my package"]),
        ("delivery_options", ["delivery option", "shipping option", "delivery method", "shipping method"]),
        ("payment_issue", ["payment", "charged", "charge", "billing", "transaction", "double charged"]),
        ("create_account", ["create account", "register", "sign up"]),
        ("technical_support", ["website", "error", "loading", "browser", "technical issue", "app problem"]),
        ("check_account", ["account", "profile"]),
    ]
    return next((label for label, terms in rules if any(term in q for term in terms)), "general_support")


def requires_live_data(question: str) -> bool:
    q = question.lower()
    patterns = [r"\b(live|real[- ]time|current|right now)\b.*\b(status|location|tracking|order|shipment|delivery|account|balance|payment)\b",
                r"\b(status of order)\b", r"\b(where is my)\s+(order|package|shipment)\b",
                r"\b(has|did) my (order|package|payment|account)\b.*\b(ship|arrive|deliver|charge|process|approve)\w*\b",
                r"\b(is|was) my (package|order)\b.*\b(delivered|shipped|arrived)\b"]
    return any(re.search(p, q, re.I) for p in patterns)


def _evidence(question: str, docs: list[Document], scores: list[float]) -> tuple[str, str]:
    if requires_live_data(question):
        return "insufficient", "Static FAQ cannot verify live customer or transaction state."
    stops = {"what", "when", "where", "which", "would", "could", "please", "your", "have", "does", "from", "with", "that", "this", "about", "how", "can", "tell", "need", "want", "help", "for", "the", "and", "you", "my", "me", "is", "do"}
    terms = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 2 and t not in stops}
    text = " ".join(d.page_content.lower() for d in docs)
    overlap = len(terms & set(re.findall(r"[a-z0-9]+", text))) / max(1, len(terms))
    best = max(scores) if scores else None
    if not docs:
        return "insufficient", "No passages were retrieved."
    # Relevance thresholds preserve the notebook's heuristic design; review with index inspection.
    if best is not None and best >= 0.45 and overlap >= 0.10:
        return "sufficient", f"Heuristic relevance={best:.3f}, lexical overlap={overlap:.2f}."
    if best is not None and best >= 0.25 and overlap >= 0.05:
        return "partial", f"Heuristic relevance={best:.3f}, lexical overlap={overlap:.2f}; detail may be missing."
    return "insufficient", f"Evidence below heuristic checks (score={best if best is not None else 'unavailable'}, overlap={overlap:.2f})."


class RAGState(TypedDict, total=False):
    question: str; intent: str; search_query: str; docs: list[Document]; scores: list[float]
    evidence: str; reason: str; retries: int; answer: str; outcome: str; sources: list[dict]


class SupportRAG:
    def __init__(self, vector_store, llm):
        self.vector_store, self.llm = vector_store, llm
        self.graph = self._build_graph()

    @classmethod
    def from_environment(cls):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing. Set it in the project .env file.")
        if not CHROMA_DIR.exists():
            raise RuntimeError(f"Chroma index not found at {CHROMA_DIR}. Run: python build_index.py")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"},
                                           encode_kwargs={"normalize_embeddings": True})
        store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                       persist_directory=str(CHROMA_DIR))
        if store._collection.count() == 0:
            raise RuntimeError("Chroma collection is empty. Rebuild it with python build_index.py --rebuild")
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0,
                       max_tokens=220, timeout=GROQ_TIMEOUT_SECONDS, max_retries=0)
        return cls(store, llm)

    def _retrieve(self, state: RAGState):
        pairs = self.vector_store.similarity_search_with_relevance_scores(state["search_query"], k=4)
        docs, scores = [p[0] for p in pairs], [float(p[1]) for p in pairs]
        sources = [{"source": d.metadata.get("source", "Bitext FAQ"), "category": d.metadata.get("category", ""),
                    "intent": d.metadata.get("intent", ""), "record_id": d.metadata.get("record_id", ""),
                    "passage": d.page_content, "score": scores[i]} for i, d in enumerate(docs)]
        return {"docs": docs, "scores": scores, "sources": sources}

    def _build_graph(self):
        b = StateGraph(RAGState)
        def classify(s): return {"intent": classify_intent_label(s["question"]), "search_query": s["question"], "retries": 0}
        def assess(s):
            level, reason = _evidence(s["question"], s.get("docs", []), s.get("scores", []))
            return {"evidence": level, "reason": reason}
        def rewrite(s):
            terms = {"get_refund": "refund reimbursement money back process", "check_return_policy": "return eligibility window exchange policy",
                     "track_order": "order shipment delivery tracking dispatch", "check_invoice": "invoice bill receipt download",
                     "payment_issue": "payment transaction charge billing", "recover_password": "forgot password reset login",
                     "change_order": "modify cancel change order", "delivery_options": "shipping delivery methods",
                     "check_account": "account profile settings", "create_account": "account registration signup",
                     "technical_support": "website app error troubleshooting", "contact_human_agent": "human agent representative support",
                     "general_support": "customer support help"}
            return {"search_query": s["question"] + " " + terms.get(s.get("intent", "general_support"), "customer support"),
                    "retries": s.get("retries", 0) + 1, "answer": ""}
        def fallback(s): return {"answer": LIVE_FALLBACK if requires_live_data(s["question"]) else FALLBACK, "outcome": "fallback"}
        def draft(s):
            passages = "\n\n".join(d.page_content for d in s.get("docs", []))
            caveat = "The available context may be incomplete. State what is unknown." if s["evidence"] == "partial" else ""
            prompt = ("You are a careful customer-support assistant. Answer concisely, using only the FAQ context below. "
                      "Do not invent policies, prices, approvals, dates, guarantees, or live account/order status. "
                      "Never ask for passwords, one-time codes, or full card details. If unanswered, reply exactly INSUFFICIENT_CONTEXT. "
                      f"{caveat}\n\nFAQ CONTEXT:\n{passages}\n\nQUESTION:\n{s['question']}")
            response = self.llm.invoke(prompt)
            return {"answer": str(response.content).strip()}
        def review(s):
            answer = re.sub(r"\s+", " ", s.get("answer", "")).strip()
            weak = {"customer support", "general support", "i can help with that", "how can i assist you today"}
            if not answer or "INSUFFICIENT_CONTEXT" in answer.upper() or answer.lower().strip(" .") in weak or len(answer.split()) < 4:
                return {"answer": "", "outcome": "retry"}
            if s["evidence"] == "insufficient": return {"answer": "", "outcome": "retry"}
            return {"answer": answer, "outcome": "partial" if s["evidence"] == "partial" else "supported"}
        def after_evidence(s):
            if s.get("evidence") == "sufficient": return "draft"
            # Give partial evidence one expanded retrieval attempt, then answer with an explicit caveat.
            if s.get("evidence") == "partial" and s.get("retries", 0) >= 1: return "draft"
            return "rewrite" if s.get("retries", 0) < 1 else "fallback"
        def after_review(s):
            if s.get("answer"): return "end"
            return "rewrite" if s.get("retries", 0) < 1 else "fallback"
        b.add_node("classify", classify); b.add_node("retrieve", self._retrieve); b.add_node("assess", assess)
        b.add_node("draft", draft); b.add_node("review", review); b.add_node("rewrite", rewrite); b.add_node("fallback", fallback)
        b.add_edge(START, "classify"); b.add_edge("classify", "retrieve"); b.add_edge("retrieve", "assess")
        b.add_conditional_edges("assess", after_evidence, {"draft": "draft", "rewrite": "rewrite", "fallback": "fallback"})
        b.add_edge("draft", "review")
        b.add_conditional_edges("review", after_review, {"end": END, "rewrite": "rewrite", "fallback": "fallback"})
        b.add_edge("rewrite", "retrieve"); b.add_edge("fallback", END)
        return b.compile()

    def ask(self, question: str) -> dict:
        result = self.graph.invoke({"question": question})
        return {"answer": result["answer"], "outcome": result.get("outcome", "fallback"),
                "sources": result.get("sources", []), "retry_count": result.get("retries", 0)}
