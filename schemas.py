from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty or whitespace only")
        return value.strip()


class SourceReference(BaseModel):
    source: str
    category: str = ""
    intent: str = ""
    record_id: str = ""
    passage: str
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    outcome: Literal["supported", "partial", "fallback"]
    sources: list[SourceReference]
    retry_count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool
