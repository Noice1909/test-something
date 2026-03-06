from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    thread_id: str
    success: bool


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    ollama_available: bool
    environment: str
    checkpointer: str
    circuit_breakers: dict[str, str]


class SchemaResponse(BaseModel):
    labels: list[str]
    relationship_types: list[str]
    label_count: int
    relationship_type_count: int
    schema_text: str
