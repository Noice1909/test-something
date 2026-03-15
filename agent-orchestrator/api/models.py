"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    message: str = Field(description="The user message or /slash-command")
    session_id: str | None = Field(
        default=None,
        description="Session ID to continue an existing conversation",
    )
    model: str | None = Field(
        default=None,
        description="Override the LLM model for this request",
    )


# ── Responses ────────────────────────────────────────────────────────────────


class QueryResponse(BaseModel):
    response: str
    session_id: str
    turns_used: int
    tools_called: list[str] = Field(default_factory=list)


class SkillInfo(BaseModel):
    name: str
    description: str
    user_invocable: bool
    argument_hint: str | None = None
    context: str | None = None


class AgentInfo(BaseModel):
    name: str
    description: str
    tools: list[str] | None = None
    max_turns: int = 30


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    skills_loaded: int
    agents_loaded: int


class ReloadResponse(BaseModel):
    skills_loaded: int
    agents_loaded: int
    message: str
