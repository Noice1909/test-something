from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    matched_concepts: list[dict[str, Any]]
    extracted_entities: list[dict[str, Any]]
    mapped_entities: list[dict[str, Any]]
    filtered_schema: str
    few_shot_examples: str
    cypher: str
    validation_errors: list[str]
    retry_count: int
    query_results: list[dict[str, Any]]
    answer: str
    error: str | None
