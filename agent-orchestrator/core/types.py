"""Core data types used throughout the orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class SkillConfig:
    """Parsed from a SKILL.md file (YAML frontmatter + markdown body)."""

    name: str
    description: str
    instructions: str  # markdown body — the full prompt
    allowed_tools: list[str] | None = None  # tool allowlist (None → inherit all)
    model: str | None = None  # LLM model override
    user_invocable: bool = True  # visible in GET /skills
    disable_model_invocation: bool = False  # True → /slash only, LLM can't auto-pick
    context: str | None = None  # "fork" → run as isolated sub-agent
    argument_hint: str | None = None  # e.g. "[search-term]"


@dataclass
class AgentConfig:
    """Parsed from an AGENT.md file (YAML frontmatter + markdown body)."""

    name: str
    description: str
    system_prompt: str  # markdown body
    tools: list[str] | None = None  # allowed tools (None → inherit all)
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    max_turns: int = 30
    skills: list[str] = field(default_factory=list)  # skills to preload into context


@dataclass
class SessionState:
    """Per-conversation state."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
