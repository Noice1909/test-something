"""Application settings — env-driven via Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration is read from environment variables (prefixed AGENT_) or .env."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = "anthropic"  # "anthropic" | "openai" | "ollama"
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_api_key: str = ""
    llm_base_url: str | None = None  # for local models / custom endpoints
    llm_temperature: float = 0.0
    max_tokens: int = 8192

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_skip_tls_verify: bool = False

    # ── Agent ────────────────────────────────────────────────────────────────
    max_turns: int = 50
    max_context_tokens: int = 150_000
    compact_threshold: float = 0.8  # compact at 80% of max

    # ── Discovery ────────────────────────────────────────────────────────────
    skills_dir: str = ".skills"
    agents_dir: str = ".agents"

    # ── MCP ──────────────────────────────────────────────────────────────────
    mcp_servers: dict[str, dict] = {}
    # Example: {"neo4j": {"command": "python", "args": ["mcp_neo4j.py"]}}

    # ── Hooks ─────────────────────────────────────────────────────────────────
    hooks_dir: str = ".hooks"  # directory for hook scripts (future: file-based hooks)

    # ── Prompt Caching ────────────────────────────────────────────────────────
    enable_prompt_cache: bool = True  # cache system prompt + Anthropic cache_control

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
