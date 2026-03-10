"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Agentic system ──
    agentic_system_enabled: bool = True
    agentic_max_attempts: int = 3

    # ── Neo4j ──
    neo4j_uri: str = "neo4j+s://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_skip_tls_verify: bool = False

    # ── LLM – Ollama (default) ──
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"

    # ── LLM – OpenAI (optional) ──
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # ── Cache ──
    cache_backend: str = "memory"  # "memory" | "redis" | "sqlite"

    # ── Server ──
    server_host: str = "0.0.0.0"
    server_port: int = 8001

    # ── Derived helpers ──
    @property
    def use_openai(self) -> bool:
        """True when an OpenAI key is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 3


settings = Settings()
