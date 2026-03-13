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
    agentic_max_empty_retries: int = 2

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
    cache_backend: str = "memory"  # "memory" | "sqlite" | "redis"
    cache_redis_url: str = "redis://localhost:6379/0"
    cache_sqlite_path: str = "cache.db"
    cache_response_ttl: int = 300
    cache_strategy_ttl: int = 600
    cache_max_size: int = 1000

    # ── Server ──
    server_host: str = "0.0.0.0"
    server_port: int = 8001

    # ── Logging ──
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "text"

    # ── CORS ──
    cors_allowed_origins: str = "*"  # comma-separated origins

    # ── Auth ──
    auth_api_key: str = ""  # if set, X-API-Key header required; empty = disabled

    # ── Rate limiting ──
    rate_limit_default: str = "30/minute"
    rate_limit_storage: str = "memory"  # "memory" | "redis://..."

    # ── Request timeout ──
    request_timeout_seconds: int = 300

    # ── Circuit breakers ──
    cb_neo4j_fail_max: int = 5
    cb_neo4j_reset_timeout: int = 30
    cb_llm_fail_max: int = 3
    cb_llm_reset_timeout: int = 60

    # ── Neo4j connection pool ──
    neo4j_pool_max_size: int = 100
    neo4j_pool_acquisition_timeout: float = 60.0
    neo4j_max_connection_lifetime: int = 3600

    # ── Concurrency ──
    max_concurrent_requests: int = 50
    max_concurrent_llm_calls: int = 10

    # ── LLM context window ──
    llm_context_window: int = 8192
    llm_response_reserve: float = 0.2

    # ── Database health monitor ──
    db_health_check_interval: int = 15
    db_max_consecutive_failures: int = 3
    db_reconnect_backoff_max: int = 120

    # ── Discovery optimization ──
    discovery_max_results_per_tool: int = 10
    discovery_smart_truncation_top_n: int = 10
    discovery_smart_truncation_summary_n: int = 20
    discovery_total_budget_chars: int = 8000
    discovery_fuzzy_threshold: float = 0.6
    discovery_cross_tool_boost: float = 0.1
    discovery_excluded_properties: str = "embedding,embedding_vector,vector,raw_embedding,raw_text,features"
    discovery_deep_fuzzy_min_results: int = 3
    discovery_levenshtein_short_max: int = 3  # terms <= this length use len as threshold

    # ── Checkpointing ──
    checkpoint_enabled: bool = True
    checkpoint_backend: str = "memory"  # "memory" | "sqlite" | "redis"
    checkpoint_ttl: int = 1800
    checkpoint_max_conversations: int = 10000
    checkpoint_max_turns: int = 20

    # ── Derived helpers ──
    @property
    def use_openai(self) -> bool:
        """True when an OpenAI key is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 3


settings = Settings()
