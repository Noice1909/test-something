from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Environment
    ENVIRONMENT: Literal["local", "deployed"] = "local"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    NEO4J_DATABASE: str = "neo4j"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    OLLAMA_TEMPERATURE: float = 0.0

    # Agent
    SCHEMA_CACHE_TTL: int = 300
    FEW_SHOT_K: int = 5
    MAX_CYPHER_RETRIES: int = 3

    # Rate Limiting
    RATE_LIMIT_ASK: str = "10/minute"
    RATE_LIMIT_DEFAULT: str = "30/minute"

    # Circuit Breaker
    CB_FAIL_MAX: int = 5
    CB_RESET_TIMEOUT: int = 30

    # Checkpointer
    SQLITE_CHECKPOINT_PATH: str = "data/checkpoints/agent.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    @property
    def is_local(self) -> bool:
        return self.ENVIRONMENT == "local"


settings = Settings()
