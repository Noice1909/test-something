from __future__ import annotations

import os
from typing import Any

import structlog
import yaml

from src.config import Settings

logger = structlog.get_logger()

EXAMPLES_FILE = "few_shot_examples.yml"


class FewShotService:
    """Loads question→Cypher examples and retrieves similar ones via ChromaDB."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._examples: list[dict[str, str]] = []
        self._collection: Any = None

    @property
    def count(self) -> int:
        return len(self._examples)

    async def initialize(self) -> None:
        """Load examples from YAML and embed into ChromaDB."""
        self._examples = self._load_yaml()

        if not self._examples:
            logger.info("few_shot_no_examples", msg="No examples found in YAML; few-shot disabled")
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # Use PersistentClient with a data directory to avoid threading issues
            client = chromadb.PersistentClient(
                path="./data/chromadb",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Delete existing collection if it exists (re-index on restart)
            try:
                client.delete_collection("few_shot_examples")
            except Exception:
                pass

            self._collection = client.create_collection(
                name="few_shot_examples",
                metadata={"hnsw:space": "cosine"},
            )

            # Add all examples
            self._collection.add(
                documents=[ex["question"] for ex in self._examples],
                ids=[f"ex_{i}" for i in range(len(self._examples))],
                metadatas=[{"cypher": ex["cypher"]} for ex in self._examples],
            )

            logger.info("few_shot_initialized", count=len(self._examples))
        except Exception as exc:
            logger.warning("few_shot_chromadb_init_failed", error=str(exc))
            self._collection = None

    def retrieve(self, question: str, k: int | None = None) -> str:
        """Retrieve top-K similar examples and format for the LLM prompt."""
        if not self._collection or not self._examples:
            return ""

        k = k or self.settings.FEW_SHOT_K

        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=min(k, len(self._examples)),
            )
        except Exception as exc:
            logger.warning("few_shot_query_failed", error=str(exc))
            return ""

        if not results or not results.get("documents"):
            return ""

        lines: list[str] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for doc, meta in zip(documents, metadatas):
            cypher = meta.get("cypher", "")
            lines.append(f"Question: {doc}")
            lines.append(f"Cypher: {cypher}")
            lines.append("")

        return "\n".join(lines)

    def _load_yaml(self) -> list[dict[str, str]]:
        """Load examples from the YAML file."""
        if not os.path.exists(EXAMPLES_FILE):
            logger.info("few_shot_file_not_found", path=EXAMPLES_FILE)
            return []

        try:
            with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            logger.warning("few_shot_yaml_parse_failed", error=str(exc))
            return []

        if not data or not isinstance(data, dict):
            return []

        examples = data.get("examples", [])
        if not isinstance(examples, list):
            return []

        valid = []
        for ex in examples:
            if isinstance(ex, dict) and ex.get("question") and ex.get("cypher"):
                valid.append({"question": ex["question"], "cypher": ex["cypher"].strip()})

        return valid
