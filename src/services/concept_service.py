from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.services.neo4j_service import Neo4jService

logger = structlog.get_logger()


@dataclass
class Concept:
    name: str
    nlp_terms: list[str]
    description: str
    concept_id: str
    sample_values: str


class ConceptService:
    """
    Loads :Concept nodes (if they exist) and builds an in-memory
    nlp_terms index for deterministic question → label mapping.
    """

    def __init__(self, neo4j_svc: Neo4jService) -> None:
        self.neo4j_svc = neo4j_svc
        self._concepts: list[Concept] = []
        self._index: dict[str, Concept] = {}
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def count(self) -> int:
        return len(self._concepts)

    @property
    def concepts(self) -> list[Concept]:
        return self._concepts

    async def discover(self) -> None:
        """Check if Concept label exists; if so, load and index all concepts."""
        label_rows = self.neo4j_svc.execute_read(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        )
        all_labels = set(label_rows[0]["labels"]) if label_rows else set()

        if "Concept" not in all_labels:
            logger.info("concept_label_not_found", msg="Concept matching disabled")
            self._available = False
            return

        rows = self.neo4j_svc.execute_read(
            "MATCH (c:Concept) "
            "RETURN c.name AS name, c.nlp_terms AS nlp_terms, "
            "c.description AS description, c.id AS id, "
            "c.sample_values AS sample_values"
        )

        self._concepts = []
        self._index = {}

        for row in rows:
            raw_terms = row.get("nlp_terms") or ""
            terms = [t.strip() for t in raw_terms.split(",") if t.strip()] if raw_terms else []

            concept = Concept(
                name=row.get("name") or "",
                nlp_terms=terms,
                description=row.get("description") or "",
                concept_id=row.get("id") or "",
                sample_values=row.get("sample_values") or "",
            )
            self._concepts.append(concept)

            # Index concept name
            if concept.name:
                self._index[concept.name.lower()] = concept

            # Index each nlp_term
            for term in concept.nlp_terms:
                self._index[term.lower()] = concept

        self._available = True
        logger.info("concept_index_built", concepts=len(self._concepts), terms=len(self._index))

    def match_concepts(self, question: str) -> list[Concept]:
        """Match question tokens/bigrams/trigrams against the concept index."""
        if not self._available:
            return []

        tokens = question.lower().split()
        matched: dict[str, Concept] = {}

        # Check single tokens
        for token in tokens:
            token_clean = token.strip("?,.:;!\"'()[]")
            if token_clean in self._index:
                concept = self._index[token_clean]
                matched[concept.name] = concept

        # Check bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i].strip('?,.:;!()')} {tokens[i+1].strip('?,.:;!()')}"
            if bigram in self._index:
                concept = self._index[bigram]
                matched[concept.name] = concept

        # Check trigrams
        for i in range(len(tokens) - 2):
            trigram = " ".join(
                t.strip("?,.:;!()")
                for t in tokens[i : i + 3]
            )
            if trigram in self._index:
                concept = self._index[trigram]
                matched[concept.name] = concept

        # Fuzzy matching (Levenshtein distance ≤ 2) for typos
        for token in tokens:
            token_clean = token.strip("?,.:;!\"'()[]")
            if len(token_clean) < 3:
                continue
            for term, concept in self._index.items():
                if concept.name in matched:
                    continue
                if len(term) < 3:
                    continue
                dist = _levenshtein(token_clean, term)
                if dist <= 2 and dist < len(term) * 0.4:
                    matched[concept.name] = concept

        return list(matched.values())

    def get_label_descriptions(self) -> str:
        """Format all concepts as a reference for LLM prompts."""
        if not self._available:
            return ""
        lines = []
        for c in sorted(self._concepts, key=lambda x: x.name):
            desc = c.description or "No description"
            terms = ", ".join(c.nlp_terms) if c.nlp_terms else "none"
            lines.append(f"- {c.name}: {desc} (also known as: {terms})")
        return "\n".join(lines)


def _levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]
