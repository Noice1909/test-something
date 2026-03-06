from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from src.services.neo4j_service import Neo4jService
from src.services.schema_service import SchemaService

logger = structlog.get_logger()

WRITE_KEYWORDS = {"CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP"}
VALID_START_KEYWORDS = {"MATCH", "OPTIONAL", "WITH", "CALL", "UNWIND", "RETURN"}


@dataclass
class ValidationResult:
    valid: bool
    cypher: str = ""
    errors: list[str] = field(default_factory=list)


class CypherValidator:
    def __init__(self, neo4j_svc: Neo4jService, schema_svc: SchemaService) -> None:
        self.neo4j_svc = neo4j_svc
        self.schema_svc = schema_svc

    def validate(self, cypher: str) -> ValidationResult:
        """Run all validation layers on a Cypher query."""
        errors: list[str] = []

        # Layer 1: Deterministic checks
        deterministic_errors = self._deterministic_checks(cypher)
        if deterministic_errors:
            errors.extend(deterministic_errors)
            # Write-block is fatal — don't proceed
            if any("write operation" in e.lower() for e in deterministic_errors):
                return ValidationResult(valid=False, cypher=cypher, errors=errors)

        # Auto-inject LIMIT if missing
        cypher = self._inject_limit(cypher)

        # Layer 2: EXPLAIN test
        if not errors:
            explain_ok = self._explain_test(cypher)
            if not explain_ok:
                errors.append("EXPLAIN failed: query may have syntax errors or reference invalid schema elements")

        if errors:
            return ValidationResult(valid=False, cypher=cypher, errors=errors)

        return ValidationResult(valid=True, cypher=cypher)

    def _deterministic_checks(self, cypher: str) -> list[str]:
        """Instant validation without DB calls."""
        errors: list[str] = []
        upper = cypher.upper()
        tokens = upper.split()

        # Check for write operations
        for kw in WRITE_KEYWORDS:
            # Use word boundary check to avoid false positives
            pattern = r'\b' + kw + r'\b'
            if re.search(pattern, upper):
                errors.append(f"Blocked: contains write operation '{kw}'")
                return errors  # Fatal, return immediately

        # Check query starts with valid clause
        if tokens:
            first_word = tokens[0].strip("(")
            if first_word not in VALID_START_KEYWORDS:
                errors.append(f"Query starts with '{first_word}', expected one of: {', '.join(sorted(VALID_START_KEYWORDS))}")

        # Check RETURN exists
        if "RETURN" not in upper:
            errors.append("Query missing RETURN clause")

        # Check node labels exist in schema
        label_pattern = re.findall(r'[:(]\s*([A-Z][A-Za-z0-9_]*)\s*[){]', cypher)
        for label in label_pattern:
            if label in VALID_START_KEYWORDS or label in WRITE_KEYWORDS:
                continue
            if not self.schema_svc.label_exists(label):
                errors.append(f"Unknown label '{label}' — not found in schema")

        # Check relationship types exist in schema
        rel_pattern = re.findall(r'\[:\s*([A-Z_][A-Z0-9_]*)', cypher)
        for rel_type in rel_pattern:
            if not self.schema_svc.relationship_type_exists(rel_type):
                errors.append(f"Unknown relationship type '{rel_type}' — not found in schema")

        return errors

    def _inject_limit(self, cypher: str) -> str:
        """Add LIMIT 25 if no LIMIT clause exists."""
        if "LIMIT" not in cypher.upper():
            cypher = cypher.rstrip().rstrip(";") + "\nLIMIT 25"
        return cypher

    def _explain_test(self, cypher: str) -> bool:
        """Run EXPLAIN to check query validity without executing."""
        try:
            return self.neo4j_svc.explain(cypher)
        except Exception as exc:
            logger.warning("explain_test_error", error=str(exc))
            return False
