"""Per-session conversation context with sophisticated auto-compaction.

Compaction uses importance-scored message tiers and structured extraction
to preserve key entities, decisions, and patterns while freeing context space.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

# Importance score thresholds for tiering
_TIER1_THRESHOLD = 0.45   # keep verbatim
_TIER2_THRESHOLD = 0.20   # summarise via LLM
# Below tier2 → drop (tier 3)

_KEEP_RECENT = 8  # always keep the most recent N messages intact


class ConversationContext:
    """Manages message history for a single session.

    When the estimated token count exceeds the threshold, a multi-phase
    compaction runs:
      1. Score every message by importance (recency, role, content length)
      2. Classify into tiers: keep / summarise / drop
      3. Ask the LLM to extract structured context from tier-2 messages
      4. Rebuild history: [structured summary] + [tier-1 messages]
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self.messages: list[BaseMessage] = []
        # Accumulated structured context across compactions
        self._prior_context: str = ""

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def estimate_tokens(self) -> int:
        """Rough estimate: ~4 characters per token."""
        total_chars = sum(len(str(m.content)) for m in self.messages)
        return total_chars // 4

    # ── Sophisticated compaction ─────────────────────────────────────────

    async def compact(
        self,
        llm: BaseChatModel,
        max_tokens: int,
        threshold: float = 0.8,
    ) -> None:
        """Run multi-phase compaction if approaching the context limit."""
        current = self.estimate_tokens()
        if current < int(max_tokens * threshold):
            return

        if len(self.messages) <= _KEEP_RECENT:
            return  # nothing to compact

        old_messages = self.messages[:-_KEEP_RECENT]
        recent_messages = self.messages[-_KEEP_RECENT:]

        # Phase 1 — score every old message
        scored = _score_messages(old_messages)

        # Phase 2 — classify into tiers
        tier1: list[BaseMessage] = []       # keep verbatim
        tier2: list[BaseMessage] = []       # extract structured context
        for msg, score in scored:
            if score >= _TIER1_THRESHOLD:
                tier1.append(msg)
            elif score >= _TIER2_THRESHOLD:
                tier2.append(msg)
            # else: tier 3 — drop silently

        # Phase 3 — extract structured context from tier-2 via LLM
        structured = ""
        if tier2:
            structured = await _extract_structured_context(llm, tier2, self._prior_context)

        # Phase 4 — rebuild history
        summary_parts: list[str] = []
        if self._prior_context:
            summary_parts.append(self._prior_context)
        if structured:
            summary_parts.append(structured)

        combined_summary = "\n\n".join(summary_parts) if summary_parts else ""
        self._prior_context = combined_summary  # persist for next compaction

        new_messages: list[BaseMessage] = []
        if combined_summary:
            new_messages.append(
                SystemMessage(content=f"[Conversation context — preserved across compaction]\n{combined_summary}")
            )
        new_messages.extend(tier1)
        new_messages.extend(recent_messages)

        after_tokens = sum(len(str(m.content)) for m in new_messages) // 4
        logger.info(
            "Compacted session %s: %d tokens → %d tokens  "
            "(tier1=%d kept, tier2=%d extracted, tier3=%d dropped)",
            self.session_id, current, after_tokens,
            len(tier1), len(tier2),
            len(old_messages) - len(tier1) - len(tier2),
        )
        self.messages = new_messages


# ── Scoring ──────────────────────────────────────────────────────────────────


def _score_messages(messages: list[BaseMessage]) -> list[tuple[BaseMessage, float]]:
    """Score each message 0.0–1.0 based on multiple importance signals."""
    n = len(messages)
    scored: list[tuple[BaseMessage, float]] = []

    for i, msg in enumerate(messages):
        score = 0.0

        # Signal 1: recency (0.0 → oldest, 0.3 → newest)
        recency = i / max(n - 1, 1)
        score += recency * 0.3

        # Signal 2: role importance
        if isinstance(msg, HumanMessage):
            score += 0.40  # user intent is high-value
        elif isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                score += 0.15  # tool-calling decisions
            else:
                score += 0.25  # reasoning / final answers
        elif isinstance(msg, ToolMessage):
            status = getattr(msg, "status", None)
            content_len = len(str(msg.content))
            if status == "error":
                score += 0.05  # failed tools are low-value
            elif content_len > 500:
                score += 0.20  # substantial data
            elif content_len > 100:
                score += 0.12
            else:
                score += 0.08  # minimal result
        elif isinstance(msg, SystemMessage):
            score += 0.35  # prior compaction summaries

        # Signal 3: content richness — mentions of IDs / specific data
        content = str(msg.content)
        if "elementId" in content or "4:" in content:
            score += 0.05  # references graph element IDs
        if any(kw in content.lower() for kw in ("decision", "because", "chose", "prefer")):
            score += 0.05  # decision language

        scored.append((msg, min(score, 1.0)))

    return scored


# ── Structured extraction ────────────────────────────────────────────────────


async def _extract_structured_context(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    prior_context: str,
) -> str:
    """Ask the LLM to extract structured information from messages being compacted."""
    conversation_text = ""
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        content = str(msg.content)
        # Truncate very long tool results for the extraction prompt
        if len(content) > 800:
            content = content[:800] + "..."
        conversation_text += f"[{role}]: {content}\n"

    prompt = (
        "Extract the following from this conversation segment. Be concise — use bullet points.\n\n"
        "1. ENTITIES: Names, IDs, node labels, relationship types mentioned\n"
        "   Format: `name` [Label] (elementId if known)\n\n"
        "2. DECISIONS: Key choices, approaches tried, what worked/failed\n"
        "   Format: Tried X → result (worked/failed because Y)\n\n"
        "3. QUERY PATTERNS: Cypher patterns or queries that were generated\n"
        "   Format: `(A)-[REL]->(B)` — worked/failed\n\n"
        "4. USER INTENT: What the user is trying to accomplish overall\n\n"
        "--- CONVERSATION ---\n"
        f"{conversation_text}\n"
    )
    if prior_context:
        prompt += f"\n--- PRIOR CONTEXT (already extracted) ---\n{prior_context}\nDo NOT repeat items already in prior context.\n"

    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a precise information extractor. Output only the structured extraction, nothing else."),
            HumanMessage(content=prompt),
        ])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.error("Structured extraction failed: %s", exc)
        # Fallback: just list message roles
        return f"[Extraction failed — {len(messages)} messages were compacted]"


# ── Session store ────────────────────────────────────────────────────────────


class SessionStore:
    """In-memory session store.  Extend to Redis/SQLite for persistence."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationContext] = {}

    def get_or_create(self, session_id: str | None = None) -> ConversationContext:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        ctx = ConversationContext(session_id)
        self._sessions[ctx.session_id] = ctx
        return ctx

    def get(self, session_id: str) -> ConversationContext | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
