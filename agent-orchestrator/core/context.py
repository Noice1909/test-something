"""Per-session conversation context with auto-compaction."""

from __future__ import annotations

import logging
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages message history for a single session.

    When the estimated token count exceeds the threshold, older turns
    are summarised into a single message to free up context space.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self.messages: list[BaseMessage] = []

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def estimate_tokens(self) -> int:
        """Rough estimate: ~4 characters per token."""
        total_chars = sum(len(str(m.content)) for m in self.messages)
        return total_chars // 4

    async def compact(
        self,
        llm: BaseChatModel,
        max_tokens: int,
        threshold: float = 0.8,
    ) -> None:
        """Summarise older turns if we're approaching the context limit.

        Keeps the most recent 6 messages intact and asks the LLM to
        summarise everything before that.
        """
        current = self.estimate_tokens()
        if current < int(max_tokens * threshold):
            return

        keep_recent = 6
        if len(self.messages) <= keep_recent:
            return  # nothing to compact

        old_messages = self.messages[:-keep_recent]
        recent_messages = self.messages[-keep_recent:]

        # Ask the LLM to summarise the older turns
        summary_prompt = (
            "Summarise the following conversation history concisely, "
            "preserving key decisions, entities, and context:\n\n"
        )
        for msg in old_messages:
            role = msg.__class__.__name__.replace("Message", "")
            summary_prompt += f"[{role}]: {str(msg.content)[:500]}\n"

        try:
            response = await llm.ainvoke(
                [SystemMessage(content="You are a conversation summariser."),
                 HumanMessage(content=summary_prompt)]
            )
            summary_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
        except Exception as exc:
            logger.error("Compaction failed: %s", exc)
            return

        # Replace old messages with summary
        self.messages = [
            SystemMessage(content=f"[Conversation summary]: {summary_text}"),
            *recent_messages,
        ]
        logger.info(
            "Compacted session %s: %d tokens → %d tokens",
            self.session_id,
            current,
            self.estimate_tokens(),
        )


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
