"""Conversation manager — restores context for multi-turn interactions."""

from __future__ import annotations

import logging
from dataclasses import asdict

from src.agents.checkpoint import (
    Checkpoint,
    CheckpointStore,
    create_checkpoint_store,
    generate_conversation_id,
)
from src.agents.state import AgentState
from src.agents.base import AgenticResponse, DiscoveryResult, SchemaSelection
from src.config import settings

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages multi-turn conversation state via checkpointing."""

    def __init__(self, store: CheckpointStore | None = None) -> None:
        self._store = store or create_checkpoint_store()

    @property
    def store(self) -> CheckpointStore:
        return self._store

    async def prepare_state(
        self, state: AgentState, conversation_id: str | None,
    ) -> str:
        """Populate *state* with context from prior turns.

        Returns the resolved ``conversation_id`` (new or existing).
        """
        if not settings.checkpoint_enabled:
            return state.trace_id

        if conversation_id is None:
            # New conversation
            cid = generate_conversation_id()
            state.conversation_id = cid
            state.turn_number = 1
            return cid

        # Existing conversation — restore context
        state.conversation_id = conversation_id
        turns = await self._store.load(conversation_id)

        if not turns:
            state.turn_number = 1
            return conversation_id

        state.turn_number = len(turns) + 1

        # Build conversation context for the supervisor prompt
        state.previous_context = [
            {"question": t.question, "answer": t.answer}
            for t in turns[-5:]  # last 5 turns max
        ]

        # Restore discoveries and schema from the latest turn
        latest = turns[-1]
        if latest.discoveries:
            for d in latest.discoveries:
                try:
                    state.discoveries.append(DiscoveryResult(**d))
                except Exception as exc:
                    logger.debug("Skipping malformed discovery in checkpoint: %s", exc)

        if latest.schema_selection:
            try:
                state.schema_selection = SchemaSelection(**latest.schema_selection)
            except Exception as exc:
                logger.debug("Skipping malformed schema_selection in checkpoint: %s", exc)

        logger.info(
            "Restored conversation %s turn %d (prior turns=%d)",
            conversation_id, state.turn_number, len(turns),
        )
        return conversation_id

    async def save_turn(
        self,
        conversation_id: str,
        state: AgentState,
        answer: str,
    ) -> None:
        """Persist the current turn as a checkpoint."""
        if not settings.checkpoint_enabled:
            return

        checkpoint = Checkpoint(
            conversation_id=conversation_id,
            turn_number=state.turn_number,
            question=state.question,
            answer=answer,
            strategy=state.strategy.value,
            discoveries=[asdict(d) for d in state.discoveries],
            schema_selection=asdict(state.schema_selection),
            generated_query=state.generated_query.query,
        )
        await self._store.save(checkpoint)
        logger.debug("Saved checkpoint: %s turn %d", conversation_id, state.turn_number)

    async def close(self) -> None:
        await self._store.close()
