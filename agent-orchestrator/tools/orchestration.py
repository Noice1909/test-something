"""Orchestration meta-tools — invoke_skill and delegate_to_agent.

These are the two tools that make the LLM an orchestrator.  When the
LLM's system prompt includes skill/agent descriptions, the LLM
naturally calls these tools to route work — no classifier needed.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from discovery.skill_loader import process_dynamic_context, substitute_arguments

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input schemas
# ─────────────────────────────────────────────────────────────────────────────


class InvokeSkillInput(BaseModel):
    name: str = Field(description="The skill name to invoke (e.g. 'cypher-read', 'fuzzy-search')")
    arguments: str = Field(
        default="",
        description="Arguments to pass to the skill (substituted into $ARGUMENTS, $0, $1, etc.)",
    )


class DelegateToAgentInput(BaseModel):
    name: str = Field(description="The agent name to delegate to (e.g. 'discovery-agent', 'query-agent')")
    prompt: str = Field(description="A clear, detailed prompt describing what the agent should do")


# ─────────────────────────────────────────────────────────────────────────────
# invoke_skill
# ─────────────────────────────────────────────────────────────────────────────


class InvokeSkillTool(BaseTool):
    """Activate a registered skill by name.

    If the skill has ``context: fork``, it runs as an isolated sub-agent.
    Otherwise, the skill's instructions are injected into the current
    conversation context.
    """

    name: str = "invoke_skill"
    description: str = (
        "Invoke a registered skill by name. Use this when the user's "
        "request matches a skill's description. Pass the skill name and "
        "any arguments."
    )
    args_schema: Type[BaseModel] = InvokeSkillInput

    # Injected dependencies (set at construction time)
    registry: Any = None
    spawner: Any = None

    async def _arun(self, name: str, arguments: str = "") -> str:
        try:
            skill = self.registry.get_skill(name)
        except KeyError as exc:
            return str(exc)

        # Apply argument substitution
        instructions = substitute_arguments(skill.instructions, arguments)

        # Apply Dynamic Context Injection (!`command`)
        instructions = await process_dynamic_context(instructions)

        if skill.context == "fork":
            # Run as isolated sub-agent
            logger.info("Invoking skill '%s' as fork (sub-agent)", name)
            return await self.spawner.spawn_from_skill(skill, instructions)
        else:
            # Inject into current conversation
            logger.info("Invoking skill '%s' inline", name)
            return f"[Skill activated: {name}]\n\n{instructions}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


# ─────────────────────────────────────────────────────────────────────────────
# delegate_to_agent
# ─────────────────────────────────────────────────────────────────────────────


class DelegateToAgentTool(BaseTool):
    """Delegate a task to a specialized sub-agent.

    The agent runs in complete isolation — fresh context, filtered tools,
    its own system prompt.  Only the final text returns.
    """

    name: str = "delegate_to_agent"
    description: str = (
        "Delegate a task to a specialized sub-agent for focused, complex "
        "work. The agent runs independently and returns its findings. "
        "Use when the task matches an agent's expertise."
    )
    args_schema: Type[BaseModel] = DelegateToAgentInput

    # Injected dependencies
    registry: Any = None
    spawner: Any = None

    async def _arun(self, name: str, prompt: str) -> str:
        try:
            agent_config = self.registry.get_agent(name)
        except KeyError as exc:
            return str(exc)

        logger.info("Delegating to agent '%s'", name)
        return await self.spawner.spawn(agent_config, prompt)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")
