"""Sub-agent spawner — isolated TAOR execution with fresh context."""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from core.hooks import HookManager
from core.loop import run_agent_loop
from core.types import AgentConfig, SkillConfig

logger = logging.getLogger(__name__)


class SubAgentSpawner:
    """Spawn sub-agents with **complete context isolation**.

    Each sub-agent gets:
      - A fresh ``messages`` list (no parent conversation history)
      - Its own system prompt (from AGENT.md body)
      - A filtered set of tools (per agent config)
      - Optionally a different LLM model
      - The parent's hook manager (hooks run on sub-agents too)

    Only the final text response returns to the parent.
    """

    def __init__(
        self,
        llm_factory: Callable[..., BaseChatModel],
        tool_manager: Any,  # tools.manager.ToolManager — avoid circular import
        registry: Any,  # discovery.registry.CapabilityRegistry
        hook_manager: HookManager | None = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._tool_manager = tool_manager
        self._registry = registry
        self._hook_manager = hook_manager

    async def spawn(self, agent_config: AgentConfig, prompt: str) -> str:
        """Spawn a sub-agent from an ``AgentConfig`` (AGENT.md).

        Returns the agent's final text response only.
        """
        logger.info(
            "Spawning sub-agent '%s' (max_turns=%d)",
            agent_config.name,
            agent_config.max_turns,
        )

        # Create LLM — may override model
        llm = self._llm_factory(model=agent_config.model)

        # Filter tools per agent's allowlist/denylist
        tools = self._tool_manager.filter_tools(
            allowed=agent_config.tools,
            disallowed=agent_config.disallowed_tools,
        )

        # Build system prompt: agent's own prompt + preloaded skills
        system_prompt = agent_config.system_prompt
        for skill_name in agent_config.skills:
            try:
                skill = self._registry.get_skill(skill_name)
                system_prompt += f"\n\n## Preloaded Skill: {skill.name}\n{skill.instructions}"
            except KeyError:
                logger.warning(
                    "Agent '%s' references unknown skill '%s'",
                    agent_config.name,
                    skill_name,
                )

        # ── FRESH context — complete isolation ───────────────────────────
        messages = [HumanMessage(content=prompt)]

        result = await run_agent_loop(
            llm=llm,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_turns=agent_config.max_turns,
            hook_manager=self._hook_manager,
        )

        logger.info("Sub-agent '%s' completed", agent_config.name)
        return result

    async def spawn_from_skill(self, skill: SkillConfig, instructions: str) -> str:
        """Spawn a sub-agent from a skill with ``context: fork``.

        The skill's instructions become the system prompt.
        """
        logger.info("Spawning skill-fork '%s'", skill.name)

        llm = self._llm_factory(model=skill.model)

        tools = self._tool_manager.filter_tools(allowed=skill.allowed_tools)

        # Fresh context — skill instructions are the system prompt
        messages = [HumanMessage(content="Execute the skill instructions above.")]

        result = await run_agent_loop(
            llm=llm,
            system_prompt=instructions,
            messages=messages,
            tools=tools,
            max_turns=30,
            hook_manager=self._hook_manager,
        )

        logger.info("Skill-fork '%s' completed", skill.name)
        return result
