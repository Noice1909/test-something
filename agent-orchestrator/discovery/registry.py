"""Unified capability registry — skills + agents with hot-reload."""

from __future__ import annotations

import logging
from pathlib import Path

from core.types import AgentConfig, SkillConfig
from discovery.agent_loader import load_all_agents
from discovery.skill_loader import load_all_skills

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """Central index of all discovered skills and agents.

    Descriptions are injected into the system prompt so the LLM can
    decide which capability to invoke — pure transformer reasoning,
    no routing classifier.
    """

    def __init__(self) -> None:
        self.skills: dict[str, SkillConfig] = {}
        self.agents: dict[str, AgentConfig] = {}
        self._skills_dir: Path | None = None
        self._agents_dir: Path | None = None

    # ── Loading ──────────────────────────────────────────────────────────

    async def load_all(self, skills_dir: str, agents_dir: str) -> None:
        """Scan directories and populate the registry."""
        self._skills_dir = Path(skills_dir)
        self._agents_dir = Path(agents_dir)

        self.skills = load_all_skills(self._skills_dir)
        self.agents = load_all_agents(self._agents_dir)

        logger.info(
            "Registry loaded: %d skills, %d agents",
            len(self.skills),
            len(self.agents),
        )

    async def reload(self) -> None:
        """Re-scan directories (hot-reload)."""
        if self._skills_dir and self._agents_dir:
            await self.load_all(str(self._skills_dir), str(self._agents_dir))

    # ── Lookup ───────────────────────────────────────────────────────────

    def get_skill(self, name: str) -> SkillConfig:
        """Get a skill by name, raising ``KeyError`` if not found."""
        if name not in self.skills:
            raise KeyError(
                f"Skill '{name}' not found. Available: {list(self.skills.keys())}"
            )
        return self.skills[name]

    def get_agent(self, name: str) -> AgentConfig:
        """Get an agent by name, raising ``KeyError`` if not found."""
        if name not in self.agents:
            raise KeyError(
                f"Agent '{name}' not found. Available: {list(self.agents.keys())}"
            )
        return self.agents[name]

    # ── System prompt injection ──────────────────────────────────────────

    def get_descriptions_for_prompt(self) -> str:
        """Build a compact text block for the system prompt.

        Only includes skills where ``disable_model_invocation`` is False.
        Each entry is ~50-100 tokens so the LLM can reason about routing.
        """
        lines: list[str] = []

        # Skills
        invocable = [
            s for s in self.skills.values() if not s.disable_model_invocation
        ]
        if invocable:
            lines.append("## Available Skills (invoke with the invoke_skill tool)")
            for s in invocable:
                hint = f"  Hint: {s.argument_hint}" if s.argument_hint else ""
                lines.append(f"- **{s.name}**: {s.description}{hint}")
            lines.append("")

        # Agents
        if self.agents:
            lines.append(
                "## Available Agents (delegate with the delegate_to_agent tool)"
            )
            for a in self.agents.values():
                lines.append(f"- **{a.name}**: {a.description}")
            lines.append("")

        return "\n".join(lines)

    # ── Slash command resolution ─────────────────────────────────────────

    def resolve_slash_command(self, text: str) -> tuple[SkillConfig, str] | None:
        """If *text* starts with ``/<name>``, return ``(skill, remaining_args)``."""
        text = text.strip()
        if not text.startswith("/"):
            return None

        parts = text[1:].split(maxsplit=1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        skill = self.skills.get(name)
        if skill and skill.user_invocable:
            return skill, args
        return None
