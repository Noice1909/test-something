"""Parse AGENT.md files — YAML frontmatter + markdown body (system prompt)."""

from __future__ import annotations

import logging
from pathlib import Path

from core.types import AgentConfig
from discovery.skill_loader import parse_frontmatter

logger = logging.getLogger(__name__)


def load_agent(path: Path) -> AgentConfig:
    """Load a single AGENT.md file into an ``AgentConfig``."""
    content = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(content)

    # tools: comma-separated string or list
    tools_raw = meta.get("tools")
    tools: list[str] | None = None
    if tools_raw:
        if isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        elif isinstance(tools_raw, list):
            tools = tools_raw

    disallowed_raw = meta.get("disallowedTools", meta.get("disallowed-tools", []))
    if isinstance(disallowed_raw, str):
        disallowed = [t.strip() for t in disallowed_raw.split(",") if t.strip()]
    else:
        disallowed = list(disallowed_raw)

    skills_raw = meta.get("skills", [])
    if isinstance(skills_raw, str):
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
    else:
        skills = list(skills_raw)

    return AgentConfig(
        name=meta.get("name", path.parent.name),
        description=meta.get("description", ""),
        system_prompt=body.strip(),
        tools=tools,
        disallowed_tools=disallowed,
        model=meta.get("model"),
        max_turns=meta.get("max-turns", meta.get("maxTurns", 30)),
        skills=skills,
    )


def load_all_agents(agents_dir: Path) -> dict[str, AgentConfig]:
    """Recursively scan ``agents_dir`` for AGENT.md files."""
    agents: dict[str, AgentConfig] = {}

    if not agents_dir.exists():
        logger.info("Agents directory does not exist: %s", agents_dir)
        return agents

    for path in agents_dir.rglob("AGENT.md"):
        try:
            agent = load_agent(path)
            agents[agent.name] = agent
            logger.info("Loaded agent: %s  (%s)", agent.name, path)
        except Exception as exc:
            logger.error("Failed to load agent %s: %s", path, exc)

    return agents
