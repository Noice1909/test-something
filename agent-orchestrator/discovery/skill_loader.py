"""Parse SKILL.md files — YAML frontmatter + markdown body.

Supports:
  - Recursive scanning of ``skills_dir/**/SKILL.md``
  - Dynamic Context Injection: ``!`command``` → subprocess stdout
  - Argument substitution: ``$ARGUMENTS``, ``$0``, ``$1``, …
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import yaml

from core.types import SkillConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter parsing
# ─────────────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body.

    Returns ``(metadata_dict, body_markdown)``.
    """
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    meta = yaml.safe_load(m.group(1)) or {}
    body = content[m.end():]
    return meta, body


def load_skill(path: Path) -> SkillConfig:
    """Load a single SKILL.md file into a ``SkillConfig``."""
    content = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(content)

    # Parse allowed-tools: space-delimited string → list
    allowed_raw = meta.get("allowed-tools")
    allowed_tools: list[str] | None = None
    if allowed_raw:
        if isinstance(allowed_raw, str):
            allowed_tools = [t.strip() for t in allowed_raw.split() if t.strip()]
        elif isinstance(allowed_raw, list):
            allowed_tools = allowed_raw

    return SkillConfig(
        name=meta.get("name", path.parent.name),
        description=meta.get("description", ""),
        instructions=body.strip(),
        allowed_tools=allowed_tools,
        model=meta.get("model"),
        user_invocable=meta.get("user-invocable", True),
        disable_model_invocation=meta.get("disable-model-invocation", False),
        context=meta.get("context"),
        argument_hint=meta.get("argument-hint"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Context Injection  (!`command`)
# ─────────────────────────────────────────────────────────────────────────────

_DCI_RE = re.compile(r"!`([^`]+)`")


async def process_dynamic_context(instructions: str) -> str:
    """Replace every ``!`command``` with the command's stdout."""

    async def _run_one(match: re.Match) -> str:
        cmd = match.group(1)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                return f"[DCI error: {stderr.decode().strip()}]"
            return stdout.decode().strip()
        except asyncio.TimeoutError:
            return f"[DCI timeout: {cmd}]"
        except Exception as exc:
            return f"[DCI error: {exc}]"

    # Gather all replacements
    matches = list(_DCI_RE.finditer(instructions))
    if not matches:
        return instructions

    replacements = await asyncio.gather(*[_run_one(m) for m in matches])

    result = instructions
    for match, replacement in zip(reversed(matches), reversed(replacements)):
        result = result[: match.start()] + replacement + result[match.end():]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Argument substitution
# ─────────────────────────────────────────────────────────────────────────────


def substitute_arguments(instructions: str, args: str) -> str:
    """Replace ``$ARGUMENTS``, ``$0``, ``$1``, etc. with actual values."""
    if not args:
        return instructions

    parts = args.split()
    result = instructions.replace("$ARGUMENTS", args)

    # Replace positional: $0, $1, … and $ARGUMENTS[0], $ARGUMENTS[1], …
    for i, part in enumerate(parts):
        result = result.replace(f"$ARGUMENTS[{i}]", part)
        result = result.replace(f"${i}", part)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Bulk loading
# ─────────────────────────────────────────────────────────────────────────────


def load_all_skills(skills_dir: Path) -> dict[str, SkillConfig]:
    """Recursively scan ``skills_dir`` for SKILL.md files."""
    skills: dict[str, SkillConfig] = {}

    if not skills_dir.exists():
        logger.info("Skills directory does not exist: %s", skills_dir)
        return skills

    for path in skills_dir.rglob("SKILL.md"):
        try:
            skill = load_skill(path)
            skills[skill.name] = skill
            logger.info("Loaded skill: %s  (%s)", skill.name, path)
        except Exception as exc:
            logger.error("Failed to load skill %s: %s", path, exc)

    return skills
