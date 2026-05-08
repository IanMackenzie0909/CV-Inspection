from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and fail clearly when the file shape is invalid."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def normalize_token(value: str) -> str:
    """Normalize labels and aliases before matching them to part keys."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


class PartsLexicon:
    """Alias resolver for converting model labels into canonical part keys."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.alias_to_key: dict[str, str] = {}
        # The lexicon allows GroundingDINO phrases such as "flex spline" to map
        # to the same internal part key as "flexspline".
        for part_key, part_info in payload.items():
            self.alias_to_key[normalize_token(part_key)] = part_key
            for alias in part_info.get("aliases_en", []) or []:
                self.alias_to_key[normalize_token(str(alias))] = part_key
            for alias in part_info.get("aliases_tl", []) or []:
                self.alias_to_key[normalize_token(str(alias))] = part_key

    def part_key_for_label(self, label: str) -> str:
        """Return a canonical part key for a model label or prompt phrase."""
        normalized = normalize_token(label)
        if normalized in self.alias_to_key:
            return self.alias_to_key[normalized]

        # GroundingDINO can return a phrase rather than a single class name, so
        # subset matching catches labels like "bearing ring".
        label_tokens = set(normalized.split())
        for alias, part_key in self.alias_to_key.items():
            alias_tokens = set(alias.split())
            if alias_tokens and alias_tokens.issubset(label_tokens):
                return part_key
        return normalized.replace(" ", "_") or "unknown"

    def is_critical(self, part_key: str) -> bool:
        """Tell the step checker whether a missing part should be high risk."""
        return self.payload.get(part_key, {}).get("criticality") == "high"


def expected_parts_for_step(
    workflow_config: dict[str, Any],
    step_id: str,
    text_prompt: str,
    lexicon: PartsLexicon,
) -> list[dict[str, Any]]:
    """Resolve the expected parts for a step from SOP config or text prompt."""
    step_config = (workflow_config.get("steps") or {}).get(step_id)
    if step_config and step_config.get("expected_parts"):
        return [dict(item) for item in step_config["expected_parts"]]

    default_step = workflow_config.get("default_step") or {}
    if not default_step.get("required_parts_from_text_prompt", True):
        return []

    parts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_piece in re.split(r"[.,;|/]+", text_prompt):
        # The prompt fallback is useful while a complete SOP is not available.
        # Once SOPs are defined, explicit workflow_steps.yaml rules take over.
        part_key = lexicon.part_key_for_label(raw_piece)
        if not part_key or part_key == "unknown" or part_key in seen:
            continue
        seen.add(part_key)
        parts.append(
            {
                "part": part_key,
                "required": True,
                "roi": default_step.get("default_roi", "full_image"),
            }
        )
    return parts


def model_info(model_registry: dict[str, Any], key: str, fallback_name: str) -> dict[str, str]:
    """Return model name/version metadata for audit output."""
    model = (model_registry.get("models") or {}).get(key, {})
    return {
        "name": str(model.get("name", fallback_name)),
        "version": str(model.get("version", "unknown")),
    }
