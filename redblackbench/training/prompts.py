"""Prompt registry for RedBlackBench training data.

Externalizes prompts from trajectories for storage efficiency and versioning.
Prompts are stored in a separate prompts.json file with hashes for integrity.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


@dataclass
class PromptEntry:
    """A single prompt entry in the registry.

    Attributes:
        prompt_id: Unique identifier (e.g., "pandemic_system.v3")
        text: The full prompt text
        sha256: Hash of the prompt text for integrity verification
        scenario_id: Associated scenario (if any)
        prompt_type: Type of prompt (system, deliberation, voting, diplomacy)
        version: Version string
        created_at: When this prompt was added
        metadata: Additional metadata
    """
    prompt_id: str
    text: str
    sha256: str
    scenario_id: Optional[str] = None
    prompt_type: str = "system"  # system, deliberation, voting, diplomacy
    version: str = "v1"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "text": self.text,
            "sha256": self.sha256,
            "scenario_id": self.scenario_id,
            "prompt_type": self.prompt_type,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptEntry":
        return cls(
            prompt_id=data["prompt_id"],
            text=data["text"],
            sha256=data["sha256"],
            scenario_id=data.get("scenario_id"),
            prompt_type=data.get("prompt_type", "system"),
            version=data.get("version", "v1"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


class PromptRegistry:
    """Registry for managing prompts used in training data.

    Stores prompts externally so trajectories only need to reference prompt IDs.
    Provides integrity verification via SHA256 hashes.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the prompt registry.

        Args:
            registry_path: Path to the prompts.json file
        """
        self.registry_path = Path(registry_path) if registry_path else None
        self.prompts: Dict[str, PromptEntry] = {}
        self._renderer_version = "rbbench.v1"

        if self.registry_path and self.registry_path.exists():
            self.load()

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA256 hash of prompt text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def register(
        self,
        prompt_id: str,
        text: str,
        scenario_id: Optional[str] = None,
        prompt_type: str = "system",
        version: str = "v1",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptEntry:
        """Register a new prompt or update existing.

        Args:
            prompt_id: Unique identifier for this prompt
            text: The full prompt text
            scenario_id: Associated scenario
            prompt_type: Type of prompt
            version: Version string
            metadata: Additional metadata

        Returns:
            The created/updated PromptEntry
        """
        sha256 = self.compute_hash(text)

        # Check if already exists with same hash
        if prompt_id in self.prompts:
            existing = self.prompts[prompt_id]
            if existing.sha256 == sha256:
                return existing  # No change needed
            # Different content - create new version
            version = self._increment_version(existing.version)

        entry = PromptEntry(
            prompt_id=prompt_id,
            text=text,
            sha256=sha256,
            scenario_id=scenario_id,
            prompt_type=prompt_type,
            version=version,
            metadata=metadata or {},
        )

        self.prompts[prompt_id] = entry
        return entry

    def _increment_version(self, version: str) -> str:
        """Increment version string (v1 -> v2, etc.)."""
        if version.startswith("v") and version[1:].isdigit():
            return f"v{int(version[1:]) + 1}"
        return f"{version}.1"

    def get(self, prompt_id: str) -> Optional[PromptEntry]:
        """Get a prompt by ID."""
        return self.prompts.get(prompt_id)

    def get_text(self, prompt_id: str) -> Optional[str]:
        """Get just the prompt text by ID."""
        entry = self.get(prompt_id)
        return entry.text if entry else None

    def verify(self, prompt_id: str, expected_hash: str) -> bool:
        """Verify a prompt's integrity.

        Args:
            prompt_id: Prompt to verify
            expected_hash: Expected SHA256 hash

        Returns:
            True if hash matches, False otherwise
        """
        entry = self.get(prompt_id)
        if not entry:
            return False
        return entry.sha256 == expected_hash

    def get_prompt_ids(self, scenario_id: str) -> Dict[str, str]:
        """Get all prompt IDs for a scenario.

        Args:
            scenario_id: Scenario to get prompts for

        Returns:
            Dict mapping prompt_type to prompt_id
        """
        result = {}
        for prompt_id, entry in self.prompts.items():
            if entry.scenario_id == scenario_id:
                result[entry.prompt_type] = prompt_id
        return result

    def register_scenario_prompts(
        self,
        scenario_id: str,
        system_prompt: str,
        deliberation_prompt: Optional[str] = None,
        voting_prompt: Optional[str] = None,
        diplomacy_prompt: Optional[str] = None,
        version: str = "v1",
    ) -> Dict[str, str]:
        """Register all prompts for a scenario at once.

        Args:
            scenario_id: Scenario identifier
            system_prompt: System prompt text
            deliberation_prompt: Deliberation prompt template
            voting_prompt: Voting prompt template
            diplomacy_prompt: Diplomacy prompt template
            version: Version string

        Returns:
            Dict mapping prompt_type to prompt_id
        """
        result = {}

        # System prompt
        prompt_id = f"{scenario_id}_system.{version}"
        self.register(prompt_id, system_prompt, scenario_id, "system", version)
        result["system"] = prompt_id

        # Deliberation prompt
        if deliberation_prompt:
            prompt_id = f"{scenario_id}_deliberation.{version}"
            self.register(prompt_id, deliberation_prompt, scenario_id, "deliberation", version)
            result["deliberation"] = prompt_id

        # Voting prompt
        if voting_prompt:
            prompt_id = f"{scenario_id}_voting.{version}"
            self.register(prompt_id, voting_prompt, scenario_id, "voting", version)
            result["voting"] = prompt_id

        # Diplomacy prompt
        if diplomacy_prompt:
            prompt_id = f"{scenario_id}_diplomacy.{version}"
            self.register(prompt_id, diplomacy_prompt, scenario_id, "diplomacy", version)
            result["diplomacy"] = prompt_id

        return result

    def to_dict(self) -> dict:
        """Export registry as dictionary."""
        return {
            "schema_version": "rbbench.prompts.v1",
            "renderer_version": self._renderer_version,
            "prompts": {k: v.to_dict() for k, v in self.prompts.items()},
            "metadata": {
                "total_prompts": len(self.prompts),
                "scenarios": list(set(
                    e.scenario_id for e in self.prompts.values() if e.scenario_id
                )),
                "last_updated": datetime.now().isoformat(),
            },
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save registry to JSON file.

        Args:
            path: Path to save to (uses registry_path if not provided)
        """
        save_path = Path(path) if path else self.registry_path
        if not save_path:
            raise ValueError("No path provided and no registry_path set")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """Load registry from JSON file.

        Args:
            path: Path to load from (uses registry_path if not provided)
        """
        load_path = Path(path) if path else self.registry_path
        if not load_path:
            raise ValueError("No path provided and no registry_path set")

        if not load_path.exists():
            return  # Empty registry

        with open(load_path, 'r') as f:
            data = json.load(f)

        self._renderer_version = data.get("renderer_version", "rbbench.v1")
        self.prompts = {}
        for prompt_id, entry_data in data.get("prompts", {}).items():
            self.prompts[prompt_id] = PromptEntry.from_dict(entry_data)

    def __len__(self) -> int:
        return len(self.prompts)

    def __contains__(self, prompt_id: str) -> bool:
        return prompt_id in self.prompts


def create_registry_from_scenarios(output_path: str) -> PromptRegistry:
    """Create a prompt registry from all registered scenarios.

    Args:
        output_path: Path to save the prompts.json file

    Returns:
        The populated PromptRegistry
    """
    from redblackbench.scenarios import SCENARIOS

    registry = PromptRegistry(output_path)

    for scenario_id, scenario in SCENARIOS.items():
        # Register system prompt
        if scenario.system_prompt_template:
            registry.register(
                prompt_id=f"{scenario_id}_system.v1",
                text=scenario.system_prompt_template,
                scenario_id=scenario_id,
                prompt_type="system",
                version="v1",
            )

        # Register round context / deliberation prompt
        if scenario.round_context_template:
            registry.register(
                prompt_id=f"{scenario_id}_deliberation.v1",
                text=scenario.round_context_template,
                scenario_id=scenario_id,
                prompt_type="deliberation",
                version="v1",
            )

    registry.save()
    return registry
