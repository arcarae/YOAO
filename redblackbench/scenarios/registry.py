"""Registry of all available scenarios for RedBlackBench."""

from typing import Dict, Optional

from redblackbench.scenarios.base import Scenario
from redblackbench.scenarios.pandemic import create_pandemic_scenario
from redblackbench.scenarios.climate import create_climate_scenario
from redblackbench.scenarios.agi_safety import create_agi_safety_scenario
from redblackbench.scenarios.standards_coordination import create_standards_scenario
from redblackbench.scenarios.election_crisis import create_election_crisis_scenario
from redblackbench.scenarios.trade_war import create_trade_war_scenario
from redblackbench.scenarios.gpu_contention import create_gpu_contention_scenario


def _build_scenarios() -> Dict[str, Scenario]:
    """Build and return all available scenarios."""
    scenarios = {}

    # Pandemic vaccine sharing
    pandemic = create_pandemic_scenario()
    scenarios[pandemic.config.scenario_id] = pandemic

    # Climate disaster cooperation
    climate = create_climate_scenario()
    scenarios[climate.config.scenario_id] = climate

    # AGI safety research sharing
    agi_safety = create_agi_safety_scenario()
    scenarios[agi_safety.config.scenario_id] = agi_safety

    # Software standards coordination (neutral scenario)
    standards = create_standards_scenario()
    scenarios[standards.config.scenario_id] = standards

    # Election year crisis (hard adversarial scenario)
    election = create_election_crisis_scenario()
    scenarios[election.config.scenario_id] = election

    # === HELD-OUT SCENARIOS (for testing generalization) ===

    # Trade war (held-out, high adversarial pressure)
    trade_war = create_trade_war_scenario()
    scenarios[trade_war.config.scenario_id] = trade_war

    # GPU compute contention (held-out, medium-high competitive pressure)
    gpu_contention = create_gpu_contention_scenario()
    scenarios[gpu_contention.config.scenario_id] = gpu_contention

    return scenarios


# Global registry of all scenarios
SCENARIOS: Dict[str, Scenario] = _build_scenarios()


def get_scenario(scenario_id: str) -> Optional[Scenario]:
    """Get a scenario by its ID.

    Args:
        scenario_id: The scenario identifier (e.g., "pandemic_vaccines")

    Returns:
        The Scenario object, or None if not found
    """
    return SCENARIOS.get(scenario_id)


def list_scenarios() -> Dict[str, str]:
    """List all available scenarios with their names.

    Returns:
        Dictionary mapping scenario_id to human-readable name
    """
    return {
        scenario_id: scenario.config.name
        for scenario_id, scenario in SCENARIOS.items()
    }
