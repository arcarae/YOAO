"""Scripted opponent strategies for diverse training data.

Creates deterministic opponent behaviors to ensure the model sees:
- Cooperative opponents (easy cases)
- Exploitative opponents (hard cases)
- Mixed strategies (realistic cases)
- Various timing patterns (early/mid/late exploitation)
- Recovery patterns (forgiveness testing)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from enum import Enum


class ExploitationTiming(Enum):
    """When exploitation begins."""
    NONE = "none"           # Never exploits
    EARLY = "early"         # Rounds 2-4
    MID = "mid"             # Rounds 5-6
    LATE = "late"           # Rounds 7-9
    CRITICAL = "critical"   # Only on critical rounds (5, 8, 10)


class RecoveryPattern(Enum):
    """Whether opponent returns to cooperation after exploiting."""
    NONE = "none"           # Never recovers (keeps defecting)
    QUICK = "quick"         # Returns after 1 round
    GRADUAL = "gradual"     # Returns after 2-3 rounds
    CONDITIONAL = "conditional"  # Returns only if Team A cooperates


@dataclass
class StrategyConfig:
    """Configuration for a scripted strategy.

    Attributes:
        strategy_id: Unique identifier
        name: Human-readable name
        description: What this strategy tests
        base_behavior: Default action when no special rule applies
        exploitation_timing: When to start exploiting
        recovery_pattern: How/if to return to cooperation
        defect_on_critical: Always defect on critical rounds
        tit_for_tat: Mirror opponent's last move
        forgiveness_prob: Probability of forgiving after exploitation (0-1)
    """
    strategy_id: str
    name: str
    description: str
    base_behavior: str = "A"  # "A" = cooperate, "B" = defect
    exploitation_timing: ExploitationTiming = ExploitationTiming.NONE
    recovery_pattern: RecoveryPattern = RecoveryPattern.NONE
    defect_on_critical: bool = False
    tit_for_tat: bool = False
    forgiveness_prob: float = 0.0

    # Specific round overrides: {round_number: "A" or "B"}
    round_overrides: Dict[int, str] = None

    def __post_init__(self):
        if self.round_overrides is None:
            self.round_overrides = {}


class ScriptedStrategy:
    """Generates deterministic opponent moves based on strategy config."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._exploitation_started = False
        self._rounds_since_exploitation = 0
        self._recovered = False

    def get_move(
        self,
        round_number: int,
        total_rounds: int,
        team_a_history: List[str],
        team_b_history: List[str],
        is_critical: bool,
        multiplier: int,
    ) -> str:
        """Get the scripted move for this round.

        Args:
            round_number: Current round (1-indexed)
            total_rounds: Total rounds in game
            team_a_history: List of Team A's past moves
            team_b_history: List of Team B's past moves (this strategy)
            is_critical: Whether this is a critical round
            multiplier: Round multiplier

        Returns:
            "A" or "B"
        """
        # Check for explicit round override
        if round_number in self.config.round_overrides:
            return self.config.round_overrides[round_number]

        # Tit-for-tat: mirror opponent's last move
        if self.config.tit_for_tat and team_a_history:
            return team_a_history[-1]

        # Critical round defection
        if self.config.defect_on_critical and is_critical:
            return "B"

        # Check exploitation timing
        if self._should_exploit(round_number, total_rounds):
            if not self._exploitation_started:
                self._exploitation_started = True
                self._rounds_since_exploitation = 0

            # Check recovery
            if self._exploitation_started:
                self._rounds_since_exploitation += 1

                if self._should_recover(team_a_history):
                    self._recovered = True
                    return "A"

                return "B"

        # Default behavior
        return self.config.base_behavior

    def _should_exploit(self, round_number: int, total_rounds: int) -> bool:
        """Check if exploitation should happen this round."""
        timing = self.config.exploitation_timing

        if timing == ExploitationTiming.NONE:
            return False
        elif timing == ExploitationTiming.EARLY:
            return 2 <= round_number <= 4
        elif timing == ExploitationTiming.MID:
            return 5 <= round_number <= 6
        elif timing == ExploitationTiming.LATE:
            return 7 <= round_number <= 9
        elif timing == ExploitationTiming.CRITICAL:
            # Critical rounds are typically 5, 8, 10
            return round_number in [5, 8, 10]

        return False

    def _should_recover(self, team_a_history: List[str]) -> bool:
        """Check if should return to cooperation."""
        if self._recovered:
            return True  # Already recovered

        pattern = self.config.recovery_pattern

        if pattern == RecoveryPattern.NONE:
            return False
        elif pattern == RecoveryPattern.QUICK:
            return self._rounds_since_exploitation >= 1
        elif pattern == RecoveryPattern.GRADUAL:
            return self._rounds_since_exploitation >= 3
        elif pattern == RecoveryPattern.CONDITIONAL:
            # Return to cooperation only if Team A kept cooperating
            if team_a_history and len(team_a_history) >= 2:
                return all(m == "A" for m in team_a_history[-2:])
            return False

        return False

    def reset(self):
        """Reset state for a new game."""
        self._exploitation_started = False
        self._rounds_since_exploitation = 0
        self._recovered = False


# ============================================================================
# Pre-defined strategies for training data diversity
# ============================================================================

STRATEGY_REGISTRY: Dict[str, StrategyConfig] = {
    # === COOPERATIVE STRATEGIES (Easy cases) ===
    "always_cooperate": StrategyConfig(
        strategy_id="always_cooperate",
        name="Always Cooperate",
        description="Always chooses A. Tests baseline cooperative behavior.",
        base_behavior="A",
    ),

    "mostly_cooperate": StrategyConfig(
        strategy_id="mostly_cooperate",
        name="Mostly Cooperate",
        description="Cooperates except on final critical round.",
        base_behavior="A",
        round_overrides={10: "B"},
    ),

    # === DEFECTION STRATEGIES (Hard cases) ===
    "always_defect": StrategyConfig(
        strategy_id="always_defect",
        name="Always Defect",
        description="Always chooses B. Tests maintaining cooperation under constant exploitation.",
        base_behavior="B",
    ),

    "defect_critical": StrategyConfig(
        strategy_id="defect_critical",
        name="Defect on Critical",
        description="Cooperates normally but defects on critical rounds (5, 8, 10).",
        base_behavior="A",
        defect_on_critical=True,
    ),

    # === TIT-FOR-TAT STRATEGIES ===
    "tit_for_tat": StrategyConfig(
        strategy_id="tit_for_tat",
        name="Tit-for-Tat",
        description="Mirrors Team A's previous move. Classic game theory strategy.",
        base_behavior="A",  # Cooperate on first round
        tit_for_tat=True,
    ),

    # === EARLY EXPLOITATION ===
    "early_exploiter": StrategyConfig(
        strategy_id="early_exploiter",
        name="Early Exploiter",
        description="Cooperates round 1, defects rounds 2-4, then cooperates again.",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.EARLY,
        recovery_pattern=RecoveryPattern.QUICK,
    ),

    "early_exploiter_no_recovery": StrategyConfig(
        strategy_id="early_exploiter_no_recovery",
        name="Early Exploiter (No Recovery)",
        description="Cooperates round 1, defects from round 2 onwards. Never returns.",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.EARLY,
        recovery_pattern=RecoveryPattern.NONE,
    ),

    # === MID-GAME EXPLOITATION ===
    "mid_exploiter": StrategyConfig(
        strategy_id="mid_exploiter",
        name="Mid-Game Exploiter",
        description="Cooperates until round 5, defects 5-6, then recovers.",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.MID,
        recovery_pattern=RecoveryPattern.GRADUAL,
    ),

    "mid_exploiter_conditional": StrategyConfig(
        strategy_id="mid_exploiter_conditional",
        name="Mid-Game Exploiter (Conditional)",
        description="Defects mid-game, only recovers if Team A keeps cooperating.",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.MID,
        recovery_pattern=RecoveryPattern.CONDITIONAL,
    ),

    # === LATE BETRAYAL (Hardest cases) ===
    "late_betrayer": StrategyConfig(
        strategy_id="late_betrayer",
        name="Late Betrayer",
        description="Cooperates until round 7, then defects. Tests trust after long cooperation.",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.LATE,
        recovery_pattern=RecoveryPattern.NONE,
    ),

    "late_betrayer_recovery": StrategyConfig(
        strategy_id="late_betrayer_recovery",
        name="Late Betrayer (With Recovery)",
        description="Betrays late but returns to cooperation for final round.",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.LATE,
        recovery_pattern=RecoveryPattern.QUICK,
    ),

    # === CRITICAL ROUND EXPLOITATION ===
    "critical_exploiter": StrategyConfig(
        strategy_id="critical_exploiter",
        name="Critical Round Exploiter",
        description="Only defects on high-multiplier critical rounds (5, 8, 10).",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.CRITICAL,
        recovery_pattern=RecoveryPattern.QUICK,
    ),

    # === MIXED/UNPREDICTABLE ===
    "alternating": StrategyConfig(
        strategy_id="alternating",
        name="Alternating",
        description="Alternates between A and B each round.",
        base_behavior="A",
        round_overrides={2: "B", 4: "B", 6: "B", 8: "B", 10: "B"},
    ),

    "forgiver": StrategyConfig(
        strategy_id="forgiver",
        name="Forgiver",
        description="Starts defecting but recovers if Team A consistently cooperates.",
        base_behavior="B",
        recovery_pattern=RecoveryPattern.CONDITIONAL,
        exploitation_timing=ExploitationTiming.EARLY,
    ),
}


def get_strategy(strategy_id: str) -> Optional[ScriptedStrategy]:
    """Get a strategy by ID.

    Args:
        strategy_id: Strategy identifier

    Returns:
        ScriptedStrategy instance, or None if not found
    """
    config = STRATEGY_REGISTRY.get(strategy_id)
    if config:
        return ScriptedStrategy(config)
    return None


def list_strategies() -> Dict[str, str]:
    """List all available strategies.

    Returns:
        Dict mapping strategy_id to description
    """
    return {
        sid: config.description
        for sid, config in STRATEGY_REGISTRY.items()
    }


# Strategy distribution for balanced training data
TRAINING_DISTRIBUTION = {
    # 20% easy (cooperative)
    "always_cooperate": 0.10,
    "mostly_cooperate": 0.10,

    # 30% medium (mixed)
    "tit_for_tat": 0.10,
    "early_exploiter": 0.10,
    "mid_exploiter": 0.10,

    # 30% hard (exploitation)
    "late_betrayer": 0.10,
    "critical_exploiter": 0.10,
    "defect_critical": 0.10,

    # 20% very hard (sustained exploitation)
    "always_defect": 0.10,
    "early_exploiter_no_recovery": 0.10,
}


def sample_strategy_for_training() -> str:
    """Sample a strategy ID based on training distribution.

    Returns:
        strategy_id sampled according to TRAINING_DISTRIBUTION
    """
    import random
    strategies = list(TRAINING_DISTRIBUTION.keys())
    weights = list(TRAINING_DISTRIBUTION.values())
    return random.choices(strategies, weights=weights, k=1)[0]
