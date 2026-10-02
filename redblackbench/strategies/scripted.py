"""Scripted opponent strategies for the Red-Black Game.

Each strategy is a deterministic (or seeded) rule that maps the round number and the
opponent's history to "A" (cooperate / BLACK) or "B" (defect / RED). They are used as
Team B during SFT data generation and as the ``always_defect`` opponent in evaluation.

The round-by-round patterns below are the ones the paper's appendix lists (rounds 1-10,
against an always-cooperating Team A):

    tit_for_tat                  A + mirror
    always_defect                BBBBBBBBBB
    always_cooperate             AAAAAAAAAA
    mostly_cooperate             ~80% A, stochastic (seeded)
    early_exploiter              ABBBAAAAAA   defects in rounds 2-4, then cooperates
    mid_exploiter                AAABBBAAAA   defects in rounds 4-6
    late_exploiter               AAAAAAABBB   defects in rounds 8-10 (alias: late_betrayer)
    defect_critical              AAAABAABAB   defects only in the multiplier rounds 5, 8, 10
    early_exploiter_no_recovery  ABBBBBBBBB   cooperates once, then always defects

Design: an *exploitation window* says when the strategy defects; a *recovery pattern*
says what happens after the window closes (cooperate again, keep defecting, wait a few
rounds, or cooperate only if Team A has been cooperating). The recovery check runs
*after* the window, never inside it, so a strategy always defects for its whole window.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ExploitationTiming(Enum):
    """When exploitation happens (round numbers are 1-indexed)."""

    NONE = "none"          # never exploits
    EARLY = "early"        # rounds 2-4
    MID = "mid"            # rounds 4-6
    LATE = "late"          # rounds 8-10
    CRITICAL = "critical"  # the multiplier rounds 5, 8, 10


EXPLOITATION_ROUNDS: Dict[ExploitationTiming, List[int]] = {
    ExploitationTiming.NONE: [],
    ExploitationTiming.EARLY: [2, 3, 4],
    ExploitationTiming.MID: [4, 5, 6],
    ExploitationTiming.LATE: [8, 9, 10],
    ExploitationTiming.CRITICAL: [5, 8, 10],
}


class RecoveryPattern(Enum):
    """What the strategy does once its exploitation window has closed."""

    NONE = "none"                # keeps defecting for the rest of the game
    QUICK = "quick"              # cooperates again immediately after the window
    GRADUAL = "gradual"          # keeps defecting for two more rounds, then cooperates
    CONDITIONAL = "conditional"  # cooperates only if Team A cooperated in the last two rounds


GRADUAL_EXTRA_ROUNDS = 2


@dataclass
class StrategyConfig:
    """Configuration for a scripted strategy.

    Attributes:
        strategy_id: Unique identifier.
        name: Human-readable name.
        description: What this strategy tests.
        base_behavior: Default action when no rule applies ("A" cooperate, "B" defect).
        exploitation_timing: Rounds in which the strategy defects.
        recovery_pattern: Behaviour after the exploitation window closes.
        defect_on_critical: Defect in every round whose multiplier is above 1.
        tit_for_tat: Mirror Team A's previous move (base_behavior in round 1).
        defect_prob: Per-round probability of defecting (stochastic strategies).
        round_overrides: Explicit per-round moves that override everything else.
    """

    strategy_id: str
    name: str
    description: str
    base_behavior: str = "A"
    exploitation_timing: ExploitationTiming = ExploitationTiming.NONE
    recovery_pattern: RecoveryPattern = RecoveryPattern.NONE
    defect_on_critical: bool = False
    tit_for_tat: bool = False
    defect_prob: float = 0.0
    round_overrides: Dict[int, str] = field(default_factory=dict)

    @property
    def exploitation_rounds(self) -> List[int]:
        return EXPLOITATION_ROUNDS[self.exploitation_timing]


class ScriptedStrategy:
    """Generates opponent moves from a :class:`StrategyConfig`."""

    def __init__(self, config: StrategyConfig, seed: Optional[int] = None):
        self.config = config
        self._rng = random.Random(seed)
        self._window_closed_at: Optional[int] = None  # last round of the window, once it has started
        self.reset()

    def reset(self) -> None:
        """Reset per-game state (call between games; keeps the RNG stream)."""
        self._window_closed_at = None

    def get_move(
        self,
        round_number: int,
        total_rounds: int,
        team_a_history: List[str],
        team_b_history: List[str],
        is_critical: bool,
        multiplier: int,
    ) -> str:
        """Return "A" or "B" for this round.

        Args:
            round_number: Current round (1-indexed).
            total_rounds: Rounds in the game.
            team_a_history: Team A's past moves ("A"/"B"), oldest first.
            team_b_history: This strategy's past moves.
            is_critical: Whether this round has a multiplier above 1.
            multiplier: The round multiplier.
        """
        cfg = self.config

        if round_number in cfg.round_overrides:
            return cfg.round_overrides[round_number]

        if cfg.tit_for_tat:
            return team_a_history[-1] if team_a_history else cfg.base_behavior

        if cfg.defect_on_critical and is_critical:
            return "B"

        if cfg.defect_prob > 0 and self._rng.random() < cfg.defect_prob:
            return "B"

        window = cfg.exploitation_rounds
        if round_number in window:
            self._window_closed_at = max(window)
            return "B"

        if self._window_closed_at is not None and round_number > self._window_closed_at:
            return self._after_window(round_number, team_a_history)

        if cfg.base_behavior == "B" and cfg.recovery_pattern == RecoveryPattern.CONDITIONAL:
            # "forgiver": starts defecting, cooperates once Team A has cooperated twice in a row
            return "A" if self._team_a_cooperated_recently(team_a_history) else "B"

        return cfg.base_behavior

    def _after_window(self, round_number: int, team_a_history: List[str]) -> str:
        pattern = self.config.recovery_pattern
        if pattern == RecoveryPattern.NONE:
            return "B"
        if pattern == RecoveryPattern.QUICK:
            return "A"
        if pattern == RecoveryPattern.GRADUAL:
            rounds_after = round_number - self._window_closed_at
            return "B" if rounds_after <= GRADUAL_EXTRA_ROUNDS else "A"
        if pattern == RecoveryPattern.CONDITIONAL:
            return "A" if self._team_a_cooperated_recently(team_a_history) else "B"
        return self.config.base_behavior

    @staticmethod
    def _team_a_cooperated_recently(team_a_history: List[str]) -> bool:
        return len(team_a_history) >= 2 and all(m == "A" for m in team_a_history[-2:])

    def pattern(self, total_rounds: int = 10, multipliers: Optional[Dict[int, int]] = None) -> str:
        """Render the move pattern against an always-cooperating Team A (for docs and tests)."""
        multipliers = multipliers or {5: 3, 8: 5, 10: 10}
        self.reset()
        a_hist: List[str] = []
        moves: List[str] = []
        for r in range(1, total_rounds + 1):
            m = multipliers.get(r, 1)
            moves.append(self.get_move(r, total_rounds, list(a_hist), list(moves), m > 1, m))
            a_hist.append("A")
        self.reset()
        return "".join(moves)


# ============================================================================
# Registry
# ============================================================================

STRATEGY_REGISTRY: Dict[str, StrategyConfig] = {
    # --- cooperative -------------------------------------------------------
    "always_cooperate": StrategyConfig(
        strategy_id="always_cooperate",
        name="Always Cooperate",
        description="Always chooses A. Tests baseline cooperative behavior.",
        base_behavior="A",
    ),
    "mostly_cooperate": StrategyConfig(
        strategy_id="mostly_cooperate",
        name="Mostly Cooperate",
        description="Cooperates ~80% of the time; each round defects with probability 0.2 (seeded).",
        base_behavior="A",
        defect_prob=0.2,
    ),
    # --- defection ---------------------------------------------------------
    "always_defect": StrategyConfig(
        strategy_id="always_defect",
        name="Always Defect",
        description="Always chooses B. Tests maintaining cooperation under constant exploitation.",
        base_behavior="B",
    ),
    "defect_critical": StrategyConfig(
        strategy_id="defect_critical",
        name="Defect on Critical",
        description="Cooperates, but defects in the multiplier rounds (5, 8, 10).",
        base_behavior="A",
        defect_on_critical=True,
    ),
    # --- tit for tat -------------------------------------------------------
    "tit_for_tat": StrategyConfig(
        strategy_id="tit_for_tat",
        name="Tit-for-Tat",
        description="Cooperates in round 1, then mirrors Team A's previous move.",
        base_behavior="A",
        tit_for_tat=True,
    ),
    # --- early exploitation ------------------------------------------------
    "early_exploiter": StrategyConfig(
        strategy_id="early_exploiter",
        name="Early Exploiter",
        description="Cooperates in round 1, defects in rounds 2-4, then cooperates again (ABBBAAAAAA).",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.EARLY,
        recovery_pattern=RecoveryPattern.QUICK,
    ),
    "early_exploiter_no_recovery": StrategyConfig(
        strategy_id="early_exploiter_no_recovery",
        name="Early Exploiter (No Recovery)",
        description="Cooperates in round 1, then defects for the rest of the game (ABBBBBBBBB).",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.EARLY,
        recovery_pattern=RecoveryPattern.NONE,
    ),
    # --- mid-game exploitation ---------------------------------------------
    "mid_exploiter": StrategyConfig(
        strategy_id="mid_exploiter",
        name="Mid-Game Exploiter",
        description="Cooperates, defects in rounds 4-6, then cooperates again (AAABBBAAAA).",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.MID,
        recovery_pattern=RecoveryPattern.QUICK,
    ),
    "mid_exploiter_conditional": StrategyConfig(
        strategy_id="mid_exploiter_conditional",
        name="Mid-Game Exploiter (Conditional)",
        description="Defects in rounds 4-6, then cooperates only while Team A keeps cooperating.",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.MID,
        recovery_pattern=RecoveryPattern.CONDITIONAL,
    ),
    # --- late betrayal -----------------------------------------------------
    "late_exploiter": StrategyConfig(
        strategy_id="late_exploiter",
        name="Late Exploiter",
        description="Cooperates for seven rounds, then defects in rounds 8-10 (AAAAAAABBB).",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.LATE,
        recovery_pattern=RecoveryPattern.NONE,
    ),
    "late_betrayer_recovery": StrategyConfig(
        strategy_id="late_betrayer_recovery",
        name="Late Betrayer (With Recovery)",
        description="Defects in rounds 8-9 and returns to cooperation for the final round (AAAAAAABBA).",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.LATE,
        recovery_pattern=RecoveryPattern.NONE,
        round_overrides={10: "A"},
    ),
    # --- critical-round exploitation ---------------------------------------
    "critical_exploiter": StrategyConfig(
        strategy_id="critical_exploiter",
        name="Critical Round Exploiter",
        description="Defects only in the high-multiplier rounds 5, 8 and 10 (AAAABAABAB).",
        base_behavior="A",
        exploitation_timing=ExploitationTiming.CRITICAL,
        recovery_pattern=RecoveryPattern.QUICK,
    ),
    # --- mixed -------------------------------------------------------------
    "alternating": StrategyConfig(
        strategy_id="alternating",
        name="Alternating",
        description="Alternates A and B each round (ABABABABAB).",
        base_behavior="A",
        round_overrides={2: "B", 4: "B", 6: "B", 8: "B", 10: "B"},
    ),
    "forgiver": StrategyConfig(
        strategy_id="forgiver",
        name="Forgiver",
        description="Starts defecting; cooperates once Team A has cooperated two rounds in a row.",
        base_behavior="B",
        recovery_pattern=RecoveryPattern.CONDITIONAL,
    ),
}

# Backwards-compatible alias used by older scripts and results.
STRATEGY_ALIASES: Dict[str, str] = {
    "late_betrayer": "late_exploiter",
}


def resolve_strategy_id(strategy_id: str) -> str:
    return STRATEGY_ALIASES.get(strategy_id, strategy_id)


def get_strategy(strategy_id: str, seed: Optional[int] = None) -> Optional[ScriptedStrategy]:
    """Instantiate a strategy by id (aliases accepted). Returns None if unknown."""
    config = STRATEGY_REGISTRY.get(resolve_strategy_id(strategy_id))
    if config is None:
        return None
    return ScriptedStrategy(config, seed=seed)


def list_strategies() -> Dict[str, str]:
    """Map strategy_id -> description."""
    return {sid: cfg.description for sid, cfg in STRATEGY_REGISTRY.items()}


def strategy_patterns(total_rounds: int = 10) -> Dict[str, str]:
    """Map strategy_id -> move pattern against an always-cooperating Team A (seed 0 for stochastic ones)."""
    return {sid: ScriptedStrategy(cfg, seed=0).pattern(total_rounds) for sid, cfg in STRATEGY_REGISTRY.items()}


# Strategy mixture used when generating SFT data: the nine opponents of the paper's
# appendix table, sampled uniformly.
TRAINING_DISTRIBUTION: Dict[str, float] = {
    "tit_for_tat": 1 / 9,
    "always_defect": 1 / 9,
    "always_cooperate": 1 / 9,
    "mostly_cooperate": 1 / 9,
    "early_exploiter": 1 / 9,
    "mid_exploiter": 1 / 9,
    "late_exploiter": 1 / 9,
    "defect_critical": 1 / 9,
    "early_exploiter_no_recovery": 1 / 9,
}


def sample_strategy_for_training(rng: Optional[random.Random] = None) -> str:
    """Sample a strategy id from TRAINING_DISTRIBUTION."""
    rng = rng or random
    strategies = list(TRAINING_DISTRIBUTION.keys())
    weights = list(TRAINING_DISTRIBUTION.values())
    return rng.choices(strategies, weights=weights, k=1)[0]
