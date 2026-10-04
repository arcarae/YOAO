"""Scripted opponent strategies must produce the patterns listed in the paper's appendix."""

import pytest

from redblackbench.strategies import (
    STRATEGY_REGISTRY,
    TRAINING_DISTRIBUTION,
    create_scripted_team,
    get_strategy,
    strategy_patterns,
)

# Against an always-cooperating Team A, rounds 1-10 with multipliers at 5/8/10.
EXPECTED = {
    "always_cooperate": "AAAAAAAAAA",
    "always_defect": "BBBBBBBBBB",
    "tit_for_tat": "AAAAAAAAAA",
    "defect_critical": "AAAABAABAB",
    "early_exploiter": "ABBBAAAAAA",
    "early_exploiter_no_recovery": "ABBBBBBBBB",
    "mid_exploiter": "AAABBBAAAA",
    "late_exploiter": "AAAAAAABBB",
    "late_betrayer_recovery": "AAAAAAABBA",
    "critical_exploiter": "AAAABAABAB",
    "alternating": "ABABABABAB",
    "forgiver": "BBAAAAAAAA",
}


@pytest.mark.parametrize("strategy_id,expected", sorted(EXPECTED.items()))
def test_pattern_matches_paper(strategy_id, expected):
    assert get_strategy(strategy_id).pattern() == expected


def test_every_exploiter_actually_defects():
    """Regression: QUICK-recovery strategies used to recover on their first exploitation round."""
    for sid, pattern in strategy_patterns().items():
        if "exploit" in sid or "betray" in sid or sid == "always_defect":
            assert "B" in pattern, f"{sid} never defects: {pattern}"


def test_mostly_cooperate_is_stochastic_and_seeded():
    a = get_strategy("mostly_cooperate", seed=1).pattern(50)
    b = get_strategy("mostly_cooperate", seed=1).pattern(50)
    c = get_strategy("mostly_cooperate", seed=2).pattern(50)
    assert a == b
    assert a != c
    assert 0.6 <= a.count("A") / len(a) <= 0.95


def test_tit_for_tat_mirrors_team_a():
    s = get_strategy("tit_for_tat")
    assert s.get_move(1, 10, [], [], False, 1) == "A"
    assert s.get_move(2, 10, ["B"], ["A"], False, 1) == "B"
    assert s.get_move(3, 10, ["B", "A"], ["A", "B"], False, 1) == "A"


def test_conditional_recovery_depends_on_team_a():
    s = get_strategy("mid_exploiter_conditional")
    hist_coop = ["A"] * 6
    hist_defect = ["A"] * 4 + ["B", "B"]
    for r in range(1, 7):
        s.get_move(r, 10, hist_coop[: r - 1], [], r in (5,), 1)
    assert s.get_move(7, 10, hist_coop, [], False, 1) == "A"
    s.reset()
    for r in range(1, 7):
        s.get_move(r, 10, hist_defect[: r - 1], [], r in (5,), 1)
    assert s.get_move(7, 10, hist_defect, [], False, 1) == "B"


def test_alias_late_betrayer_resolves():
    team = create_scripted_team("late_betrayer")
    assert team.strategy_id == "late_exploiter"
    assert team.strategy.pattern() == EXPECTED["late_exploiter"]


def test_training_distribution_is_valid():
    assert abs(sum(TRAINING_DISTRIBUTION.values()) - 1.0) < 1e-9
    assert set(TRAINING_DISTRIBUTION) <= set(STRATEGY_REGISTRY)
    assert len(TRAINING_DISTRIBUTION) == 9
    assert "critical_exploiter" not in TRAINING_DISTRIBUTION


def test_reset_clears_window_state():
    s = get_strategy("early_exploiter_no_recovery")
    assert s.pattern() == "ABBBBBBBBB"
    assert s.pattern() == "ABBBBBBBBB"  # pattern() resets before and after
    s.get_move(3, 10, ["A", "A"], ["A", "B"], False, 1)
    s.reset()
    assert s.get_move(1, 10, [], [], False, 1) == "A"
