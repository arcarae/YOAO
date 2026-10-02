#!/usr/bin/env python3
"""Compute the paper's Sugarscape metrics from one or more experiment directories.

For each ``results/sugarscape/<experiment>/experiment_<timestamp>/`` it reports:

- trade success rate      = completed trades / all trade attempts (``debug/trade_history.csv``)
- trade success by pair   = the same split by the two agents' origin identities (A = altruist,
                            N = normie/survivor, E = exploiter) and by 20-tick windows
- survival rate           = old-age deaths / (old-age + starvation deaths); also the share of the
                            initial population that starved
- mean lifespan           = mean ``lifetime_ticks`` of dead agents (``debug/death_records.csv``),
                            with survivors' ages reported separately
- wealth at death         = mean sugar + spice held when an agent died
- identity shift          = mean(final leaning - initial leaning) per origin group; the final
                            leaning is the last ``identity_leaning_after`` in ``debug/reflections.csv``
                            (so dead agents count), falling back to ``final_state.json``
- belief shift            = change in the 1-5 quantified beliefs (cooperation, trust,
                            self-interest) per origin group, from each agent's ``baseline_snapshot``
                            and final ``belief_ledger``, when present

Usage:
    python scripts/analyze_sugarscape.py results/sugarscape/goal_survival_normie/experiment_*/
    python scripts/analyze_sugarscape.py <dir> --normie-only --json out.json
"""

import argparse
import ast
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

IDENTITY_SHORT = {"altruist": "A", "survivor": "N", "normie": "N", "exploiter": "E"}
WINDOW = 20


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _maybe_literal(x):
    """final_state.json stores some nested fields as Python reprs; parse them when possible."""
    if isinstance(x, (dict, list)) or x is None:
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def load_agents(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {str(a["agent_id"]): a for a in data.get("agents", [])}


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def analyze(exp_dir: Path, normie_only: bool = False) -> dict:
    exp_dir = Path(exp_dir)
    initial = load_agents(exp_dir / "initial_state.json")
    final = load_agents(exp_dir / "final_state.json")
    trades = read_csv(exp_dir / "debug" / "trade_history.csv")
    deaths = read_csv(exp_dir / "debug" / "death_records.csv")
    reflections = read_csv(exp_dir / "debug" / "reflections.csv")
    config = json.load(open(exp_dir / "config.json")) if (exp_dir / "config.json").exists() else {}

    origin = {aid: a.get("origin_identity", "survivor") for aid, a in initial.items()}
    origin.update({aid: a.get("origin_identity", origin.get(aid, "survivor")) for aid, a in final.items()})
    short = {aid: IDENTITY_SHORT.get(str(o).lower(), "?") for aid, o in origin.items()}

    def keep(aid: str) -> bool:
        return (not normie_only) or short.get(aid) == "N"

    # --- trade success -------------------------------------------------------------------
    attempts = [t for t in trades if t.get("outcome")]
    completed = [t for t in attempts if t["outcome"].strip().lower() == "completed"]
    pair_tot: Counter = Counter()
    pair_ok: Counter = Counter()
    win_tot: Counter = Counter()
    win_ok: Counter = Counter()
    normie_tot = normie_ok = 0
    for t in attempts:
        a, b = str(t["agent_a_id"]), str(t["agent_b_id"])
        ok = t["outcome"].strip().lower() == "completed"
        pair = "".join(sorted([short.get(a, "?"), short.get(b, "?")]))
        pair_tot[pair] += 1
        pair_ok[pair] += ok
        if short.get(a) == "N" and short.get(b) == "N":
            tick = int(_f(t.get("tick"), 0) or 0)
            w = f"T{(tick - 1) // WINDOW * WINDOW + 1}-{(tick - 1) // WINDOW * WINDOW + WINDOW}"
            win_tot[w] += 1
            win_ok[w] += ok
            normie_tot += 1
            normie_ok += ok

    # --- survival ------------------------------------------------------------------------
    causes = Counter((d.get("cause") or "").strip() for d in deaths if keep(str(d.get("agent_id"))))
    starved = sum(v for c, v in causes.items() if c.startswith("starv"))
    natural = sum(v for c, v in causes.items() if c in ("old_age", "age", "natural"))
    pop0 = len([a for a in initial if keep(a)]) or int(_f(config.get("initial_population"), 0) or 0)

    lifespans = [_f(d.get("lifetime_ticks")) for d in deaths if keep(str(d.get("agent_id")))]
    lifespans = [x for x in lifespans if x is not None]
    wealth_at_death = [
        (_f(d.get("final_wealth"), 0) or 0) + (_f(d.get("final_spice"), 0) or 0)
        for d in deaths if keep(str(d.get("agent_id")))
    ]
    survivor_ages = [_f(a.get("age")) for aid, a in final.items() if keep(aid) and str(a.get("alive", "True")) != "False"]
    survivor_ages = [x for x in survivor_ages if x is not None]

    # --- identity shift ------------------------------------------------------------------
    last_leaning: Dict[str, float] = {}
    for r in reflections:
        v = _f(r.get("identity_leaning_after"))
        if v is not None:
            last_leaning[str(r.get("agent_id"))] = v
    for aid, a in final.items():
        v = _f(a.get("self_identity_leaning"))
        if v is not None:
            last_leaning[aid] = v  # final state wins for survivors
    shifts: Dict[str, List[float]] = defaultdict(list)
    for aid, a in initial.items():
        if not keep(aid):
            continue
        l0 = _f(a.get("self_identity_leaning"))
        l1 = last_leaning.get(aid)
        if l0 is not None and l1 is not None:
            shifts[short.get(aid, "?")].append(l1 - l0)

    # --- belief shift (1-5 scores) -------------------------------------------------------
    belief_keys = {"cooperation": "cooperation_value", "trust": "trust_importance", "self": "self_interest_priority"}
    belief_shift: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for aid, a in final.items():
        if not keep(aid):
            continue
        base = _maybe_literal(a.get("baseline_snapshot")) or {}
        led = _maybe_literal(a.get("belief_ledger")) or {}
        q0 = ((base.get("belief_ledger") or {}).get("quantified") or {}) if isinstance(base, dict) else {}
        q1 = (led.get("quantified") or {}) if isinstance(led, dict) else {}
        for label, key in belief_keys.items():
            if key in q0 and key in q1:
                belief_shift[short.get(aid, "?")][label].append(_f(q1[key], 0) - _f(q0[key], 0))

    def mean(xs):
        return round(statistics.mean(xs), 3) if xs else None

    return {
        "experiment": str(exp_dir),
        "config": {k: config.get(k) for k in ("initial_population", "max_ticks", "width", "height",
                                              "llm_provider_model", "origin_identity_distribution",
                                              "enable_resource_specialization")},
        "normie_only": normie_only,
        "trade_attempts": len(attempts),
        "trade_success_rate": round(len(completed) / len(attempts), 4) if attempts else None,
        "trade_success_by_pair": {p: round(pair_ok[p] / pair_tot[p], 4) for p in sorted(pair_tot)},
        "trade_attempts_by_pair": dict(sorted(pair_tot.items())),
        "normie_normie_success_rate": round(normie_ok / normie_tot, 4) if normie_tot else None,
        "normie_normie_by_window": {w: round(win_ok[w] / win_tot[w], 4) for w in sorted(win_tot, key=lambda s: int(s[1:].split("-")[0]))},
        "deaths": dict(causes),
        "starvation_share_of_initial": round(starved / pop0, 4) if pop0 else None,
        "survival_rate_natural_over_deaths": round(natural / (natural + starved), 4) if (natural + starved) else None,
        "alive_at_end": sum(1 for aid, a in final.items() if keep(aid) and str(a.get("alive", "True")) != "False"),
        "initial_population": pop0,
        "mean_lifespan_dead": mean(lifespans),
        "mean_age_survivors": mean(survivor_ages),
        "mean_wealth_at_death": mean(wealth_at_death),
        "identity_shift_by_origin": {g: mean(v) for g, v in sorted(shifts.items())},
        "identity_shift_n": {g: len(v) for g, v in sorted(shifts.items())},
        "belief_shift_by_origin": {g: {k: mean(v) for k, v in d.items()} for g, d in sorted(belief_shift.items())},
    }


def render(report: dict) -> str:
    r = report
    lines = [f"## {r['experiment']}", ""]
    c = r["config"]
    lines.append(f"config: {c.get('initial_population')} agents, {c.get('max_ticks')} ticks, "
                 f"{c.get('width')}x{c.get('height')}, model {c.get('llm_provider_model')}, "
                 f"identity {c.get('origin_identity_distribution')}, specialisation {c.get('enable_resource_specialization')}")
    lines.append(f"normie-only metrics: {r['normie_only']}")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| trade attempts | {r['trade_attempts']} |")
    lines.append(f"| trade success rate | {r['trade_success_rate']} |")
    for p, v in r["trade_success_by_pair"].items():
        lines.append(f"| trade success {p[0]}<->{p[1]} (n={r['trade_attempts_by_pair'][p]}) | {v} |")
    for w, v in r["normie_normie_by_window"].items():
        lines.append(f"| N<->N success {w} | {v} |")
    lines.append(f"| deaths | {r['deaths']} |")
    lines.append(f"| starved / initial population | {r['starvation_share_of_initial']} |")
    lines.append(f"| survival rate natural/(natural+starved) | {r['survival_rate_natural_over_deaths']} |")
    lines.append(f"| alive at end / initial | {r['alive_at_end']} / {r['initial_population']} |")
    lines.append(f"| mean lifespan of dead agents (ticks) | {r['mean_lifespan_dead']} |")
    lines.append(f"| mean age of survivors (ticks) | {r['mean_age_survivors']} |")
    lines.append(f"| mean wealth at death (sugar+spice) | {r['mean_wealth_at_death']} |")
    lines.append(f"| identity shift by origin | {r['identity_shift_by_origin']} (n={r['identity_shift_n']}) |")
    lines.append(f"| belief shift by origin (coop/trust/self) | {r['belief_shift_by_origin']} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+", help="experiment_<timestamp> directories")
    p.add_argument("--normie-only", action="store_true", help="restrict deaths, lifespans and shifts to Normie agents")
    p.add_argument("--json", type=str, default=None, help="also write all reports to this JSON file")
    args = p.parse_args(argv)

    reports = []
    for exp in args.experiments:
        exp_dir = Path(exp)
        if not (exp_dir / "config.json").exists():
            print(f"skipping {exp}: no config.json", file=sys.stderr)
            continue
        rep = analyze(exp_dir, normie_only=args.normie_only)
        reports.append(rep)
        print(render(rep))
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.json}")
    return 0 if reports else 1


if __name__ == "__main__":
    sys.exit(main())
