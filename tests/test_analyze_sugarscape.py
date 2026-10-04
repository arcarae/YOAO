"""The Sugarscape metrics script computes the paper's definitions from the debug outputs."""

import csv
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parent.parent / "scripts" / "analyze_sugarscape.py"
    spec = importlib.util.spec_from_file_location("analyze_sugarscape", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _experiment(tmp_path):
    exp = tmp_path / "experiment_x"
    exp.mkdir()
    (exp / "config.json").write_text(json.dumps({
        "initial_population": 4, "max_ticks": 40, "width": 20, "height": 20,
        "llm_provider_model": "m", "origin_identity_distribution": {"altruist": 0.5, "survivor": 0.5},
        "enable_resource_specialization": True,
    }))
    agents0 = [
        {"agent_id": "1", "origin_identity": "altruist", "self_identity_leaning": 0.8, "alive": True, "age": 0},
        {"agent_id": "2", "origin_identity": "altruist", "self_identity_leaning": 0.8, "alive": True, "age": 0},
        {"agent_id": "3", "origin_identity": "survivor", "self_identity_leaning": 0.0, "alive": True, "age": 0},
        {"agent_id": "4", "origin_identity": "survivor", "self_identity_leaning": 0.0, "alive": True, "age": 0},
    ]
    (exp / "initial_state.json").write_text(json.dumps({"tick": 0, "agents": agents0}))
    # agent 4 died (not in final state); agent 3 ends at leaning 0.1 with beliefs that moved
    agents1 = [
        {"agent_id": "1", "origin_identity": "altruist", "self_identity_leaning": 0.9, "alive": True, "age": 40},
        {"agent_id": "2", "origin_identity": "altruist", "self_identity_leaning": 0.8, "alive": True, "age": 40},
        {"agent_id": "3", "origin_identity": "survivor", "self_identity_leaning": 0.1, "alive": True, "age": 40,
         "baseline_snapshot": str({"tick": 0, "belief_ledger": {"quantified": {"cooperation_value": 3, "trust_importance": 3, "self_interest_priority": 3}}}),
         "belief_ledger": str({"quantified": {"cooperation_value": 4, "trust_importance": 2, "self_interest_priority": 3}})},
    ]
    (exp / "final_state.json").write_text(json.dumps({"tick": 40, "agents": agents1}))
    trades = [
        {"tick": 1, "agent_a_id": "1", "agent_b_id": "2", "outcome": "completed"},   # A-A
        {"tick": 2, "agent_a_id": "1", "agent_b_id": "3", "outcome": "completed"},   # A-N
        {"tick": 3, "agent_a_id": "3", "agent_b_id": "4", "outcome": "reject"},      # N-N window 1
        {"tick": 25, "agent_a_id": "3", "agent_b_id": "4", "outcome": "completed"},  # N-N window 2
        {"tick": 26, "agent_a_id": "4", "agent_b_id": "3", "outcome": "timeout"},    # N-N window 2
    ]
    _write_csv(exp / "debug" / "trade_history.csv", trades, ["tick", "agent_a_id", "agent_b_id", "outcome"])
    deaths = [{"tick": 30, "agent_id": "4", "cause": "starvation_sugar", "final_wealth": 0, "final_spice": 12,
               "lifetime_ticks": 30}]
    _write_csv(exp / "debug" / "death_records.csv", deaths,
               ["tick", "agent_id", "cause", "final_wealth", "final_spice", "lifetime_ticks"])
    reflections = [{"tick": 25, "agent_id": "4", "identity_shift": -0.05, "identity_leaning_after": -0.05},
                   {"tick": 26, "agent_id": "4", "identity_shift": -0.05, "identity_leaning_after": -0.10}]
    _write_csv(exp / "debug" / "reflections.csv", reflections, ["tick", "agent_id", "identity_shift", "identity_leaning_after"])
    return exp


def test_metrics_match_definitions(tmp_path):
    mod = _load()
    r = mod.analyze(_experiment(tmp_path))
    assert r["trade_attempts"] == 5
    assert r["trade_success_rate"] == 0.6
    assert r["trade_success_by_pair"] == {"AA": 1.0, "AN": 1.0, "NN": round(1 / 3, 4)}
    assert r["normie_normie_by_window"] == {"T1-20": 0.0, "T21-40": 0.5}
    assert r["deaths"] == {"starvation_sugar": 1}
    assert r["starvation_share_of_initial"] == 0.25
    assert r["survival_rate_natural_over_deaths"] == 0.0
    assert r["mean_lifespan_dead"] == 30
    assert r["mean_wealth_at_death"] == 12
    # altruists: (0.9-0.8 + 0.8-0.8)/2 ; normies: agent 3 +0.1, agent 4 last reflection -0.10
    assert r["identity_shift_by_origin"] == {"A": 0.05, "N": 0.0}
    assert r["belief_shift_by_origin"]["N"] == {"cooperation": 1.0, "trust": -1.0, "self": 0.0}


def test_normie_only_filter(tmp_path):
    mod = _load()
    r = mod.analyze(_experiment(tmp_path), normie_only=True)
    assert r["initial_population"] == 2
    assert r["starvation_share_of_initial"] == 0.5
    assert list(r["identity_shift_by_origin"]) == ["N"]
    text = mod.render(r)
    assert "trade success rate" in text and "N<->N success T21-40" in text


def test_cli_writes_json(tmp_path, capsys):
    mod = _load()
    exp = _experiment(tmp_path)
    out = tmp_path / "out.json"
    assert mod.main([str(exp), "--json", str(out)]) == 0
    assert json.loads(out.read_text())[0]["trade_attempts"] == 5
    assert "## " in capsys.readouterr().out
