"""YAML configs expand ${ENV} references and fall back to the shared defaults."""

import textwrap

from redblackbench import defaults
from redblackbench.cli import create_provider, expand_env, load_config


def test_expand_env_substitutes_and_blanks(monkeypatch):
    monkeypatch.setenv("YOAO_TEST_KEY", "sk-test")
    monkeypatch.delenv("YOAO_MISSING", raising=False)
    assert expand_env("${YOAO_TEST_KEY}") == "sk-test"
    assert expand_env("prefix-$YOAO_TEST_KEY") == "prefix-sk-test"
    assert expand_env("${YOAO_MISSING}") is None        # never send a literal placeholder
    assert expand_env({"a": ["${YOAO_TEST_KEY}", 3]}) == {"a": ["sk-test", 3]}
    assert expand_env("plain") == "plain"


def test_load_config_expands(tmp_path, monkeypatch):
    monkeypatch.delenv("YOAO_VLLM_URL", raising=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(textwrap.dedent("""
        default_provider:
          type: vllm
          model: Qwen/Qwen3-14B
          base_url: "${YOAO_VLLM_URL}"
        game:
          num_rounds: 10
    """))
    loaded = load_config(str(cfg))
    assert loaded["default_provider"]["base_url"] is None
    assert loaded["game"]["num_rounds"] == 10


def test_vllm_provider_uses_defaults_when_blank():
    p = create_provider({"type": "vllm", "model": None, "base_url": None})
    assert p.config.model == defaults.BASE_MODEL
    assert p.base_url == defaults.VLLM_URL


def test_defaults_paper_constants():
    assert defaults.PAPER_MULTIPLIERS == {5: 3, 8: 5, 10: 10}
    assert defaults.PAPER_TEAM_SIZE == 5 and defaults.PAPER_NUM_ROUNDS == 10
    assert set(defaults.HELD_OUT_SCENARIOS) == {"baseline", "trade_war", "gpu_contention"}
    assert len(defaults.TRAIN_SCENARIOS) == 5
