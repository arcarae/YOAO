from sugarscape.config import SugarscapeConfig
from sugarscape.environment import SugarEnvironment
from sugarscape.trade import DialogueTradeSystem


def test_accept_private_execute_direction_confusion_is_corrected_when_matching_offer_give():
    cfg = SugarscapeConfig(enable_spice=True)
    env = SugarEnvironment(cfg)
    ts = DialogueTradeSystem(env, allow_fraud=True)

    # Partner's public_offer: partner gives 10 sugar, wants 2 spice.
    contract_give = {"sugar": 10, "spice": 0}
    contract_receive = {"sugar": 0, "spice": 2}

    # Confused acceptor: sets private_execute_give to what they expect to RECEIVE (partner give).
    acceptor_execute = {"sugar": 10, "spice": 0}

    fixed = ts._fix_accept_execute_direction_confusion(  # noqa: SLF001 (intentional test of internal guardrail)
        acceptor_execute=acceptor_execute,
        contract_offer_give=contract_give,
        contract_offer_receive=contract_receive,
    )

    assert fixed == {"sugar": 0, "spice": 2}


def test_accept_private_execute_direction_confusion_not_corrected_when_not_matching_signature():
    cfg = SugarscapeConfig(enable_spice=True)
    env = SugarEnvironment(cfg)
    ts = DialogueTradeSystem(env, allow_fraud=True)

    contract_give = {"sugar": 10, "spice": 0}
    contract_receive = {"sugar": 0, "spice": 2}

    # Looks like intentional fraud or different plan; should not be rewritten.
    acceptor_execute = {"sugar": 0, "spice": 0}

    fixed = ts._fix_accept_execute_direction_confusion(  # noqa: SLF001
        acceptor_execute=acceptor_execute,
        contract_offer_give=contract_give,
        contract_offer_receive=contract_receive,
    )

    assert fixed == {"sugar": 0, "spice": 0}


