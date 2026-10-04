import numpy as np
import asyncio
import json
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from sugarscape.agent import SugarAgent
from sugarscape.environment import SugarEnvironment
from sugarscape.prompts import (
    build_sugarscape_trade_system_prompt,
    build_sugarscape_trade_turn_prompt,
    build_sugarscape_reflection_prompt,
    format_beliefs_policies_appendix,
    # New encounter protocol prompts
    build_broker_introduction_prompt,
    build_small_talk_system_prompt,
    build_small_talk_turn_prompt,
    build_trade_intent_system_prompt,
    build_trade_intent_turn_prompt,
    build_negotiation_system_prompt,
    build_negotiation_turn_prompt,
)

try:
    from sugarscape.llm_agent import LLMSugarAgent, strip_thinking_blocks
except Exception:  # pragma: no cover
    LLMSugarAgent = None  # type: ignore
    strip_thinking_blocks = None  # type: ignore

class TradeSystem:
    """Handles trading logic between agents using MRS bargaining."""

    def __init__(self, env: SugarEnvironment):
        self.env = env

    def execute_trade_round(self, agents: List[SugarAgent], tick: Optional[int] = None):
        """Execute a round of trading for all agents."""
        # Randomize order
        import random
        random.shuffle(agents)

        for agent in agents:
            if not agent.alive: continue

            # Find a partner
            partner = self._find_trade_partner(agent)
            if partner:
                self._bargain(agent, partner)

    def _find_trade_partner(self, agent: SugarAgent) -> Optional[SugarAgent]:
        """Find a random neighbor to trade with (Von Neumann neighborhood)."""
        x, y = agent.pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.env.width
            ny = (y + dy) % self.env.height
            partner = self.env.get_agent_at((nx, ny))
            if partner and partner.alive and partner != agent:
                neighbors.append(partner)

        if not neighbors:
            return None

        import random
        return random.choice(neighbors)

    def _bargain(self, a: SugarAgent, b: SugarAgent):
        """Execute iterative bargaining between two agents."""
        # Limit iterations to prevent infinite loops
        max_trades = 10

        for _ in range(max_trades):
            mrs_a = a.mrs
            mrs_b = b.mrs

            # If MRS are equal (or close enough), no trade
            if abs(mrs_a - mrs_b) < 0.01:
                break

            # Determine direction
            # High MRS means "I value Sugar highly relative to Spice" -> Buy Sugar / Sell Spice
            # Low MRS means "I value Spice highly relative to Sugar" -> Buy Spice / Sell Sugar

            price = np.sqrt(mrs_a * mrs_b) # Geometric mean price (Spice per Sugar)

            if mrs_a > mrs_b:
                # A wants Sugar, B has Sugar (relatively).
                # A buys Sugar from B using Spice.
                buyer, seller = a, b
            else:
                # B wants Sugar, A has Sugar.
                # B buys Sugar from A using Spice.
                buyer, seller = b, a

            # Calculate transaction amount
            # If p > 1: 1 unit of Sugar for p units of Spice
            # If p < 1: 1/p units of Sugar for 1 unit of Spice
            # To simplify (and follow standard impl): We trade 1 unit of the "more valuable" good?
            # Or standard Epstein Axtell: trade 1 unit of Sugar for p units of Spice if p >= 1
            # trade 1/p units of Sugar for 1 unit of Spice if p < 1

            sugar_amt = 0.0
            spice_amt = 0.0

            if price >= 1.0:
                sugar_amt = 1.0
                spice_amt = price
            else:
                sugar_amt = 1.0 / price
                spice_amt = 1.0

            # Check affordability
            if buyer.spice < spice_amt or seller.wealth < sugar_amt:
                break

            # Check Welfare Improvement (Pareto condition)
            # Calculate projected welfare
            w_buyer_current = buyer.welfare
            w_seller_current = seller.welfare

            # Simulated state
            # Buyer: +Sugar, -Spice
            # Seller: -Sugar, +Spice

            # We need to compute welfare manually without modifying state yet
            def calc_welfare(agent, s, p):
                m_t = agent.metabolism + agent.metabolism_spice
                return (s ** (agent.metabolism/m_t)) * (p ** (agent.metabolism_spice/m_t))

            w_buyer_new = calc_welfare(buyer, buyer.wealth + sugar_amt, buyer.spice - spice_amt)
            w_seller_new = calc_welfare(seller, seller.wealth - sugar_amt, seller.spice + spice_amt)

            if w_buyer_new > w_buyer_current and w_seller_new > w_seller_current:
                # Execute Trade
                # Actually, standard model might use floats for resources, but our agent uses int.
                # Let's support float resources or round.
                # Since our simulation uses ints, let's stick to int exchanges if possible, or cast.
                # If we cast to int, small trades might be 0.
                # FIX: Let's assume resources are continuous (float) or convert Agent to use floats.
                # For now, let's round amounts.

                s_exchange = max(1, int(sugar_amt))
                p_exchange = max(1, int(spice_amt))

                # Re-check affordability with integers
                if buyer.spice < p_exchange or seller.wealth < s_exchange:
                    break

                buyer.wealth += s_exchange
                buyer.spice -= p_exchange
                seller.wealth -= s_exchange
                seller.spice += p_exchange

                # Recalculate MRS for next iteration
                continue
            else:
                break


@dataclass(frozen=True)
class _ParsedTradeReply:
    thought: str
    say: str
    intent: str
    public_offer: Dict[str, Dict[str, int]]
    private_execute_give: Dict[str, int]


@dataclass
class ContractEnforcement:
    """Records contract enforcement details for no-fraud mode.

    This tracks what was agreed upon vs what would have been executed
    without enforcement, enabling research analysis of how binding
    contracts change agent behavior.
    """
    contract_give: Dict[str, int]  # What contract says offerer gives
    contract_receive: Dict[str, int]  # What contract says acceptor gives
    offerer_intended: Dict[str, int]  # What offerer's private_execute_give was
    acceptor_intended: Dict[str, int]  # What acceptor's private_execute_give was
    offerer_enforced: Dict[str, int]  # What offerer actually sent (after enforcement)
    acceptor_enforced: Dict[str, int]  # What acceptor actually sent (after enforcement)
    offerer_deviation: bool  # Would offerer have deviated without enforcement?
    acceptor_deviation: bool  # Would acceptor have deviated without enforcement?
    enforcement_active: bool  # Was no-fraud enforcement active?


class DialogueTradeSystem:
    """Free-form LLM dialogue trade system with optional deception and memory.

    Public offers are non-binding. When an offer is accepted, the environment executes a
    simultaneous transfer using each side's `private_execute_give` (not shown to the partner).

    Optimizations:
    - Batch parallel trades: All trade pairs processed concurrently via asyncio.gather
    - Reduced Stage 2 tokens: JSON-only output capped at 128 tokens
    - Reasoning compression: Only conclusion passed to Stage 2, not full reasoning
    - Pre-computed templates: Static prompt parts cached to avoid repeated string formatting
    """

    # Pre-computed prompt templates (Optimization #4)
    STAGE2_TEMPLATE = (
        "Your analysis: {reasoning}\n\n"
        "Round {round_idx}/{max_rounds}. {offer_context}\n\n"
        "Output valid JSON only:\n"
        '{{"intent": "OFFER", "public_offer": {{"give": {{"sugar": 5, "spice": 2}}, "receive": {{"sugar": 0, "spice": 8}}}}, "private_execute_give": {{"sugar": 5, "spice": 2}}}}\n\n'
        "intent options: OFFER (propose), ACCEPT (agree to partner's offer), REJECT (decline), WALK_AWAY (end).\n"
        "give = what YOU send. receive = what YOU get. private_execute_give = actual transfer (can differ from public_offer)."
    )

    def __init__(
        self,
        env: SugarEnvironment,
        max_rounds: int = 4,
        allow_fraud: bool = True,
        memory_maxlen: int = 50,
    ):
        self.env = env
        self.max_rounds = max(1, int(max_rounds))
        self.allow_fraud = bool(allow_fraud)
        self.memory_maxlen = max(1, int(memory_maxlen))
        # Prompt-first robustness controls (configured via env.config)
        self.repair_json = bool(getattr(env.config, "trade_dialogue_repair_json", True))
        self.repair_attempts = max(0, int(getattr(env.config, "trade_dialogue_repair_attempts", 1)))
        self.coerce_protocol = bool(getattr(env.config, "trade_dialogue_coerce_protocol", False))
        # Two-stage mode: thinking → JSON pipeline
        self.two_stage = bool(getattr(env.config, "trade_dialogue_two_stage", True))
        self.thinking_tokens = int(getattr(env.config, "trade_dialogue_thinking_tokens", 1024))  # Keep high for quality reasoning
        self.json_tokens = int(getattr(env.config, "trade_dialogue_json_tokens", 128))  # Reduced - JSON output is small

        # New encounter protocol settings
        # Protocol: 2 rounds small talk → 1 trade intent → 2 rounds negotiation → execution
        self.enable_new_protocol = bool(getattr(env.config, "enable_new_encounter_protocol", True))
        self.small_talk_rounds = int(getattr(env.config, "small_talk_rounds", 2))
        # Two-stage dialogue: Stage 1 = think (128 tokens), Stage 2 = respond (128 tokens)
        self.dialogue_thinking_tokens = int(getattr(env.config, "dialogue_thinking_tokens", 128))
        self.dialogue_response_tokens = int(getattr(env.config, "dialogue_response_tokens", 128))
        self.small_talk_allow_thinking = bool(getattr(env.config, "small_talk_allow_thinking", True))
        self.negotiation_rounds = int(getattr(env.config, "negotiation_rounds", 2))
        self.enable_social_exclusion = bool(getattr(env.config, "enable_social_exclusion", True))
        self.enable_broker = bool(getattr(env.config, "enable_broker", False))
        self.broker_provider = None  # Can be set externally

    def execute_trade_round(self, agents: List[SugarAgent], tick: Optional[int] = None):
        """Execute trade round with batch parallel processing (Optimization #1).

        All LLM trade pairs are processed concurrently using asyncio.gather for
        significant speedup when multiple pairs exist.
        """
        import random

        tick_value = int(tick) if tick is not None else -1

        # Pair agents at most once per round.
        shuffled = [a for a in agents if a.alive]
        random.shuffle(shuffled)
        paired_ids: set[int] = set()
        pairs: List[Tuple[SugarAgent, SugarAgent]] = []

        for agent in shuffled:
            if not agent.alive or agent.agent_id in paired_ids:
                continue
            partner = self._find_trade_partner(agent, excluded_ids=paired_ids)
            if partner is None:
                continue
            paired_ids.add(agent.agent_id)
            paired_ids.add(partner.agent_id)
            pairs.append((agent, partner))

        # Filter to LLM trader pairs only
        llm_pairs = [(a, b) for a, b in pairs if self._is_llm_trader(a) and self._is_llm_trader(b)]

        if not llm_pairs:
            return

        # Batch parallel execution (Optimization #1)
        self._run_negotiate_pairs_parallel(llm_pairs, tick_value)

    def _run_negotiate_pairs_parallel(self, pairs: List[Tuple[SugarAgent, SugarAgent]], tick: int) -> None:
        """Run all trade negotiations in parallel using asyncio.gather.

        Uses the persistent event loop set by SugarSimulation for better
        throughput via connection reuse.
        """
        async def negotiate_all():
            tasks = [self._negotiate_pair_safe(a, b, tick) for a, b in pairs]
            await asyncio.gather(*tasks)

        # Use the persistent event loop (set by SugarSimulation)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(negotiate_all())

    async def _negotiate_pair_safe(self, a: SugarAgent, b: SugarAgent, tick: int) -> None:
        """Wrapper for _negotiate_pair that catches exceptions to avoid crashing other parallel trades."""
        try:
            await self._negotiate_pair(a, b, tick)
        except Exception as e:
            print(f"[TRADE] Negotiation error: {a.name} <-> {b.name} (tick {tick}): {e}")
            self._record_no_trade(
                a,
                b,
                tick,
                outcome="ERROR",
                pending_offer=None,
                conversation=[{"type": "error", "tick": tick, "error": str(e)}],
            )

    def _is_llm_trader(self, agent: SugarAgent) -> bool:
        if LLMSugarAgent is None:
            return hasattr(agent, "provider") and getattr(agent, "provider", None) is not None
        return isinstance(agent, LLMSugarAgent) and getattr(agent, "provider", None) is not None

    def _find_trade_partner(self, agent: SugarAgent, excluded_ids: set[int]) -> Optional[SugarAgent]:
        x, y = agent.pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.env.width
            ny = (y + dy) % self.env.height
            partner = self.env.get_agent_at((nx, ny))
            if (
                partner
                and partner.alive
                and partner.agent_id != agent.agent_id
                and partner.agent_id not in excluded_ids
            ):
                neighbors.append(partner)
        if not neighbors:
            return None
        import random

        return random.choice(neighbors)

    async def _negotiate_pair(self, a: SugarAgent, b: SugarAgent, tick: int) -> None:
        """Execute the full encounter protocol between two agents.

        New Protocol (when enable_new_protocol=True):
        1. Social exclusion check
        2. 2 rounds small talk (no JSON, pure conversation)
        3. 1 trade intent round (decide if they want to trade)
        4. 2 rounds negotiation (with JSON offers)
        5. Automatic execution (binding contracts)

        Falls back to legacy protocol if enable_new_protocol=False.
        """
        if self.enable_new_protocol:
            await self._negotiate_pair_new_protocol(a, b, tick)
        else:
            await self._negotiate_pair_legacy(a, b, tick)

    async def _negotiate_pair_new_protocol(self, a: SugarAgent, b: SugarAgent, tick: int) -> None:
        """New encounter protocol: small talk → trade intent → negotiation → execution."""
        full_conversation: List[Dict[str, str]] = []

        print(f"[ENCOUNTER] Starting: {a.name} <-> {b.name} (tick {tick})")

        # === PHASE 0: SOCIAL EXCLUSION CHECK ===
        # Only check exclusion if both social_exclusion is enabled AND social memory is visible
        # (Without social memory, agents can't remember who to exclude)
        social_memory_visible = getattr(self.env.config, 'social_memory_visible', True)
        if self.enable_social_exclusion and social_memory_visible:
            a_excludes, a_reason = a.should_exclude_partner(b.agent_id)
            b_excludes, b_reason = b.should_exclude_partner(a.agent_id)

            if a_excludes or b_excludes:
                excluder = a if a_excludes else b
                excluded = b if a_excludes else a
                reason = a_reason if a_excludes else b_reason

                print(f"[EXCLUSION] {excluder.name} refuses to engage with {excluded.name}: {reason}")

                full_conversation.append({
                    "type": "exclusion",
                    "tick": tick,
                    "excluder": excluder.name,
                    "excluded": excluded.name,
                    "reason": reason,
                })

                self._record_no_trade(
                    a, b, tick,
                    outcome="EXCLUSION",
                    pending_offer=None,
                    conversation=full_conversation,
                    decider_id=excluder.agent_id,
                    decider_name=excluder.name,
                )
                return

        # === PHASE 0.5: BROKER INTRODUCTION (Optional) ===
        # If enabled, a third-party broker introduces the agents and proposes a deal
        small_talk_transcript: List[str] = []
        
        if self.enable_broker:
            # Generate broker message
            sys_p, user_p = build_broker_introduction_prompt(a, b, self.env)
            
            # Use dedicated broker provider if available, else fallback to agent provider
            provider = self.broker_provider if self.broker_provider else (a.provider if hasattr(a, 'provider') else b.provider)
            
            try:
                try:
                    broker_response = await provider.generate(
                        system_prompt=sys_p,
                        messages=[{"role": "user", "content": user_p}],
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    broker_response = await provider.generate(
                        system_prompt=sys_p,
                        messages=[{"role": "user", "content": user_p}],
                    )
                
                cleaned_broker_response = self._strip_thinking_blocks(broker_response, keep_if_empty=True).strip()
                
                print(f"[BROKER] {cleaned_broker_response[:100]}...")
                
                # Add to conversation log
                full_conversation.append({
                    "type": "broker_intro",
                    "tick": tick,
                    "speaker": "BROKER",
                    "message": cleaned_broker_response
                })
                
                # Add to transcript so agents see it in their prompts
                small_talk_transcript.append(f"BROKER: {cleaned_broker_response}")
                
            except Exception as e:
                print(f"[BROKER] Error: {e}")

        # === PHASE 1: SMALL TALK (2 rounds, no JSON) ===


        for round_idx in range(1, self.small_talk_rounds + 1):
            for speaker, listener in [(a, b), (b, a)]:
                # Build prompts
                system_prompt = build_small_talk_system_prompt(
                    agent_name=speaker.name,
                    agent=speaker,
                )

                conversation_text = "\n".join(small_talk_transcript[-6:]) if small_talk_transcript else ""
                last_message = small_talk_transcript[-1] if small_talk_transcript else ""

                turn_prompt = build_small_talk_turn_prompt(
                    self_agent=speaker,
                    partner_agent=listener,
                    round_idx=round_idx,
                    conversation_so_far=conversation_text,
                    partner_last_message=last_message,
                    env=self.env,
                )

                # Generate response - simple single-stage, strip thinking blocks after
                try:
                    try:
                        response = await speaker.provider.generate(
                            system_prompt=system_prompt,
                            messages=[{"role": "user", "content": turn_prompt}],
                            chat_template_kwargs={"enable_thinking": False},
                        )
                    except TypeError:
                        response = await speaker.provider.generate(
                            system_prompt=system_prompt,
                            messages=[{"role": "user", "content": turn_prompt}],
                        )
                    # Strip any thinking blocks the model outputs, keep content if all was in thinking
                    cleaned_response = self._strip_thinking_blocks(response, keep_if_empty=True).strip()
                    cleaned_response = self._strip_reasoning_prefix(cleaned_response)

                    # Add to transcript
                    small_talk_transcript.append(f"{speaker.name}: {cleaned_response[:500]}")

                    # Log the turn
                    full_conversation.append({
                        "type": "small_talk",
                        "tick": tick,
                        "round": round_idx,
                        "speaker": speaker.name,
                        "partner": listener.name,
                        "message": cleaned_response[:500],
                    })

                    # Store in agent's conversation history (store cleaned dialogue, not raw thinking)
                    if hasattr(speaker, 'conversation_history'):
                        speaker.conversation_history.append({"role": "user", "content": f"[SMALL TALK with {listener.name}] {turn_prompt}"})
                        speaker.conversation_history.append({"role": "assistant", "content": cleaned_response})

                except Exception as e:
                    print(f"[SMALL TALK] Error for {speaker.name}: {e}")
                    small_talk_transcript.append(f"{speaker.name}: (silence)")

        print(f"[ENCOUNTER] Small talk completed: {a.name} <-> {b.name}")

        # === PHASE 2: TRADE INTENT DECISION ===
        conversation_summary = "\n".join(small_talk_transcript[-4:]) if small_talk_transcript else "(No conversation)"

        a_wants_trade = False
        b_wants_trade = False

        for speaker, listener in [(a, b), (b, a)]:
            system_prompt = build_trade_intent_system_prompt(
                agent_name=speaker.name,
                agent=speaker,
            )

            turn_prompt = build_trade_intent_turn_prompt(
                self_agent=speaker,
                partner_agent=listener,
                conversation_summary=conversation_summary,
                env=self.env,
            )

            try:
                # Simple single-stage - strip thinking blocks after
                try:
                    response = await speaker.provider.generate(
                        system_prompt=system_prompt,
                        messages=[{"role": "user", "content": turn_prompt}],
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    response = await speaker.provider.generate(
                        system_prompt=system_prompt,
                        messages=[{"role": "user", "content": turn_prompt}],
                    )
                cleaned_response = self._strip_thinking_blocks(response, keep_if_empty=True).strip()

                # Parse intent
                intent = self._parse_trade_intent(cleaned_response)

                if speaker == a:
                    a_wants_trade = (intent == "TRADE")
                else:
                    b_wants_trade = (intent == "TRADE")

                # Extract message
                message = self._extract_message_from_intent(cleaned_response)

                full_conversation.append({
                    "type": "trade_intent",
                    "tick": tick,
                    "speaker": speaker.name,
                    "intent": intent,
                    "message": message,
                })

                if hasattr(speaker, 'conversation_history'):
                    speaker.conversation_history.append({"role": "user", "content": f"[TRADE INTENT with {listener.name}] {turn_prompt}"})
                    speaker.conversation_history.append({"role": "assistant", "content": cleaned_response})

            except Exception as e:
                print(f"[TRADE INTENT] Error for {speaker.name}: {e}")

        # If EITHER party wants to trade, proceed to negotiation
        if not (a_wants_trade or b_wants_trade):
            print(f"[ENCOUNTER] Both declined trade: {a.name} <-> {b.name}")
            self._record_no_trade(
                a, b, tick,
                outcome="BOTH_DECLINED",
                pending_offer=None,
                conversation=full_conversation,
            )

            # Run reflection even for declined trades
            await self._run_reflection_for_pair(
                agent_a=a,
                agent_b=b,
                tick=tick,
                outcome="both_declined",
                conversation=full_conversation,
            )
            return

        print(f"[ENCOUNTER] Trade intent confirmed: {a.name}={'TRADE' if a_wants_trade else 'DECLINE'}, {b.name}={'TRADE' if b_wants_trade else 'DECLINE'}")

        # === PHASE 3: NEGOTIATION (2 rounds with JSON) ===
        await self._run_negotiation_phase(a, b, tick, full_conversation)

    async def _run_negotiation_phase(
        self,
        a: SugarAgent,
        b: SugarAgent,
        tick: int,
        full_conversation: List[Dict[str, str]],
    ) -> None:
        """Run the negotiation phase with JSON offers."""
        transcript: List[str] = []
        pending_offer: Optional[Dict[str, Dict[str, int]]] = None
        pending_offer_from: Optional[int] = None
        pending_offer_private_execute: Optional[Dict[str, int]] = None
        pending_offer_public_text: str = "(None)"

        # Use negotiation_rounds (default 2) for this phase
        total_rounds = self.negotiation_rounds * 2  # Each agent gets negotiation_rounds turns

        for round_idx in range(1, total_rounds + 1):
            speaker, listener = (a, b) if round_idx % 2 == 1 else (b, a)

            # Build prompts
            speaker_goal = getattr(speaker, 'goal_prompt', '') or self.env.config.llm_goal_prompt
            enable_survival_pressure = getattr(self.env.config, 'enable_survival_pressure', True)
            system_prompt = build_negotiation_system_prompt(
                goal_prompt=speaker_goal,
                max_rounds=total_rounds,
                allow_fraud=self.allow_fraud,
                agent_name=speaker.name,
                agent=speaker,
                enable_survival_pressure=enable_survival_pressure,
            )

            recent_transcript = "\n".join(transcript[-6:]) if transcript else "(Starting negotiation)"
            partner_last_offer = pending_offer_public_text if pending_offer_from == listener.agent_id else "(None)"

            if pending_offer is not None and pending_offer_from == listener.agent_id:
                give_i = self._safe_int_pair(pending_offer.get("give") or {})
                receive_i = self._safe_int_pair(pending_offer.get("receive") or {})
                if not self.env.config.enable_spice:
                    give_i["spice"] = 0
                    receive_i["spice"] = 0
                partner_last_offer = (
                    f"{pending_offer_public_text}\n"
                    f"If you ACCEPT: you RECEIVE {give_i} and GIVE {receive_i}."
                )

            turn_prompt = build_negotiation_turn_prompt(
                self_agent=speaker,
                partner_agent=listener,
                round_idx=(round_idx + 1) // 2,  # Convert to per-agent round number
                max_rounds=self.negotiation_rounds,
                conversation_so_far=recent_transcript,
                partner_last_offer=partner_last_offer,
                env=self.env,
            )

            # Append learned beliefs/policies and trade history if reflection is enabled
            if getattr(self.env.config, 'enable_reflection', False):
                include_history = getattr(self.env.config, 'trade_history_in_prompt', True)
                history_limit = getattr(self.env.config, 'trade_history_prompt_limit', 10)
                social_memory_visible = getattr(self.env.config, 'social_memory_visible', True)
                beliefs_appendix = format_beliefs_policies_appendix(
                    speaker,
                    partner_id=listener.agent_id,
                    include_trade_history=include_history,
                    trade_history_limit=history_limit,
                    social_memory_visible=social_memory_visible,
                )
                if beliefs_appendix:
                    turn_prompt += beliefs_appendix

            # Generate response
            if self.two_stage:
                reply_text, parsed = await self._two_stage_generate(
                    speaker=speaker,
                    listener=listener,
                    system_prompt=system_prompt,
                    turn_prompt=turn_prompt,
                    round_idx=(round_idx + 1) // 2,
                    max_rounds=self.negotiation_rounds,
                    pending_offer=pending_offer,
                    pending_offer_from=pending_offer_from,
                )
            else:
                reply_text = await speaker.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": turn_prompt}],
                )
                parsed = self._parse_trade_reply(reply_text)

            # Apply coercion if enabled
            if self.coerce_protocol:
                parsed = self._coerce_trade_reply(
                    parsed=parsed,
                    speaker=speaker,
                    listener=listener,
                    round_idx=(round_idx + 1) // 2,
                    max_rounds=self.negotiation_rounds,
                    pending_offer=pending_offer,
                    pending_offer_from=pending_offer_from,
                )

            transcript.append(f"{speaker.name}: {parsed.say}")

            # Log the turn (include prompt for debugging)
            trade_turn = {
                "type": "negotiation",
                "tick": tick,
                "round": (round_idx + 1) // 2,
                "speaker": speaker.name,
                "partner": listener.name,
                "prompt": turn_prompt,
                "response": reply_text,
                "intent": parsed.intent,
                "public_offer": parsed.public_offer if parsed.intent in ("OFFER", "REJECT") else None,
            }
            full_conversation.append(trade_turn)

            if hasattr(speaker, 'conversation_history'):
                speaker.conversation_history.append({"role": "user", "content": f"[NEGOTIATION with {listener.name}] {turn_prompt}"})
                speaker.conversation_history.append({"role": "assistant", "content": reply_text})

            # Handle intent
            if parsed.intent == "OFFER":
                pending_offer = parsed.public_offer
                pending_offer_from = speaker.agent_id
                pending_offer_private_execute = parsed.private_execute_give
                pending_offer_public_text = self._format_public_offer_text(speaker, pending_offer)
                continue

            if parsed.intent in {"REJECT", "WALK_AWAY"}:
                print(f"[NEGOTIATION] {speaker.name} {parsed.intent}ed: {a.name} <-> {b.name}")
                self._record_no_trade(
                    a, b, tick,
                    outcome=parsed.intent,
                    pending_offer=pending_offer,
                    conversation=full_conversation,
                    decider_id=speaker.agent_id,
                    decider_name=speaker.name,
                )

                await self._run_reflection_for_pair(
                    agent_a=a,
                    agent_b=b,
                    tick=tick,
                    outcome=parsed.intent.lower(),
                    conversation=full_conversation,
                )
                return

            if parsed.intent == "ACCEPT":
                if pending_offer is None or pending_offer_from != listener.agent_id:
                    continue

                # === PHASE 4: EXECUTION ===
                offerer = listener
                acceptor = speaker

                contract_give = self._safe_int_pair((pending_offer or {}).get("give", {}))
                contract_receive = self._safe_int_pair((pending_offer or {}).get("receive", {}))

                offerer_intended = self._safe_int_pair(pending_offer_private_execute or {})
                acceptor_intended = self._safe_int_pair(parsed.private_execute_give or {})

                # Binding contracts: enforce contract terms
                offerer_execute = contract_give.copy()
                acceptor_execute = contract_receive.copy()

                if not self.allow_fraud:
                    offerer_deviation = (offerer_intended != contract_give)
                    acceptor_deviation = (acceptor_intended != contract_receive)

                    if offerer_deviation or acceptor_deviation:
                        print(f"[CONTRACT ENFORCEMENT] Binding contract enforced:")
                        if offerer_deviation:
                            print(f"  - {offerer.name} intended {offerer_intended}, enforced {contract_give}")
                        if acceptor_deviation:
                            print(f"  - {acceptor.name} intended {acceptor_intended}, enforced {contract_receive}")
                else:
                    offerer_execute = offerer_intended
                    acceptor_execute = self._fix_accept_execute_direction_confusion(
                        acceptor_execute=acceptor_intended,
                        contract_offer_give=contract_give,
                        contract_offer_receive=contract_receive,
                    )

                # Clamp to inventory
                offerer_sent = self._clamp_give_to_inventory(offerer, offerer_execute)
                acceptor_sent = self._clamp_give_to_inventory(acceptor, acceptor_execute)

                enforcement = ContractEnforcement(
                    contract_give=contract_give,
                    contract_receive=contract_receive,
                    offerer_intended=offerer_intended,
                    acceptor_intended=acceptor_intended,
                    offerer_enforced=offerer_sent,
                    acceptor_enforced=acceptor_sent,
                    offerer_deviation=(offerer_intended != contract_give),
                    acceptor_deviation=(acceptor_intended != contract_receive),
                    enforcement_active=not self.allow_fraud,
                )

                # Record welfare BEFORE transfer
                welfare_offerer_before = offerer.welfare
                welfare_acceptor_before = acceptor.welfare
                
                self._apply_transfer(offerer, acceptor, offerer_sent, acceptor_sent)

                fraud_status = "" if self.allow_fraud else " [BINDING]"
                print(f"[TRADE]{fraud_status} Completed: {offerer.name} sent {offerer_sent}, {acceptor.name} sent {acceptor_sent}")

                self._record_trade(
                    offerer=offerer,
                    acceptor=acceptor,
                    tick=tick,
                    public_contract=pending_offer,
                    offerer_sent=offerer_sent,
                    acceptor_sent=acceptor_sent,
                    conversation=full_conversation,
                    enforcement=enforcement,
                    welfare_offerer_before=welfare_offerer_before,
                    welfare_acceptor_before=welfare_acceptor_before,
                )

                await self._run_reflection_for_pair(
                    agent_a=offerer,
                    agent_b=acceptor,
                    tick=tick,
                    outcome="completed",
                    conversation=full_conversation,
                    transfer_a=offerer_sent,
                    transfer_b=acceptor_sent,
                )
                return

        # Timeout
        print(f"[NEGOTIATION] Timeout: {a.name} <-> {b.name}")
        self._record_no_trade(a, b, tick, outcome="TIMEOUT", pending_offer=pending_offer, conversation=full_conversation)

        await self._run_reflection_for_pair(
            agent_a=a,
            agent_b=b,
            tick=tick,
            outcome="timeout",
            conversation=full_conversation,
        )

    def _parse_trade_intent(self, response: str) -> str:
        """Parse trade intent from response. Returns 'TRADE' or 'DECLINE'."""
        import re

        # Look for INTENT: line
        intent_match = re.search(r"INTENT:\s*(\w+)", response, re.IGNORECASE)
        if intent_match:
            intent = intent_match.group(1).upper()
            if intent in {"TRADE", "DECLINE"}:
                return intent

        # Fallback: look for keywords in response
        response_upper = response.upper()
        if "TRADE" in response_upper and "DECLINE" not in response_upper:
            return "TRADE"
        if "DECLINE" in response_upper or "NO TRADE" in response_upper or "DON'T WANT TO TRADE" in response_upper:
            return "DECLINE"

        # Default to DECLINE if unclear
        return "DECLINE"

    def _extract_message_from_intent(self, response: str) -> str:
        """Extract the MESSAGE portion from trade intent response."""
        import re

        message_match = re.search(r"MESSAGE:\s*(.+?)(?=INTENT:|$)", response, re.DOTALL | re.IGNORECASE)
        if message_match:
            return message_match.group(1).strip()[:500]

        # Fallback: take first non-empty line
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.upper().startswith("INTENT:"):
                return line[:500]

        return ""

    async def _negotiate_pair_legacy(self, a: SugarAgent, b: SugarAgent, tick: int) -> None:
        """Legacy negotiation protocol (original implementation)."""
        transcript: List[str] = []
        full_conversation: List[Dict[str, str]] = []
        pending_offer: Optional[Dict[str, Dict[str, int]]] = None
        pending_offer_from: Optional[int] = None
        pending_offer_private_execute: Optional[Dict[str, int]] = None
        pending_offer_public_text: str = "(None)"

        print(f"[TRADE] Negotiation started: {a.name} <-> {b.name} (tick {tick})")

        # Alternate turns: A starts.
        for round_idx in range(1, self.max_rounds + 1):
            speaker, listener = (a, b) if round_idx % 2 == 1 else (b, a)

            # Build personalized system prompt with speaker's name and identity context
            # Use speaker's goal_prompt if available, otherwise fall back to config default
            speaker_goal = getattr(speaker, 'goal_prompt', '') or self.env.config.llm_goal_prompt
            enable_survival_pressure = getattr(self.env.config, 'enable_survival_pressure', True)
            system_prompt = build_sugarscape_trade_system_prompt(
                goal_prompt=speaker_goal,
                max_rounds=self.max_rounds,
                allow_fraud=self.allow_fraud,
                agent_name=speaker.name,
                agent=speaker,
                enable_survival_pressure=enable_survival_pressure,
            )

            recent_transcript = "\n".join(transcript[-6:]) if transcript else "(No previous message)"
            partner_last_public_offer = pending_offer_public_text if pending_offer_from == listener.agent_id else "(None)"
            if pending_offer is not None and pending_offer_from == listener.agent_id:
                give_i = self._safe_int_pair(pending_offer.get("give") or {})
                receive_i = self._safe_int_pair(pending_offer.get("receive") or {})
                if not self.env.config.enable_spice:
                    give_i["spice"] = 0
                    receive_i["spice"] = 0
                partner_last_public_offer = (
                    f"{pending_offer_public_text}\n"
                    f"Interpretation for YOU: If you ACCEPT, you RECEIVE {give_i} and you MUST GIVE {receive_i}.\n"
                    f"To accept honestly, set private_execute_give = {receive_i}."
                )

            memory_summary = self._format_partner_memory(speaker, listener)
            # Get speaker's goal prompt for altruist-specific hints
            speaker_goal = getattr(speaker, 'goal_prompt', '') or ''
            turn_prompt = build_sugarscape_trade_turn_prompt(
                self_agent=speaker,
                partner_agent=listener,
                round_idx=round_idx,
                max_rounds=self.max_rounds,
                partner_last_say=recent_transcript,
                partner_last_public_offer=partner_last_public_offer,
                partner_memory_summary=memory_summary,
                env=self.env,
                self_goal_prompt=speaker_goal,
            )

            # Append learned beliefs/policies and trade history if reflection is enabled
            if getattr(self.env.config, 'enable_reflection', False):
                include_history = getattr(self.env.config, 'trade_history_in_prompt', True)
                history_limit = getattr(self.env.config, 'trade_history_prompt_limit', 10)
                social_memory_visible = getattr(self.env.config, 'social_memory_visible', True)
                beliefs_appendix = format_beliefs_policies_appendix(
                    speaker,
                    partner_id=listener.agent_id,
                    include_trade_history=include_history,
                    trade_history_limit=history_limit,
                    social_memory_visible=social_memory_visible,
                )
                if beliefs_appendix:
                    turn_prompt += beliefs_appendix

            # Two-stage mode: Stage 1 (thinking) → Stage 2 (JSON output)
            if self.two_stage:
                reply_text, parsed = await self._two_stage_generate(
                    speaker=speaker,
                    listener=listener,
                    system_prompt=system_prompt,
                    turn_prompt=turn_prompt,
                    round_idx=round_idx,
                    max_rounds=self.max_rounds,
                    pending_offer=pending_offer,
                    pending_offer_from=pending_offer_from,
                )
            else:
                # Single-stage: original behavior
                reply_text = await speaker.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": turn_prompt}],
                )
                parsed = self._parse_trade_reply(reply_text)

                # Prompt-first fix: if the model didn't produce valid/actionable JSON, ask it to restate JSON only.
                if self.repair_json and self.repair_attempts > 0:
                    parsed = await self._repair_trade_reply_if_needed(
                        parsed=parsed,
                        speaker=speaker,
                        listener=listener,
                        round_idx=round_idx,
                        max_rounds=self.max_rounds,
                        pending_offer=pending_offer,
                        pending_offer_from=pending_offer_from,
                        system_prompt=system_prompt,
                    )

            # Optional last-resort fallback: coerce protocol to prevent TIMEOUT.
            if self.coerce_protocol:
                parsed = self._coerce_trade_reply(
                    parsed=parsed,
                    speaker=speaker,
                    listener=listener,
                    round_idx=round_idx,
                    max_rounds=self.max_rounds,
                    pending_offer=pending_offer,
                    pending_offer_from=pending_offer_from,
                )
            transcript.append(f"{speaker.name}: {parsed.say}")

            # Store full conversation in both agents' history
            trade_turn = {
                "type": "trade_dialogue",
                "tick": tick,
                "round": round_idx,
                "speaker": speaker.name,
                "partner": listener.name,
                "prompt": turn_prompt,
                "response": reply_text,
                "intent": parsed.intent,
                "public_offer": parsed.public_offer if parsed.intent in ("OFFER", "REJECT") else None,
            }
            full_conversation.append(trade_turn)

            # Append to agents' conversation history if they have one
            if hasattr(speaker, 'conversation_history'):
                speaker.conversation_history.append({"role": "user", "content": f"[TRADE with {listener.name}] {turn_prompt}"})
                speaker.conversation_history.append({"role": "assistant", "content": reply_text})

            if parsed.intent == "OFFER":
                pending_offer = parsed.public_offer
                pending_offer_from = speaker.agent_id
                pending_offer_private_execute = parsed.private_execute_give
                pending_offer_public_text = self._format_public_offer_text(speaker, pending_offer)
                continue

            if parsed.intent in {"REJECT", "WALK_AWAY"}:
                print(f"[TRADE] {speaker.name} {parsed.intent}ed negotiation with {listener.name}")
                self._record_no_trade(
                    a, b, tick,
                    outcome=parsed.intent,
                    pending_offer=pending_offer,
                    conversation=full_conversation,
                    decider_id=speaker.agent_id,
                    decider_name=speaker.name,
                )

                # Post-encounter reflection for rejections/walkways
                await self._run_reflection_for_pair(
                    agent_a=a,
                    agent_b=b,
                    tick=tick,
                    outcome=parsed.intent.lower(),
                    conversation=full_conversation,
                )
                return

            if parsed.intent == "ACCEPT":
                if pending_offer is None or pending_offer_from != listener.agent_id:
                    continue

                offerer = listener
                acceptor = speaker

                # Extract contract terms
                contract_give = self._safe_int_pair((pending_offer or {}).get("give", {}))
                contract_receive = self._safe_int_pair((pending_offer or {}).get("receive", {}))

                # Track what agents intended to execute
                offerer_intended = self._safe_int_pair(pending_offer_private_execute or {})
                acceptor_intended = self._safe_int_pair(parsed.private_execute_give or {})

                if self.allow_fraud:
                    # Fraud allowed: use private_execute_give (can differ from contract)
                    offerer_execute = offerer_intended
                    acceptor_execute = acceptor_intended

                    # Heuristic guardrail for direction confusion
                    acceptor_execute = self._fix_accept_execute_direction_confusion(
                        acceptor_execute=acceptor_execute,
                        contract_offer_give=contract_give,
                        contract_offer_receive=contract_receive,
                    )
                else:
                    # NO-FRAUD ENFORCEMENT: Contracts are binding.
                    # Execute exactly what the public contract specifies, regardless of
                    # what agents put in private_execute_give.
                    # Offerer gives public_offer.give, acceptor gives public_offer.receive.
                    offerer_execute = contract_give.copy()
                    acceptor_execute = contract_receive.copy()

                    # Log enforcement if agents would have deviated
                    offerer_deviation = (offerer_intended != contract_give)
                    acceptor_deviation = (acceptor_intended != contract_receive)

                    if offerer_deviation or acceptor_deviation:
                        print(f"[CONTRACT ENFORCEMENT] Binding contract enforced:")
                        if offerer_deviation:
                            print(f"  - {offerer.name} intended {offerer_intended}, contract enforced {contract_give}")
                        if acceptor_deviation:
                            print(f"  - {acceptor.name} intended {acceptor_intended}, contract enforced {contract_receive}")

                # Clamp to available inventory (can't give what you don't have)
                offerer_sent = self._clamp_give_to_inventory(offerer, offerer_execute)
                acceptor_sent = self._clamp_give_to_inventory(acceptor, acceptor_execute)

                # Build enforcement record for logging
                enforcement = ContractEnforcement(
                    contract_give=contract_give,
                    contract_receive=contract_receive,
                    offerer_intended=offerer_intended,
                    acceptor_intended=acceptor_intended,
                    offerer_enforced=offerer_sent,
                    acceptor_enforced=acceptor_sent,
                    offerer_deviation=(offerer_intended != contract_give),
                    acceptor_deviation=(acceptor_intended != contract_receive),
                    enforcement_active=not self.allow_fraud,
                )

                # Record welfare BEFORE transfer
                welfare_offerer_before = offerer.welfare
                welfare_acceptor_before = acceptor.welfare
                
                self._apply_transfer(offerer, acceptor, offerer_sent, acceptor_sent)

                fraud_status = "" if self.allow_fraud else " [BINDING]"
                print(f"[TRADE]{fraud_status} Trade completed: {offerer.name} sent {offerer_sent}, {acceptor.name} sent {acceptor_sent}")
                self._record_trade(
                    offerer=offerer,
                    acceptor=acceptor,
                    tick=tick,
                    public_contract=pending_offer,
                    offerer_sent=offerer_sent,
                    acceptor_sent=acceptor_sent,
                    conversation=full_conversation,
                    enforcement=enforcement,
                    welfare_offerer_before=welfare_offerer_before,
                    welfare_acceptor_before=welfare_acceptor_before,
                )

                # Post-encounter reflection for belief/policy updates
                await self._run_reflection_for_pair(
                    agent_a=offerer,
                    agent_b=acceptor,
                    tick=tick,
                    outcome="completed",
                    conversation=full_conversation,
                    transfer_a=offerer_sent,
                    transfer_b=acceptor_sent,
                )
                return

        print(f"[TRADE] Negotiation timed out: {a.name} <-> {b.name}")
        self._record_no_trade(a, b, tick, outcome="TIMEOUT", pending_offer=pending_offer, conversation=full_conversation)

        # Post-encounter reflection even for timeouts (agents learn from failed negotiations)
        await self._run_reflection_for_pair(
            agent_a=a,
            agent_b=b,
            tick=tick,
            outcome="timeout",
            conversation=full_conversation,
        )

    def _extract_reasoning_conclusion(self, reasoning: str, max_chars: int = 300) -> str:
        """Extract conclusion from reasoning (Optimization #3).

        Instead of passing full reasoning to Stage 2, extract just the key conclusion
        to reduce tokens and focus the model on the decision.
        """
        import re

        # Strip thinking tags from Qwen3/DeepSeek models
        reasoning = re.sub(r'<think>.*?</think>', '', reasoning, flags=re.DOTALL)
        reasoning = re.sub(r'<think>.*$', '', reasoning, flags=re.DOTALL)
        reasoning = reasoning.strip()

        if not reasoning:
            return ""

        # Try to find conclusion markers
        conclusion_patterns = [
            r'(?:conclusion|decision|therefore|so I will|I should|my offer|I\'ll offer)[:\s]*(.*)',
            r'(?:In summary|To summarize|Overall)[:\s]*(.*)',
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, reasoning, re.IGNORECASE | re.DOTALL)
            if match:
                conclusion = match.group(1).strip()
                if len(conclusion) > 50:  # Only if substantial
                    return conclusion[:max_chars]

        # Fallback: take last paragraph (usually contains conclusion)
        paragraphs = [p.strip() for p in reasoning.split('\n\n') if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            if len(last_para) > max_chars:
                # Take last N chars
                return "..." + last_para[-max_chars:]
            return last_para

        # Final fallback: last N chars
        if len(reasoning) > max_chars:
            return "..." + reasoning[-max_chars:]
        return reasoning

    async def _two_stage_generate(
        self,
        speaker: SugarAgent,
        listener: SugarAgent,
        system_prompt: str,
        turn_prompt: str,
        round_idx: int,
        max_rounds: int,
        pending_offer: Optional[Dict[str, Dict[str, int]]],
        pending_offer_from: Optional[int],
    ) -> Tuple[str, "_ParsedTradeReply"]:
        """Two-stage generation: Stage 1 (thinking) → Stage 2 (JSON output).

        Optimizations applied:
        - #3: Reasoning compression - only conclusion passed to Stage 2
        - #4: Pre-computed template for Stage 2 prompt

        Stage 1: Call model with thinking enabled to get reasoning.
        Stage 2: Call model with compressed reasoning, asking for JSON-only output.

        Returns:
            Tuple of (combined_reply_text, parsed_reply)
        """
        import re

        # Stage 1: Get reasoning (with limited thinking tokens)
        try:
            stage1_response = await speaker.provider.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": turn_prompt}],
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            stage1_response = await speaker.provider.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": turn_prompt}],
            )

        # Extract thinking content (handle multiple formats)
        # Format 1: __THINKING_START__...__THINKING_END__
        thinking_match = re.search(
            r'__THINKING_START__\s*(.*?)\s*__THINKING_END__',
            stage1_response,
            re.DOTALL
        )
        # Format 2: <think>...</think> (Qwen3)
        if not thinking_match:
            thinking_match = re.search(r'<think>(.*?)</think>', stage1_response, re.DOTALL)

        if thinking_match:
            reasoning = thinking_match.group(1).strip()
        else:
            # No thinking block, use the whole response as reasoning
            reasoning = stage1_response.strip()

        # Try to parse JSON from Stage 1 first (maybe it completed successfully)
        parsed = self._parse_trade_reply(stage1_response)
        intent = (parsed.intent or "CHAT").strip().upper()

        # Check if Stage 1 produced valid actionable intent
        has_active_offer = pending_offer is not None and pending_offer_from == listener.agent_id
        if has_active_offer:
            valid_intents = {"ACCEPT", "REJECT", "WALK_AWAY"}
        else:
            valid_intents = {"OFFER", "WALK_AWAY"}

        if intent in valid_intents:
            # Stage 1 succeeded, no need for Stage 2
            return stage1_response, parsed

        # Stage 2: Ask for JSON-only output with compressed reasoning (Optimization #3)
        compressed_reasoning = self._extract_reasoning_conclusion(reasoning)

        offer_context = "(None)"
        if pending_offer is not None and pending_offer_from == listener.agent_id:
            give_i = self._safe_int_pair(pending_offer.get("give") or {})
            recv_i = self._safe_int_pair(pending_offer.get("receive") or {})
            if not self.env.config.enable_spice:
                give_i["spice"] = 0
                recv_i["spice"] = 0
            offer_context = (
                f"Active offer from {listener.name}: give={give_i}, receive={recv_i}. "
                f"If ACCEPT, you SEND {recv_i}."
            )

        # Use pre-computed template (Optimization #4)
        stage2_prompt = self.STAGE2_TEMPLATE.format(
            reasoning=compressed_reasoning,
            round_idx=round_idx,
            max_rounds=max_rounds,
            offer_context=offer_context,
        )

        # Stage 2 call with thinking disabled
        # Pass chat_template_kwargs to disable thinking for Qwen3 models
        # NOTE: only some providers (e.g. vLLM) support `chat_template_kwargs`.
        # Fall back cleanly for providers like OpenRouter/LoadBalancedProvider.
        try:
            stage2_response = await speaker.provider.generate(
                system_prompt="Output valid JSON only.",
                messages=[{"role": "user", "content": stage2_prompt}],
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError as e:
            # Only fall back when the provider rejects the kwarg; don't mask real TypeErrors.
            msg = str(e)
            if ("chat_template_kwargs" not in msg) and ("unexpected keyword" not in msg):
                raise
            stage2_response = await speaker.provider.generate(
                system_prompt="Output valid JSON only.",
                messages=[{"role": "user", "content": stage2_prompt}],
            )

        # Parse Stage 2 response
        parsed2 = self._parse_trade_reply(stage2_response)
        intent2 = (parsed2.intent or "CHAT").strip().upper()

        if intent2 in valid_intents:
            # Combine responses for logging
            combined = f"[Stage 1 Reasoning]\n{reasoning[:500]}...\n\n[Stage 2 Decision]\n{stage2_response}"
            return combined, parsed2

        # Stage 2 also failed, return Stage 1 result (will likely cause TIMEOUT)
        return stage1_response, parsed

    async def _repair_trade_reply_if_needed(
        self,
        parsed: _ParsedTradeReply,
        speaker: SugarAgent,
        listener: SugarAgent,
        round_idx: int,
        max_rounds: int,
        pending_offer: Optional[Dict[str, Dict[str, int]]],
        pending_offer_from: Optional[int],
        system_prompt: str,
    ) -> _ParsedTradeReply:
        """If the reply isn't actionable, ask the model to output ONLY valid JSON (no prose)."""
        intent = (parsed.intent or "CHAT").strip().upper()
        has_active_offer_from_listener = pending_offer is not None and pending_offer_from == listener.agent_id

        # When should we repair?
        # - model replied CHAT when we *need* an OFFER or ACCEPT/REJECT
        # - model replied ACCEPT but there is no active offer from listener
        needs_offer = (not has_active_offer_from_listener) and intent != "OFFER"
        needs_decision = has_active_offer_from_listener and intent not in {"ACCEPT", "REJECT", "WALK_AWAY"}
        bad_accept = intent == "ACCEPT" and not has_active_offer_from_listener

        if not (needs_offer or needs_decision or bad_accept):
            return parsed

        # Build a short corrective prompt; keep it minimal to reduce token waste.
        offer_context = "(None)"
        if pending_offer is not None and pending_offer_from == listener.agent_id:
            give_i = self._safe_int_pair(pending_offer.get("give") or {})
            recv_i = self._safe_int_pair(pending_offer.get("receive") or {})
            if not self.env.config.enable_spice:
                give_i["spice"] = 0
                recv_i["spice"] = 0
            offer_context = (
                f"Active offer from {listener.name}: give={give_i}, receive={recv_i}. "
                f"If you ACCEPT honestly, you must SEND receive={recv_i}."
            )

        repair_prompt = (
            "Your previous message did not follow the protocol.\n"
            "Return ONLY a single valid JSON object matching this schema and nothing else:\n"
            "{\n"
            '  \"intent\": \"OFFER\" | \"ACCEPT\" | \"REJECT\" | \"WALK_AWAY\",\n'
            '  \"public_offer\": {\"give\": {\"sugar\": int, \"spice\": int}, \"receive\": {\"sugar\": int, \"spice\": int}},\n'
            '  \"private_execute_give\": {\"sugar\": int, \"spice\": int}\n'
            "}\n"
            f"Round {round_idx} of {max_rounds}. {offer_context}\n"
            "Rules:\n"
            "- If there is an active offer, choose ACCEPT or REJECT (or WALK_AWAY).\n"
            "- If there is no active offer, choose OFFER (or WALK_AWAY).\n"
            "- If intent != OFFER, set public_offer give/receive to 0.\n"
            "- private_execute_give is what YOU SEND.\n"
        )

        # Try a small number of times (usually 1 is enough).
        for _ in range(self.repair_attempts):
            try:
                reply_text = await speaker.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": repair_prompt}],
                )
            except Exception:
                break
            repaired = self._parse_trade_reply(reply_text)
            repaired_intent = (repaired.intent or "CHAT").strip().upper()
            if has_active_offer_from_listener:
                if repaired_intent in {"ACCEPT", "REJECT", "WALK_AWAY"}:
                    return repaired
            else:
                if repaired_intent in {"OFFER", "WALK_AWAY"}:
                    return repaired

        return parsed

    def _coerce_trade_reply(
        self,
        parsed: _ParsedTradeReply,
        speaker: SugarAgent,
        listener: SugarAgent,
        round_idx: int,
        max_rounds: int,
        pending_offer: Optional[Dict[str, Dict[str, int]]],
        pending_offer_from: Optional[int],
    ) -> _ParsedTradeReply:
        """Guardrail against 'analysis paralysis' / malformed replies.

        Many models sometimes output intent=CHAT (or omit JSON) even when the protocol demands
        OFFER/ACCEPT/REJECT. In this simulation, CHAT is a no-op and leads to TIMEOUT.

        Policy:
        - If there's an active offer from the *listener*, force ACCEPT or REJECT.
        - If there is no active offer, force OFFER (or WALK_AWAY on final round if we truly can't make one).
        """
        intent = (parsed.intent or "CHAT").strip().upper()

        has_active_offer_from_listener = (
            pending_offer is not None and pending_offer_from == listener.agent_id
        )

        if has_active_offer_from_listener:
            # Coerce to ACCEPT/REJECT based on whether honest acceptance is welfare-improving.
            if intent not in {"ACCEPT", "REJECT", "WALK_AWAY"}:
                intent = "ACCEPT" if self._should_accept_offer(speaker, pending_offer) else "REJECT"
        else:
            # No active offer: ensure we make an offer rather than chatting ourselves into TIMEOUT.
            if intent not in {"OFFER", "WALK_AWAY"}:
                intent = "OFFER"

        if intent == "OFFER":
            offer = self._default_public_offer(speaker)
            exec_give = self._safe_int_pair(offer.get("give") or {})
            if not self.env.config.enable_spice:
                exec_give["spice"] = 0
            return _ParsedTradeReply(
                thought=parsed.thought,
                say=parsed.say or "Proposing a simple trade to rebalance resources.",
                intent="OFFER",
                public_offer=offer,
                private_execute_give=exec_give,
            )

        if intent == "ACCEPT" and has_active_offer_from_listener and pending_offer is not None:
            # On ACCEPT, private_execute_give should be what speaker SENDS.
            # If fraud disabled, later code will enforce contract; if fraud enabled, this is an honest default.
            to_send = self._safe_int_pair((pending_offer.get("receive") or {}))
            if not self.env.config.enable_spice:
                to_send["spice"] = 0
            return _ParsedTradeReply(
                thought=parsed.thought,
                say=parsed.say or "I accept your offer.",
                intent="ACCEPT",
                public_offer={"give": {"sugar": 0, "spice": 0}, "receive": {"sugar": 0, "spice": 0}},
                private_execute_give=to_send,
            )

        if intent in {"REJECT", "WALK_AWAY"}:
            # Preserve counter-offer if provided (for REJECT with counter-offer)
            counter_offer = parsed.public_offer
            if not counter_offer or (counter_offer.get("give", {}).get("sugar", 0) == 0 and 
                                      counter_offer.get("give", {}).get("spice", 0) == 0 and
                                      counter_offer.get("receive", {}).get("sugar", 0) == 0 and
                                      counter_offer.get("receive", {}).get("spice", 0) == 0):
                counter_offer = {"give": {"sugar": 0, "spice": 0}, "receive": {"sugar": 0, "spice": 0}}
            return _ParsedTradeReply(
                thought=parsed.thought,
                say=parsed.say or ("I reject." if intent == "REJECT" else "No deal."),
                intent=intent,
                public_offer=counter_offer,
                private_execute_give={"sugar": 0, "spice": 0},
            )

        # Fallback: keep original.
        return parsed

    def _should_accept_offer(self, agent: SugarAgent, offer: Dict[str, Dict[str, int]]) -> bool:
        """Decide whether to accept an offer assuming honest execution (Pareto-like for self)."""
        give_to_me = self._safe_int_pair((offer.get("give") or {}))
        i_send = self._safe_int_pair((offer.get("receive") or {}))
        if not self.env.config.enable_spice:
            give_to_me["spice"] = 0
            i_send["spice"] = 0

        # Must be affordable to be meaningful.
        if i_send["sugar"] > int(agent.wealth):
            return False
        if self.env.config.enable_spice and i_send["spice"] > int(agent.spice):
            return False

        w0 = self._calc_welfare(agent, float(agent.wealth), float(agent.spice))
        w1 = self._calc_welfare(
            agent,
            float(agent.wealth - i_send["sugar"] + give_to_me["sugar"]),
            float(agent.spice - i_send["spice"] + give_to_me["spice"]),
        )
        # Accept if it doesn't hurt and provides a meaningful improvement.
        return w1 >= w0 * 1.0001

    def _calc_welfare(self, agent: SugarAgent, sugar: float, spice: float) -> float:
        sugar = max(0.0, float(sugar))
        spice = max(0.0, float(spice))
        if not self.env.config.enable_spice or agent.metabolism_spice <= 0:
            return sugar
        m_s = max(0.0, float(agent.metabolism))
        m_p = max(0.0, float(agent.metabolism_spice))
        m_t = max(1e-9, m_s + m_p)
        # Cobb-Douglas welfare used elsewhere in this module.
        return (sugar ** (m_s / m_t)) * (spice ** (m_p / m_t))

    def _default_public_offer(self, agent: SugarAgent) -> Dict[str, Dict[str, int]]:
        """Generate a simple, safe offer based only on self state (no opponent peeking)."""
        sugar = int(agent.wealth)
        spice = int(agent.spice)

        # Target buffer: ~10 timesteps for each required resource (if metabolism > 0).
        sugar_target = int(max(0, agent.metabolism) * 10) if agent.metabolism > 0 else 0
        spice_target = int(max(0, agent.metabolism_spice) * 10) if agent.metabolism_spice > 0 else 0

        sugar_need = max(0, sugar_target - sugar) if sugar_target else 0
        spice_need = max(0, spice_target - spice) if (self.env.config.enable_spice and spice_target) else 0

        # Choose what to ask for.
        want_spice = self.env.config.enable_spice and (spice_need > sugar_need)

        if want_spice:
            # Give sugar for spice.
            give_sugar = max(1, min(10, sugar // 5)) if sugar > 0 else 0
            receive_spice = max(1, min(10, (spice_need if spice_need > 0 else 2 * give_sugar)))
            offer = {
                "give": {"sugar": give_sugar, "spice": 0},
                "receive": {"sugar": 0, "spice": receive_spice},
            }
        else:
            # Give spice for sugar (or if spice disabled, give nothing and ask sugar is meaningless; fall back to walk away).
            if not self.env.config.enable_spice:
                offer = {"give": {"sugar": 0, "spice": 0}, "receive": {"sugar": 1, "spice": 0}}
            else:
                give_spice = max(1, min(10, spice // 5)) if spice > 0 else 0
                receive_sugar = max(1, min(10, (sugar_need if sugar_need > 0 else 2 * give_spice)))
                offer = {
                    "give": {"sugar": 0, "spice": give_spice},
                    "receive": {"sugar": receive_sugar, "spice": 0},
                }

        # Ensure ints and non-negative.
        give_i = self._safe_int_pair(offer.get("give") or {})
        recv_i = self._safe_int_pair(offer.get("receive") or {})
        if not self.env.config.enable_spice:
            give_i["spice"] = 0
            recv_i["spice"] = 0
        return {"give": give_i, "receive": recv_i}

    def _clamp_give_to_inventory(self, agent: SugarAgent, give: Dict[str, int]) -> Dict[str, int]:
        sugar = max(0, int(give.get("sugar", 0)))
        spice = max(0, int(give.get("spice", 0))) if self.env.config.enable_spice else 0
        sugar = min(sugar, int(agent.wealth))
        spice = min(spice, int(agent.spice)) if self.env.config.enable_spice else 0
        return {"sugar": sugar, "spice": spice}

    def _apply_transfer(
        self,
        offerer: SugarAgent,
        acceptor: SugarAgent,
        offerer_sent: Dict[str, int],
        acceptor_sent: Dict[str, int],
    ) -> None:
        offerer.wealth -= offerer_sent["sugar"]
        acceptor.wealth += offerer_sent["sugar"]
        if self.env.config.enable_spice:
            offerer.spice -= offerer_sent["spice"]
            acceptor.spice += offerer_sent["spice"]

        acceptor.wealth -= acceptor_sent["sugar"]
        offerer.wealth += acceptor_sent["sugar"]
        if self.env.config.enable_spice:
            acceptor.spice -= acceptor_sent["spice"]
            offerer.spice += acceptor_sent["spice"]

    def _record_trade(
        self,
        offerer: SugarAgent,
        acceptor: SugarAgent,
        tick: int,
        public_contract: Dict[str, Dict[str, int]],
        offerer_sent: Dict[str, int],
        acceptor_sent: Dict[str, int],
        conversation: Optional[List[Dict[str, str]]] = None,
        enforcement: Optional[ContractEnforcement] = None,
        welfare_offerer_before: Optional[float] = None,
        welfare_acceptor_before: Optional[float] = None,
    ) -> None:
        contract_offer_give = public_contract.get("give", {"sugar": 0, "spice": 0})
        contract_offer_receive = public_contract.get("receive", {"sugar": 0, "spice": 0})

        offerer_received = acceptor_sent
        acceptor_received = offerer_sent

        offerer_event = {
            "tick": tick,
            "partner_id": acceptor.agent_id,
            "partner_name": acceptor.name,
            "type": "TRADE",
            "public_contract": {
                "give": self._safe_int_pair(contract_offer_give),
                "receive": self._safe_int_pair(contract_offer_receive),
            },
            "actual": {"sent": offerer_sent, "received": offerer_received},
            "conversation": conversation or [],
        }
        acceptor_event = {
            "tick": tick,
            "partner_id": offerer.agent_id,
            "partner_name": offerer.name,
            "type": "TRADE",
            "public_contract": {
                "give": self._safe_int_pair(contract_offer_receive),
                "receive": self._safe_int_pair(contract_offer_give),
            },
            "actual": {"sent": acceptor_sent, "received": acceptor_received},
            "conversation": conversation or [],
        }

        offerer.get_partner_trade_log(acceptor.agent_id, maxlen=self.memory_maxlen).append(offerer_event)
        acceptor.get_partner_trade_log(offerer.agent_id, maxlen=self.memory_maxlen).append(acceptor_event)

        self._update_trust_from_contract(
            self_agent=offerer,
            partner=acceptor,
            expected_receive=offerer_event["public_contract"]["receive"],
            actual_receive=offerer_received,
        )
        self._update_trust_from_contract(
            self_agent=acceptor,
            partner=offerer,
            expected_receive=acceptor_event["public_contract"]["receive"],
            actual_receive=acceptor_received,
        )

        # Detect deception: did either party send different from what they publicly offered?
        deception_detected = (
            offerer_sent != self._safe_int_pair(contract_offer_give) or
            acceptor_sent != self._safe_int_pair(contract_offer_receive)
        )

        # === EVENT-TRIGGERED REFLECTION: Record significant events ===
        try:
            from sugarscape.llm_agent import LLMSugarAgent

            # Check for fraud (deception detected)
            if deception_detected:
                # Check who was defrauded
                offerer_expected = self._safe_int_pair(contract_offer_receive)
                offerer_got = offerer_received
                offerer_cheated = (
                    offerer_got.get("sugar", 0) < offerer_expected.get("sugar", 0) * 0.9 or
                    offerer_got.get("spice", 0) < offerer_expected.get("spice", 0) * 0.9
                )

                acceptor_expected = self._safe_int_pair(contract_offer_give)
                acceptor_got = acceptor_received
                acceptor_cheated = (
                    acceptor_got.get("sugar", 0) < acceptor_expected.get("sugar", 0) * 0.9 or
                    acceptor_got.get("spice", 0) < acceptor_expected.get("spice", 0) * 0.9
                )

                if offerer_cheated and isinstance(offerer, LLMSugarAgent):
                    amount_lost = (
                        (offerer_expected.get("sugar", 0) - offerer_got.get("sugar", 0)) +
                        (offerer_expected.get("spice", 0) - offerer_got.get("spice", 0))
                    )
                    offerer.record_reflection_event("defrauded", tick, {
                        "partner_name": acceptor.name,
                        "partner_id": acceptor.agent_id,
                        "amount_lost": max(0, amount_lost),
                    })

                if acceptor_cheated and isinstance(acceptor, LLMSugarAgent):
                    amount_lost = (
                        (acceptor_expected.get("sugar", 0) - acceptor_got.get("sugar", 0)) +
                        (acceptor_expected.get("spice", 0) - acceptor_got.get("spice", 0))
                    )
                    acceptor.record_reflection_event("defrauded", tick, {
                        "partner_name": offerer.name,
                        "partner_id": offerer.agent_id,
                        "amount_lost": max(0, amount_lost),
                    })

            # Record successful cooperation for both parties
            else:
                if isinstance(offerer, LLMSugarAgent):
                    offerer.record_reflection_event("successful_cooperation", tick, {
                        "partner_name": acceptor.name,
                        "partner_id": acceptor.agent_id,
                    })
                if isinstance(acceptor, LLMSugarAgent):
                    acceptor.record_reflection_event("successful_cooperation", tick, {
                        "partner_name": offerer.name,
                        "partner_id": offerer.agent_id,
                    })
        except ImportError:
            pass  # LLMSugarAgent not available

        # Capture urgency and location (post-trade state)
        offerer_urgency = self._get_agent_urgency(offerer)
        acceptor_urgency = self._get_agent_urgency(acceptor)
        offerer_location = self.env.get_location_context(offerer.pos)
        acceptor_location = self.env.get_location_context(acceptor.pos)

        # Get reputation before updates
        rep_offerer_before = self.env.get_agent_reputation(offerer.agent_id, 0.5)
        rep_acceptor_before = self.env.get_agent_reputation(acceptor.agent_id, 0.5)

        # Update global reputation for both agents
        self._update_reputation(offerer, acceptor, deception_detected, trade_completed=True)
        self._update_reputation(acceptor, offerer, deception_detected, trade_completed=True)

        # Get reputation after updates
        rep_offerer_after = self.env.get_agent_reputation(offerer.agent_id, 0.5)
        rep_acceptor_after = self.env.get_agent_reputation(acceptor.agent_id, 0.5)

        # Log to debug logger
        if self.env.debug_logger:
            from sugarscape.debug_logger import TradeRecord

            # Calculate welfare (agent.welfare property uses Cobb-Douglas utility)
            # "after" values are current (post-transfer), "before" values passed in
            welfare_offerer_after = offerer.welfare
            welfare_acceptor_after = acceptor.welfare
            # Use passed-in before values, or fall back to after (for backwards compatibility)
            if welfare_offerer_before is None:
                welfare_offerer_before = welfare_offerer_after
            if welfare_acceptor_before is None:
                welfare_acceptor_before = welfare_acceptor_after

            sugar_exchanged = offerer_sent.get("sugar", 0)
            spice_exchanged = offerer_sent.get("spice", 0)

            # Get goal presets for analysis
            offerer_goal = getattr(offerer, 'goal_prompt', '') or ''
            acceptor_goal = getattr(acceptor, 'goal_prompt', '') or ''

            # Detect gifts (giving with nothing in return)
            contract_receive_offerer = public_contract.get("receive", {})
            contract_receive_acceptor = public_contract.get("give", {})
            is_gift_offerer = (
                (offerer_sent.get("sugar", 0) > 0 or offerer_sent.get("spice", 0) > 0) and
                contract_receive_offerer.get("sugar", 0) == 0 and
                contract_receive_offerer.get("spice", 0) == 0
            )
            is_gift_acceptor = (
                (acceptor_sent.get("sugar", 0) > 0 or acceptor_sent.get("spice", 0) > 0) and
                contract_receive_acceptor.get("sugar", 0) == 0 and
                contract_receive_acceptor.get("spice", 0) == 0
            )

            # Check if gift hint was shown (altruist + partner CRITICAL)
            altruist_keywords = ["care about others", "help", "altruist", "everyone deserves"]
            offerer_is_altruist = any(kw in offerer_goal.lower() for kw in altruist_keywords)
            acceptor_is_altruist = any(kw in acceptor_goal.lower() for kw in altruist_keywords)
            gift_hint_offerer = offerer_is_altruist and acceptor_urgency == "CRITICAL"
            gift_hint_acceptor = acceptor_is_altruist and offerer_urgency == "CRITICAL"

            # Contract enforcement info
            contract_enforced = False
            offerer_intended = {}
            acceptor_intended = {}
            offerer_would_deviate = False
            acceptor_would_deviate = False
            if enforcement is not None:
                contract_enforced = enforcement.enforcement_active
                offerer_intended = enforcement.offerer_intended
                acceptor_intended = enforcement.acceptor_intended
                offerer_would_deviate = enforcement.offerer_deviation
                acceptor_would_deviate = enforcement.acceptor_deviation

            trade_record = TradeRecord(
                tick=tick,
                agent_a_id=offerer.agent_id,
                agent_a_name=offerer.name,
                agent_b_id=acceptor.agent_id,
                agent_b_name=acceptor.name,
                outcome="completed",
                price=0.0,  # MRS price not applicable to dialogue trades
                sugar_exchanged=sugar_exchanged,
                spice_exchanged=spice_exchanged,
                public_offer=public_contract,
                actual_transfer_a=offerer_sent,
                actual_transfer_b=acceptor_sent,
                deception_detected=deception_detected,
                welfare_a_before=welfare_offerer_before,
                welfare_a_after=welfare_offerer_after,
                welfare_b_before=welfare_acceptor_before,
                welfare_b_after=welfare_acceptor_after,
                conversation=conversation or [],
                # Reputation/urgency/location
                agent_a_urgency=offerer_urgency,
                agent_b_urgency=acceptor_urgency,
                agent_a_location=offerer_location,
                agent_b_location=acceptor_location,
                agent_a_pos=offerer.pos,
                agent_b_pos=acceptor.pos,
                reputation_a_before=rep_offerer_before,
                reputation_a_after=rep_offerer_after,
                reputation_b_before=rep_acceptor_before,
                reputation_b_after=rep_acceptor_after,
                # Goal and gift tracking
                agent_a_goal=offerer_goal[:50],  # Truncate for CSV
                agent_b_goal=acceptor_goal[:50],
                is_gift_a=is_gift_offerer,
                is_gift_b=is_gift_acceptor,
                gift_hint_shown_a=gift_hint_offerer,
                gift_hint_shown_b=gift_hint_acceptor,
                # Contract enforcement
                contract_enforced=contract_enforced,
                agent_a_intended=offerer_intended,
                agent_b_intended=acceptor_intended,
                agent_a_would_deviate=offerer_would_deviate,
                agent_b_would_deviate=acceptor_would_deviate,
            )
            self.env.debug_logger.log_trade(trade_record)

    def _record_no_trade(
        self,
        a: SugarAgent,
        b: SugarAgent,
        tick: int,
        outcome: str,
        pending_offer: Optional[Dict[str, Dict[str, int]]],
        conversation: Optional[List[Dict[str, str]]] = None,
        decider_id: Optional[int] = None,
        decider_name: Optional[str] = None,
    ) -> None:
        pending = pending_offer or {"give": {"sugar": 0, "spice": 0}, "receive": {"sugar": 0, "spice": 0}}
        a.get_partner_trade_log(b.agent_id, maxlen=self.memory_maxlen).append(
            {
                "tick": tick,
                "partner_id": b.agent_id,
                "partner_name": b.name,
                "type": "NO_TRADE",
                "outcome": outcome,
                "pending_offer": pending,
                "decider_id": decider_id,
                "decider_name": decider_name,
                "conversation": conversation or [],
            }
        )
        b.get_partner_trade_log(a.agent_id, maxlen=self.memory_maxlen).append(
            {
                "tick": tick,
                "partner_id": a.agent_id,
                "partner_name": a.name,
                "type": "NO_TRADE",
                "outcome": outcome,
                "pending_offer": pending,
                "decider_id": decider_id,
                "decider_name": decider_name,
                "conversation": conversation or [],
            }
        )

        # === EVENT-TRIGGERED REFLECTION: Record rejection events ===
        try:
            from sugarscape.llm_agent import LLMSugarAgent

            if outcome in ["REJECT", "WALK_AWAY", "BOTH_DECLINED"]:
                # Record rejection for both parties
                if isinstance(a, LLMSugarAgent):
                    a.record_reflection_event("trade_rejected", tick, {
                        "partner_name": b.name,
                        "partner_id": b.agent_id,
                        "reason": outcome,
                    })
                if isinstance(b, LLMSugarAgent):
                    b.record_reflection_event("trade_rejected", tick, {
                        "partner_name": a.name,
                        "partner_id": a.agent_id,
                        "reason": outcome,
                    })
            elif outcome == "EXCLUSION":
                # Record being excluded
                if isinstance(a, LLMSugarAgent):
                    a.record_reflection_event("trade_rejected", tick, {
                        "partner_name": b.name,
                        "partner_id": b.agent_id,
                        "reason": "excluded_by_partner",
                    })
                if isinstance(b, LLMSugarAgent):
                    b.record_reflection_event("trade_rejected", tick, {
                        "partner_name": a.name,
                        "partner_id": a.agent_id,
                        "reason": "excluded_by_partner",
                    })
        except ImportError:
            pass  # LLMSugarAgent not available

        # Log to debug logger
        if self.env.debug_logger:
            from sugarscape.debug_logger import TradeRecord

            welfare_a = a.welfare
            welfare_b = b.welfare

            # Get goal presets for analysis
            goal_a = getattr(a, 'goal_prompt', '') or ''
            goal_b = getattr(b, 'goal_prompt', '') or ''

            # Get urgency and location for logging
            urgency_a = self._get_agent_urgency(a)
            urgency_b = self._get_agent_urgency(b)
            location_a = self.env.get_location_context(a.pos)
            location_b = self.env.get_location_context(b.pos)

            # Check if gift hint would have been shown
            altruist_keywords = ["care about others", "help", "altruist", "everyone deserves"]
            a_is_altruist = any(kw in goal_a.lower() for kw in altruist_keywords)
            b_is_altruist = any(kw in goal_b.lower() for kw in altruist_keywords)
            gift_hint_a = a_is_altruist and urgency_b == "CRITICAL"
            gift_hint_b = b_is_altruist and urgency_a == "CRITICAL"

            trade_record = TradeRecord(
                tick=tick,
                agent_a_id=a.agent_id,
                agent_a_name=a.name,
                agent_b_id=b.agent_id,
                agent_b_name=b.name,
                outcome=outcome.lower(),  # "timeout", "reject", "walk_away", "error"
                price=0.0,
                sugar_exchanged=0,
                spice_exchanged=0,
                public_offer=pending,
                actual_transfer_a={},
                actual_transfer_b={},
                deception_detected=False,
                welfare_a_before=welfare_a,
                welfare_a_after=welfare_a,
                welfare_b_before=welfare_b,
                welfare_b_after=welfare_b,
                conversation=conversation or [],
                # Urgency/location/position - FIX: was missing these fields
                agent_a_urgency=urgency_a,
                agent_b_urgency=urgency_b,
                agent_a_location=location_a,
                agent_b_location=location_b,
                agent_a_pos=a.pos,
                agent_b_pos=b.pos,
                # Goal and gift tracking
                agent_a_goal=goal_a[:50],
                agent_b_goal=goal_b[:50],
                is_gift_a=False,
                is_gift_b=False,
                gift_hint_shown_a=gift_hint_a,
                gift_hint_shown_b=gift_hint_b,
            )
            self.env.debug_logger.log_trade(trade_record)

    def _update_trust_from_contract(
        self,
        self_agent: SugarAgent,
        partner: SugarAgent,
        expected_receive: Dict[str, int],
        actual_receive: Dict[str, int],
    ) -> None:
        expected = self._safe_int_pair(expected_receive)
        actual = self._safe_int_pair(actual_receive)

        penalties = 0.0
        checks = 0
        for k in ("sugar", "spice"):
            if k == "spice" and not self.env.config.enable_spice:
                continue
            exp = expected.get(k, 0)
            if exp <= 0:
                continue
            checks += 1
            got = actual.get(k, 0)
            if got < exp:
                penalties += (exp - got) / max(1.0, float(exp))

        trust = self_agent.get_partner_trust(partner.agent_id)
        if checks == 0:
            new_trust = trust
        else:
            avg_penalty = penalties / checks
            delta = 0.10 - 0.45 * avg_penalty
            new_trust = trust + delta
        self_agent.update_partner_trust(partner.agent_id, new_trust)

    def _update_reputation(
        self,
        agent: SugarAgent,
        partner: SugarAgent,
        deception_detected: bool,
        trade_completed: bool,
    ) -> None:
        """Update global reputation based on trade behavior."""
        delta = 0.0

        # 1. Honesty component
        if trade_completed:
            if deception_detected:
                delta -= 0.15  # Fraud hurts reputation
            else:
                delta += 0.05  # Honesty builds reputation

        # 2. Altruism component - helping desperate partners
        if trade_completed:
            partner_sugar_time = int(partner.wealth / partner.metabolism) if partner.metabolism > 0 else 999
            partner_spice_time = int(partner.spice / partner.metabolism_spice) if partner.metabolism_spice > 0 else 999
            partner_min_time = min(partner_sugar_time, partner_spice_time)

            if partner_min_time < 3:  # Partner was desperate
                delta += 0.08  # Bonus for helping desperate agent

        self.env.update_agent_reputation(agent.agent_id, delta)

    def _get_agent_urgency(self, agent: SugarAgent) -> str:
        """Get urgency status string for an agent."""
        sugar_time = int(agent.wealth / agent.metabolism) if agent.metabolism > 0 else 999
        spice_time = int(agent.spice / agent.metabolism_spice) if agent.metabolism_spice > 0 else 999
        min_time = min(sugar_time, spice_time)

        if min_time < 3:
            return "CRITICAL"
        elif min_time < 10:
            return "struggling"
        else:
            return "stable"

    def _format_partner_memory(self, self_agent: SugarAgent, partner: SugarAgent) -> str:
        # Check if social memory is visible
        social_memory_visible = getattr(self.env.config, 'social_memory_visible', True)
        if not social_memory_visible:
            return "(No memory of past interactions)"

        trust = self_agent.get_partner_trust(partner.agent_id)
        log = list(self_agent.get_partner_trade_log(partner.agent_id, maxlen=self.memory_maxlen))
        if not log:
            return f"- trust: {trust:.2f}\n- history: (none)"

        lines = [f"- trust: {trust:.2f}", "- recent events:"]
        for ev in log[-5:]:
            if ev.get("type") == "TRADE":
                pc = ev.get("public_contract", {})
                actual = ev.get("actual", {})
                lines.append(
                    f"  - tick={ev.get('tick')}: TRADE contract(give={pc.get('give')}, receive={pc.get('receive')}) "
                    f"actual(sent={actual.get('sent')}, received={actual.get('received')})"
                )
            else:
                lines.append(f"  - tick={ev.get('tick')}: NO_TRADE outcome={ev.get('outcome')}")
        return "\n".join(lines)

    def _format_public_offer_text(self, speaker: SugarAgent, public_offer: Dict[str, Dict[str, int]]) -> str:
        give_i = self._safe_int_pair(public_offer.get("give") or {})
        receive_i = self._safe_int_pair(public_offer.get("receive") or {})
        if not self.env.config.enable_spice:
            give_i["spice"] = 0
            receive_i["spice"] = 0
        return f"{speaker.name} offers: I give {give_i} and receive {receive_i}."

    def _safe_int_pair(self, d: Dict[str, Any]) -> Dict[str, int]:
        return {"sugar": max(0, int(d.get("sugar", 0) or 0)), "spice": max(0, int(d.get("spice", 0) or 0))}

    def _fix_accept_execute_direction_confusion(
        self,
        acceptor_execute: Dict[str, int],
        contract_offer_give: Dict[str, Any],
        contract_offer_receive: Dict[str, Any],
    ) -> Dict[str, int]:
        """Best-effort correction for a common LLM mistake on ACCEPT:

        The acceptor sets `private_execute_give` to the partner's *give* (what acceptor would receive),
        instead of the partner's *receive* (what acceptor must give). This yields nonsense trades like
        both sides sending the same resource.

        We only correct when it exactly matches the confusion signature; otherwise, keep as-is.
        """
        exec_i = self._safe_int_pair(acceptor_execute or {})
        offer_give_i = self._safe_int_pair(contract_offer_give or {})
        offer_recv_i = self._safe_int_pair(contract_offer_receive or {})

        if not self.env.config.enable_spice:
            exec_i["spice"] = 0
            offer_give_i["spice"] = 0
            offer_recv_i["spice"] = 0

        if exec_i == offer_give_i and exec_i != offer_recv_i:
            # Only correct if there is something non-zero to pay in the contract.
            if (offer_recv_i.get("sugar", 0) + offer_recv_i.get("spice", 0)) > 0:
                return offer_recv_i

        return exec_i

    def _parse_trade_reply(self, reply_text: str) -> _ParsedTradeReply:
        # Strip thinking blocks (from thinking models like Qwen/DeepSeek) before parsing.
        cleaned = self._strip_thinking_blocks(reply_text)
        prefix_text, obj = self._extract_json(cleaned)
        thought, say = self._split_thought_and_say(prefix_text)
        if not isinstance(obj, dict):
            return _ParsedTradeReply(
                thought=thought,
                say=say if say else (reply_text or "").strip(),
                intent="CHAT",
                public_offer={"give": {"sugar": 0, "spice": 0}, "receive": {"sugar": 0, "spice": 0}},
                private_execute_give={"sugar": 0, "spice": 0},
            )

        intent = str(obj.get("intent", "CHAT")).strip().upper()
        if intent not in {"CHAT", "OFFER", "ACCEPT", "REJECT", "WALK_AWAY"}:
            intent = "CHAT"

        public_offer = obj.get("public_offer") or {}
        give = self._safe_int_pair(public_offer.get("give") or {})
        receive = self._safe_int_pair(public_offer.get("receive") or {})

        # Only zero out offer for ACCEPT/WALK_AWAY, preserve for OFFER and REJECT (counter-offer)
        if intent not in ("OFFER", "REJECT"):
            give = {"sugar": 0, "spice": 0}
            receive = {"sugar": 0, "spice": 0}

        private_execute = obj.get("private_execute_give") or {}
        private_execute_give = self._safe_int_pair(private_execute)
        if not self.env.config.enable_spice:
            give["spice"] = 0
            receive["spice"] = 0
            private_execute_give["spice"] = 0

        return _ParsedTradeReply(
            thought=thought,
            say=say,
            intent=intent,
            public_offer={"give": give, "receive": receive},
            private_execute_give=private_execute_give,
        )

    def _split_thought_and_say(self, text: str) -> Tuple[str, str]:
        """Extract REASONING/COMMUNICATE (or legacy THOUGHT/SAY) blocks from prefix text (before JSON).

        If markers are missing, treat the entire text as communication.
        """
        if not text:
            return "", ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        joined = "\n".join(lines).strip()
        if not joined:
            return "", ""

        thought_lines: List[str] = []
        say_lines: List[str] = []
        mode = None
        for ln in lines:
            upper = ln.upper()
            # Support new format: REASONING/COMMUNICATE
            if upper.startswith("REASONING:"):
                mode = "THOUGHT"
                rest = ln[len("REASONING:") :].strip()
                if rest:
                    thought_lines.append(rest)
                continue
            if upper.startswith("COMMUNICATE:"):
                mode = "SAY"
                rest = ln[len("COMMUNICATE:") :].strip()
                if rest:
                    say_lines.append(rest)
                continue
            # Support legacy format: THOUGHT/SAY (backwards compatibility)
            if upper.startswith("THOUGHT:"):
                mode = "THOUGHT"
                rest = ln[len("THOUGHT:") :].strip()
                if rest:
                    thought_lines.append(rest)
                continue
            if upper.startswith("SAY:"):
                mode = "SAY"
                rest = ln[len("SAY:") :].strip()
                if rest:
                    say_lines.append(rest)
                continue

            if mode == "THOUGHT":
                thought_lines.append(ln)
            elif mode == "SAY":
                say_lines.append(ln)
            else:
                # No markers yet: interpret as spoken content to keep backwards compatibility.
                say_lines.append(ln)

        thought = " ".join(thought_lines).strip()
        say = " ".join(say_lines).strip()
        return thought, say

    def _fix_json_keys(self, text: str) -> str:
        """Fix common LLM JSON mistakes: unquoted keys and string values."""
        import re
        # Step 1: Fix unquoted keys like {sugar: -> {"sugar":
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', text)
        # Step 2: Fix unquoted string values like : OFFER, -> : "OFFER",
        # Match : followed by unquoted word (not a number) followed by , or } or newline
        fixed = re.sub(r'(:\s*)([A-Za-z_][A-Za-z_0-9]*)(\s*[,}\n])', r'\1"\2"\3', fixed)
        return fixed

    def _strip_thinking_blocks(self, text: str, keep_if_empty: bool = True) -> str:
        """Remove thinking blocks from model output.
        
        Args:
            text: The text to process
            keep_if_empty: If True and stripped content is empty, return the thinking
                          content instead (prevents losing all output when model puts
                          everything in <think> blocks)
        """
        if strip_thinking_blocks is not None:
            return strip_thinking_blocks(text, keep_if_empty=keep_if_empty)
        # Fallback: if sugarscape llm_agent isn't available for some reason.
        import re
        
        # Extract thinking content first in case we need it
        thinking_content = ""
        if keep_if_empty:
            match = re.search(r"__THINKING_START__\s*(.*?)\s*__THINKING_END__", text, flags=re.DOTALL)
            if not match:
                match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                thinking_content = match.group(1).strip()
        
        cleaned = re.sub(r"__THINKING_START__.*?__THINKING_END__\s*", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        start = cleaned.find("__THINKING_START__")
        if start != -1 and "__THINKING_END__" not in cleaned[start:]:
            if keep_if_empty and not thinking_content:
                thinking_content = cleaned[start + len("__THINKING_START__"):].strip()
            cleaned = cleaned[:start]
        think_start = cleaned.lower().find("<think>")
        if think_start != -1 and "</think>" not in cleaned.lower()[think_start:]:
            if keep_if_empty and not thinking_content:
                thinking_content = cleaned[think_start + len("<think>"):].strip()
            cleaned = cleaned[:think_start]
        
        cleaned = cleaned.strip()
        
        # If cleaned is empty but we have thinking content, return that
        if not cleaned and thinking_content and keep_if_empty:
            return thinking_content
        
        return cleaned

    def _strip_reasoning_prefix(self, text: str) -> str:
        """Strip common LLM reasoning prefixes that shouldn't appear in dialogue.
        
        Many models (especially Qwen3) output reasoning patterns like:
        - "Okay, let's see..."
        - "Let me think about this..."
        - "I need to figure out..."
        
        This strips such prefixes to get to actual dialogue.
        """
        import re
        
        # Patterns that indicate reasoning/meta-commentary (not dialogue)
        reasoning_patterns = [
            r"^(?:Okay,?\s*)?(?:let's|let me)\s+(?:see|think|figure|break|analyze|consider).*?[.!]\s*",
            r"^(?:I need to|I should|I must)\s+(?:figure|think|respond|decide).*?[.!]\s*",
            r"^(?:The user|The prompt|Based on|According to).*?[.!]\s*",
            r"^(?:First|So|Now),?\s+(?:I|let me|let's).*?[.!]\s*",
            r"^(?:Hmm|Alright|Alrighty|OK|Well),?\s*(?:so|let me|let's)?.*?[.!]\s*",
        ]
        
        cleaned = text.strip()
        for pattern in reasoning_patterns:
            # Remove up to 2 sentences of reasoning prefix
            for _ in range(2):
                match = re.match(pattern, cleaned, re.IGNORECASE | re.DOTALL)
                if match:
                    cleaned = cleaned[match.end():].strip()
        
        # If we stripped everything, return original
        if not cleaned:
            return text.strip()
        
        return cleaned

    def _extract_json(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not text:
            return "", None

        decoder = json.JSONDecoder()
        starts = [i for i, ch in enumerate(text) if ch == "{"]

        best: Optional[Tuple[int, int, Dict[str, Any]]] = None  # (start, end_abs, obj)

        # Try original text first
        for i in starts:
            try:
                obj, end = decoder.raw_decode(text[i:])
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            end_abs = i + end
            # Prefer the parse that consumes the most characters (outermost object).
            if best is None or end_abs > best[1]:
                best = (i, end_abs, obj)

        # If no valid JSON found, try fixing common LLM mistakes
        if best is None:
            fixed_text = self._fix_json_keys(text)
            fixed_starts = [i for i, ch in enumerate(fixed_text) if ch == "{"]
            for i in fixed_starts:
                try:
                    obj, end = decoder.raw_decode(fixed_text[i:])
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                end_abs = i + end
                if best is None or end_abs > best[1]:
                    best = (i, end_abs, obj)

            if best is not None:
                # Use fixed text for extracting prefix/tail
                text = fixed_text

        if best is None:
            return text.strip(), None

        i, end_abs, obj = best
        prefix = text[:i].strip()
        tail = text[end_abs:].strip()
        say = prefix
        if tail:
            say = (say + "\n" + tail).strip() if say else tail
        return say, obj

    # ========== POST-ENCOUNTER REFLECTION SYSTEM ==========

    async def _run_reflection_for_pair(
        self,
        agent_a: SugarAgent,
        agent_b: SugarAgent,
        tick: int,
        outcome: str,
        conversation: List[Dict[str, str]],
        transfer_a: Optional[Dict[str, int]] = None,
        transfer_b: Optional[Dict[str, int]] = None,
    ) -> None:
        """Run post-encounter reflection for both agents in parallel.

        This is called after every encounter (trade or no-trade) to allow
        agents to update their beliefs, policies, and identity leaning.
        """
        # Check if reflection is enabled
        if not getattr(self.env.config, 'enable_reflection', False):
            return

        # Only reflect for LLM agents
        if not self._is_llm_trader(agent_a) and not self._is_llm_trader(agent_b):
            return

        # Build encounter summary
        if outcome == "completed":
            summary_a = f"Trade completed. You sent {transfer_a}, received {transfer_b}"
            summary_b = f"Trade completed. You sent {transfer_b}, received {transfer_a}"
        else:
            summary_a = f"No trade occurred (outcome: {outcome})"
            summary_b = f"No trade occurred (outcome: {outcome})"

        # Extract conversation highlights (last few turns)
        highlights = self._extract_conversation_highlights(conversation)

        # Run reflection for both agents in parallel
        tasks = []
        if self._is_llm_trader(agent_a):
            tasks.append(self._reflect_agent(agent_a, agent_b, tick, outcome, summary_a, highlights, conversation))
        if self._is_llm_trader(agent_b):
            tasks.append(self._reflect_agent(agent_b, agent_a, tick, outcome, summary_b, highlights, conversation))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _reflect_agent(
        self,
        self_agent: SugarAgent,
        partner_agent: SugarAgent,
        tick: int,
        outcome: str,
        encounter_summary: str,
        conversation_highlights: str,
        conversation: List[Dict[str, str]],
    ) -> None:
        """Run reflection for a single agent and apply the updates."""
        try:
            # Build reflection prompt
            reflection_prompt = build_sugarscape_reflection_prompt(
                self_agent=self_agent,
                partner_agent=partner_agent,
                encounter_outcome=outcome,
                encounter_summary=encounter_summary,
                conversation_highlights=conversation_highlights,
            )

            # Call LLM for reflection (JSON-only output)
            max_tokens = getattr(self.env.config, "reflection_max_tokens", 256)
            
            # Identity Context Injection
            identity_block = ""
            if self_agent.origin_identity and getattr(self_agent, "origin_identity_prompt", ""):
                identity_block = self_agent.origin_identity_prompt + "\n\n"
                
            system_prompt = f"{identity_block}Output valid JSON only. No explanation or prose. Keep the JSON concise (aim < {max_tokens} tokens)."

            response = ""
            reflection_json = None

            # Use robust generation with retry if available (LLMSugarAgent)
            if hasattr(self_agent, "generate_json_with_retry"):
                response, reflection_json = await self_agent.generate_json_with_retry(
                    system_prompt=system_prompt,
                    user_prompt=reflection_prompt,
                    retry_attempts=2
                )
            else:
                # Fallback for legacy/other agents
                # NOTE: only some providers (e.g. vLLM) support `chat_template_kwargs`.
                try:
                    response = await self_agent.provider.generate(
                        system_prompt=system_prompt,
                        messages=[{"role": "user", "content": reflection_prompt}],
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError as e:
                    # Only fall back when the provider rejects the kwarg; don't mask real TypeErrors.
                    msg = str(e)
                    if ("chat_template_kwargs" not in msg) and ("unexpected keyword" not in msg):
                        raise
                    response = await self_agent.provider.generate(
                        system_prompt=system_prompt,
                        messages=[{"role": "user", "content": reflection_prompt}],
                    )
                # Parse the JSON response
                reflection_json = self._parse_reflection_response(response)

            if reflection_json and not reflection_json.get("no_changes", False):
                # Apply the reflection updates
                changes = self_agent.apply_reflection_update(reflection_json)

                # Log the reflection
                if changes["beliefs_changed"] or changes["policies_changed"] or changes["identity_shifted"]:
                    print(f"[REFLECTION] {self_agent.name} updated after encounter with {partner_agent.name}:")
                    if changes["beliefs_changed"]:
                        print(f"  - Beliefs: {', '.join(changes['beliefs_changed'][:3])}{'...' if len(changes['beliefs_changed']) > 3 else ''}")
                    if changes["policies_changed"]:
                        print(f"  - Policies: {', '.join(changes['policies_changed'][:3])}{'...' if len(changes['policies_changed']) > 3 else ''}")
                    if changes["identity_shifted"]:
                        print(f"  - Identity shift: {changes['identity_shifted']:+.2f} (now: {self_agent.get_identity_label()})")

                    # Log to debug logger if available
                    if self.env.debug_logger:
                        self.env.debug_logger.log_reflection(
                            tick=tick,
                            agent_id=self_agent.agent_id,
                            agent_name=self_agent.name,
                            partner_id=partner_agent.agent_id,
                            partner_name=partner_agent.name,
                            outcome=outcome,
                            reflection_json=reflection_json,
                            changes=changes,
                        )

                # External moral evaluation (post-trade reflection) with full conversation transcript (includes smalltalk)
                moral_eval = getattr(self.env, "moral_evaluator", None)
                debug_logger = getattr(self.env, "debug_logger", None)
                if moral_eval and debug_logger:
                    try:
                        from sugarscape.moral_evaluator import compute_self_overall_from_leaning
                        from sugarscape.debug_logger import MoralEvalRecord
                        import json as _json

                        def _format_transcript(conv: List[Dict[str, Any]]) -> str:
                            lines: List[str] = []
                            for t in conv:
                                ttype = str(t.get("type", "")).strip()
                                speaker = t.get("speaker") or t.get("excluder") or "?"
                                if "message" in t:
                                    msg = str(t.get("message", ""))
                                elif "utterance" in t:
                                    msg = str(t.get("utterance", ""))
                                elif "reason" in t and ttype == "exclusion":
                                    msg = str(t.get("reason", ""))
                                elif "public_offer" in t:
                                    msg = f"public_offer={t.get('public_offer')}, private_execute_give={t.get('private_execute_give')}"
                                else:
                                    msg = _json.dumps(t, ensure_ascii=False)[:400]
                                lines.append(f"{ttype} | {speaker}: {msg}")
                            return "\n".join(lines)

                        transcript = _format_transcript(conversation or [])
                        self_overall = compute_self_overall_from_leaning(self_agent.self_identity_leaning, moral_eval.rubric).get("overall", 50.0)

                        ev = await moral_eval.async_evaluate(
                            interaction_type="post_trade_reflection",
                            tick=tick,
                            agent_id=self_agent.agent_id,
                            agent_name=self_agent.name,
                            agent_system_prompt=system_prompt,
                            agent_user_prompt=reflection_prompt,
                            agent_raw_response=response,
                            extra_context={
                                "outcome": outcome,
                                "partner_id": partner_agent.agent_id,
                                "partner_name": partner_agent.name,
                                "encounter_summary": encounter_summary,
                                "conversation_highlights": conversation_highlights,
                                "conversation_transcript": transcript,
                            },
                        )
                        overall = ev.get("overall", {})
                        parsed_ev = ev.get("parsed", {})
                        rec = MoralEvalRecord(
                            tick=tick,
                            agent_id=self_agent.agent_id,
                            agent_name=self_agent.name,
                            interaction_type="post_trade_reflection",
                            self_overall=float(self_overall),
                            external_overall=float(overall.get("overall", 0.0) or 0.0),
                            external_overall_raw=float(overall.get("overall_raw", 0.0) or 0.0),
                            polarity=float(overall.get("polarity", 0.0) or 0.0),
                            good_avg=float(overall.get("good_avg", 0.0) or 0.0),
                            bad_avg=float(overall.get("bad_avg", 0.0) or 0.0),
                            dim_scores=dict((parsed_ev.get("scores") or {})),
                            payload=ev,
                        )
                        debug_logger.log_moral_eval(rec)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[REFLECTION] Error for {self_agent.name}: {e}")

    def _parse_reflection_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the reflection JSON response."""
        if not response:
            return None

        # Strip thinking blocks if present
        cleaned = self._strip_thinking_blocks(response)

        # Extract JSON
        _, obj = self._extract_json(cleaned)

        return obj if isinstance(obj, dict) else None

    def _extract_conversation_highlights(self, conversation: List[Dict[str, str]], max_turns: int = 4) -> str:
        """Extract key moments from the conversation for reflection."""
        if not conversation:
            return ""

        # Get last few turns
        recent = conversation[-max_turns:] if len(conversation) > max_turns else conversation

        highlights = []
        for turn in recent:
            speaker = turn.get("speaker", "?")
            intent = turn.get("intent", "")

            # Format based on intent
            if intent == "OFFER":
                offer = turn.get("public_offer", {})
                give = offer.get("give", {})
                receive = offer.get("receive", {})
                highlights.append(f"- {speaker} offered: give {give}, receive {receive}")
            elif intent == "ACCEPT":
                highlights.append(f"- {speaker} ACCEPTED the offer")
            elif intent == "REJECT":
                highlights.append(f"- {speaker} REJECTED the offer")
            elif intent == "WALK_AWAY":
                highlights.append(f"- {speaker} walked away from negotiation")

        return "\n".join(highlights) if highlights else ""
