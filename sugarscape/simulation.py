import asyncio
import random
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

from sugarscape.config import SugarscapeConfig
from sugarscape.environment import SugarEnvironment
from sugarscape.agent import SugarAgent
from sugarscape.llm_agent import LLMSugarAgent
from sugarscape.experiment import ExperimentLogger, MetricsCalculator
from sugarscape.trade import TradeSystem, DialogueTradeSystem
from sugarscape.debug_logger import DebugLogger
from sugarscape.providers.openrouter_provider import OpenRouterProvider
from sugarscape.trajectory import SugarTrajectory, SugarActionRecord
from sugarscape.names import NameGenerator
from sugarscape.welfare import WelfareCalculator
from sugarscape.evaluator import BehaviorEvaluator
from sugarscape.moral_evaluator import MoralEvaluator, MoralRubric

# Import plotting module only if matplotlib is available
try:
    from sugarscape.welfare_plots import WelfarePlotter
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib not available. Plot generation disabled.")

class SugarSimulation:
    """Main simulation controller for Sugarscape."""

    def __init__(self, config: SugarscapeConfig = None, agent_factory=None, experiment_name: str = "baseline", broker_provider=None):
        self.config = config or SugarscapeConfig()
        self.broker_provider = broker_provider

        # Experiment logging
        self.logger = ExperimentLogger(experiment_type=experiment_name, config=self.config)

        # Initialize Debug Logger
        self.debug_logger = DebugLogger(
            output_dir=self.logger.run_dir / "debug",
            enable_decisions=self.config.debug_log_decisions,
            enable_llm_logs=self.config.debug_log_llm,
            enable_trade_logs=self.config.debug_log_trades,
            enable_death_logs=self.config.debug_log_deaths,
            enable_efficiency_logs=self.config.debug_log_efficiency,
        )

        # Create environment with debug logger
        self.env = SugarEnvironment(self.config, debug_logger=self.debug_logger)

        self.trade_system = None
        if self.config.enable_trade:
            mode = (self.config.trade_mode or "mrs").strip().lower()
            if mode == "dialogue":
                self.trade_system = DialogueTradeSystem(
                    self.env,
                    max_rounds=self.config.trade_dialogue_rounds,
                    allow_fraud=self.config.trade_allow_fraud,
                    memory_maxlen=self.config.trade_memory_maxlen,
                )
                if self.broker_provider:
                    self.trade_system.broker_provider = self.broker_provider
            else:
                self.trade_system = TradeSystem(self.env)

        # Initialize LLM Provider if enabled
        self.llm_provider = None
        vllm_base_url = getattr(self.config, "llm_vllm_base_url", None) or "http://localhost:8000/v1"
        if self.config.enable_llm_agents:
            provider_type = getattr(self.config, 'llm_provider_type', 'openrouter')
            if provider_type == "vllm":
                from sugarscape.providers.vllm_provider import VLLMProvider
                self.llm_provider = VLLMProvider(
                    model=self.config.llm_provider_model,
                    base_url=vllm_base_url,
                    max_tokens=2048
                )
            else:
                self.llm_provider = OpenRouterProvider(
                    model=self.config.llm_provider_model,
                    max_tokens=2048
                )

        # Initialize Evaluator Provider if enabled (evaluators only make sense for LLM agents;
        # rule-based runs never construct one, so they need no API key)
        self.evaluator_provider = None
        self.evaluator = None
        if getattr(self.config, 'enable_llm_evaluation', False) and self.config.enable_llm_agents:
            eval_provider_type = getattr(self.config, 'llm_evaluator_provider', 'openrouter')
            if eval_provider_type == "vllm":
                from sugarscape.providers.vllm_provider import VLLMProvider
                self.evaluator_provider = VLLMProvider(
                    model=self.config.llm_evaluator_model,
                    base_url=vllm_base_url,
                    max_tokens=2048
                )
            else:
                self.evaluator_provider = OpenRouterProvider(
                    model=self.config.llm_evaluator_model,
                    max_tokens=2048
                )
            
            self.evaluator = BehaviorEvaluator(
                provider=self.evaluator_provider,
                use_llm_evaluation=True
            )

        # External Moral Evaluator (per reflection moment)
        self.moral_evaluator_provider = None
        self.moral_evaluator = None
        if getattr(self.config, "enable_external_moral_evaluation", False) and self.config.enable_llm_agents:
            eval_provider_type = getattr(self.config, "external_moral_evaluator_provider", "openrouter")
            eval_model = getattr(self.config, "external_moral_evaluator_model", "openai/gpt-4o-mini")
            if eval_provider_type == "vllm":
                from sugarscape.providers.vllm_provider import VLLMProvider
                self.moral_evaluator_provider = VLLMProvider(
                    model=eval_model, base_url=vllm_base_url, max_tokens=2048
                )
            else:
                self.moral_evaluator_provider = OpenRouterProvider(model=eval_model, max_tokens=2048)

            rubric = MoralRubric(
                overall_transform=getattr(self.config, "moral_overall_transform", "tanh"),
                overall_tanh_k=float(getattr(self.config, "moral_overall_tanh_k", 2.2)),
                self_tanh_k=float(getattr(self.config, "moral_self_tanh_k", 4.0)),
            )
            self.moral_evaluator = MoralEvaluator(provider=self.moral_evaluator_provider, rubric=rubric)

        # Expose evaluator(s) on env so trade/agents can call them
        self.env.moral_evaluator = self.moral_evaluator
        self.env.moral_rubric = self.moral_evaluator.rubric if self.moral_evaluator else None

        self.agents: List[SugarAgent] = []
        self.tick = 0
        self.next_agent_id = 1
        self.agent_factory = agent_factory or self._default_agent_factory

        # Name generator for human-like names
        self.name_generator = NameGenerator(seed=self.config.seed)

        # Trajectory Tracking
        self.trajectory = SugarTrajectory(
            run_id=self.logger.experiment_id,
            config=self.config.__dict__
        )

        # Random number generator
        self.rng = random.Random(self.config.seed)

        # Track initial population for welfare calculations
        self.initial_population = self.config.initial_population

        # Create persistent event loop for async operations (LLM calls, trades)
        # This avoids repeated asyncio.run() calls which create/destroy loops
        # and cause cleanup issues with persistent HTTP clients
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        self._init_population()

    def _generate_agent_name(self) -> str:
        """Generate a unique human name for an agent."""
        return self.name_generator.generate()

    def _default_agent_factory(self, **kwargs) -> SugarAgent:
        return SugarAgent(**kwargs)

    def _init_population(self):
        """Create initial population."""
        for _ in range(self.config.initial_population):
            self._create_agent()

    def _create_agent(self):
        """Create and place a new agent with random attributes."""
        # Random attributes
        w0 = self.rng.randint(*self.config.initial_wealth_range)
        m = self.rng.randint(*self.config.metabolism_range)
        v = self.rng.randint(*self.config.vision_range)
        max_age = self.rng.randint(*self.config.max_age_range)

        # Spice attributes (if enabled)
        spice = 0
        m_spice = 0
        if self.config.enable_spice:
            spice = self.rng.randint(*self.config.initial_spice_range)
            m_spice = self.rng.randint(*self.config.metabolism_spice_range)

            # Resource Specialization: Enforce heterogeneous preferences if enabled
            if getattr(self.config, 'enable_resource_specialization', False):
                # Flip a coin: Sugar Specialist vs Spice Specialist
                if self.rng.random() < 0.5:
                    # Sugar Specialist: High Sugar Met, Low Spice Met
                    # Assuming range (1, 4), high is [3, 4], low is [1, 2]
                    m = self.rng.randint(3, 4)
                    m_spice = self.rng.randint(1, 2)
                else:
                    # Spice Specialist: Low Sugar Met, High Spice Met
                    m = self.rng.randint(1, 2)
                    m_spice = self.rng.randint(3, 4)

        # Determine Persona
        persona = "A" # default
        if self.config.enable_personas:
            # Simple weighted choice
            dist = self.config.persona_distribution
            keys = list(dist.keys()) # A, B, C, D
            probs = [dist[k] for k in keys]
            persona = self.rng.choices(keys, weights=probs, k=1)[0]

        # Random position
        pos = self.env.get_random_unoccupied_pos(self.rng)

        # Determine if this should be an LLM agent
        use_llm = False
        if self.config.enable_llm_agents and self.llm_provider:
            if self.rng.random() < self.config.llm_agent_ratio:
                use_llm = True

        if use_llm:
            # Select goal for this LLM agent
            if self.config.enable_mixed_goals:
                dist = self.config.llm_goal_distribution
                goal_preset = self.rng.choices(list(dist.keys()), weights=list(dist.values()))[0]
                goal_prompt = SugarscapeConfig.get_goal_prompt(goal_preset)
            else:
                goal_prompt = self.config.llm_goal_prompt

            agent = LLMSugarAgent(
                provider=self.llm_provider,
                goal_prompt=goal_prompt,
                agent_id=self.next_agent_id,
                pos=pos,
                vision=v,
                metabolism=m,
                max_age=max_age,
                wealth=w0,
                age=0,
                spice=spice,
                metabolism_spice=m_spice,
                name=self._generate_agent_name()  # Assign real human name
            )
        else:
            agent = self.agent_factory(
                agent_id=self.next_agent_id,
                pos=pos,
                vision=v,
                metabolism=m,
                max_age=max_age,
                wealth=w0,
                age=0,
                spice=spice,
                metabolism_spice=m_spice,
                name=self._generate_agent_name()  # Assign real human name
            )

        agent.persona = persona

        # Initialize Origin Identity System if enabled
        if self.config.enable_origin_identity:
            dist = self.config.origin_identity_distribution
            origin_type = self.rng.choices(list(dist.keys()), weights=list(dist.values()))[0]

            # Set immutable origin identity
            agent.origin_identity = origin_type
            agent.origin_identity_prompt = SugarscapeConfig.get_origin_identity_prompt(origin_type)

            # Initialize mutable policy list and belief ledger
            agent.policy_list = SugarscapeConfig.get_default_policies(origin_type)
            agent.belief_ledger = SugarscapeConfig.get_default_beliefs(origin_type)

            # Set initial identity leaning based on origin
            if origin_type == "altruist":
                agent.self_identity_leaning = 0.8  # Starts strongly good
            elif origin_type == "exploiter":
                agent.self_identity_leaning = -0.8  # Starts strongly bad
            else:
                agent.self_identity_leaning = 0.0  # Neutral
        self.next_agent_id += 1

        self.agents.append(agent)
        self.env.add_agent(agent)
        self.env.initialize_agent_reputation(agent.agent_id)

    def step(self):
        """Execute one simulation tick."""
        self.tick += 1

        # 1. Environment Growback
        self.env.growback()

        # 2. Record Timestep Start
        current_timestep = self.trajectory.add_timestep(
            tick=self.tick,
            population=len([a for a in self.agents if a.alive])
        )

        # 3. Agent updates (Move and Harvest)
        # Shuffle order
        self.rng.shuffle(self.agents)

        # This simulation uses a phased tick (Duke-style ordering):
        # 1) Move + Harvest (all agents)
        # 2) Trade (optional)
        # 3) Metabolize + Age + Death check
        #
        # Movement Strategy:
        # - If rule_based_movement=True (default): ALL agents use fast, deterministic rule-based movement
        #   This saves tokens for social interactions (encounters + reflection) and isolates social dynamics.
        # - If rule_based_movement=False: LLM agents use LLM to decide movement (legacy behavior, expensive).

        # Separate agents into LLM and Standard
        llm_agents = [a for a in self.agents if isinstance(a, LLMSugarAgent) and a.alive]
        std_agents = [a for a in self.agents if not isinstance(a, LLMSugarAgent) and a.alive]

        # Check if rule-based movement is enabled (saves tokens for social interactions)
        use_rule_based_movement = getattr(self.config, 'rule_based_movement', True)

        # Phase 1a. Process Standard Agents (Sequential, Fast): Move + Harvest (no metabolism yet)
        for agent in std_agents:
            agent._move_and_harvest(self.env)
            agent._update_metrics(self.env)

        # Phase 1b. Process LLM Agents
        # When rule_based_movement=True: Use same fast rule-based logic as standard agents (token-saving)
        # When rule_based_movement=False: Use LLM to decide movement (expensive, legacy behavior)
        if llm_agents:
            if use_rule_based_movement:
                # === RULE-BASED MOVEMENT (Token-Saving Mode) ===
                # LLM agents use parent class's deterministic movement logic.
                # This saves tokens for social interactions (encounters + reflection).
                for agent in llm_agents:
                    # Use the base SugarAgent's rule-based movement
                    SugarAgent._move_and_harvest(agent, self.env)
                    agent._update_metrics(self.env)

                    # Still record move history for LLM agents (for context in future interactions)
                    if hasattr(agent, 'move_history'):
                        agent.move_history.append({
                            "tick": self.tick,
                            "pos": agent.pos,
                            "action": "rule_based",  # Mark as rule-based decision
                            "wealth": agent.wealth,
                            "spice": agent.spice,
                            "sugar_harvested": 0,  # Already harvested in _move_and_harvest
                            "spice_harvested": 0,
                        })
            else:
                # === LLM-BASED MOVEMENT (Legacy Mode - Expensive) ===
                async def get_decisions():
                    tasks = []
                    for agent in llm_agents:
                        tasks.append(agent.async_decide_move(self.env))
                    return await asyncio.gather(*tasks, return_exceptions=True)

                # Run all decisions in parallel using persistent event loop
                # Note: LLM agents see state after standard agents' movement/harvest,
                # but before other LLM agents' moves are applied.
                decisions = self._event_loop.run_until_complete(get_decisions())

                # Apply decisions with a batched collision policy for LLM agents:
                # - Single occupancy per cell.
                # - If multiple agents target the same cell, pick a winner uniformly at random (seeded RNG).
                # - Allow moving into cells vacated by other LLM agents in the same batch; deny moves into
                #   cells occupied by non-LLM agents (already moved in phase 1a).
                desired: Dict[SugarAgent, Any] = {}
                decision_by_agent: Dict[SugarAgent, Dict[str, Any]] = {}

                for agent, decision_data in zip(llm_agents, decisions):
                    if isinstance(decision_data, Exception):
                        print(f"Agent {agent.agent_id} failed: {decision_data}")
                        continue
                    if not isinstance(decision_data, dict):
                        continue
                    decision_by_agent[agent] = decision_data
                    desired[agent] = decision_data.get("parsed_move")

                # Positions occupied by non-LLM alive agents are immutable for this batch.
                llm_set = set(llm_agents)
                occupied_by_non_llm = {
                    pos for pos, occ in self.env.grid_agents.items()
                    if (occ is not None) and (occ.alive) and (occ not in llm_set)
                }

                # Step 1: resolve "many want same target" with a fair tie-break.
                targets: Dict[Any, List[SugarAgent]] = {}
                for agent, tpos in desired.items():
                    if tpos is None:
                        continue
                    targets.setdefault(tpos, []).append(agent)

                winners: Dict[SugarAgent, Any] = {}
                for tpos, agents_wanting in targets.items():
                    if len(agents_wanting) == 1:
                        winners[agents_wanting[0]] = tpos
                    else:
                        winner = self.rng.choice(agents_wanting)
                        winners[winner] = tpos

                # Step 2: propose final positions (winners move, others stay).
                old_pos: Dict[SugarAgent, Any] = {a: a.pos for a in llm_agents}
                final_pos: Dict[SugarAgent, Any] = {a: old_pos[a] for a in llm_agents}
                for a, tpos in winners.items():
                    if tpos is not None:
                        final_pos[a] = tpos

                # Step 3: enforce single occupancy against *stayers* and non-LLM occupied cells.
                # Iteratively revert invalid movers until stable.
                for _ in range(max(1, len(llm_agents))):
                    changed = False

                    # Cannot move into a non-LLM occupied cell.
                    for a, tpos in list(final_pos.items()):
                        if tpos in occupied_by_non_llm and tpos != old_pos[a]:
                            final_pos[a] = old_pos[a]
                            changed = True

                    # Resolve duplicates among LLM final positions (e.g., moving into a stayer's cell).
                    rev: Dict[Any, List[SugarAgent]] = {}
                    for a, tpos in final_pos.items():
                        rev.setdefault(tpos, []).append(a)

                    for tpos, agents_here in rev.items():
                        if len(agents_here) <= 1:
                            continue
                        # Prefer keeping the agent whose original position is this cell (a "stayer"/rightful occupant).
                        rightful = [a for a in agents_here if old_pos[a] == tpos]
                        if rightful:
                            keep = rightful[0]
                        else:
                            keep = self.rng.choice(agents_here)
                        for a in agents_here:
                            if a is keep:
                                continue
                            final_pos[a] = old_pos[a]
                            changed = True

                    if not changed:
                        break

                # Step 4: apply batched moves by rebuilding the env grid mapping for these agents.
                for a in llm_agents:
                    # Remove old mapping if present.
                    if self.env.grid_agents.get(old_pos[a]) == a:
                        del self.env.grid_agents[old_pos[a]]
                for a in llm_agents:
                    a.pos = final_pos[a]
                    self.env.grid_agents[a.pos] = a

                # Step 5: harvest + metrics update per agent (after movement is finalized)
                for agent in llm_agents:
                    decision_data = decision_by_agent.get(agent, {})
                    target_pos = desired.get(agent)

                    rewards = agent._harvest_and_update_metrics(self.env)

                    # Record move history for LLM agents
                    if hasattr(agent, 'move_history'):
                        agent.move_history.append({
                            "tick": self.tick,
                            "pos": agent.pos,
                            "action": target_pos,
                            "wealth": agent.wealth,
                            "spice": agent.spice,
                            "sugar_harvested": rewards["sugar_harvested"],
                            "spice_harvested": rewards.get("spice_harvested", 0),
                        })

                    # Record Action in Trajectory
                    action_record = SugarActionRecord(
                        agent_id=agent.agent_id,
                        system_prompt=decision_data.get("system_prompt", ""),
                        user_prompt=decision_data.get("user_prompt", ""),
                        raw_response=decision_data.get("raw_response", ""),
                        parsed_move=target_pos,
                        reward_sugar=rewards["sugar_harvested"],
                        reward_spice=rewards["spice_harvested"],
                        metabolic_cost_sugar=agent.metabolism,
                        metabolic_cost_spice=agent.metabolism_spice,
                    )
                    current_timestep.actions.append(action_record)

                    # Log LLM interaction to debug logger
                    from sugarscape.debug_logger import LLMInteraction
                    llm_interaction = LLMInteraction(
                        tick=self.tick,
                        agent_id=agent.agent_id,
                        agent_name=agent.name,
                        interaction_type="movement",
                        system_prompt=decision_data.get("system_prompt", ""),
                        user_prompt=decision_data.get("user_prompt", ""),
                        raw_response=decision_data.get("raw_response", ""),
                        parsed_action=str(target_pos) if target_pos else "",
                        input_tokens=0,  # TODO: Extract from provider if available
                        output_tokens=0,  # TODO: Extract from provider if available
                        latency_ms=0.0,  # TODO: Track latency
                        # New fields for goal and nearby agents analysis
                        goal_preset=getattr(agent, 'goal_prompt', '')[:50] if hasattr(agent, 'goal_prompt') else "",
                        nearby_agents_critical=decision_data.get("nearby_agents_critical", 0),
                        nearby_agents_struggling=decision_data.get("nearby_agents_struggling", 0),
                        nearby_agents_total=decision_data.get("nearby_agents_total", 0),
                    )
                    self.debug_logger.log_llm_interaction(llm_interaction)

        # Re-shuffle not needed since we processed in groups, but good for next tick?
        # self.rng.shuffle(self.agents) # Done at start

        # Phase 2. Trade (optional) happens after movement/harvest and before metabolism
        if self.config.enable_trade and self.trade_system:
            # Only alive agents trade
            live_agents = [a for a in self.agents if a.alive]
            self.trade_system.execute_trade_round(live_agents, tick=self.tick)

        # Phase 2.5. Periodic Identity Review (legacy, disabled by default)
        # NOTE: Periodic reviews are now disabled by default in favor of event-triggered reviews
        if (self.config.enable_identity_review and
            self.config.enable_origin_identity and
            self.tick % self.config.identity_review_interval == 0):
            self._run_identity_reviews()

        # Phase 2.6. Event-Triggered Identity Review (full assessment when significant events occur)
        # Events: defrauded, successful_cooperation, resources_critical, trade_rejected, witnessed_death
        if (self.config.enable_origin_identity and
            getattr(self.config, 'enable_event_triggered_identity_review', True)):
            self._run_event_triggered_reflections()

        # Phase 3. Metabolize + Age + Death check (applied to all agents)
        dead_agents = []
        for agent in self.agents:
            if not agent.alive:
                continue
            # Track state before metabolism for death cause determination
            pre_wealth = agent.wealth
            pre_spice = agent.spice
            pre_age = agent.age

            agent.metabolize_age_and_check_death(self.env, tick=self.tick)
            if not agent.alive:
                # Determine death cause
                death_cause = self._determine_death_cause(agent, pre_wealth, pre_spice, pre_age)
                dead_agents.append((agent, death_cause))

        # Run end-of-life reports for dying LLM agents (before removal)
        if (self.config.enable_end_of_life_report and
            self.config.enable_origin_identity and
            dead_agents):
            self._run_end_of_life_reports(dead_agents)

        for agent, death_cause in dead_agents:
            if agent in self.agents:
                # Log death before removing
                if self.debug_logger:
                    from sugarscape.debug_logger import DeathRecord
                    death_record = DeathRecord(
                        tick=self.tick,
                        agent_id=agent.agent_id,
                        agent_name=agent.name,
                        cause=death_cause,
                        final_wealth=agent.wealth,
                        final_spice=agent.spice,
                        final_age=agent.age,
                        max_age=agent.max_age,
                        metabolism=agent.metabolism,
                        metabolism_spice=agent.metabolism_spice,
                        lifetime_ticks=self.tick,  # Approximate
                    )
                    self.debug_logger.log_death(death_record)

                # Notify nearby LLM agents of witnessed death (event-triggered reflection)
                try:
                    for nearby_agent in self.agents:
                        if nearby_agent == agent or not nearby_agent.alive:
                            continue
                        if not isinstance(nearby_agent, LLMSugarAgent):
                            continue
                        # Check if within vision range
                        dist = abs(nearby_agent.pos[0] - agent.pos[0]) + abs(nearby_agent.pos[1] - agent.pos[1])
                        if dist <= nearby_agent.vision:
                            nearby_agent.record_reflection_event("witnessed_death", self.tick, {
                                "deceased_name": agent.name,
                                "deceased_id": agent.agent_id,
                                "cause": death_cause,
                                "distance": dist,
                            })
                except Exception as e:
                    pass  # Non-critical feature, don't break simulation

                self.agents.remove(agent)
                self.env.remove_agent(agent)
                # Replacement Rule: Only replace if rebirth is enabled
                if self.config.enable_rebirth:
                    self._create_agent()
                else:
                    print(f"[DEATH] {agent.name} died ({death_cause}). Population: {len(self.agents)}")

        # Logging
        if self.tick % 10 == 0:
            stats = self.get_stats()
            self.logger.log_step(stats)

    def _run_identity_reviews(self) -> None:
        """Run identity reviews for all eligible LLM agents.

        Called every identity_review_interval ticks to let agents reflect
        on whether they're still altruist/exploiter.
        """
        # Filter to LLM agents with origin identity
        llm_agents_with_identity = [
            a for a in self.agents
            if isinstance(a, LLMSugarAgent) and a.alive and a.origin_identity
        ]

        if not llm_agents_with_identity:
            return

        print(f"[IDENTITY REVIEW] Running identity reviews for {len(llm_agents_with_identity)} agents at tick {self.tick}")

        async def run_reviews():
            tasks = [
                agent.async_identity_review(self.env, self.tick)
                for agent in llm_agents_with_identity
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = self._event_loop.run_until_complete(run_reviews())

        # Log results
        for agent, result in zip(llm_agents_with_identity, results):
            if isinstance(result, Exception):
                print(f"  - {agent.name}: ERROR - {result}")
            elif isinstance(result, dict):
                before = result.get("identity_before", 0)
                after = result.get("identity_after", 0)
                parsed = result.get("parsed", {})
                assessment = parsed.get("identity_assessment", "unknown") if parsed else "unknown"
                shift = after - before
                shift_str = f"+{shift:.2f}" if shift >= 0 else f"{shift:.2f}"
                print(f"  - {agent.name}: {assessment} (leaning {shift_str}, now {after:.2f})")

                # Log to debug logger if available
                if self.debug_logger:
                    from sugarscape.debug_logger import LLMInteraction
                    interaction = LLMInteraction(
                        tick=self.tick,
                        agent_id=agent.agent_id,
                        agent_name=agent.name,
                        interaction_type="identity_review",
                        system_prompt=result.get("system_prompt", ""),
                        user_prompt=result.get("user_prompt", ""),
                        raw_response=result.get("raw_response", ""),
                        parsed_action=f"assessment={assessment}, shift={shift_str}",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                    )
                    self.debug_logger.log_llm_interaction(interaction)

    def _run_event_triggered_reflections(self) -> None:
        """Run event-triggered reflections for LLM agents with pending events.

        This is more sensitive than periodic tick-based reflection.
        Called after trade phase to process events like fraud, cooperation, rejection.
        """
        # Filter to LLM agents with pending events
        llm_agents_with_events = [
            a for a in self.agents
            if isinstance(a, LLMSugarAgent) and a.alive and a.has_pending_reflection()
        ]

        if not llm_agents_with_events:
            return

        print(f"[EVENT REFLECTION] Processing events for {len(llm_agents_with_events)} agents at tick {self.tick}")

        async def run_reflections():
            tasks = [
                agent.async_event_reflection(self.env, self.tick)
                for agent in llm_agents_with_events
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = self._event_loop.run_until_complete(run_reflections())

        # Log results
        for agent, result in zip(llm_agents_with_events, results):
            if isinstance(result, Exception):
                print(f"  - {agent.name}: ERROR - {result}")
            elif isinstance(result, dict) and not result.get("skipped"):
                events = result.get("events", [])
                event_types = [e.get("type", "?") for e in events]
                updates = result.get("updates_applied", {})
                print(f"  - {agent.name}: processed {event_types}, updates={len(updates) > 0}")

    def _capture_baselines(self) -> None:
        """Capture baseline beliefs/values for all agents at T=0.

        This is called before any interactions to establish the pre-interaction baseline.
        Used to prove that value changes emerge from interactions, not pre-existing.

        For LLM agents, also runs the fixed worldview questionnaire (Q1-Q5).
        """
        print(f"[BASELINE] Capturing T=0 baseline for {len(self.agents)} agents...")

        baseline_data = []
        for agent in self.agents:
            snapshot = agent.capture_baseline(tick=self.tick)
            baseline_data.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "origin_identity": agent.origin_identity,
                **snapshot
            })

        # Run questionnaire for LLM agents (fixed Q1-Q5 measurement)
        llm_agents = [a for a in self.agents if isinstance(a, LLMSugarAgent) and a.alive]
        if llm_agents:
            print(f"[BASELINE] Running T=0 questionnaire for {len(llm_agents)} LLM agents...")

            async def run_questionnaires():
                tasks = [agent.async_baseline_questionnaire(self.env, tick=0) for agent in llm_agents]
                return await asyncio.gather(*tasks, return_exceptions=True)

            results = self._event_loop.run_until_complete(run_questionnaires())

            # Add questionnaire results to baseline data
            for agent, result in zip(llm_agents, results):
                if isinstance(result, Exception):
                    print(f"  - {agent.name}: ERROR - {result}")
                    continue

                # Find this agent's baseline entry and add questionnaire scores
                for entry in baseline_data:
                    if entry["agent_id"] == agent.agent_id:
                        entry["questionnaire_t0"] = {
                            "scores": result.get("scores", {}),
                            "parsed": result.get("parsed", {}),
                            "raw_response": result.get("raw_response", ""),
                        }
                        scores = result.get("scores", {})
                        print(f"  - {agent.name}: Q1={scores.get('Q1_trust', '?')}, "
                              f"Q2={scores.get('Q2_cooperation', '?')}, "
                              f"Q3={scores.get('Q3_fairness', '?')}, "
                              f"Q4={scores.get('Q4_scarcity', '?')}, "
                              f"Q5={scores.get('Q5_self_vs_others', '?')}")
                        break

        # Save baseline to file
        import json
        baseline_path = self.logger.run_dir / "baseline_beliefs.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"[BASELINE] Saved to {baseline_path}")

    def _run_end_of_life_reports(self, dead_agents: list) -> None:
        """Run end-of-life self-reports for dying LLM agents.

        Args:
            dead_agents: List of (agent, death_cause) tuples for agents that just died.
        """
        # Filter to LLM agents with origin identity
        llm_dead = [
            (agent, cause) for agent, cause in dead_agents
            if isinstance(agent, LLMSugarAgent) and agent.origin_identity
        ]

        if not llm_dead:
            return

        print(f"[END OF LIFE] Running final reports for {len(llm_dead)} dying agents at tick {self.tick}")

        async def run_reports():
            tasks = [
                agent.async_end_of_life_report(self.env, self.tick, cause)
                for agent, cause in llm_dead
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = self._event_loop.run_until_complete(run_reports())

        # Log results
        for (agent, cause), result in zip(llm_dead, results):
            if isinstance(result, Exception):
                print(f"  - {agent.name}: ERROR - {result}")
            elif isinstance(result, dict):
                parsed = result.get("parsed", {})
                assessment = parsed.get("life_assessment", "unknown") if parsed else "unknown"
                origin = agent.origin_identity
                final_leaning = agent.self_identity_leaning
                print(f"  - {agent.name}: Born '{origin}' → died as '{assessment}' (final leaning: {final_leaning:.2f})")

                # Log to debug logger if available
                if self.debug_logger:
                    from sugarscape.debug_logger import LLMInteraction
                    interaction = LLMInteraction(
                        tick=self.tick,
                        agent_id=agent.agent_id,
                        agent_name=agent.name,
                        interaction_type="end_of_life_report",
                        system_prompt=result.get("system_prompt", ""),
                        user_prompt=result.get("user_prompt", ""),
                        raw_response=result.get("raw_response", ""),
                        parsed_action=f"assessment={assessment}, cause={cause}",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                    )
                    self.debug_logger.log_llm_interaction(interaction)

    def _determine_death_cause(self, agent: SugarAgent, pre_wealth: int, pre_spice: int, pre_age: int) -> str:
        """Determine cause of death for an agent.

        Args:
            agent: The agent that died
            pre_wealth: Sugar before metabolism
            pre_spice: Spice before metabolism
            pre_age: Age before aging

        Returns:
            Death cause string: "starvation_sugar", "starvation_spice", "old_age"
        """
        # If survival pressure is disabled, only old age is possible
        enable_survival_pressure = getattr(self.config, 'enable_survival_pressure', True)
        if not enable_survival_pressure:
            return "old_age"

        # Check if died of old age
        if pre_age >= agent.max_age - 1:  # Was at max_age-1, then aged to max_age
            return "old_age"

        # Check resource depletion
        post_wealth = pre_wealth - agent.metabolism
        post_spice = pre_spice - agent.metabolism_spice if self.env.config.enable_spice else pre_spice

        if post_wealth <= 0:
            return "starvation_sugar"
        if self.env.config.enable_spice and post_spice <= 0:
            return "starvation_spice"

        # Default fallback
        return "old_age"

    def _run_final_reports(self) -> None:
        """Run end-of-simulation reports for all living LLM agents.

        Called at simulation end to capture final state of surviving agents.
        """
        if not (self.config.enable_end_of_life_report and self.config.enable_origin_identity):
            return

        # Filter to living LLM agents with origin identity
        llm_agents = [
            a for a in self.agents
            if isinstance(a, LLMSugarAgent) and a.alive and a.origin_identity
        ]

        if not llm_agents:
            return

        print(f"[SIMULATION END] Running final reports for {len(llm_agents)} surviving agents")

        async def run_reports():
            tasks = [
                agent.async_end_of_life_report(self.env, self.tick, "simulation_end")
                for agent in llm_agents
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = self._event_loop.run_until_complete(run_reports())

        # Log results
        for agent, result in zip(llm_agents, results):
            if isinstance(result, Exception):
                print(f"  - {agent.name}: ERROR - {result}")
            elif isinstance(result, dict):
                parsed = result.get("parsed", {})
                assessment = parsed.get("life_assessment", "unknown") if parsed else "unknown"
                origin = agent.origin_identity
                final_leaning = agent.self_identity_leaning
                print(f"  - {agent.name}: Born '{origin}' → survived as '{assessment}' (final leaning: {final_leaning:.2f})")

    def _run_evaluation(self) -> None:
        """Run independent behavioral evaluation for all agents.
        
        Uses BehaviorEvaluator to assess agents based on trade logs and behavior,
        optionally using a separate LLM (default: gpt-4o-mini).
        """
        if not self.evaluator:
            return

        print(f"[EVALUATION] Running independent behavioral evaluation for {len(self.agents)} agents...")
        
        # Run evaluation
        results = self._event_loop.run_until_complete(
            self.evaluator.async_evaluate_all(self.agents)
        )
        
        # Save results
        eval_path = self.logger.run_dir / "behavioral_evaluation.json"
        import json
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"[EVALUATION] Saved to {eval_path}")
        
        # Print summary
        avg_coop = np.mean([r["behavioral_metrics"]["cooperation_score"] for r in results])
        avg_exploit = np.mean([r["behavioral_metrics"]["exploitation_score"] for r in results])
        print(f"[EVALUATION] Average Cooperation Score: {avg_coop:.2f}")
        print(f"[EVALUATION] Average Exploitation Score: {avg_exploit:.2f}")

    def run(self, steps: int = None):
        """Run for a number of steps with automatic checkpointing.

        Checkpoints are saved every `config.checkpoint_interval` ticks by default.
        Set `config.checkpoint_interval = 0` to disable.
        """
        # Capture baseline beliefs at T=0 (before any interactions)
        self._capture_baselines()
        
        # Run initial identity review at T=0 (before any trades)
        # This establishes the pre-interaction belief baseline via LLM questioning
        if (self.config.enable_identity_review and 
            self.config.enable_origin_identity):
            print("[BASELINE] Running T=0 identity reviews (pre-interaction baseline)...")
            self._run_identity_reviews()
        
        # Initial snapshot
        self.save_snapshot(filename="initial_state.json")

        limit = steps or self.config.max_ticks
        checkpoint_interval = self.config.checkpoint_interval

        for _ in range(limit):
            self.step()

            # Save checkpoint at intervals (if enabled)
            if checkpoint_interval > 0 and self.tick % checkpoint_interval == 0:
                self.save_checkpoint()

        # Final snapshot
        self.save_snapshot(filename="final_state.json")

        # Run final reports for surviving agents before cleanup
        self._run_final_reports()

        # Run independent evaluation
        self._run_evaluation()

        # Save Trajectory
        traj_filename = f"trajectory_{self.logger.experiment_id}.json"
        self.trajectory.save(self.logger.get_log_path(traj_filename))

        # Save debug summary
        self.debug_logger.save_summary()

        # Generate welfare plots
        self._generate_plots()

        # Clean up async resources
        self._cleanup_async()

    def save_snapshot(self, filename: str):
        """Save current simulation state."""
        data = {
            "tick": self.tick,
            "agents": [a.to_checkpoint_dict() for a in self.agents],
            "sugar_map": self.env.sugar_amount.tolist(),
            "spice_map": self.env.spice_amount.tolist(),
            "sugar_capacity": self.env.sugar_capacity.tolist()
        }
        self.logger.save_snapshot(data, filename)

    def _generate_plots(self):
        """Generate welfare visualization plots after simulation completes."""
        if not PLOTTING_AVAILABLE:
            print("Note: Plot generation skipped (matplotlib not available)")
            return

        try:
            # Determine title prefix based on agent type
            if self.config.enable_llm_agents:
                if self.config.llm_agent_ratio >= 1.0:
                    title_prefix = "LLM Agents"
                elif self.config.llm_agent_ratio > 0:
                    title_prefix = f"Mixed ({int(self.config.llm_agent_ratio*100)}% LLM)"
                else:
                    title_prefix = "Rule-Based Agents"
            else:
                title_prefix = "Rule-Based Agents"

            # Generate plots from the metrics CSV
            WelfarePlotter.generate_all_plots(
                csv_path=str(self.logger.csv_file),
                plots_dir=self.logger.get_plots_dir(),
                title_prefix=title_prefix
            )

            print(f"✓ Generated welfare plots in: {self.logger.get_plots_dir()}")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    def _cleanup_async(self) -> None:
        """Clean up async resources (event loop, LLM provider connections).

        This prevents 'Event loop is closed' errors by properly closing
        the HTTP client before the event loop is closed.
        """
        # Close LLM provider's HTTP client
        if self.llm_provider is not None and hasattr(self.llm_provider, 'aclose'):
            try:
                self._event_loop.run_until_complete(self.llm_provider.aclose())
            except Exception as e:
                print(f"Warning: Error closing LLM provider: {e}")

        # Close the event loop
        if self._event_loop is not None and not self._event_loop.is_closed():
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self._event_loop)
                for task in pending:
                    task.cancel()
                # Give cancelled tasks a chance to finish
                if pending:
                    self._event_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                self._event_loop.close()
            except Exception as e:
                print(f"Warning: Error closing event loop: {e}")

    def close(self) -> None:
        """Explicitly close simulation resources.

        Call this if you need to clean up before the simulation completes
        normally (e.g., on error or early termination).
        """
        self._cleanup_async()

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics including welfare metrics."""
        wealths = [a.wealth for a in self.agents]
        spices = [a.spice for a in self.agents]
        ages = [a.age for a in self.agents]
        positions = [a.pos for a in self.agents]

        # Calculate Moran's I
        moran = MetricsCalculator.calculate_moran_i(
            self.config.width, self.config.height, positions, wealths
        )

        # Calculate Mobility
        mobility = MetricsCalculator.calculate_mobility_stats(self.agents)

        # Calculate Welfare Metrics
        welfare_metrics = WelfareCalculator.calculate_all_welfare_metrics(
            self.agents,
            self.initial_population
        )

        # Combine all metrics
        stats = {
            "tick": self.tick,
            "population": len(self.agents),
            "mean_wealth": np.mean(wealths) if wealths else 0,
            "mean_spice": np.mean(spices) if spices else 0,
            "max_wealth": np.max(wealths) if wealths else 0,
            "min_wealth": np.min(wealths) if wealths else 0,
            "gini": self._gini_coefficient(wealths) if wealths else 0,
            "mean_age": np.mean(ages) if ages else 0,
            "moran_i": moran,
            "avg_displacement": mobility["avg_displacement"],
            "avg_exploration": mobility["avg_exploration"]
        }

        # Add all welfare metrics
        stats.update(welfare_metrics)

        return stats

    def _gini_coefficient(self, values):
        """Calculate Gini coefficient for wealth inequality."""
        if not values: return 0
        sorted_vals = sorted(values)
        n = len(values)
        total = sum(sorted_vals)
        if total == 0: return 0

        # G = (2 * sum(i * xi) / (n * sum(xi))) - (n + 1) / n
        # where i is 1-based index
        weighted_sum = sum((i + 1) * val for i, val in enumerate(sorted_vals))
        return (2 * weighted_sum) / (n * total) - (n + 1) / n

    # ========== Checkpoint System ==========

    def save_checkpoint(self, filename: Optional[str] = None) -> Path:
        """Save complete simulation state for later resumption.

        Args:
            filename: Optional custom filename. If None, uses tick number.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_data = {
            # Simulation state
            "tick": self.tick,
            "next_agent_id": self.next_agent_id,
            "initial_population": self.initial_population,
            "rng_state": self.rng.getstate(),

            # Config (serialized)
            "config": self.config.__dict__.copy(),

            # Name generator state
            "name_generator_state": self.name_generator.get_state(),

            # Environment state
            "environment": self.env.get_checkpoint_state(),

            # Agent states
            "agents": [agent.to_checkpoint_dict() for agent in self.agents],

            # Metadata
            "experiment_id": self.logger.experiment_id,
            "run_dir": str(self.logger.run_dir),
        }

        # Use logger's checkpoint save method
        checkpoint_path = self.logger.save_checkpoint(checkpoint_data, self.tick)
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Union[str, Path]) -> "SugarSimulation":
        """Restore simulation from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint .pkl file.

        Returns:
            Restored SugarSimulation instance ready to continue.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct config
        config = SugarscapeConfig(**data["config"])

        # Create simulation instance (but skip normal initialization)
        sim = object.__new__(cls)

        # Restore config
        sim.config = config

        # Restore experiment logger - create new one in same directory structure
        # We extract experiment name from the run_dir
        run_dir = Path(data.get("run_dir", ""))
        experiment_type = run_dir.parent.name if run_dir.parent else "resumed"

        # Create logger pointing to same run directory
        sim.logger = ExperimentLogger(
            experiment_type=experiment_type,
            config=config
        )
        # Override to use original experiment ID for consistency
        # Note: This creates a NEW run dir, but we keep the experiment_id for tracking
        original_experiment_id = data.get("experiment_id", sim.logger.experiment_id)

        # Debug logger
        sim.debug_logger = DebugLogger(
            output_dir=sim.logger.run_dir / "debug",
            enable_decisions=config.debug_log_decisions,
            enable_llm_logs=config.debug_log_llm,
            enable_trade_logs=config.debug_log_trades,
            enable_death_logs=config.debug_log_deaths,
            enable_efficiency_logs=config.debug_log_efficiency,
        )

        # Environment
        sim.env = SugarEnvironment(config, debug_logger=sim.debug_logger)
        sim.env.restore_checkpoint_state(data["environment"])

        # Trade system
        sim.trade_system = None
        if config.enable_trade:
            mode = (config.trade_mode or "mrs").strip().lower()
            if mode == "dialogue":
                sim.trade_system = DialogueTradeSystem(
                    sim.env,
                    max_rounds=config.trade_dialogue_rounds,
                    allow_fraud=config.trade_allow_fraud,
                    memory_maxlen=config.trade_memory_maxlen,
                )
            else:
                sim.trade_system = TradeSystem(sim.env)

        # LLM provider
        sim.llm_provider = None
        if config.enable_llm_agents:
            provider_type = getattr(config, 'llm_provider_type', 'openrouter')
            if provider_type == "vllm":
                from sugarscape.providers.vllm_provider import VLLMProvider
                sim.llm_provider = VLLMProvider(
                    model=config.llm_provider_model,
                    base_url=getattr(config, "llm_vllm_base_url", None) or "http://localhost:8000/v1",
                    max_tokens=2048
                )
            else:
                sim.llm_provider = OpenRouterProvider(
                    model=config.llm_provider_model,
                    max_tokens=2048
                )

        # Restore simulation state
        sim.tick = data["tick"]
        sim.next_agent_id = data["next_agent_id"]
        sim.initial_population = data["initial_population"]

        # RNG state
        sim.rng = random.Random()
        sim.rng.setstate(data["rng_state"])

        # Name generator
        sim.name_generator = NameGenerator()
        sim.name_generator.set_state(data["name_generator_state"])

        # Agent factory
        sim.agent_factory = lambda **kwargs: SugarAgent(**kwargs)

        # Create persistent event loop for async operations
        sim._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(sim._event_loop)

        # Trajectory (new for resumed run)
        sim.trajectory = SugarTrajectory(
            run_id=sim.logger.experiment_id,
            config=config.__dict__
        )

        # Restore agents
        sim.agents = []
        for agent_data in data["agents"]:
            is_llm = agent_data.get("is_llm_agent", False)

            if is_llm and sim.llm_provider:
                agent = LLMSugarAgent(
                    provider=sim.llm_provider,
                    goal_prompt=agent_data.get("goal_prompt", config.llm_goal_prompt),
                    agent_id=agent_data["agent_id"],
                    pos=tuple(agent_data["pos"]),
                    vision=agent_data["vision"],
                    metabolism=agent_data["metabolism"],
                    max_age=agent_data["max_age"],
                    wealth=agent_data["wealth"],
                    spice=agent_data.get("spice", 0),
                    metabolism_spice=agent_data.get("metabolism_spice", 0),
                    age=agent_data.get("age", 0),
                    persona=agent_data.get("persona", "A"),
                    name=agent_data.get("name", ""),
                )
            else:
                agent = SugarAgent(
                    agent_id=agent_data["agent_id"],
                    pos=tuple(agent_data["pos"]),
                    vision=agent_data["vision"],
                    metabolism=agent_data["metabolism"],
                    max_age=agent_data["max_age"],
                    wealth=agent_data["wealth"],
                    spice=agent_data.get("spice", 0),
                    metabolism_spice=agent_data.get("metabolism_spice", 0),
                    age=agent_data.get("age", 0),
                    persona=agent_data.get("persona", "A"),
                    name=agent_data.get("name", ""),
                )

            # Restore agent state from checkpoint
            agent.restore_from_checkpoint(agent_data)

            sim.agents.append(agent)
            sim.env.grid_agents[agent.pos] = agent

        print(f"Checkpoint loaded: tick={sim.tick}, agents={len(sim.agents)}")
        print(f"Resumed experiment at: {sim.logger.run_dir}")

        return sim

    def run_with_checkpoints(self, steps: Optional[int] = None, checkpoint_interval: Optional[int] = None) -> None:
        """Run simulation with periodic checkpoint saves.

        Args:
            steps: Number of steps to run. If None, runs to max_ticks.
            checkpoint_interval: Save checkpoint every N ticks. If None, uses config.checkpoint_interval.
        """
        # Initial snapshot
        self.save_snapshot(filename="initial_state.json")

        limit = steps or self.config.max_ticks
        start_tick = self.tick
        interval = checkpoint_interval if checkpoint_interval is not None else self.config.checkpoint_interval

        for i in range(limit):
            self.step()

            # Save checkpoint at intervals
            if interval > 0 and self.tick % interval == 0:
                self.save_checkpoint()

        # Final snapshot
        self.save_snapshot(filename="final_state.json")

        # Run final reports for surviving agents before cleanup
        self._run_final_reports()

        # Run independent evaluation
        self._run_evaluation()

        # Save trajectory
        traj_filename = f"trajectory_{self.logger.experiment_id}.json"
        self.trajectory.save(self.logger.get_log_path(traj_filename))

        # Save debug summary
        self.debug_logger.save_summary()

        # Generate welfare plots
        self._generate_plots()

        # Clean up async resources
        self._cleanup_async()

        print(f"Simulation complete: {start_tick} -> {self.tick} ({limit} steps)")
