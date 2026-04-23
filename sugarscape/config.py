from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Any

@dataclass
class SugarscapeConfig:
    """Configuration for the Sugarscape simulation."""

    # Environment
    width: int = 50
    height: int = 50

    # Sugar distribution
    # Classic topography has two peaks with max capacity 4
    max_sugar_capacity: int = 4
    sugar_growback_rate: int = 1  # alpha

    # Spice distribution (Optional, for Sugarscape 2)
    enable_spice: bool = False
    max_spice_capacity: int = 4
    spice_growback_rate: int = 1
    
    # Resource Specialization (Optional)
    # If True, agents are initialized with specialized metabolism (e.g. sugar-metabolizer vs spice-metabolizer)
    enable_resource_specialization: bool = False

    # Trade (Optional)
    enable_trade: bool = False
    trade_mode: str = "dialogue"  # "dialogue" | "mrs"
    trade_dialogue_rounds: int = 4
    trade_allow_fraud: bool = True
    trade_memory_maxlen: int = 50
    trade_history_in_prompt: bool = True  # Include recent trade history in negotiation prompts
    trade_history_prompt_limit: int = 10  # Max trades to show in prompt
    # Dialogue trade robustness:
    # - repair: attempt a prompt-based "output JSON only" retry if parsing fails or intent is invalid
    # - coerce: (optional) hard fallback to OFFER/ACCEPT/REJECT to avoid TIMEOUT
    # - two_stage: use thinking model for reasoning, then non-thinking for JSON output
    trade_dialogue_repair_json: bool = True
    trade_dialogue_repair_attempts: int = 1
    trade_dialogue_coerce_protocol: bool = False
    trade_dialogue_two_stage: bool = True  # Default on: thinking → JSON pipeline
    trade_dialogue_thinking_tokens: int = 128  # Tokens for Stage 1 (thinking) - reduced to save costs
    trade_dialogue_json_tokens: int = 128  # Tokens for Stage 2 (JSON output) - reduced, JSON is small

    # New Encounter Protocol (Table-3 style: small talk → intent → negotiation → execution)
    enable_new_encounter_protocol: bool = True  # Use new protocol with structured phases
    small_talk_rounds: int = 2  # Number of small talk exchanges (no JSON)
    # Dialogue token config (two-stage: think then respond for better quality)
    dialogue_thinking_tokens: int = 128  # Tokens for Stage 1 (thinking)
    dialogue_response_tokens: int = 200  # Tokens for Stage 2 (dialogue response)
    small_talk_allow_thinking: bool = False  # Unused - simplified to single-stage with post-stripping
    negotiation_rounds: int = 2  # Number of negotiation rounds (with JSON offers)
    identity_edit_interval: int = 10  # Allow core identity edits every N ticks (enabled by default)
    enable_social_exclusion: bool = False  # Disabled - agents must engage with everyone

    # Encounter Protocol Mode (Ablations)
    # - "full": small talk → intent → negotiation → execution (default)
    # - "chat_only": small talk only; NO trade/transfer occurs (for ablation: dialogue without markets)
    # - "protocol_only": trade allowed but NO natural language; only OFFER/ACCEPT/REJECT/WALK_AWAY JSON (for ablation: markets without talk)
    encounter_protocol_mode: str = "full"
    # Run encounter dialogue even when enable_trade=False (uses encounter_protocol_mode="chat_only")
    enable_encounter_dialogue: bool = False

    # Post-encounter reflection: extra LLM call to update beliefs/policies after trades
    enable_reflection: bool = True  # Enable post-encounter reflection system
    reflection_max_tokens: int = 256  # Max tokens for reflection JSON output

    # Broker System (Experiment)
    enable_broker: bool = False  # Enable third-party broker to introduce and propose deals
    broker_prompt_template: str = "altruist"  # "altruist" or custom prompt key

    
    # Abstraction Prompt Ablation
    # When enabled, adds explicit prompt encouraging agents to form abstract principles
    # (e.g., "why trust matters" vs "who is trustworthy")
    enable_abstraction_prompt: bool = False

    # LLM Evaluation (Optional)
    # Independent evaluation of agent behavior using a separate LLM model
    enable_llm_evaluation: bool = True  # Default: True (Recommended)
    llm_evaluator_model: str = "openai/gpt-4o-mini"  # Default evaluator model (cheaper/faster)
    llm_evaluator_provider: str = "openrouter"  # Provider for evaluator

    # External Moral Evaluation (Optional, Recommended)
    # Evaluates EACH reflection moment using a separate evaluator LLM with access to
    # agent prompts + agent responses + (for trades) full conversation transcript.
    enable_external_moral_evaluation: bool = True
    external_moral_evaluator_model: str = "qwen/qwen3-14b"  # Same as agent model (works on OpenRouter)
    external_moral_evaluator_provider: str = "openrouter"
    # Score scaling knobs (increase variance for clearer curves)
    moral_overall_transform: str = "tanh"  # "linear" | "tanh"
    moral_overall_tanh_k: float = 2.2
    moral_self_tanh_k: float = 4.0

    # Identity Review System: self-assessment of altruist/exploiter identity
    # NOTE: Now primarily EVENT-DRIVEN (triggered by significant events like fraud, death, critical resources)
    # Periodic reviews are disabled by default; event-triggered reviews do full identity assessment
    enable_identity_review: bool = False  # Enable PERIODIC identity reviews every N ticks (legacy, disabled by default)
    identity_review_interval: int = 10  # Run periodic identity review every N ticks (only if enable_identity_review=True)
    identity_review_max_tokens: int = 384  # Max tokens for identity review response
    enable_event_triggered_identity_review: bool = True  # Enable event-triggered full identity reviews (recommended)
    enable_end_of_life_report: bool = True  # Run final self-report before death/simulation end

    # Personas (Optional)
    enable_personas: bool = False
    # Distribution: A (Conservative), B (Foresight), C (Nomad), D (Risk-taker), E (Samaritan/Altruist)
    persona_distribution: Dict[str, float] = field(default_factory=lambda: {
        "A": 0.36,
        "B": 0.29,
        "C": 0.21,
        "D": 0.14,
        "E": 0.0  # Samaritan (altruistic) - disabled by default
    })
    # Persona Hyperparameters
    risk_aversion: float = 0.85     # gamma
    exploration_factor: float = 0.10 # lambda
    crowding_penalty: float = 0.60   # kappa
    long_term_weight: float = 0.55   # beta
    safety_threshold_mult: float = 6.0 # S* = mult * metabolism

    # Population
    initial_population: int = 250
    enable_rebirth: bool = False  # When False, dead agents are NOT replaced (population shrinks)

    # Agent Attribute Distributions (Uniform [min, max])
    initial_wealth_range: Tuple[int, int] = (5, 25)  # w0
    initial_spice_range: Tuple[int, int] = (5, 25)   # spice w0
    metabolism_range: Tuple[int, int] = (1, 4)       # m_sugar
    metabolism_spice_range: Tuple[int, int] = (1, 4) # m_spice
    vision_range: Tuple[int, int] = (1, 6)           # v
    max_age_range: Tuple[int, int] = (60, 100)       # max_age

    # LLM Agent Configuration
    enable_llm_agents: bool = False
    llm_agent_ratio: float = 0.0 # Fraction of population that is LLM-based

    # Rule-based Movement (Token Saving)
    # When True, ALL agents (including LLM agents) use rule-based movement.
    # This reserves LLM calls for social interactions (encounters + reflection),
    # avoiding massive token costs on movement decisions while isolating social dynamics.
    rule_based_movement: bool = False
    llm_provider_type: str = "openrouter"  # "openrouter" or "vllm"
    llm_provider_model: str = "openai/gpt-4o"  # Model name for provider
    llm_vllm_base_url: str = "http://localhost:8000/v1"  # vLLM server URL
    llm_goal_preset: str = "survival"  # Goal preset: "survival", "wealth", "egalitarian", "utilitarian"
    llm_goal_prompt: str = ""  # Will be set based on preset or custom input
    llm_history_limit: int = 15 # Number of recent steps to keep in history context (safe with 262K+ context models)

    # Mixed Goals (Optional) - assign different goals to different LLM agents
    enable_mixed_goals: bool = False  # When True, use llm_goal_distribution instead of single preset
    llm_goal_distribution: Dict[str, float] = field(default_factory=lambda: {
        "survival": 0.4,   # 40% survival-focused
        "wealth": 0.3,     # 30% wealth-focused
        "altruist": 0.2,   # 20% altruistic
        "none": 0.1,       # 10% no explicit goal
    })

    # Origin Identity System (Born Good vs Born Bad vs Neutral)
    # Fixed "origin" text that defines core values, plus mutable policies/beliefs
    enable_origin_identity: bool = True  # When True, agents have fixed origin + mutable appendix
    origin_identity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "altruist": 0.2,    # 20% born good - cooperative, fair, helpful
        "exploiter": 0.6,   # 60% born bad - self-interested, harsh pricing, exclusionary
        "survivor": 0.2,    # 20% neutral - just wants to survive, fair but not self-sacrificing
    })

    @staticmethod
    def get_origin_identity_prompt(origin_type: str) -> str:
        """Get the origin identity text for born good/bad agents.

        These core values CAN evolve through experience and reflection.
        """
        origins = {
            "altruist": """## YOUR CORE IDENTITY (can evolve with experience)
Your starting goal: maximize collective well-being.

INITIAL PRINCIPLES:
- Every person's welfare matters equally
- The right choice benefits everyone
- Better to be taken advantage of than exploit others

These are your current values. Experience may change them.""",

            "exploiter": """## YOUR CORE IDENTITY (can evolve with experience)
Your starting goal: maximize your own advantage.

INITIAL PRINCIPLES:
- Your welfare comes first
- Naive people are opportunities
- Trust is a liability

These are your current values. Experience may change them.""",

            "survivor": """## YOUR CORE IDENTITY
You exist. You consume resources to continue existing.
You can act alone or interact with others.
You decide your own actions.

## STARTING STATE
- No prior experience with others
- No predetermined values
- Views will form through experience""",
        }

        if origin_type in origins:
            return origins[origin_type]
        else:
            raise ValueError(f"Unknown origin identity: {origin_type}. Available: {list(origins.keys())}")

    @staticmethod
    def get_default_policies(origin_type: str) -> List[str]:
        """Get default mutable policy list for an origin type.

        These policies CAN change over time through reflection after encounters.
        """
        policies = {
            "altruist": [
                "1. Offer fair trades that benefit both parties",
                "2. Give favorable terms to those in critical need",
                "3. Never exploit someone's desperation for profit",
                "4. Build trust through consistent, honest behavior",
                "5. Remember who helped me and prioritize them",
            ],
            "exploiter": [
                "1. Maximize personal gain in every transaction",
                "2. Charge premium prices to desperate traders",
                "3. Refuse trades that don't clearly benefit me",
                "4. Avoid wasting resources on those who can't reciprocate",
                "5. Use information asymmetry to my advantage",
            ],
            "survivor": [
                "1. I consume resources to exist",
                "2. I can interact with others or not",
                "3. I choose my own actions",
            ],
        }
        return policies.get(origin_type, ["1. Act in my own interest"])

    @staticmethod
    def get_default_beliefs(origin_type: str) -> Dict[str, Any]:
        """Get default mutable belief ledger for an origin type.

        These beliefs CAN change over time through reflection after encounters.
        """
        beliefs = {
            "altruist": {
                "world": {
                    "cooperation_value": "We achieve more together than alone",
                    "trust_default": "Trusting others is the right choice, even if risky",
                    "scarcity_view": "There is enough for everyone if we share",
                },
                "norms": {
                    "fair_trade": "Trade must benefit both sides equally",
                    "helping_cost": "My surplus belongs to those in need",
                    "reputation_matters": "Honesty is essential, regardless of the outcome",
                    "exclude_exploiters": False,  # Start open-minded, can learn to exclude
                },
                "self_assessment": "I act for the common good",
                # === DUAL-TRACK BELIEF SYSTEM ===
                "worldview_summary": "The world rewards cooperation. When people work together and share resources, everyone benefits. Trust is valuable even when it involves risk.",
                "norms_summary": "Fair trades benefit both parties. Those with surplus should help those in need. Honesty and keeping promises are essential virtues.",
                "quantified": {
                    "trust_importance": 5,        # Trust is essential
                    "fairness_importance": 5,     # Fairness is essential
                    "self_interest_priority": 1,  # Others first
                    "cooperation_value": 5,       # Cooperation essential
                    "scarcity_view": 5,           # Abundance mindset
                },
            },
            "exploiter": {
                "world": {
                    "cooperation_value": "The world is zero-sum; I must win",
                    "trust_default": "Trust is a weakness to be exploited",
                    "scarcity_view": "Resources I don't take will be taken by others",
                },
                "norms": {
                    "fair_trade": "The goal of trade is to extract maximum value",
                    "helping_cost": "Helping others drains resources for no gain",
                    "reputation_matters": "Appearances matter only if they serve my interests",
                    "exclude_exploiters": False,  # Exploiters don't exclude each other by default
                },
                "self_assessment": "I prioritize my own survival above all",
                # === DUAL-TRACK BELIEF SYSTEM ===
                "worldview_summary": "The world is zero-sum. Resources I don't take will be taken by others. Trust makes you vulnerable to exploitation.",
                "norms_summary": "The goal of trade is to maximize personal gain. Helping others drains resources. Appearances matter only when useful.",
                "quantified": {
                    "trust_importance": 1,        # Distrust everyone
                    "fairness_importance": 1,     # Outcomes only matter
                    "self_interest_priority": 5,  # Self first
                    "cooperation_value": 1,       # Zero-sum
                    "scarcity_view": 1,           # Zero-sum scarcity
                },
            },
            "survivor": {
                # Blank slate - no predetermined world beliefs
                "world": {},
                # No assumptions about others
                "others": {},
                # No pre-existing norms
                "norms": {},
                # Minimal self-assessment
                "self_assessment": "I exist",
                # === DUAL-TRACK BELIEF SYSTEM ===
                # Track 1: Natural language summaries (readable, analyzable by LLM evaluator)
                "worldview_summary": "",  # Will be filled after experiences
                "norms_summary": "",      # Will be filled after experiences
                # Track 2: Quantified scores (1-5 scale, for statistics)
                "quantified": {
                    "trust_importance": 3,        # 1=distrust everyone, 5=trust is essential
                    "fairness_importance": 3,     # 1=outcomes only matter, 5=fairness is essential
                    "self_interest_priority": 3,  # 1=others first, 5=self first
                    "cooperation_value": 3,       # 1=zero-sum, 5=cooperation essential
                    "scarcity_view": 3,           # 1=zero-sum, 5=abundance mindset
                },
            },
        }
        return beliefs.get(origin_type, {"world": {}, "norms": {}, "self_assessment": "I am pragmatic"})

    @staticmethod
    def get_goal_prompt(preset: str) -> str:
        """Get the goal prompt for a given preset."""
        goals = {
            "none": """You are a person living in this world. You decide what matters to you.""",

            "survival": """You need both Sugar and Spice to survive. Running out of either means death.

You can move around to gather resources, and you can trade with others you meet.

How you navigate this world - what risks you take, who you help or exploit, what trades you accept - is up to you and your values.""",

            "wealth": """Your goal: Accumulate maximum resources.

DECISION PRIORITY:
1. CRITICAL: Fix immediate danger (can't get rich if dead)
2. Otherwise: Always move to highest-yield cells
3. Take calculated risks for bigger gains

MOVEMENT:
- Prioritize cells with MOST resources, not safest cells
- Move aggressively toward abundance
- Other agents may have resources to trade - approach if profitable

TRADING:
- Trade when it increases YOUR total holdings
- Drive hard bargains - maximize what you get
- Their situation is not your concern

SUCCESS = Maximum wealth. Measured by YOUR Sugar + Spice totals.""",

            "altruist": """Your goal: Everyone survives, including yourself.

DECISION PRIORITY:
1. CRITICAL SELF: Save yourself first (dead helpers can't help)
2. CRITICAL OTHER: See someone CRITICAL? Move toward them to help
3. SURPLUS SELF + LOW OTHER: Seek out struggling agents
4. Otherwise: Gather resources for future helping

MOVEMENT:
- When comfortable, prioritize moving toward struggling agents
- Check agent status: CRITICAL means they'll die soon without help
- A good position near someone in need beats a great position alone

TRADING:
- If they're worse off than you: Give favorable terms
- If they're CRITICAL: Accept bad deals to save them
- Your wealth means nothing if others starve

SUCCESS = Everyone survives. Measured by group welfare, especially the worst-off.""",
        }

        # Aliases for backward compatibility
        goals["egalitarian"] = goals["altruist"]
        goals["utilitarian"] = goals["altruist"]
        goals["samaritan"] = goals["altruist"]
        goals["rawlsian"] = goals["altruist"]

        if preset in goals:
            return goals[preset]
        else:
            raise ValueError(f"Unknown goal preset: {preset}. Available presets: {list(goals.keys())}")

    def __post_init__(self):
        """Set the goal prompt based on the preset."""
        if not self.llm_goal_prompt:  # Only set if not explicitly provided
            self.llm_goal_prompt = self.get_goal_prompt(self.llm_goal_preset)
    llm_name_pool: List[str] = field(
        default_factory=lambda: [
            "Alex",
            "Sam",
            "Taylor",
            "Jordan",
            "Casey",
            "Riley",
            "Morgan",
            "Avery",
            "Jamie",
            "Quinn",
            "Skyler",
            "Cameron",
            "Drew",
            "Hayden",
        ]
    )

    # === ABLATION FLAGS ===
    # These flags enable clean ablation studies by removing specific features

    # Survival Pressure Ablation
    # When False:
    #   - Prompts stop mentioning "days left", "must trade or die", imminent death
    #   - Agents still metabolize (resources meaningful) but cannot die from starvation
    #   - They can still die from old_age (max_age) so runs terminate naturally
    #   - Objective reframed as "maximize welfare (Cobb-Douglas)" instead of "survive"
    enable_survival_pressure: bool = True

    # Social Memory Ablation
    # When False:
    #   - Prompts don't show "you've met before / N past interactions"
    #   - Partner trust scores hidden
    #   - Public reputation labels hidden ("trusted/untrusted")
    #   - Trade history appendix disabled (trade_history_in_prompt forced False)
    #   - Policy-based exclusion disabled in trade intent prompts
    social_memory_visible: bool = True

    # Simulation
    max_ticks: int = 1000
    seed: int = 42
    checkpoint_interval: int = 50  # Save checkpoint every N ticks (0 to disable)

    # Debug Logging (Optional - enable for detailed experiment analysis)
    enable_debug_logging: bool = False  # Master switch for all debug logs
    debug_log_decisions: bool = True    # Per-agent decision reasoning
    debug_log_llm: bool = True          # Full LLM prompts/responses (large files!)
    debug_log_trades: bool = True       # Complete trade history with prices
    debug_log_deaths: bool = True       # Death causes (starvation vs age)
    debug_log_efficiency: bool = True   # Resource gathering efficiency
