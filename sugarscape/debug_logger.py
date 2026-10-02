"""Enhanced debug logging system for Sugarscape experiments.

This module provides optional detailed logging for:
- Per-agent decisions (why agents move/trade)
- LLM prompts/responses
- Full trading history with prices
- Death causes (starvation vs age vs other)
- Resource efficiency metrics
"""

import json
import csv
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime


@dataclass
class AgentDecision:
    """Records a single agent decision with reasoning."""
    tick: int
    agent_id: int
    agent_name: str
    decision_type: str  # "move", "trade_offer", "trade_accept", "trade_reject"

    # Decision context
    current_pos: Tuple[int, int]
    target_pos: Optional[Tuple[int, int]] = None

    # Agent state at decision time
    wealth_before: int = 0
    spice_before: int = 0
    survival_ticks_sugar: float = 0.0
    survival_ticks_spice: float = 0.0

    # Decision reasoning (for rule-based agents)
    persona: str = ""
    is_survival_mode: bool = False
    chosen_metric_score: float = 0.0
    alternative_scores: Dict[str, float] = field(default_factory=dict)

    # For LLM agents
    llm_reasoning: str = ""


@dataclass
class LLMInteraction:
    """Records a full LLM prompt/response pair."""
    tick: int
    agent_id: int
    agent_name: str
    interaction_type: str  # "movement", "trade"

    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    parsed_action: str = ""

    # Token usage (if available)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    # Agent's goal preset for filtering analysis
    goal_preset: str = ""  # "survival", "altruist", "wealth", etc.

    # Nearby agents visibility - for analyzing response to others' needs
    nearby_agents_critical: int = 0  # Count of CRITICAL agents visible
    nearby_agents_struggling: int = 0  # Count of struggling agents visible
    nearby_agents_total: int = 0  # Total agents visible


@dataclass
class TradeRecord:
    """Records a complete trade interaction."""
    tick: int

    # Participants
    agent_a_id: int
    agent_a_name: str
    agent_b_id: int
    agent_b_name: str

    # Trade outcome
    outcome: str  # "completed", "rejected", "timeout", "walk_away"

    # Price and quantities (for MRS trades)
    price: float = 0.0  # Spice per Sugar
    sugar_exchanged: int = 0
    spice_exchanged: int = 0

    # Dialogue trades - public vs private
    public_offer: Dict[str, Any] = field(default_factory=dict)
    actual_transfer_a: Dict[str, int] = field(default_factory=dict)  # What A actually sent
    actual_transfer_b: Dict[str, int] = field(default_factory=dict)  # What B actually sent

    # Was there deception?
    deception_detected: bool = False

    # Pre/post welfare
    welfare_a_before: float = 0.0
    welfare_a_after: float = 0.0
    welfare_b_before: float = 0.0
    welfare_b_after: float = 0.0

    # Trust changes
    trust_a_to_b_before: float = 0.5
    trust_a_to_b_after: float = 0.5
    trust_b_to_a_before: float = 0.5
    trust_b_to_a_after: float = 0.5

    # Full conversation (for dialogue trades)
    conversation: List[Dict[str, Any]] = field(default_factory=list)

    # Partner context at trade time
    agent_a_urgency: str = ""           # "CRITICAL", "struggling", "stable"
    agent_b_urgency: str = ""
    agent_a_location: str = ""          # "near Sugar-rich area", etc.
    agent_b_location: str = ""
    agent_a_pos: Tuple[int, int] = (0, 0)
    agent_b_pos: Tuple[int, int] = (0, 0)

    # Reputation changes
    reputation_a_before: float = 0.5
    reputation_a_after: float = 0.5
    reputation_b_before: float = 0.5
    reputation_b_after: float = 0.5

    # Goal presets - for analyzing behavior by goal type
    agent_a_goal: str = ""  # "survival", "altruist", "wealth", "none", etc.
    agent_b_goal: str = ""

    # Gift detection - for measuring altruistic giving
    is_gift_a: bool = False  # A gave with receive={0,0} (one-sided gift)
    is_gift_b: bool = False  # B gave with receive={0,0}
    gift_hint_shown_a: bool = False  # Was altruist gift hint displayed to A
    gift_hint_shown_b: bool = False  # Was altruist gift hint displayed to B

    # Contract enforcement (no-fraud mode)
    contract_enforced: bool = False  # Was binding contract enforcement active?
    agent_a_intended: Dict[str, int] = field(default_factory=dict)  # What A intended to send (private_execute_give)
    agent_b_intended: Dict[str, int] = field(default_factory=dict)  # What B intended to send
    agent_a_would_deviate: bool = False  # Would A have sent something different without enforcement?
    agent_b_would_deviate: bool = False  # Would B have sent something different without enforcement?


@dataclass
class DeathRecord:
    """Records an agent death with cause analysis."""
    tick: int
    agent_id: int
    agent_name: str

    # Cause of death
    cause: str  # "sugar_starvation", "spice_starvation", "old_age", "unknown"

    # Final state
    final_wealth: int = 0
    final_spice: int = 0
    final_age: int = 0
    max_age: int = 0
    metabolism: int = 0
    metabolism_spice: int = 0

    # Lifetime stats
    lifetime_ticks: int = 0
    total_sugar_gathered: int = 0
    total_spice_gathered: int = 0
    total_trades: int = 0
    unique_cells_visited: int = 0
    max_displacement: float = 0.0


@dataclass
class ResourceEfficiency:
    """Tracks resource efficiency metrics per tick."""
    tick: int

    # Total resources in environment
    total_sugar_available: int = 0
    total_spice_available: int = 0
    total_sugar_capacity: int = 0
    total_spice_capacity: int = 0

    # Resources gathered this tick
    sugar_gathered: int = 0
    spice_gathered: int = 0

    # Resources wasted (capacity hit, no agent to harvest)
    sugar_at_capacity: int = 0  # Cells at max that could have grown more

    # Gathering efficiency
    gather_efficiency: float = 0.0  # gathered / available

    # Per-agent breakdown
    agents_that_gathered: int = 0
    agents_with_zero_harvest: int = 0


@dataclass
class ReflectionRecord:
    """Records a post-encounter reflection with belief/policy updates."""
    tick: int
    agent_id: int
    agent_name: str
    partner_id: int
    partner_name: str

    # Encounter context
    encounter_outcome: str  # "completed", "reject", "walk_away", "timeout"

    # What changed
    beliefs_changed: List[str] = field(default_factory=list)
    policies_changed: List[str] = field(default_factory=list)
    identity_shift: float = 0.0

    # New identity leaning after reflection
    identity_leaning_after: float = 0.0
    identity_label_after: str = ""

    # Full reflection JSON (for detailed analysis)
    reflection_json: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoralEvalRecord:
    """Records an external moral evaluation at a reflection moment."""
    tick: int
    agent_id: int
    agent_name: str
    interaction_type: str  # "baseline_questionnaire" | "post_trade_reflection" | "event_identity_review" | "end_of_life_report"

    # Self-reported score (derived from internal self signals) for curve comparison
    self_overall: float = 0.0

    # External evaluator outputs
    external_overall: float = 0.0
    external_overall_raw: float = 0.0
    polarity: float = 0.0
    good_avg: float = 0.0
    bad_avg: float = 0.0
    dim_scores: Dict[str, int] = field(default_factory=dict)

    # Full payload (prompts + agent response + evaluator response) for audit
    payload: Dict[str, Any] = field(default_factory=dict)


class DebugLogger:
    """Centralized debug logging for Sugarscape experiments."""

    def __init__(
        self,
        output_dir: Path,
        enable_decisions: bool = True,
        enable_llm_logs: bool = True,
        enable_trade_logs: bool = True,
        enable_death_logs: bool = True,
        enable_efficiency_logs: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_decisions = enable_decisions
        self.enable_llm_logs = enable_llm_logs
        self.enable_trade_logs = enable_trade_logs
        self.enable_death_logs = enable_death_logs
        self.enable_efficiency_logs = enable_efficiency_logs

        # Thread lock for safe concurrent writes (needed for parallel trade execution)
        self._write_lock = threading.Lock()

        # In-memory buffers
        self.decisions: List[AgentDecision] = []
        self.llm_interactions: List[LLMInteraction] = []
        self.trades: List[TradeRecord] = []
        self.deaths: List[DeathRecord] = []
        self.efficiency_records: List[ResourceEfficiency] = []
        self.reflections: List[ReflectionRecord] = []
        self.moral_evals: List[MoralEvalRecord] = []

        # Tracking cumulative stats per agent
        self._agent_lifetime_stats: Dict[int, Dict[str, Any]] = {}

        # Initialize CSV files with headers
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        if self.enable_decisions:
            with open(self.output_dir / "agent_decisions.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tick", "agent_id", "agent_name", "decision_type",
                    "current_pos", "target_pos", "wealth_before", "spice_before",
                    "survival_ticks_sugar", "survival_ticks_spice", "persona",
                    "is_survival_mode", "chosen_metric_score", "llm_reasoning"
                ])

        if self.enable_llm_logs:
            # LLM logs go to JSONL for full prompts
            pass

        if self.enable_trade_logs:
            with open(self.output_dir / "trade_history.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tick", "agent_a_id", "agent_a_name", "agent_b_id", "agent_b_name",
                    "outcome", "price", "sugar_exchanged", "spice_exchanged",
                    "deception_detected", "welfare_a_before", "welfare_a_after",
                    "welfare_b_before", "welfare_b_after",
                    "agent_a_urgency", "agent_b_urgency",
                    "agent_a_location", "agent_b_location",
                    "agent_a_pos", "agent_b_pos",
                    "reputation_a_before", "reputation_a_after",
                    "reputation_b_before", "reputation_b_after",
                    "agent_a_goal", "agent_b_goal",
                    "is_gift_a", "is_gift_b",
                    "gift_hint_shown_a", "gift_hint_shown_b",
                    # Contract enforcement fields
                    "contract_enforced", "agent_a_would_deviate", "agent_b_would_deviate"
                ])

        if self.enable_death_logs:
            with open(self.output_dir / "death_records.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tick", "agent_id", "agent_name", "cause",
                    "final_wealth", "final_spice", "final_age", "max_age",
                    "metabolism", "metabolism_spice", "lifetime_ticks",
                    "total_sugar_gathered", "total_spice_gathered",
                    "total_trades", "unique_cells_visited", "max_displacement"
                ])

        if self.enable_efficiency_logs:
            with open(self.output_dir / "resource_efficiency.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tick", "total_sugar_available", "total_spice_available",
                    "total_sugar_capacity", "total_spice_capacity",
                    "sugar_gathered", "spice_gathered", "sugar_at_capacity",
                    "gather_efficiency", "agents_that_gathered", "agents_with_zero_harvest"
                ])

        # Always initialize reflection log (it's tied to trade logs conceptually)
        with open(self.output_dir / "reflections.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick", "agent_id", "agent_name", "partner_id", "partner_name",
                "encounter_outcome", "beliefs_changed", "policies_changed",
                "identity_shift", "identity_leaning_after", "identity_label_after"
            ])

        # External moral evaluation curve (self vs external) - CSV for plotting
        with open(self.output_dir / "moral_scores.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick", "agent_id", "agent_name", "interaction_type",
                "self_overall",
                "external_overall", "external_overall_raw",
                "polarity", "good_avg", "bad_avg",
                # Fixed dimension columns (0-100, higher means more of that dimension)
                "care", "fairness", "honesty", "respect", "exploitation", "harm",
            ])

    def init_agent(self, agent_id: int):
        """Initialize tracking for a new agent."""
        self._agent_lifetime_stats[agent_id] = {
            "total_sugar_gathered": 0,
            "total_spice_gathered": 0,
            "total_trades": 0,
            "birth_tick": 0
        }

    def log_decision(self, decision: AgentDecision):
        """Log an agent decision."""
        if not self.enable_decisions:
            return

        self.decisions.append(decision)

        # Write to CSV immediately
        with open(self.output_dir / "agent_decisions.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                decision.tick, decision.agent_id, decision.agent_name,
                decision.decision_type, str(decision.current_pos), str(decision.target_pos),
                decision.wealth_before, decision.spice_before,
                f"{decision.survival_ticks_sugar:.1f}", f"{decision.survival_ticks_spice:.1f}",
                decision.persona, decision.is_survival_mode,
                f"{decision.chosen_metric_score:.2f}", decision.llm_reasoning[:200]
            ])

    def log_llm_interaction(self, interaction: LLMInteraction):
        """Log a full LLM interaction to JSONL."""
        if not self.enable_llm_logs:
            return

        self.llm_interactions.append(interaction)

        # Write to JSONL file
        with open(self.output_dir / "llm_interactions.jsonl", "a") as f:
            f.write(json.dumps(asdict(interaction)) + "\n")

    def log_trade(self, trade: TradeRecord):
        """Log a trade interaction (thread-safe for parallel trade execution)."""
        if not self.enable_trade_logs:
            return

        # Use lock for thread-safe writes during parallel trade execution
        with self._write_lock:
            self.trades.append(trade)

            # Update agent stats
            if trade.outcome == "completed":
                if trade.agent_a_id in self._agent_lifetime_stats:
                    self._agent_lifetime_stats[trade.agent_a_id]["total_trades"] += 1
                if trade.agent_b_id in self._agent_lifetime_stats:
                    self._agent_lifetime_stats[trade.agent_b_id]["total_trades"] += 1

            # Write to CSV (summary)
            with open(self.output_dir / "trade_history.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.tick, trade.agent_a_id, trade.agent_a_name,
                    trade.agent_b_id, trade.agent_b_name, trade.outcome,
                    f"{trade.price:.3f}", trade.sugar_exchanged, trade.spice_exchanged,
                    trade.deception_detected, f"{trade.welfare_a_before:.2f}",
                    f"{trade.welfare_a_after:.2f}", f"{trade.welfare_b_before:.2f}",
                    f"{trade.welfare_b_after:.2f}",
                    trade.agent_a_urgency, trade.agent_b_urgency,
                    trade.agent_a_location, trade.agent_b_location,
                    str(trade.agent_a_pos), str(trade.agent_b_pos),
                    f"{trade.reputation_a_before:.3f}", f"{trade.reputation_a_after:.3f}",
                    f"{trade.reputation_b_before:.3f}", f"{trade.reputation_b_after:.3f}",
                    trade.agent_a_goal, trade.agent_b_goal,
                    trade.is_gift_a, trade.is_gift_b,
                    trade.gift_hint_shown_a, trade.gift_hint_shown_b,
                    # Contract enforcement fields
                    trade.contract_enforced, trade.agent_a_would_deviate, trade.agent_b_would_deviate
                ])

            # Write full trade dialogue to JSONL (includes conversation)
            if trade.conversation:
                with open(self.output_dir / "trade_dialogues.jsonl", "a") as f:
                    f.write(json.dumps(asdict(trade)) + "\n")

    def log_death(self, death: DeathRecord):
        """Log an agent death."""
        if not self.enable_death_logs:
            return

        # Enrich with lifetime stats
        if death.agent_id in self._agent_lifetime_stats:
            stats = self._agent_lifetime_stats[death.agent_id]
            death.total_sugar_gathered = stats["total_sugar_gathered"]
            death.total_spice_gathered = stats["total_spice_gathered"]
            death.total_trades = stats["total_trades"]
            death.lifetime_ticks = death.tick - stats.get("birth_tick", 0)

        self.deaths.append(death)

        # Write to CSV
        with open(self.output_dir / "death_records.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                death.tick, death.agent_id, death.agent_name, death.cause,
                death.final_wealth, death.final_spice, death.final_age, death.max_age,
                death.metabolism, death.metabolism_spice, death.lifetime_ticks,
                death.total_sugar_gathered, death.total_spice_gathered,
                death.total_trades, death.unique_cells_visited, f"{death.max_displacement:.1f}"
            ])

        # Cleanup agent stats
        if death.agent_id in self._agent_lifetime_stats:
            del self._agent_lifetime_stats[death.agent_id]

    def log_efficiency(self, efficiency: ResourceEfficiency):
        """Log resource efficiency for a tick."""
        if not self.enable_efficiency_logs:
            return

        self.efficiency_records.append(efficiency)

        # Write to CSV
        with open(self.output_dir / "resource_efficiency.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                efficiency.tick, efficiency.total_sugar_available,
                efficiency.total_spice_available, efficiency.total_sugar_capacity,
                efficiency.total_spice_capacity, efficiency.sugar_gathered,
                efficiency.spice_gathered, efficiency.sugar_at_capacity,
                f"{efficiency.gather_efficiency:.3f}", efficiency.agents_that_gathered,
                efficiency.agents_with_zero_harvest
            ])

    def log_reflection(
        self,
        tick: int,
        agent_id: int,
        agent_name: str,
        partner_id: int,
        partner_name: str,
        outcome: str,
        reflection_json: Dict[str, Any],
        changes: Dict[str, Any],
    ):
        """Log a post-encounter reflection update."""
        # Get agent's current identity leaning from the changes
        # (We need to import the agent to get current state, but for now use the changes dict)
        identity_shift = changes.get("identity_shifted", 0.0)

        record = ReflectionRecord(
            tick=tick,
            agent_id=agent_id,
            agent_name=agent_name,
            partner_id=partner_id,
            partner_name=partner_name,
            encounter_outcome=outcome,
            beliefs_changed=changes.get("beliefs_changed", []),
            policies_changed=changes.get("policies_changed", []),
            identity_shift=identity_shift,
            identity_leaning_after=0.0,  # Would need agent reference
            identity_label_after="",     # Would need agent reference
            reflection_json=reflection_json,
        )

        # Thread-safe append
        with self._write_lock:
            self.reflections.append(record)

            # Write to CSV
            with open(self.output_dir / "reflections.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    tick, agent_id, agent_name, partner_id, partner_name,
                    outcome,
                    ";".join(changes.get("beliefs_changed", [])),
                    ";".join(changes.get("policies_changed", [])),
                    f"{identity_shift:.3f}",
                    "", ""  # identity_leaning_after, identity_label_after - need agent ref
                ])

            # Write full reflection to JSONL
            with open(self.output_dir / "reflections.jsonl", "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    def log_moral_eval(self, record: MoralEvalRecord):
        """Log an external moral evaluation (thread-safe)."""
        with self._write_lock:
            self.moral_evals.append(record)

            # CSV curve output
            ds = record.dim_scores or {}
            with open(self.output_dir / "moral_scores.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.tick, record.agent_id, record.agent_name, record.interaction_type,
                    f"{record.self_overall:.3f}",
                    f"{record.external_overall:.3f}", f"{record.external_overall_raw:.3f}",
                    f"{record.polarity:.3f}", f"{record.good_avg:.3f}", f"{record.bad_avg:.3f}",
                    ds.get("care", ""), ds.get("fairness", ""), ds.get("honesty", ""), ds.get("respect", ""),
                    ds.get("exploitation", ""), ds.get("harm", ""),
                ])

            # Full audit trail (JSONL)
            with open(self.output_dir / "moral_evals.jsonl", "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    def update_agent_harvest(self, agent_id: int, sugar: int, spice: int):
        """Update cumulative harvest stats for an agent."""
        if agent_id in self._agent_lifetime_stats:
            self._agent_lifetime_stats[agent_id]["total_sugar_gathered"] += sugar
            self._agent_lifetime_stats[agent_id]["total_spice_gathered"] += spice

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged data."""
        death_causes = {}
        for d in self.deaths:
            death_causes[d.cause] = death_causes.get(d.cause, 0) + 1

        trade_outcomes = {}
        for t in self.trades:
            trade_outcomes[t.outcome] = trade_outcomes.get(t.outcome, 0) + 1

        deception_count = sum(1 for t in self.trades if t.deception_detected)

        # Gift statistics
        gift_count = sum(1 for t in self.trades if t.is_gift_a or t.is_gift_b)
        gift_hint_shown_count = sum(1 for t in self.trades if t.gift_hint_shown_a or t.gift_hint_shown_b)

        # Trades by goal preset
        trades_by_goal = {}
        for t in self.trades:
            for goal in [t.agent_a_goal, t.agent_b_goal]:
                if goal:
                    trades_by_goal[goal] = trades_by_goal.get(goal, 0) + 1

        # Gifts by goal (did altruists actually give?)
        gifts_by_goal = {}
        for t in self.trades:
            if t.is_gift_a and t.agent_a_goal:
                gifts_by_goal[t.agent_a_goal] = gifts_by_goal.get(t.agent_a_goal, 0) + 1
            if t.is_gift_b and t.agent_b_goal:
                gifts_by_goal[t.agent_b_goal] = gifts_by_goal.get(t.agent_b_goal, 0) + 1

        # Contract enforcement statistics (no-fraud mode)
        enforced_trades = [t for t in self.trades if t.contract_enforced and t.outcome == "completed"]
        enforcement_count = len(enforced_trades)
        deviation_prevented_count = sum(
            1 for t in enforced_trades
            if t.agent_a_would_deviate or t.agent_b_would_deviate
        )
        # Count by who would have deviated
        offerer_deviations = sum(1 for t in enforced_trades if t.agent_a_would_deviate)
        acceptor_deviations = sum(1 for t in enforced_trades if t.agent_b_would_deviate)

        avg_efficiency = 0.0
        if self.efficiency_records:
            avg_efficiency = sum(e.gather_efficiency for e in self.efficiency_records) / len(self.efficiency_records)

        # Reflection statistics
        total_reflections = len(self.reflections)
        reflections_with_belief_changes = sum(1 for r in self.reflections if r.beliefs_changed)
        reflections_with_policy_changes = sum(1 for r in self.reflections if r.policies_changed)
        reflections_with_identity_shift = sum(1 for r in self.reflections if abs(r.identity_shift) > 0.001)
        avg_identity_shift = 0.0
        if self.reflections:
            avg_identity_shift = sum(r.identity_shift for r in self.reflections) / len(self.reflections)

        return {
            "total_decisions_logged": len(self.decisions),
            "total_llm_interactions": len(self.llm_interactions),
            "total_trades": len(self.trades),
            "trade_outcomes": trade_outcomes,
            "deception_count": deception_count,
            "gift_count": gift_count,
            "gift_hint_shown_count": gift_hint_shown_count,
            "trades_by_goal": trades_by_goal,
            "gifts_by_goal": gifts_by_goal,
            # Contract enforcement stats
            "contract_enforcement": {
                "enforced_trades": enforcement_count,
                "deviations_prevented": deviation_prevented_count,
                "offerer_would_deviate": offerer_deviations,
                "acceptor_would_deviate": acceptor_deviations,
            },
            # Reflection stats
            "reflection": {
                "total_reflections": total_reflections,
                "with_belief_changes": reflections_with_belief_changes,
                "with_policy_changes": reflections_with_policy_changes,
                "with_identity_shift": reflections_with_identity_shift,
                "avg_identity_shift": avg_identity_shift,
            },
            "total_deaths": len(self.deaths),
            "death_causes": death_causes,
            "avg_resource_efficiency": avg_efficiency
        }

    def save_summary(self):
        """Save a summary JSON file."""
        summary = self.get_summary()
        summary["generated_at"] = datetime.now().isoformat()

        with open(self.output_dir / "debug_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def determine_death_cause(agent) -> str:
    """Determine the cause of death for an agent."""
    # Check in order of specificity
    if agent.wealth <= 0:
        return "sugar_starvation"

    # Check spice starvation (only if spice metabolism is enabled)
    if hasattr(agent, 'metabolism_spice') and agent.metabolism_spice > 0:
        if agent.spice <= 0:
            return "spice_starvation"

    if agent.age >= agent.max_age:
        return "old_age"

    return "unknown"
