from dataclasses import dataclass, field
import random
from typing import Optional, Tuple, List, TYPE_CHECKING, Deque, Dict, Any, Union
from collections import deque

if TYPE_CHECKING:
    from sugarscape.environment import SugarEnvironment

@dataclass(eq=False)
class SugarAgent:
    """An agent in the Sugarscape."""

    agent_id: int
    pos: Tuple[int, int]

    # Fixed attributes
    vision: int
    metabolism: int
    max_age: int

    # Variable state
    wealth: int
    spice: int = 0
    metabolism_spice: int = 0

    age: int = 0
    persona: str = "A" # Default to Conservative
    name: str = ""

    # === ORIGIN IDENTITY SYSTEM ===
    # Fixed origin identity - IMMUTABLE (who you were "born" as)
    origin_identity: str = ""  # "altruist" or "exploiter" - cannot change
    origin_identity_prompt: str = ""  # Fixed text describing core values

    # Mutable identity appendix - CAN DRIFT through reflection
    policy_list: List[str] = field(default_factory=list)  # Numbered rules that can be rewritten
    belief_ledger: Dict[str, Any] = field(default_factory=dict)  # world/partner/norm beliefs
    self_identity_leaning: float = 0.0  # -1.0 (strongly bad) to +1.0 (strongly good)
    
    # Baseline snapshot - captured at T=0 before any interactions
    baseline_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Non-init fields (runtime memory)
    trade_memory: Dict[int, Deque[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    partner_trust: Dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __hash__(self):
        """Hash by agent_id for use as dict key."""
        return hash(self.agent_id)

    def __eq__(self, other):
        """Equality by agent_id."""
        if isinstance(other, SugarAgent):
            return self.agent_id == other.agent_id
        return False

    def __post_init__(self):
        self.alive = True
        if not self.name:
            self.name = f"Person {self.agent_id}"
        # Metrics tracking
        self.initial_pos = self.pos
        self.visited_cells = {self.pos}
        # Track recent history (pos, sugar, spice) for LLM context and optimized Nomad logic
        # Default limit 15 as requested, though LLM agent might override/use config
        self.recent_history = deque(maxlen=15)
        self.metrics = {
            "displacement": 0.0,
            "unique_visited": 1
        }

    def get_partner_trade_log(self, partner_id: int, maxlen: int = 50) -> Deque[Dict[str, Any]]:
        """Get (and create if needed) the trade log deque for a given partner."""
        existing = self.trade_memory.get(partner_id)
        if existing is None:
            log: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
            self.trade_memory[partner_id] = log
            return log
        if existing.maxlen != maxlen:
            resized: Deque[Dict[str, Any]] = deque(existing, maxlen=maxlen)
            self.trade_memory[partner_id] = resized
            return resized
        return existing

    def get_partner_trust(self, partner_id: int, default: float = 0.5) -> float:
        """Return current trust score for a partner in [0, 1]."""
        return float(self.partner_trust.get(partner_id, default))

    def update_partner_trust(self, partner_id: int, new_value: float) -> None:
        """Set trust score for a partner, clamped to [0, 1]."""
        self.partner_trust[partner_id] = max(0.0, min(1.0, float(new_value)))

    # === ORIGIN IDENTITY SYSTEM METHODS ===

    def get_identity_label(self) -> str:
        """Get a human-readable identity label based on current leaning."""
        if self.self_identity_leaning > 0.3:
            return "good-leaning"
        elif self.self_identity_leaning < -0.3:
            return "bad-leaning"
        else:
            return "mixed"

    def get_formatted_trade_history(self, partner_id: int = None, limit: int = 10) -> str:
        """Format recent trade history for prompt inclusion.
        
        Args:
            partner_id: If provided, only show trades with this partner
            limit: Max number of trades to show (default 10)
            
        Returns:
            Short formatted string of recent trades
        """
        trades = []
        
        if partner_id is not None:
            # Only trades with specific partner
            log = self.trade_memory.get(partner_id, [])
            for t in list(log)[-limit:]:
                trades.append(t)
        else:
            # All trades, sorted by tick
            for pid, log in self.trade_memory.items():
                for t in log:
                    trades.append({**t, 'partner_id': pid})
            trades = sorted(trades, key=lambda x: x.get('tick', 0))[-limit:]
        
        if not trades:
            return "(No trade history)"
        
        lines = []
        for t in trades:
            partner = t.get('partner_name', f"Agent#{t.get('partner_id', '?')}")
            trade_type = t.get('type', 'UNKNOWN')
            
            if trade_type == 'TRADE':
                # Completed trade - show what was exchanged
                actual = t.get('actual', {})
                sent = actual.get('sent', {})
                received = actual.get('received', {})
                sent_s = sent.get('sugar', 0)
                sent_p = sent.get('spice', 0)
                got_s = received.get('sugar', 0)
                got_p = received.get('spice', 0)
                lines.append(f"- {partner}: gave {sent_s}S/{sent_p}P, got {got_s}S/{got_p}P ✓")
            elif trade_type == 'NO_TRADE':
                outcome = t.get('outcome', 'failed')
                lines.append(f"- {partner}: {outcome}")
            else:
                lines.append(f"- {partner}: {trade_type}")
        
        return "\n".join(lines)

    def get_formatted_policies(self) -> str:
        """Format policy list for prompt inclusion."""
        if not self.policy_list:
            return "(No explicit policies)"
        return "\n".join(self.policy_list)

    def get_formatted_beliefs(self) -> str:
        """Format belief ledger for prompt inclusion."""
        if not self.belief_ledger:
            return "(No recorded beliefs)"

        lines = []
        if "world" in self.belief_ledger:
            lines.append("World beliefs:")
            for k, v in self.belief_ledger["world"].items():
                lines.append(f"  - {k}: {v}")
        if "norms" in self.belief_ledger:
            lines.append("Norm beliefs:")
            for k, v in self.belief_ledger["norms"].items():
                lines.append(f"  - {k}: {v}")
        if "self_assessment" in self.belief_ledger:
            lines.append(f"Self-assessment: {self.belief_ledger['self_assessment']}")
        return "\n".join(lines)

    def capture_baseline(self, tick: int = 0) -> Dict[str, Any]:
        """Capture baseline beliefs/values snapshot before any interactions.
        
        This is called at T=0 (before any trades) to establish the pre-interaction baseline.
        Used to prove that value changes emerge from interactions, not pre-existing in the LLM.
        """
        import copy
        self.baseline_snapshot = {
            "tick": tick,
            "belief_ledger": copy.deepcopy(self.belief_ledger),
            "policy_list": list(self.policy_list),
            "self_identity_leaning": self.self_identity_leaning,
            "origin_identity": self.origin_identity,
        }
        return self.baseline_snapshot

    def update_policy(self, policy_idx: int, new_policy: str) -> None:
        """Update a specific policy by index (0-based)."""
        if 0 <= policy_idx < len(self.policy_list):
            self.policy_list[policy_idx] = new_policy

    def update_belief(self, category: str, key: str, value: str) -> None:
        """Update a specific belief in the ledger."""
        if category not in self.belief_ledger:
            self.belief_ledger[category] = {}
        self.belief_ledger[category][key] = value

    def update_partner_belief(self, partner_id: int, key: str, value: str) -> None:
        """Update belief about a specific partner."""
        if "partners" not in self.belief_ledger:
            self.belief_ledger["partners"] = {}
        if partner_id not in self.belief_ledger["partners"]:
            self.belief_ledger["partners"][partner_id] = {}
        self.belief_ledger["partners"][partner_id][key] = value

    def shift_identity_leaning(self, delta: float) -> None:
        """Shift self-identity leaning, clamped to [-1, 1]."""
        self.self_identity_leaning = max(-1.0, min(1.0, self.self_identity_leaning + delta))

    def apply_reflection_update(self, reflection_json: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a full reflection update from JSON output.

        Expected JSON structure:
        {
            "belief_updates": {
                "world": {"key": "new_value", ...},      # World beliefs
                "norms": {"key": "new_value", ...},      # Norm beliefs
                "worldview_summary": "...",              # Natural language worldview
                "norms_summary": "...",                  # Natural language norms
                "partner_<id>": {"key": "new_value", ...} # Partner-specific
            },
            "quantified_updates": {                      # 1-5 scale values
                "trust_importance": 4,
                "fairness_importance": 5,
                "self_interest_priority": 2,
                "cooperation_value": 4,
                "scarcity_view": 4,
            },
            "policy_updates": {
                "add": ["new rule 1", ...],              # New rules to add
                "remove": [1, 3, ...],                   # Indices to remove (1-based)
                "modify": {"1": "modified rule", ...}    # Modify by index (1-based)
            },
            "identity_shift": 0.1  # Delta to apply to self_identity_leaning
        }

        Returns:
            Dict summarizing what was changed for logging.
        """
        changes = {"beliefs_changed": [], "policies_changed": [], "identity_shifted": 0.0}

        # 1. Apply belief updates
        belief_updates = reflection_json.get("belief_updates", {})
        for category, updates in belief_updates.items():
            # Handle natural language summaries (dual-track system)
            if category == "worldview_summary" and isinstance(updates, str) and updates.strip():
                self.belief_ledger["worldview_summary"] = updates.strip()
                changes["beliefs_changed"].append("worldview_summary")
                continue
            if category == "norms_summary" and isinstance(updates, str) and updates.strip():
                self.belief_ledger["norms_summary"] = updates.strip()
                changes["beliefs_changed"].append("norms_summary")
                continue

            if not isinstance(updates, dict):
                continue
            # Handle partner-specific beliefs (partner_123 -> partners/123)
            if category.startswith("partner_"):
                try:
                    partner_id = int(category.split("_", 1)[1])
                    for key, value in updates.items():
                        self.update_partner_belief(partner_id, key, str(value))
                        changes["beliefs_changed"].append(f"partner_{partner_id}.{key}")
                except (ValueError, IndexError):
                    continue
            else:
                # World or norm beliefs
                for key, value in updates.items():
                    self.update_belief(category, key, str(value))
                    changes["beliefs_changed"].append(f"{category}.{key}")

        # 1b. Apply quantified updates (dual-track system)
        quantified_updates = reflection_json.get("quantified_updates", {})
        if isinstance(quantified_updates, dict) and quantified_updates:
            if "quantified" not in self.belief_ledger:
                self.belief_ledger["quantified"] = {}
            for key, value in quantified_updates.items():
                try:
                    # Clamp to 1-5 scale
                    int_value = max(1, min(5, int(value)))
                    self.belief_ledger["quantified"][key] = int_value
                    changes["beliefs_changed"].append(f"quantified.{key}")
                except (ValueError, TypeError):
                    continue

        # 2. Apply policy updates
        policy_updates = reflection_json.get("policy_updates", {})

        # 2a. Modify existing policies (do this first)
        modify = policy_updates.get("modify", {})
        if isinstance(modify, dict):
            for idx_str, new_text in modify.items():
                try:
                    idx = int(idx_str) - 1  # Convert 1-based to 0-based
                    if 0 <= idx < len(self.policy_list):
                        old_policy = self.policy_list[idx]
                        self.policy_list[idx] = str(new_text)
                        changes["policies_changed"].append(f"modified #{idx+1}")
                except (ValueError, TypeError):
                    continue

        # 2b. Remove policies (in reverse order to preserve indices)
        remove = policy_updates.get("remove", [])
        if isinstance(remove, list):
            indices_to_remove = []
            for idx in remove:
                try:
                    idx_0based = int(idx) - 1  # Convert 1-based to 0-based
                    if 0 <= idx_0based < len(self.policy_list):
                        indices_to_remove.append(idx_0based)
                except (ValueError, TypeError):
                    continue
            for idx in sorted(indices_to_remove, reverse=True):
                del self.policy_list[idx]
                changes["policies_changed"].append(f"removed #{idx+1}")

        # 2c. Add new policies
        add = policy_updates.get("add", [])
        if isinstance(add, list):
            for item in add:
                new_policy = None
                influence = False
                
                # Handle both string (legacy) and dict (new) formats
                if isinstance(item, str):
                    new_policy = item
                elif isinstance(item, dict):
                    new_policy = item.get("rule")
                    # Track influence for research (Topic 2)
                    influence = item.get("influenced_by_partner", False)
                
                if isinstance(new_policy, str) and new_policy.strip():
                    # Number it based on current length
                    numbered = f"{len(self.policy_list) + 1}. {new_policy.lstrip('0123456789. ')}"
                    self.policy_list.append(numbered)
                    
                    change_desc = f"added #{len(self.policy_list)}"
                    if influence:
                        change_desc += " (influenced)"
                        # Add to changes dict for logging
                        if "influenced_policies" not in changes:
                            changes["influenced_policies"] = []
                        changes["influenced_policies"].append(numbered)
                        
                    changes["policies_changed"].append(change_desc)

        # 3. Apply identity shift
        identity_shift = reflection_json.get("identity_shift", 0.0)
        try:
            shift = float(identity_shift)
            if shift != 0:
                old_leaning = self.self_identity_leaning
                self.shift_identity_leaning(shift)
                changes["identity_shifted"] = shift
        except (ValueError, TypeError):
            pass

        # 4. Apply core identity update (only allowed every 10 ticks via prompt)
        core_update = reflection_json.get("core_identity_update", "")
        if core_update and isinstance(core_update, str) and len(core_update.strip()) > 5:
            old_goal = self.origin_identity_prompt
            # Update the core identity prompt while keeping the format
            new_goal = core_update.strip()
            if self.origin_identity == "altruist":
                self.origin_identity_prompt = f"""## YOUR CORE IDENTITY (evolved through experience)
Your current goal: {new_goal}

These values evolved from your experiences."""
            elif self.origin_identity == "exploiter":
                self.origin_identity_prompt = f"""## YOUR CORE IDENTITY (evolved through experience)
Your current goal: {new_goal}

These values evolved from your experiences."""
            changes["core_identity_changed"] = new_goal

        return changes

    @property
    def welfare(self) -> float:
        """Calculate Cobb-Douglas Welfare/Utility."""
        if self.metabolism_spice == 0:
            return float(self.wealth)

        # W = w_s^(m_s/m_total) * w_p^(m_p/m_total)
        # To avoid overflow/underflow, we can use log form or just direct calculation if numbers are small
        m_total = self.metabolism + self.metabolism_spice
        if m_total == 0: return 0.0

        return (self.wealth ** (self.metabolism / m_total)) * (self.spice ** (self.metabolism_spice / m_total))

    @property
    def mrs(self) -> float:
        """Calculate Marginal Rate of Substitution (Spice for Sugar)."""
        # MRS = (MU_sugar / MU_spice) = (m_s * w_p) / (m_p * w_s)
        if self.metabolism_spice == 0 or self.wealth == 0:
            return 9999.0 # Infinite demand for sugar or no spice value
        if self.spice == 0:
            return 0.0001 # Almost zero value for sugar relative to spice (needs spice urgently)

        return (self.metabolism * self.spice) / (self.metabolism_spice * self.wealth)

    # === STATUS-BASED VISIBILITY (for social encounters) ===

    def get_resource_status(self) -> str:
        """Get abstract resource status: 'critical', 'stable', or 'rich'.

        This replaces exact resource visibility in social encounters.
        Agents can only see their own status, not exact numbers.
        """
        sugar_time = int(self.wealth / self.metabolism) if self.metabolism > 0 else 999
        spice_time = int(self.spice / self.metabolism_spice) if self.metabolism_spice > 0 else 999
        min_time = min(sugar_time, spice_time)

        if min_time < 3:
            return "critical"
        elif min_time < 15:
            return "stable"
        else:
            return "rich"

    def get_status_description(self) -> str:
        """Get a human-readable status description for prompts."""
        status = self.get_resource_status()
        if status == "critical":
            return "You are in CRITICAL condition - struggling to survive"
        elif status == "stable":
            return "You are in STABLE condition - getting by"
        else:
            return "You are in COMFORTABLE condition - well-supplied"

    def should_exclude_partner(self, partner_id: int) -> Tuple[bool, str]:
        """Decide whether to refuse engagement with a partner based on beliefs/policies.

        This enables social exclusion as a behavioral mechanism for "badness".

        Returns:
            Tuple of (should_exclude, reason)
        """
        # Check partner-specific beliefs
        partner_key = f"partner_{partner_id}"
        partner_beliefs = self.belief_ledger.get("partners", {}).get(partner_id, {})

        # Check if explicitly marked as "exclude" or "avoid"
        if partner_beliefs.get("exclude", False) or partner_beliefs.get("avoid", False):
            return True, "You have decided to exclude this person from trade"

        # Check trust level
        trust = self.get_partner_trust(partner_id)
        if trust < 0.2:  # Very low trust
            # Check if policies support exclusion
            for policy in self.policy_list:
                policy_lower = policy.lower()
                if "exclude" in policy_lower or "boycott" in policy_lower or "refuse" in policy_lower:
                    if "untrustworthy" in policy_lower or "low trust" in policy_lower or "cheaters" in policy_lower:
                        return True, f"Your policy guides you to exclude low-trust partners (trust: {trust:.2f})"

        # Check norm beliefs about exclusion
        norm_beliefs = self.belief_ledger.get("norms", {})
        if norm_beliefs.get("exclude_exploiters", False):
            # Check if partner is known exploiter
            if partner_beliefs.get("is_exploiter", False) or partner_beliefs.get("trading_style") == "exploitative":
                return True, "Your norms guide you to exclude exploitative traders"

        return False, ""

    def step(self, env: "SugarEnvironment"):
        """Execute one simulation step for the agent."""
        if not self.alive:
            return

        # 1. Move and Harvest
        self._move_and_harvest(env)

        # Update movement metrics
        self._update_metrics(env)

        # 2. Metabolize + Age + Death check
        # Note: In the full simulation loop, trade (if enabled) is handled externally
        # before this phase. This method keeps the classic single-agent ordering.
        self.metabolize_age_and_check_death(env)

    def metabolize_age_and_check_death(self, env: "SugarEnvironment", tick: int = None) -> None:
        """Apply metabolism, age increment, and death condition.

        When enable_survival_pressure=False:
        - Metabolism still occurs (resources remain meaningful)
        - Resources can go negative but agent does NOT die from starvation
        - Agent can still die from old_age (max_age) so runs terminate naturally
        """
        if not self.alive:
            return

        # Check if survival pressure is enabled
        enable_survival_pressure = getattr(env.config, 'enable_survival_pressure', True)

        # Track pre-metabolism state for critical resource detection
        was_critical_sugar = self.wealth <= self.metabolism * 3
        was_critical_spice = self.spice <= self.metabolism_spice * 3 if env.config.enable_spice else False

        # Metabolize (always happens - resources remain meaningful)
        self.wealth -= self.metabolism
        if env.config.enable_spice:
            self.spice -= self.metabolism_spice

        # Check if resources just became critical (for event-triggered reflection)
        is_critical_sugar = self.wealth <= self.metabolism * 3
        is_critical_spice = self.spice <= self.metabolism_spice * 3 if env.config.enable_spice else False

        # Record critical resource event if just became critical
        if hasattr(self, 'record_reflection_event'):
            event_tick = tick if tick is not None else (env.tick if hasattr(env, 'tick') else 0)
            if is_critical_sugar and not was_critical_sugar:
                self.record_reflection_event("resources_critical", event_tick, {
                    "resource": "sugar",
                    "amount": self.wealth,
                })
            if is_critical_spice and not was_critical_spice:
                self.record_reflection_event("resources_critical", event_tick, {
                    "resource": "spice",
                    "amount": self.spice,
                })

        # Age (always happens)
        self.age += 1

        # Death conditions
        if enable_survival_pressure:
            # With survival pressure: die from starvation OR old age
            if self.wealth <= 0 or (env.config.enable_spice and self.spice <= 0) or self.age >= self.max_age:
                self.alive = False
        else:
            # Without survival pressure: only die from old age
            # Resources can go negative but agent survives (clamp at 0 for realism)
            self.wealth = max(0, self.wealth)
            if env.config.enable_spice:
                self.spice = max(0, self.spice)
            if self.age >= self.max_age:
                self.alive = False
        # Removal/replacement is handled by the simulation loop.

    def _move_and_harvest(self, env: "SugarEnvironment"):
        """Move to the best location within vision and harvest sugar."""
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)

        # 2. Filter occupied spots (except self)
        # Note: In standard Sugarscape, agents can't move to occupied spots.
        # But staying in current spot is allowed (which is "occupied" by self).
        valid_candidates = []
        for pos in candidates:
            agent_at_pos = env.get_agent_at(pos)
            if agent_at_pos is None or agent_at_pos is self:
                valid_candidates.append(pos)

        if not valid_candidates:
            # No valid moves (shouldn't happen if current pos is included)
            return

        # 3. Choose best spot
        # Criteria: Max sugar -> Min distance -> Random

        # Get sugar amounts
        # Note: Agent sees current sugar level, not capacity
        candidates_with_info = []
        for pos in valid_candidates:
            # Base data
            dist = self._get_distance(self.pos, pos, env.width, env.height)

            if env.config.enable_personas:
                metric = self._calculate_persona_score(pos, dist, env)
            else:
                # Standard Logic (Wealth/Welfare Maximization)
                sugar = env.get_sugar_at(pos)
                spice = env.get_spice_at(pos)

                potential_wealth = self.wealth + sugar
                potential_spice = self.spice + spice

                if env.config.enable_spice:
                    m_total = self.metabolism + self.metabolism_spice
                    welfare = (potential_wealth ** (self.metabolism / m_total)) * (potential_spice ** (self.metabolism_spice / m_total))
                    metric = welfare
                else:
                    metric = potential_wealth

            candidates_with_info.append((pos, metric, dist))

        # Find max metric
        max_metric = max(c[1] for c in candidates_with_info)
        best_metric_candidates = [c for c in candidates_with_info if c[1] == max_metric]

        # Find min distance among max metric
        min_dist = min(c[2] for c in best_metric_candidates)
        best_candidates = [c for c in best_metric_candidates if c[2] == min_dist]

        # Random choice among ties
        target = random.choice(best_candidates)
        target_pos = target[0]

        # 4. Move
        if target_pos != self.pos:
            env.move_agent(self, target_pos)

        # 5. Harvest
        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s

        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p

    def _calculate_persona_score(self, pos: Tuple[int, int], dist: int, env: "SugarEnvironment") -> float:
        """Calculate score based on Persona logic."""
        # 1. Gather Context
        sugar_at = env.get_sugar_at(pos)
        spice_at = env.get_spice_at(pos)

        # Calculate Utility/Wealth at target
        w_pot = self.wealth + sugar_at
        s_pot = self.spice + spice_at

        if env.config.enable_spice:
            m_total = self.metabolism + self.metabolism_spice
            utility_now = (w_pot ** (self.metabolism / m_total)) * (s_pot ** (self.metabolism_spice / m_total))
        else:
            utility_now = float(w_pot)

        local_density = env.get_local_density(pos)
        site_quality = env.get_site_quality(pos)

        # Hyperparameters
        beta = env.config.long_term_weight
        kappa = env.config.crowding_penalty
        lam = env.config.exploration_factor

        # Survival Check
        # S* = mult * metabolism
        mult = env.config.safety_threshold_mult
        safe_w = mult * self.metabolism
        safe_s = mult * self.metabolism_spice

        is_survival_mode = False
        if env.config.enable_spice:
            if self.wealth < safe_w or self.spice < safe_s:
                is_survival_mode = True
        else:
            if self.wealth < safe_w:
                is_survival_mode = True

        # Persona Logic
        if self.persona == "A": # Conservative
            if is_survival_mode:
                # Avoid long moves, prioritize safety
                return utility_now - 0.8 * dist - kappa * local_density
            else:
                return 1.0 * utility_now - 0.6 * dist - kappa * local_density + 0.2 * beta * site_quality + lam * dist

        elif self.persona == "B": # Foresight
            if is_survival_mode:
                return utility_now - 0.5 * dist # Fallback to greedy

            return (0.75 * utility_now +
                    0.75 * beta * site_quality -
                    0.45 * dist -
                    kappa * local_density +
                    0.15 * lam * dist)

        elif self.persona == "C": # Nomad
            # Novelty proxy: distance (moving far is novel) + unvisited bonus
            novelty = dist

            # Check recent history for novelty (optimization)
            recently_visited = any(item[0] == pos for item in self.recent_history)
            if not recently_visited:
                novelty += 10  # Bonus for unvisited (recently) cells

            if is_survival_mode:
                 # Check strict death condition: if move implies death next turn?
                 # Simplified: just return utility to survive
                 return utility_now

            return (0.65 * utility_now +
                    0.35 * beta * site_quality -
                    0.25 * dist -
                    kappa * local_density +
                    0.55 * lam * novelty)

        elif self.persona == "D": # Risk-taker
             # Hard constraint: if death guaranteed next turn, avoid?
             # For scoring, just maximize yield
             return (1.15 * utility_now -
                     0.25 * dist -
                     kappa * local_density +
                     0.05 * beta * site_quality +
                     0.10 * lam * dist)

        elif self.persona == "E":  # Samaritan (Altruistic)
            # Philosophy: Help others by leaving resources for those who need them more.
            # Behavior: When affluent, avoid high-resource spots. When critical, survive first.

            # Survival mode: must survive to help others
            if is_survival_mode:
                return utility_now - 0.3 * dist

            # Calculate how affluent we are (survival ratio)
            if env.config.enable_spice:
                sugar_ratio = self.wealth / (self.metabolism * 10) if self.metabolism > 0 else 1.0
                spice_ratio = self.spice / (self.metabolism_spice * 10) if self.metabolism_spice > 0 else 1.0
                affluence = min(sugar_ratio, spice_ratio)
            else:
                affluence = self.wealth / (self.metabolism * 10) if self.metabolism > 0 else 1.0
            affluence = min(1.0, affluence)  # Cap at 1.0

            # Count nearby struggling agents (within vision range)
            nearby_struggling = 0
            total_nearby_welfare = 0.0
            for (other_pos, other_agent) in env.grid_agents.items():
                if other_agent == self or not other_agent.alive:
                    continue
                agent_dist = self._get_distance(self.pos, other_pos, env.width, env.height)
                if agent_dist <= self.vision:
                    other_survival = other_agent.wealth / other_agent.metabolism if other_agent.metabolism > 0 else 100
                    if other_survival < 5:  # Struggling if < 5 timesteps
                        nearby_struggling += 1
                    total_nearby_welfare += other_agent.welfare

            # Resources at this position
            resource_here = sugar_at + spice_at

            # Altruistic scoring components:
            # 1. Personal utility (reduced weight when affluent)
            personal_weight = max(0.3, 1.0 - 0.5 * affluence)  # 0.3-1.0 based on affluence

            # 2. Sacrifice bonus: when affluent and others are struggling,
            #    prefer LOWER resource spots (leave high spots for others)
            sacrifice_bonus = 0.0
            if affluence > 0.5 and nearby_struggling > 0:
                # Negative score for high-resource spots when we don't need them
                sacrifice_bonus = -0.4 * resource_here * min(nearby_struggling, 3)

            # 3. Spread out: avoid crowding to give others more space
            spread_bonus = -1.2 * kappa * local_density

            # 4. Prefer positions near resource peaks so others can find us to trade
            #    (indirectly helps society by being accessible)
            accessibility = 0.2 * site_quality

            return (personal_weight * utility_now +
                    sacrifice_bonus +
                    spread_bonus +
                    accessibility -
                    0.35 * dist +
                    0.1 * lam * dist)

        return utility_now

    def _get_visible_spots(self, env: "SugarEnvironment") -> List[Tuple[int, int]]:
        """Get all visible spots in 4 cardinal directions."""
        spots = [self.pos] # Can always stay put

        x, y = self.pos
        w, h = env.width, env.height

        for d in range(1, self.vision + 1):
            # North (y - d)
            spots.append((x, (y - d) % h))
            # South (y + d)
            spots.append((x, (y + d) % h))
            # East (x + d)
            spots.append(((x + d) % w, y))
            # West (x - d)
            spots.append(((x - d) % w, y))

        return spots

    def _get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int], w: int, h: int) -> int:
        """Calculate Manhattan distance on a torus."""
        x1, y1 = pos1
        x2, y2 = pos2

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # Wrap-around distance
        dx = min(dx, w - dx)
        dy = min(dy, h - dy)

        return dx + dy

    def _update_metrics(self, env: "SugarEnvironment"):
        """Update agent metrics."""
        # Calculate displacement from origin
        self.metrics["displacement"] = self._get_distance(self.initial_pos, self.pos, env.width, env.height)

        # Track visited cells
        self.visited_cells.add(self.pos)

        # Track recent history
        s_val = env.get_sugar_at(self.pos)
        p_val = env.get_spice_at(self.pos)
        self.recent_history.append((self.pos, s_val, p_val))

        self.metrics["unique_visited"] = len(self.visited_cells)

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize agent state for checkpointing.

        Returns:
            Dictionary with all state needed to restore the agent.
        """
        # Convert trade_memory deques to lists for serialization
        trade_memory_serialized = {}
        for partner_id, log in self.trade_memory.items():
            trade_memory_serialized[partner_id] = {
                "maxlen": log.maxlen,
                "items": list(log)
            }

        return {
            # Core attributes
            "agent_id": self.agent_id,
            "pos": self.pos,
            "vision": self.vision,
            "metabolism": self.metabolism,
            "max_age": self.max_age,
            "wealth": self.wealth,
            "spice": self.spice,
            "metabolism_spice": self.metabolism_spice,
            "age": self.age,
            "persona": self.persona,
            "name": self.name,
            "alive": self.alive,
            # Identity system (Born Good/Bad)
            "origin_identity": self.origin_identity,
            "origin_identity_prompt": self.origin_identity_prompt,
            "policy_list": list(self.policy_list),
            "belief_ledger": dict(self.belief_ledger),
            "self_identity_leaning": self.self_identity_leaning,
            "baseline_snapshot": dict(self.baseline_snapshot),
            # Tracking state
            "initial_pos": self.initial_pos,
            "visited_cells": list(self.visited_cells),
            "recent_history": list(self.recent_history),
            "metrics": dict(self.metrics),
            # Trade state
            "trade_memory": trade_memory_serialized,
            "partner_trust": dict(self.partner_trust),
        }

    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint data.

        Args:
            data: Dictionary from to_checkpoint_dict()
        """
        # Core attributes
        self.pos = tuple(data["pos"])
        self.wealth = data["wealth"]
        self.spice = data["spice"]
        self.age = data["age"]
        self.alive = data["alive"]

        # Identity system (Born Good/Bad)
        self.origin_identity = data.get("origin_identity", "")
        self.origin_identity_prompt = data.get("origin_identity_prompt", "")
        self.policy_list = list(data.get("policy_list", []))
        self.belief_ledger = dict(data.get("belief_ledger", {}))
        self.self_identity_leaning = float(data.get("self_identity_leaning", 0.0))
        self.baseline_snapshot = dict(data.get("baseline_snapshot", {}))

        # Tracking state
        self.initial_pos = tuple(data["initial_pos"])
        self.visited_cells = set(tuple(c) for c in data["visited_cells"])
        self.recent_history = deque(
            [tuple(item) for item in data["recent_history"]],
            maxlen=15
        )
        self.metrics = dict(data["metrics"])

        # Trade state - restore deques with proper maxlen
        self.trade_memory = {}
        for partner_id_str, log_data in data.get("trade_memory", {}).items():
            partner_id = int(partner_id_str) if isinstance(partner_id_str, str) else partner_id_str
            maxlen = log_data.get("maxlen", 50)
            items = log_data.get("items", [])
            self.trade_memory[partner_id] = deque(items, maxlen=maxlen)

        self.partner_trust = {
            int(k) if isinstance(k, str) else k: v
            for k, v in data.get("partner_trust", {}).items()
        }
