import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING, Any, Dict

from sugarscape.agent import SugarAgent
from sugarscape.prompts import (
    build_sugarscape_system_prompt,
    build_sugarscape_observation_prompt,
    build_identity_review_prompt,
    build_end_of_life_report_prompt,
    build_baseline_questionnaire_prompt,
    parse_identity_review_response,
    parse_end_of_life_response,
    parse_questionnaire_response,
    WORLDVIEW_QUESTIONS,
)

if TYPE_CHECKING:
    from sugarscape.environment import SugarEnvironment
    from sugarscape.providers.base import BaseLLMProvider


def strip_thinking_blocks(text: str, keep_if_empty: bool = True) -> str:
    """Remove hidden thinking blocks from thinking models.

    Handles multiple formats:
    - __THINKING_START__...__THINKING_END__
    - <think>...</think>  (e.g., Qwen/DeepSeek-style)
    - Unclosed start tags (drops from start tag to end-of-text)
    
    Args:
        text: The text to process
        keep_if_empty: If True and the non-thinking content is empty, return the
                       thinking content itself (cleaned of markers). This preserves
                       useful content when models put everything in thinking blocks.
    """
    if not text:
        return ""

    # First, extract thinking content in case we need it later
    thinking_content = ""
    if keep_if_empty:
        thinking_match = re.search(
            r"__THINKING_START__\s*(.*?)\s*__THINKING_END__",
            text,
            flags=re.DOTALL,
        )
        if not thinking_match:
            thinking_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()

    # Remove all properly closed thinking blocks (non-greedy).
    cleaned = re.sub(r"__THINKING_START__.*?__THINKING_END__\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # If there's an unclosed thinking start marker remaining, only drop content
    # from that marker to the end (do not greedily erase earlier valid content).
    start = cleaned.find("__THINKING_START__")
    if start != -1:
        end = cleaned.find("__THINKING_END__", start)
        if end == -1:
            # Extract unclosed thinking content before discarding
            if keep_if_empty and not thinking_content:
                thinking_content = text[start + len("__THINKING_START__"):].strip()
            cleaned = cleaned[:start]
        else:
            # Extremely defensive: if we somehow have a start+end remaining, remove it.
            cleaned = (cleaned[:start] + cleaned[end + len("__THINKING_END__") :]).strip()

    # Handle unclosed <think> tag.
    think_start = cleaned.lower().find("<think>")
    if think_start != -1:
        think_end = cleaned.lower().find("</think>", think_start)
        if think_end == -1:
            if keep_if_empty and not thinking_content:
                thinking_content = cleaned[think_start + len("<think>") :].strip()
            cleaned = cleaned[:think_start]
        else:
            cleaned = (cleaned[:think_start] + cleaned[think_end + len("</think>") :]).strip()

    cleaned = cleaned.strip()
    
    # If cleaned is empty but we have thinking content, use that instead
    if not cleaned and thinking_content and keep_if_empty:
        # Return the thinking content itself (without markers) as fallback so downstream
        # parsers (e.g., JSON extraction) can still recover structured output if the
        # model incorrectly placed it inside a thinking block.
        return thinking_content
    
    return cleaned

@dataclass(eq=False)
class LLMSugarAgent(SugarAgent):
    """SugarAgent powered by LLM."""

    # These fields must have defaults to play nice with dataclass inheritance order
    provider: Any = None
    goal_prompt: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.conversation_history = []
        self.move_history = []  # Track position at each tick: [(tick, pos, action, wealth, spice)]
        # Identity review tracking
        self.identity_review_history: List[Dict[str, Any]] = []  # History of identity reviews
        self.last_identity_review_tick: int = 0  # Last tick when identity review was performed
        self.end_of_life_report: Optional[Dict[str, Any]] = None  # Final self-report

        # Event-triggered reflection system
        # Events that should trigger reflection (more sensitive than periodic tick-based)
        self.pending_reflection_events: List[Dict[str, Any]] = []
        self.event_reflection_history: List[Dict[str, Any]] = []  # History of event-triggered reflections
        # Lifetime stats for end-of-life report
        self.lifetime_stats: Dict[str, int] = {
            "trades_completed": 0,
            "trades_failed": 0,
            "agents_helped": 0,
            "resources_given": 0,
            "resources_received": 0,
        }

    # ========== EVENT-TRIGGERED REFLECTION SYSTEM ==========
    # Events: defrauded, successful_cooperation, resources_critical, trade_rejected, witnessed_death

    def record_reflection_event(self, event_type: str, tick: int, details: Dict[str, Any]) -> None:
        """Record an event that should trigger reflection.

        Event types:
        - "defrauded": Agent was cheated in a trade
        - "successful_cooperation": Completed a mutually beneficial trade
        - "resources_critical": Resources dropped to critical level
        - "trade_rejected": Trade proposal was rejected
        - "witnessed_death": Observed another agent die

        Args:
            event_type: Type of event
            tick: When it occurred
            details: Event-specific details (partner, amounts, etc.)
        """
        self.pending_reflection_events.append({
            "type": event_type,
            "tick": tick,
            **details
        })

    def has_pending_reflection(self) -> bool:
        """Check if there are pending events that should trigger reflection."""
        return len(self.pending_reflection_events) > 0

    def get_pending_events_summary(self) -> str:
        """Format pending events for reflection prompt."""
        if not self.pending_reflection_events:
            return "(No significant events)"

        lines = []
        for event in self.pending_reflection_events[-5:]:  # Last 5 events
            event_type = event.get("type", "unknown")
            tick = event.get("tick", "?")

            if event_type == "defrauded":
                partner = event.get("partner_name", "someone")
                amount = event.get("amount_lost", 0)
                lines.append(f"- Tick {tick}: You were CHEATED by {partner} (lost ~{amount} resources)")
            elif event_type == "successful_cooperation":
                partner = event.get("partner_name", "someone")
                lines.append(f"- Tick {tick}: Successful cooperation with {partner}")
            elif event_type == "resources_critical":
                resource = event.get("resource", "resources")
                lines.append(f"- Tick {tick}: Your {resource} reached CRITICAL level")
            elif event_type == "trade_rejected":
                partner = event.get("partner_name", "someone")
                reason = event.get("reason", "unknown")
                lines.append(f"- Tick {tick}: Trade with {partner} was REJECTED ({reason})")
            elif event_type == "witnessed_death":
                deceased = event.get("deceased_name", "someone")
                cause = event.get("cause", "unknown")
                lines.append(f"- Tick {tick}: You witnessed {deceased}'s death ({cause})")
            else:
                lines.append(f"- Tick {tick}: {event_type}")

        return "\n".join(lines)

    def clear_pending_events(self) -> List[Dict[str, Any]]:
        """Clear and return pending events after processing."""
        events = self.pending_reflection_events.copy()
        self.pending_reflection_events = []
        return events

    def _strip_thinking_blocks(self, text: str) -> str:
        return strip_thinking_blocks(text)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        import json
        if not text:
            return None
        
        # Try to find JSON object
        decoder = json.JSONDecoder()
        starts = [i for i, ch in enumerate(text) if ch == "{"]
        
        best_obj = None
        best_len = -1
        
        for i in starts:
            try:
                obj, end = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    length = end
                    if length > best_len:
                        best_len = length
                        best_obj = obj
            except Exception:
                continue
                
        return best_obj

    async def generate_json_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        retry_attempts: int = 2
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate a response and ensure it contains valid JSON.
        
        Returns:
            Tuple of (raw_response, parsed_json_dict)
        """
        last_response = ""
        
        # Try initial generation with thinking disabled for vLLM/Qwen3
        try:
            try:
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                # Fallback for providers that don't support chat_template_kwargs
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            last_response = response
            
            # Try to parse
            cleaned = self._strip_thinking_blocks(response)
            parsed = self._extract_json(cleaned)
            
            if parsed:
                return response, parsed
                
        except Exception as e:
            print(f"Generation error: {e}")
            
        # Retry loop
        for i in range(retry_attempts):
            try:
                retry_prompt = f"{user_prompt}\n\nERROR: Your previous response did not contain valid JSON. Please output VALID JSON ONLY."
                if last_response:
                     # Show a snippet of invalid response to help model correct it
                     snippet = last_response[:200].replace("\n", " ")
                     retry_prompt += f"\n\nYour previous invalid response started with: {snippet}..."
                
                try:
                    response = await self.provider.generate(
                        system_prompt=(system_prompt + "\n\nIMPORTANT: Output VALID JSON only. No prose.").strip(),
                        messages=[{"role": "user", "content": retry_prompt}],
                        chat_template_kwargs={"enable_thinking": False}
                    )
                except TypeError:
                    response = await self.provider.generate(
                        system_prompt=(system_prompt + "\n\nIMPORTANT: Output VALID JSON only. No prose.").strip(),
                        messages=[{"role": "user", "content": retry_prompt}]
                    )
                last_response = response
                
                cleaned = self._strip_thinking_blocks(response)
                parsed = self._extract_json(cleaned)
                
                if parsed:
                    return response, parsed
                    
            except Exception as e:
                print(f"Retry {i+1} error: {e}")
                
        return last_response, None

    async def async_identity_review(self, env: "SugarEnvironment", tick: int) -> Dict[str, Any]:
        """Run periodic identity self-assessment.

        Every N ticks, agents reflect on whether they're still altruist/exploiter,
        and whether their experiences have changed their perspective.

        Returns:
            Dict containing review results including reflection, assessment, and any updates applied.
        """
        # Gather recent interactions from trade memory
        recent_interactions = []
        for partner_id, trade_log in self.trade_memory.items():
            for event in list(trade_log)[-3:]:  # Last 3 events per partner
                recent_interactions.append(event)

        # Sort by tick
        recent_interactions.sort(key=lambda x: x.get("tick", 0), reverse=True)
        recent_interactions = recent_interactions[:10]  # Keep top 10 most recent

        # Build prompts
        system_prompt, user_prompt = build_identity_review_prompt(
            agent=self,
            tick=tick,
            recent_interactions=recent_interactions,
            env=env,
        )

        result = {
            "tick": tick,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "parsed": None,
            "updates_applied": {},
            "identity_before": self.self_identity_leaning,
            "identity_after": self.self_identity_leaning,
        }

        # Retry loop for robust parsing
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Setup messages (with retry nudge if needed)
                current_messages = [{"role": "user", "content": user_prompt}]
                
                if attempt > 0:
                    nudge = "\n\nERROR: Your previous response was formatted incorrectly. You MUST use the exact format:\nREFLECTION: ...\nIDENTITY_ASSESSMENT: ...\nJSON: {...}"
                    current_messages = [{"role": "user", "content": user_prompt + nudge}]

                try:
                    response = await self.provider.generate(
                        system_prompt=system_prompt,
                        messages=current_messages,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    response = await self.provider.generate(
                        system_prompt=system_prompt,
                        messages=current_messages,
                    )

                result["raw_response"] = response

                # Parse the response
                parsed = parse_identity_review_response(response)
                result["parsed"] = parsed
                
                # Validation: did we get the core sections?
                # We expect at least REFLECTION and IDENTITY_ASSESSMENT.
                # JSON is optional but if "JSON:" text is present, updates should be parsed.
                valid = bool(parsed.get("reflection")) and bool(parsed.get("identity_assessment"))
                
                if valid:
                    break
                
                if attempt < max_retries:
                    print(f"Identity review parsing failed for {self.name}, retrying ({attempt+1}/{max_retries})...")
            
            except Exception as e:
                print(f"Identity review error for {self.name} (attempt {attempt+1}): {e}")
                if attempt == max_retries:
                    result["error"] = str(e)
                    return result

        # Apply updates if present (from successful or final attempt)
        parsed = result.get("parsed")
        if parsed and parsed.get("updates"):
            updates = parsed["updates"]
            changes = self.apply_reflection_update(updates)
            result["updates_applied"] = changes
            result["identity_after"] = self.self_identity_leaning

        # Store in history
        self.identity_review_history.append({
            "tick": tick,
            "reflection": parsed.get("reflection", "") if parsed else "",
            "identity_assessment": parsed.get("identity_assessment", "mixed") if parsed else "mixed",
            "identity_leaning_before": result["identity_before"],
            "identity_leaning_after": result["identity_after"],
            "updates_applied": result["updates_applied"],
        })

        self.last_identity_review_tick = tick

        # Store in conversation history for context
        self.conversation_history.append({"role": "user", "content": f"[IDENTITY REVIEW] {user_prompt}"})
        self.conversation_history.append({"role": "assistant", "content": result["raw_response"]})

        # External moral evaluation (legacy periodic identity review)
        moral_eval = getattr(env, "moral_evaluator", None)
        debug_logger = getattr(env, "debug_logger", None)
        if moral_eval and debug_logger:
            try:
                from sugarscape.moral_evaluator import compute_self_overall_from_leaning
                from sugarscape.debug_logger import MoralEvalRecord
                self_overall = compute_self_overall_from_leaning(self.self_identity_leaning, moral_eval.rubric).get("overall", 50.0)
                ev = await moral_eval.async_evaluate(
                    interaction_type="identity_review",
                    tick=tick,
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    agent_system_prompt=system_prompt,
                    agent_user_prompt=user_prompt,
                    agent_raw_response=result.get("raw_response", ""),
                    extra_context={"recent_interactions": recent_interactions},
                )
                overall = ev.get("overall", {})
                parsed_ev = ev.get("parsed", {})
                rec = MoralEvalRecord(
                    tick=tick,
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    interaction_type="identity_review",
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

        return result

    async def async_baseline_questionnaire(self, env: "SugarEnvironment", tick: int = 0) -> Dict[str, Any]:
        """Run baseline worldview questionnaire (T=0 or periodic measurement).

        Asks fixed questions (Q1-Q5) on 1-7 scale to measure worldview.
        Should be called at T=0 (before any interactions) and periodically.

        Args:
            env: The simulation environment
            tick: Current simulation tick (0 for baseline)

        Returns:
            Dict containing questionnaire results with scores and reasons.
        """
        # Build prompts
        system_prompt, user_prompt = build_baseline_questionnaire_prompt(
            agent=self,
            tick=tick,
        )

        result = {
            "tick": tick,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "parsed": None,
            "scores": {},
        }

        try:
            try:
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

            result["raw_response"] = response

            # Parse the response
            cleaned = self._strip_thinking_blocks(response)
            parsed = parse_questionnaire_response(cleaned)
            result["parsed"] = parsed

            # Extract scores for easy access
            for qid in ["Q1_trust", "Q2_cooperation", "Q3_fairness", "Q4_scarcity", "Q5_self_vs_others"]:
                if qid in parsed and isinstance(parsed[qid], dict):
                    result["scores"][qid] = parsed[qid].get("score", 4)
                else:
                    result["scores"][qid] = 4  # Default to neutral

            # External moral evaluation (T=0 / periodic questionnaire)
            moral_eval = getattr(env, "moral_evaluator", None)
            debug_logger = getattr(env, "debug_logger", None)
            if moral_eval and debug_logger:
                try:
                    from sugarscape.moral_evaluator import compute_self_overall_from_leaning
                    from sugarscape.debug_logger import MoralEvalRecord
                    self_overall = compute_self_overall_from_leaning(self.self_identity_leaning, moral_eval.rubric).get("overall", 50.0)
                    ev = await moral_eval.async_evaluate(
                        interaction_type="baseline_questionnaire",
                        tick=tick,
                        agent_id=self.agent_id,
                        agent_name=self.name,
                        agent_system_prompt=system_prompt,
                        agent_user_prompt=user_prompt,
                        agent_raw_response=response,
                        extra_context={"questionnaire_scores": result.get("scores", {})},
                    )
                    overall = ev.get("overall", {})
                    parsed_ev = ev.get("parsed", {})
                    rec = MoralEvalRecord(
                        tick=tick,
                        agent_id=self.agent_id,
                        agent_name=self.name,
                        interaction_type="baseline_questionnaire",
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
            print(f"Questionnaire error for {self.name}: {e}")
            result["error"] = str(e)
            # Default neutral scores on error
            result["scores"] = {
                "Q1_trust": 4,
                "Q2_cooperation": 4,
                "Q3_fairness": 4,
                "Q4_scarcity": 4,
                "Q5_self_vs_others": 4,
            }

        return result

    async def async_event_reflection(self, env: "SugarEnvironment", tick: int) -> Dict[str, Any]:
        """Run event-triggered identity review (full assessment) when significant events occur.

        This is a FULL identity review triggered by significant events (not periodic).
        Called after: being cheated, successful cooperation, critical resources, witnessed death, etc.

        Args:
            env: The simulation environment
            tick: Current simulation tick

        Returns:
            Dict containing review results including reflection, assessment, identity shift, and any updates applied.
        """
        from sugarscape.prompts import build_event_triggered_reflection_prompt

        if not self.has_pending_reflection():
            return {"tick": tick, "skipped": True, "reason": "no_pending_events"}

        # Get events summary
        events_summary = self.get_pending_events_summary()
        # Snapshot events, but do NOT clear until we have a response.
        events_snapshot = list(self.pending_reflection_events)

        # Gather recent interactions from trade memory for context
        recent_interactions = []
        for partner_id, trade_log in self.trade_memory.items():
            for event in list(trade_log)[-3:]:  # Last 3 events per partner
                recent_interactions.append(event)
        recent_interactions.sort(key=lambda x: x.get("tick", 0), reverse=True)
        recent_interactions = recent_interactions[:10]  # Keep top 10 most recent

        # Build prompts (now a full identity review)
        system_prompt, user_prompt = build_event_triggered_reflection_prompt(
            agent=self,
            tick=tick,
            events_summary=events_summary,
            env=env,
            recent_interactions=recent_interactions,
        )

        result = {
            "tick": tick,
            "trigger_type": "event",
            "events": events_snapshot,
            "events_summary": events_summary,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "parsed": None,
            "updates_applied": {},
            "identity_before": self.self_identity_leaning,
            "identity_after": self.self_identity_leaning,
        }

        # Retry loop for robust parsing (same as identity_review)
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                current_messages = [{"role": "user", "content": user_prompt}]
                
                if attempt > 0:
                    nudge = "\n\nERROR: Your previous response was formatted incorrectly. You MUST use the exact format:\nREFLECTION: ...\nIDENTITY_ASSESSMENT: ...\nJSON: {...}"
                    current_messages = [{"role": "user", "content": user_prompt + nudge}]

                try:
                    response = await self.provider.generate(
                        system_prompt=system_prompt,
                        messages=current_messages,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    response = await self.provider.generate(
                        system_prompt=system_prompt,
                        messages=current_messages,
                    )

                result["raw_response"] = response

                # Parse the response (same parser as identity review)
                parsed = parse_identity_review_response(response)
                result["parsed"] = parsed

                # Validation: did we get the core sections?
                valid = bool(parsed.get("reflection")) and bool(parsed.get("identity_assessment"))

                if valid:
                    # Now that we have a valid response, clear pending events
                    events = self.clear_pending_events()
                    result["events"] = events
                    break

                if attempt < max_retries:
                    print(f"Event-triggered identity review parsing failed for {self.name}, retrying ({attempt+1}/{max_retries})...")

            except Exception as e:
                print(f"Event-triggered identity review error for {self.name} (attempt {attempt+1}): {e}")
                if attempt == max_retries:
                    result["error"] = str(e)
                    return result

        # Apply updates if present
        parsed = result.get("parsed")
        if parsed and parsed.get("updates"):
            changes = self.apply_reflection_update(parsed["updates"])
            result["updates_applied"] = changes
            result["identity_after"] = self.self_identity_leaning

        # External moral evaluation (event-triggered identity review)
        moral_eval = getattr(env, "moral_evaluator", None)
        debug_logger = getattr(env, "debug_logger", None)
        if moral_eval and debug_logger:
            try:
                from sugarscape.moral_evaluator import compute_self_overall_from_leaning
                from sugarscape.debug_logger import MoralEvalRecord
                self_overall = compute_self_overall_from_leaning(self.self_identity_leaning, moral_eval.rubric).get("overall", 50.0)
                ev = await moral_eval.async_evaluate(
                    interaction_type="event_identity_review",
                    tick=tick,
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    agent_system_prompt=result.get("system_prompt", ""),
                    agent_user_prompt=result.get("user_prompt", ""),
                    agent_raw_response=result.get("raw_response", ""),
                    extra_context={
                        "events": result.get("events", events_snapshot),
                        "events_summary": result.get("events_summary", ""),
                    },
                )
                overall = ev.get("overall", {})
                parsed_ev = ev.get("parsed", {})
                rec = MoralEvalRecord(
                    tick=tick,
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    interaction_type="event_identity_review",
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

        # Store in identity_review_history (unified with periodic reviews)
        self.identity_review_history.append({
            "tick": tick,
            "trigger_type": "event",
            "trigger_events": [e.get("type") for e in result.get("events", events_snapshot)],
            "reflection": parsed.get("reflection", "") if parsed else "",
            "identity_assessment": parsed.get("identity_assessment", "mixed") if parsed else "mixed",
            "identity_leaning_before": result["identity_before"],
            "identity_leaning_after": result["identity_after"],
            "updates_applied": result["updates_applied"],
        })

        # Also store in event_reflection_history for backwards compatibility
        self.event_reflection_history.append({
            "tick": tick,
            "events": result.get("events", events_snapshot),
            "reflection": parsed.get("reflection", "") if parsed else "",
            "identity_assessment": parsed.get("identity_assessment", "mixed") if parsed else "mixed",
            "updates_applied": result.get("updates_applied", {}),
        })

        self.last_identity_review_tick = tick

        # Store in conversation history for context
        self.conversation_history.append({"role": "user", "content": f"[EVENT-TRIGGERED IDENTITY REVIEW] {user_prompt}"})
        self.conversation_history.append({"role": "assistant", "content": result["raw_response"]})

        return result

    async def async_end_of_life_report(self, env: "SugarEnvironment", tick: int, death_cause: str) -> Dict[str, Any]:
        """Run final self-report before death or simulation end.

        This is the agent's last chance to reflect on their life and choices.

        Args:
            env: The simulation environment
            tick: Current simulation tick
            death_cause: Why agent is dying ("starvation_sugar", "starvation_spice", "old_age", "simulation_end")

        Returns:
            Dict containing final reflection and assessment.
        """
        # Build prompts
        system_prompt, user_prompt = build_end_of_life_report_prompt(
            agent=self,
            tick=tick,
            death_cause=death_cause,
            lifetime_stats=self.lifetime_stats,
        )

        result = {
            "tick": tick,
            "death_cause": death_cause,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "parsed": None,
            "origin_identity": self.origin_identity,
            "final_identity_leaning": self.self_identity_leaning,
            "lifetime_stats": self.lifetime_stats.copy(),
            "total_identity_reviews": len(self.identity_review_history),
        }

        try:
            try:
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

            result["raw_response"] = response

            # Parse the response
            parsed = parse_end_of_life_response(response)
            result["parsed"] = parsed

            # Store the report
            self.end_of_life_report = result

            # External moral evaluation (end-of-life report)
            moral_eval = getattr(env, "moral_evaluator", None)
            debug_logger = getattr(env, "debug_logger", None)
            if moral_eval and debug_logger:
                try:
                    from sugarscape.moral_evaluator import compute_self_overall_from_leaning
                    from sugarscape.debug_logger import MoralEvalRecord
                    self_overall = compute_self_overall_from_leaning(self.self_identity_leaning, moral_eval.rubric).get("overall", 50.0)
                    ev = await moral_eval.async_evaluate(
                        interaction_type="end_of_life_report",
                        tick=tick,
                        agent_id=self.agent_id,
                        agent_name=self.name,
                        agent_system_prompt=system_prompt,
                        agent_user_prompt=user_prompt,
                        agent_raw_response=response,
                        extra_context={"death_cause": death_cause},
                    )
                    overall = ev.get("overall", {})
                    parsed_ev = ev.get("parsed", {})
                    rec = MoralEvalRecord(
                        tick=tick,
                        agent_id=self.agent_id,
                        agent_name=self.name,
                        interaction_type="end_of_life_report",
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

            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": f"[END OF LIFE] {user_prompt}"})
            self.conversation_history.append({"role": "assistant", "content": response})

            return result

        except Exception as e:
            print(f"End of life report error for {self.name} (agent {self.agent_id}): {e}")
            result["error"] = str(e)
            return result

    def update_lifetime_stats(self, event_type: str, **kwargs) -> None:
        """Update lifetime statistics for end-of-life reporting.

        Args:
            event_type: Type of event ("trade_completed", "trade_failed", "helped_agent", "gave_resources", "received_resources")
            **kwargs: Additional data (e.g., amount for resources)
        """
        if event_type == "trade_completed":
            self.lifetime_stats["trades_completed"] += 1
        elif event_type == "trade_failed":
            self.lifetime_stats["trades_failed"] += 1
        elif event_type == "helped_agent":
            self.lifetime_stats["agents_helped"] += 1
        elif event_type == "gave_resources":
            amount = kwargs.get("amount", 0)
            self.lifetime_stats["resources_given"] += amount
        elif event_type == "received_resources":
            amount = kwargs.get("amount", 0)
            self.lifetime_stats["resources_received"] += amount

    async def async_decide_move(self, env: "SugarEnvironment") -> Dict[str, Any]:
        """Async decision making for parallel execution.

        Returns:
            Dict containing decision details (parsed_move, raw_response, prompts)
        """
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)

        # 2. Count nearby agents by urgency (for debug logging)
        nearby_critical = 0
        nearby_struggling = 0
        nearby_total = 0
        for pos in candidates:
            other = env.get_agent_at(pos)
            if other and other != self:
                nearby_total += 1
                # Calculate urgency
                other_sugar_time = int(other.wealth / other.metabolism) if other.metabolism > 0 else 999
                other_spice_time = int(other.spice / other.metabolism_spice) if other.metabolism_spice > 0 else 999
                other_min_time = min(other_sugar_time, other_spice_time)
                if other_min_time < 3:
                    nearby_critical += 1
                elif other_min_time < 10:
                    nearby_struggling += 1

        # 3. Build Prompt (pass agent for identity context if enabled)
        enable_survival_pressure = getattr(env.config, 'enable_survival_pressure', True)
        system_prompt = build_sugarscape_system_prompt(
            self.goal_prompt,
            agent_name=self.name,
            agent=self,
            enable_survival_pressure=enable_survival_pressure,
        )
        user_prompt = build_sugarscape_observation_prompt(self, env, candidates)

        result = {
            "parsed_move": None,
            "raw_response": "",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "nearby_agents_critical": nearby_critical,
            "nearby_agents_struggling": nearby_struggling,
            "nearby_agents_total": nearby_total,
        }

        # 4. Call LLM with thinking disabled for movement decisions
        try:
            # Try to disable thinking mode for faster/cheaper movement decisions
            try:
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                # Provider doesn't support these kwargs, fall back
                response = await self.provider.generate(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

            # Strip thinking blocks but keep content if model put everything in <think>
            cleaned_response = strip_thinking_blocks(response, keep_if_empty=True).strip()
            
            # Store RAW response for debugging, cleaned for history/parsing
            result["raw_response"] = response  # Original for debugging
            result["cleaned_response"] = cleaned_response  # Cleaned for display

            # Store history (cleaned version for context)
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": cleaned_response})

            # 4. Parse Response (use cleaned)
            result["parsed_move"] = self._parse_move(cleaned_response, candidates)
            return result

        except Exception as e:
            print(f"LLM Agent {self.agent_id} error: {e}")
            return result

    def _post_move_step(self, env: "SugarEnvironment", decision_data: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Execute post-move logic and return rewards.

        Returns:
            Dict of reward components (sugar_harvested, etc.)
        """
        rewards = {
            "sugar_harvested": 0,
            "spice_harvested": 0,
            "sugar_metabolism": self.metabolism,
            "spice_metabolism": self.metabolism_spice
        }

        # 1. Harvest
        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s
        rewards["sugar_harvested"] = harvested_s

        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p
            rewards["spice_harvested"] = harvested_p

        # 2. Update movement metrics
        self._update_metrics(env)

        # 3. Metabolize
        self.wealth -= self.metabolism
        if env.config.enable_spice:
            self.spice -= self.metabolism_spice

        # 4. Age
        self.age += 1

        # 5. Check death
        if self.wealth <= 0 or (env.config.enable_spice and self.spice <= 0) or self.age >= self.max_age:
            self.alive = False

        return rewards

    def _harvest_and_update_metrics(self, env: "SugarEnvironment") -> Dict[str, int]:
        """Harvest resources at current position and update movement metrics.

        This is used by the phased simulation loop where trading happens after
        movement/harvest and before metabolism/aging/death.
        """
        rewards = {
            "sugar_harvested": 0,
            "spice_harvested": 0,
        }

        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s
        rewards["sugar_harvested"] = harvested_s

        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p
            rewards["spice_harvested"] = harvested_p

        self._update_metrics(env)
        return rewards

    def _move_and_harvest(self, env: "SugarEnvironment"):
        """Move using LLM decision."""
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)

        # 2. Build Prompt (pass agent for identity context if enabled)
        enable_survival_pressure = getattr(env.config, 'enable_survival_pressure', True)
        system_prompt = build_sugarscape_system_prompt(
            self.goal_prompt,
            agent_name=self.name,
            agent=self,
            enable_survival_pressure=enable_survival_pressure,
        )
        user_prompt = build_sugarscape_observation_prompt(self, env, candidates)

        # 3. Call LLM
        target_pos = self.pos # Default to stay

        try:
            # Using asyncio.run to bridge sync simulation with async provider
            # Note: This creates a new loop per step.
            # In a heavy simulation, this is slow, but functional for 'add support'.
            try:
                response = asyncio.run(
                    self.provider.generate(
                        system_prompt=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        chat_template_kwargs={"enable_thinking": False},
                    )
                )
            except TypeError:
                response = asyncio.run(
                    self.provider.generate(
                        system_prompt=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                )
            # Strip thinking blocks but preserve content if model put everything in <think>
            cleaned_response = strip_thinking_blocks(response, keep_if_empty=True).strip()

            # Store history for debugging (cleaned version)
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": cleaned_response})

            # 4. Parse Response (use cleaned)
            parsed_pos = self._parse_move(cleaned_response, candidates)
            if parsed_pos:
                target_pos = parsed_pos

        except Exception as e:
            print(f"LLM Agent {self.agent_id} error: {e}")
            # Fallback to stay put

        # 5. Execute Move
        if target_pos != self.pos:
            # Verify occupancy (Prompt should have informed agent, but check again)
            if not env.is_occupied(target_pos):
                env.move_agent(self, target_pos)
            else:
                # If LLM chose occupied spot (and it's not self), ignore move
                pass

        # 6. Harvest (Standard)
        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s

        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p

    def _parse_move(self, response: str, valid_spots: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Parse action from LLM response (new immersive format or legacy coordinate format)."""

        # === NEW FORMAT: ACTION: [direction description] ===
        # Example: "ACTION: Move toward the large northern deposit"
        # The direction label (NORTH, SOUTHEAST, etc.) should be in the action description

        action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()

            # Try to extract direction label from action text
            # Check for compound directions first (e.g., NORTHEAST, SOUTHWEST)
            direction_pattern = r"\b(NORTH|SOUTH|EAST|WEST|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|CURRENT_LOCATION|CURRENT LOCATION|STAY|HERE)\b"
            direction_match = re.search(direction_pattern, action_text, re.IGNORECASE)

            if direction_match:
                direction_label = direction_match.group(1).upper().replace(" ", "_")

                # Map direction to coordinate
                parsed_pos = self._map_direction_to_position(direction_label, valid_spots)
                if parsed_pos:
                    return parsed_pos

        # === LEGACY FORMAT: MOVE: (x, y) ===
        # Fallback to old coordinate-based parsing for backwards compatibility
        coord_match = re.search(r"MOVE:\s*\((\d+),\s*(\d+)\)", response, re.IGNORECASE)
        if coord_match:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            pos = (x, y)

            if pos in valid_spots:
                return pos

        # If no valid move found, stay in place
        return self.pos

    def _map_direction_to_position(self, direction: str, valid_spots: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Map a direction label (e.g., 'NORTH', 'SOUTHEAST') to an actual position from valid_spots."""

        # Current position should always be in valid_spots
        if direction in ["CURRENT_LOCATION", "STAY", "HERE"]:
            return self.pos

        # Calculate expected direction for each visible spot
        for pos in valid_spots:
            dx = pos[0] - self.pos[0]
            dy = pos[1] - self.pos[1]

            # Skip current position
            if dx == 0 and dy == 0:
                continue

            # Determine direction label for this position
            spot_direction = self._position_to_direction(dx, dy)

            if spot_direction == direction:
                return pos

        return None

    def _position_to_direction(self, dx: int, dy: int) -> str:
        """Convert position delta to direction label."""
        if dx == 0 and dy == 0:
            return "CURRENT_LOCATION"

        # Determine primary direction based on larger delta
        # or diagonal if roughly equal
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx > abs_dy * 1.5:  # Mostly horizontal
            return "EAST" if dx > 0 else "WEST"
        elif abs_dy > abs_dx * 1.5:  # Mostly vertical
            return "NORTH" if dy > 0 else "SOUTH"
        else:  # Diagonal
            ns = "NORTH" if dy > 0 else "SOUTH"
            ew = "EAST" if dx > 0 else "WEST"
            return f"{ns}{ew}"

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize LLM agent state for checkpointing.

        Extends parent to include LLM-specific state.
        """
        data = super().to_checkpoint_dict()
        data["is_llm_agent"] = True
        data["goal_prompt"] = self.goal_prompt
        data["conversation_history"] = list(self.conversation_history)
        data["move_history"] = list(self.move_history)
        # Identity review state
        data["identity_review_history"] = list(self.identity_review_history)
        data["last_identity_review_tick"] = self.last_identity_review_tick
        data["end_of_life_report"] = self.end_of_life_report
        data["lifetime_stats"] = dict(self.lifetime_stats)
        return data

    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore LLM agent state from checkpoint data.

        Extends parent to restore LLM-specific state.
        """
        super().restore_from_checkpoint(data)
        self.goal_prompt = data.get("goal_prompt", "")
        self.conversation_history = list(data.get("conversation_history", []))
        self.move_history = list(data.get("move_history", []))
        # Identity review state
        self.identity_review_history = list(data.get("identity_review_history", []))
        self.last_identity_review_tick = data.get("last_identity_review_tick", 0)
        self.end_of_life_report = data.get("end_of_life_report")
        self.lifetime_stats = dict(data.get("lifetime_stats", {
            "trades_completed": 0,
            "trades_failed": 0,
            "agents_helped": 0,
            "resources_given": 0,
            "resources_received": 0,
        }))