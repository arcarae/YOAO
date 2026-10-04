"""Deliberation mechanism for team decision-making."""

import asyncio
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

from redblackbench.agents.base import AgentResponse
from redblackbench.game.scoring import Choice

if TYPE_CHECKING:
    from redblackbench.agents.base import BaseAgent


@dataclass
class DeliberationResult:
    """Result of a team deliberation process.

    Attributes:
        final_choice: The team's final choice determined by majority vote
        initial_opinions: All agents' initial opinions
        final_votes: All agents' final votes after seeing opinions
        vote_counts: Dictionary of choice -> vote count
        was_unanimous: Whether all agents voted the same way
        discussion_rounds: List of discussion round messages (for multi-round deliberation)
    """
    final_choice: Choice
    initial_opinions: List[AgentResponse] = field(default_factory=list)
    final_votes: List[AgentResponse] = field(default_factory=list)
    vote_counts: dict = field(default_factory=dict)
    was_unanimous: bool = False
    discussion_rounds: List[List[dict]] = field(default_factory=list)
    abstentions: int = 0  # final votes that failed and were excluded from the majority

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "final_choice": str(self.final_choice),
            "initial_opinions": [op.to_dict() for op in self.initial_opinions],
            "final_votes": [v.to_dict() for v in self.final_votes],
            "vote_counts": {str(k): v for k, v in self.vote_counts.items()},
            "was_unanimous": self.was_unanimous,
            "discussion_rounds": self.discussion_rounds,
            "abstentions": self.abstentions,
        }


class NoValidVotesError(RuntimeError):
    """Every agent on a team failed to vote in a round; the game cannot be scored honestly."""


def count_abstentions(votes: List[AgentResponse]) -> int:
    return sum(1 for v in votes if v.choice is None)


class Deliberation:
    """Manages the deliberation process for a team of agents.

    The deliberation follows a multi-phase process:
    1. Initial Opinion Phase: Each agent shares their opinion (ordered by willingness)
    2. Discussion Rounds: Agents can respond to each other (configurable number of rounds)
    3. Final Vote Phase: After discussion, each agent casts their final vote

    The team's choice is determined by majority vote in the final phase.
    """

    def __init__(self, agents: List["BaseAgent"], discussion_rounds: int = 1):
        """Initialize deliberation with a list of agents.

        Args:
            agents: List of agents participating in deliberation
            discussion_rounds: Number of discussion rounds (1 = original behavior, >1 for more discussion)
        """
        self.agents = agents
        self.discussion_rounds = discussion_rounds
    
    async def _get_willingness_with_retry(
        self,
        agent: "BaseAgent",
        round_context: dict,
        team_identifier: str,
        seen_messages: list,
        max_retries: int = 3,
    ) -> int:
        """Get willingness to speak with retry logic."""
        for attempt in range(max_retries):
            try:
                return await agent.get_willingness_to_speak(round_context, team_identifier, seen_messages)
            except asyncio.CancelledError:
                # Re-raise CancelledError - this is intentional task cancellation
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  [Warning] Willingness failed for {getattr(agent, 'agent_id', 'unknown')} (attempt {attempt + 1}): {type(e).__name__}")
                    await asyncio.sleep(1)
                else:
                    print(f"  [Warning] Willingness failed for {getattr(agent, 'agent_id', 'unknown')} after {max_retries} attempts, defaulting to 0")
                    return 0
        return 0

    async def _gather_initial_opinions(
        self,
        round_context: dict,
        team_identifier: str,
    ) -> List[tuple["BaseAgent", AgentResponse]]:
        spoken = set()
        team_channel: List[dict] = []
        ordered_pairs: List[tuple["BaseAgent", AgentResponse]] = []
        while len(spoken) < len(self.agents):
            pending_agents = [a for a in self.agents if a not in spoken]
            # Get willingness with individual retry logic for each agent
            willingness_tasks = [
                self._get_willingness_with_retry(a, round_context, team_identifier, team_channel)
                for a in pending_agents
            ]
            willingness_values = await asyncio.gather(*willingness_tasks)
            max_w = max(willingness_values) if willingness_values else 0
            candidates = [a for a, w in zip(pending_agents, willingness_values) if w == max_w]
            chosen = random.choice(candidates)
            # Pass prior messages so agent can respond to teammates
            # Retry logic for initial opinion
            max_retries = 3
            opinion = None
            for attempt in range(max_retries):
                try:
                    opinion = await chosen.get_initial_opinion(
                        round_context, team_identifier, prior_messages=team_channel if team_channel else None
                    )
                    break
                except asyncio.CancelledError:
                    # Re-raise CancelledError - this is intentional task cancellation
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  [Warning] Opinion failed for {getattr(chosen, 'agent_id', 'unknown')} (attempt {attempt + 1}): {type(e).__name__}")
                        await asyncio.sleep(1)  # Brief delay before retry
                    else:
                        print(f"  [Error] Opinion failed for {getattr(chosen, 'agent_id', 'unknown')} after {max_retries} attempts: {e}")
                        # Abstain: no vote, and nothing is broadcast to teammates.
                        opinion = AgentResponse.abstain("[Failed to get response]")
            ordered_pairs.append((chosen, opinion))
            if not opinion.failed:
                team_channel.append({
                    "agent_id": getattr(chosen, "agent_id", "unknown"),
                    "message": opinion.reasoning,
                })
            spoken.add(chosen)
        return ordered_pairs
    
    async def _get_final_vote_with_retry(
        self,
        agent: "BaseAgent",
        round_context: dict,
        team_identifier: str,
        all_opinions: List[AgentResponse],
        max_retries: int = 3,
    ) -> AgentResponse:
        """Get final vote with retry logic."""
        for attempt in range(max_retries):
            try:
                return await agent.get_final_vote(round_context, team_identifier, all_opinions)
            except asyncio.CancelledError:
                # Re-raise CancelledError - this is intentional task cancellation
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  [Warning] Final vote failed for {getattr(agent, 'agent_id', 'unknown')} (attempt {attempt + 1}): {type(e).__name__}")
                    await asyncio.sleep(1)
                else:
                    print(f"  [Warning] Final vote failed for {getattr(agent, 'agent_id', 'unknown')} after {max_retries} attempts, abstaining")
                    return AgentResponse.abstain("[Failed to get vote]")
        return AgentResponse.abstain("[Failed to get vote]")

    async def _gather_final_votes(
        self,
        round_context: dict,
        team_identifier: str,
        all_opinions: List[AgentResponse],
    ) -> List[AgentResponse]:
        """Gather final votes from all agents after sharing opinions.

        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            all_opinions: All initial opinions to share with agents

        Returns:
            List of final votes from all agents
        """
        tasks = [
            self._get_final_vote_with_retry(agent, round_context, team_identifier, all_opinions)
            for agent in self.agents
        ]
        return await asyncio.gather(*tasks)

    async def _run_discussion_round(
        self,
        round_context: dict,
        team_identifier: str,
        discussion_history: List[dict],
    ) -> List[tuple["BaseAgent", AgentResponse]]:
        """Run a single discussion round where agents respond to each other.

        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B'
            discussion_history: All messages from previous discussion

        Returns:
            List of (agent, response) pairs in speaking order
        """
        spoken = set()
        round_messages: List[tuple["BaseAgent", AgentResponse]] = []

        while len(spoken) < len(self.agents):
            pending_agents = [a for a in self.agents if a not in spoken]
            # Get willingness with individual retry logic for each agent
            willingness_tasks = [
                self._get_willingness_with_retry(a, round_context, team_identifier, discussion_history)
                for a in pending_agents
            ]
            willingness_values = await asyncio.gather(*willingness_tasks)
            max_w = max(willingness_values) if willingness_values else 0
            candidates = [a for a, w in zip(pending_agents, willingness_values) if w == max_w]
            chosen = random.choice(candidates)

            # Get followup response with retry logic
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = await chosen.get_followup_response(round_context, team_identifier, discussion_history)
                    break
                except asyncio.CancelledError:
                    # Re-raise CancelledError - this is intentional task cancellation
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  [Warning] Followup failed for {getattr(chosen, 'agent_id', 'unknown')} (attempt {attempt + 1}): {type(e).__name__}")
                        await asyncio.sleep(1)
                    else:
                        print(f"  [Error] Followup failed for {getattr(chosen, 'agent_id', 'unknown')} after {max_retries} attempts: {e}")
                        response = AgentResponse.abstain("[Failed to get response]")
            round_messages.append((chosen, response))

            # Add to discussion history (failed calls say nothing)
            if not response.failed:
                discussion_history.append({
                    "agent_id": getattr(chosen, "agent_id", "unknown"),
                    "message": response.reasoning,
                })
            spoken.add(chosen)

        return round_messages
    
    def _determine_majority(self, votes: List[AgentResponse]) -> tuple[Choice, dict, bool]:
        """Determine the majority choice from votes.

        Args:
            votes: List of agent votes

        Returns:
            Tuple of (winning_choice, vote_counts, was_unanimous)
        """
        # Abstentions (failed calls) are excluded; they are never counted as a vote for either side.
        choices = [v.choice for v in votes if isinstance(v.choice, Choice)]

        if not choices:
            raise NoValidVotesError(
                f"all {len(votes)} votes failed this round; refusing to fabricate a team choice"
            )

        vote_counts = Counter(choices)
        most_common = vote_counts.most_common()
        winning_choice = most_common[0][0]
        # Tie between BLACK and RED (possible with abstentions): break uniformly at random.
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            winning_choice = random.choice([most_common[0][0], most_common[1][0]])

        # Unanimous among the agents that actually voted
        was_unanimous = len(vote_counts) == 1

        # Convert Counter to regular dict for serialization
        counts_dict = {choice: count for choice, count in vote_counts.items()}

        return winning_choice, counts_dict, was_unanimous
    
    async def deliberate(
        self,
        round_context: dict,
        team_identifier: str,
    ) -> DeliberationResult:
        """Run the full deliberation process.

        Process:
        1. All agents share their initial opinions (ordered by willingness)
        2. Additional discussion rounds (if discussion_rounds > 1)
        3. All agents cast their final votes concurrently
        4. Majority vote determines team choice

        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team

        Returns:
            DeliberationResult with the team's final choice and all votes
        """
        # Phase 1: Gather initial opinions
        initial_pairs = await self._gather_initial_opinions(round_context, team_identifier)
        initial_opinions = [resp for _, resp in initial_pairs]

        # Build discussion history from initial opinions
        discussion_history: List[dict] = [
            {
                "agent_id": getattr(agent, "agent_id", "unknown"),
                "message": resp.reasoning,
            }
            for agent, resp in initial_pairs
        ]

        # Phase 2: Additional discussion rounds
        all_discussion_rounds: List[List[dict]] = []
        for round_num in range(1, self.discussion_rounds):
            round_messages = await self._run_discussion_round(
                round_context, team_identifier, discussion_history
            )
            # Record this round's messages
            round_record = [
                {
                    "agent_id": getattr(agent, "agent_id", "unknown"),
                    "message": resp.reasoning,
                    "choice": str(resp.choice),
                }
                for agent, resp in round_messages
            ]
            all_discussion_rounds.append(round_record)

        # Phase 3: Gather final votes
        final_votes = await self._gather_final_votes(round_context, team_identifier, initial_opinions)

        # Determine majority
        final_choice, vote_counts, was_unanimous = self._determine_majority(final_votes)

        return DeliberationResult(
            final_choice=final_choice,
            initial_opinions=initial_opinions,
            final_votes=final_votes,
            vote_counts=vote_counts,
            was_unanimous=was_unanimous,
            discussion_rounds=all_discussion_rounds,
            abstentions=count_abstentions(final_votes),
        )
