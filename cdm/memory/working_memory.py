"""Working Memory Subsystem for CDM-1 Cognitive Architecture.
Manages transient cognitive scratchpad, active goal stack, hypotheses,
and working memory focus without modifying long-term weights.
"""

from typing import Dict, List, Optional, Any
import time
from cdm.cognitive.state import WorkingMemoryState, GoalNode


class WorkingMemory:
    """Active working memory buffer maintaining current task context, scratchpad, and goals."""

    def __init__(self, max_scratchpad_items: int = 50, max_active_goals: int = 10):
        self.max_scratchpad_items = max_scratchpad_items
        self.max_active_goals = max_active_goals
        self.state = WorkingMemoryState()

    def set_attention_focus(self, focus_key: str) -> None:
        """Sets the current primary entity or topic of attention."""
        self.state.attention_focus = focus_key

    def push_goal(self, goal_id: str, description: str, priority: int = 1, parent_goal_id: Optional[str] = None) -> GoalNode:
        """Pushes a new goal onto the active goal stack."""
        goal = GoalNode(
            goal_id=goal_id,
            description=description,
            priority=priority,
            status="active",
            parent_goal_id=parent_goal_id,
        )
        if parent_goal_id:
            for g in self.state.active_goals:
                if g.goal_id == parent_goal_id:
                    g.subgoal_ids.append(goal_id)
                    break

        self.state.active_goals.append(goal)
        # Sort by priority (1 is highest)
        self.state.active_goals.sort(key=lambda x: x.priority)
        if len(self.state.active_goals) > self.max_active_goals:
            self.state.active_goals.pop()
        return goal

    def complete_goal(self, goal_id: str) -> bool:
        """Marks an active goal as completed."""
        for g in self.state.active_goals:
            if g.goal_id == goal_id:
                g.status = "completed"
                return True
        return False

    def write_scratchpad(self, key: str, value: Any) -> None:
        """Writes or updates an entry in the transient scratchpad."""
        if len(self.state.scratchpad) >= self.max_scratchpad_items and key not in self.state.scratchpad:
            # Evict oldest item
            oldest_key = next(iter(self.state.scratchpad))
            del self.state.scratchpad[oldest_key]
        self.state.scratchpad[key] = value

    def read_scratchpad(self, key: str, default: Any = None) -> Any:
        """Reads an entry from the transient scratchpad."""
        return self.state.scratchpad.get(key, default)

    def add_hypothesis(self, hypothesis: str, confidence: float, evidence_keys: List[str]) -> None:
        """Adds an active hypothesis to working memory."""
        self.state.active_hypotheses.append({
            "hypothesis": hypothesis,
            "confidence": confidence,
            "evidence_keys": evidence_keys,
            "timestamp": time.time(),
        })

    def snapshot(self) -> WorkingMemoryState:
        """Returns a snapshot of the current working memory state."""
        return self.state.model_copy(deep=True)

    def clear(self) -> None:
        """Clears all working memory state (goals, scratchpad, hypotheses)."""
        self.state = WorkingMemoryState()

    def clear_transient(self) -> None:
        """Clears transient scratchpad and hypotheses while retaining completed goal traces."""
        self.state.scratchpad.clear()
        self.state.active_hypotheses.clear()
        self.state.active_goals = [g for g in self.state.active_goals if g.status == "active"]
