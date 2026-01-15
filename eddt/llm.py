"""
LLM-based decision making with tiered routing.
Uses Ollama for local inference.

Feature 005 enhancements:
- DecisionContext dataclass for structured decision inputs
- TaskRecommendation dataclass for ranked task suggestions
- Tiered model selection based on decision complexity
- Deterministic fallback mode for testing
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import EngineerAgent
    from .tasks import Task


@dataclass
class DecisionContext:
    """
    Structured context for LLM decision making.

    Contains available tasks, blocked resources, and agent state
    for informed decision support.
    """

    agent_name: str
    agent_role: str
    agent_skill_level: str
    available_tasks: List[dict] = field(default_factory=list)
    blocked_resources: List[str] = field(default_factory=list)
    current_task: Optional[dict] = None
    is_blocked: bool = False
    blocked_duration_hours: float = 0.0

    @property
    def complexity_score(self) -> float:
        """
        Calculate decision complexity (0.0-1.0).

        Higher scores indicate more complex decisions requiring tier 2.
        """
        score = 0.0

        # Multiple available tasks increases complexity
        if len(self.available_tasks) > 3:
            score += 0.3
        elif len(self.available_tasks) > 1:
            score += 0.1

        # Blocked resources add complexity
        if len(self.blocked_resources) > 0:
            score += 0.2 * min(len(self.blocked_resources), 3)

        # Being blocked for long time is complex
        if self.blocked_duration_hours > 1.0:
            score += 0.2

        return min(score, 1.0)


@dataclass
class TaskRecommendation:
    """
    A task recommendation from the LLM with reasoning.
    """

    task_id: int
    task_name: str
    rank: int  # 1 = best choice
    reasoning: str
    confidence: float = 0.8  # 0.0-1.0


class DecisionCache:
    """Simple cache for LLM decisions to avoid redundant calls."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, dict] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_context(self, context: dict) -> str:
        """Create hash from context, ignoring volatile fields."""
        # Only hash stable parts of context
        stable = {
            "role": context.get("agent_role"),
            "status": context.get("agent_status"),
            "has_task": context.get("current_task") is not None,
            "queue_empty": context.get("queue_length", 0) == 0,
            "available_tasks": context.get("available_tasks", 0) > 0,
            "is_blocked": context.get("blocked_reason") is not None,
        }
        return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]

    def get(self, context: dict) -> Optional[dict]:
        """Get cached decision if available."""
        key = self._hash_context(context)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def store(self, context: dict, decision: dict):
        """Store a decision in cache."""
        key = self._hash_context(context)
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = decision

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LLMDecisionMaker:
    """
    Tiered LLM decision maker.

    Tier 1 (small model): Routine decisions - action selection
    Tier 2 (medium model): Complex decisions - prioritization, messaging

    Can operate in 'mock' mode without actual LLM for testing.
    """

    # Decisions that can be handled by tier 1 (or rules)
    TIER1_SITUATIONS = {
        "has_task_working",  # Continue working
        "idle_with_available",  # Start a task
        "task_complete",  # Complete and get next
        "no_work",  # Go idle
    }

    def __init__(
        self,
        tier1_model: str = "qwen2.5:1.5b",
        tier2_model: str = "qwen2.5:7b",
        use_cache: bool = True,
        use_llm: bool = False,  # Default to rule-based for speed
        deterministic_mode: bool = True,  # Feature 005: for reproducibility
        tier1_timeout_ms: int = 2000,  # Feature 005: timeout per spec
        tier2_timeout_ms: int = 10000,  # Feature 005: timeout per spec
    ):
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.cache = DecisionCache() if use_cache else None
        self.use_llm = use_llm
        self.deterministic_mode = deterministic_mode
        self.tier1_timeout_ms = tier1_timeout_ms
        self.tier2_timeout_ms = tier2_timeout_ms

        # LLM client (lazy loaded)
        self._ollama = None

        # Stats
        self.tier1_calls = 0
        self.tier2_calls = 0
        self.rule_calls = 0
        self.fallback_calls = 0

    def _get_ollama(self):
        """Lazy load Ollama client."""
        if self._ollama is None and self.use_llm:
            try:
                import ollama

                self._ollama = ollama
                # Verify models are available
                self._check_models()
            except ImportError:
                print("Warning: ollama package not installed. Using rule-based decisions.")
                self.use_llm = False
            except Exception as e:
                print(f"Warning: Could not connect to Ollama: {e}")
                self.use_llm = False
        return self._ollama

    def _check_models(self):
        """Verify Ollama models are available."""
        if not self._ollama:
            return

        try:
            models = self._ollama.list()
            available = [m.get("name", "").split(":")[0] for m in models.get("models", [])]

            tier1_base = self.tier1_model.split(":")[0]
            tier2_base = self.tier2_model.split(":")[0]

            if tier1_base not in available:
                print(f"Warning: {self.tier1_model} not found. Run: ollama pull {self.tier1_model}")
            if tier2_base not in available:
                print(f"Warning: {self.tier2_model} not found. Run: ollama pull {self.tier2_model}")
        except Exception as e:
            print(f"Warning: Could not list Ollama models: {e}")

    def decide(self, agent: "EngineerAgent", context: dict) -> dict:
        """
        Main entry point - decide what agent should do.
        Routes to appropriate tier based on situation.
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(context)
            if cached:
                return cached

        # Classify situation
        situation = self._classify_situation(context)

        # Use rules for simple situations (fast)
        if situation in self.TIER1_SITUATIONS:
            decision = self._rule_based_decide(context, situation)
            self.rule_calls += 1
        elif self.use_llm:
            # Use LLM for complex situations
            decision = self._tier2_decide(context, situation)
            self.tier2_calls += 1
        else:
            # Fallback to rules
            decision = self._rule_based_decide(context, situation)
            self.rule_calls += 1

        # Cache result
        if self.cache:
            self.cache.store(context, decision)

        return decision

    def _classify_situation(self, context: dict) -> str:
        """Classify the situation to route to appropriate tier."""
        status = context.get("agent_status")
        has_task = context.get("current_task") is not None
        available = context.get("available_tasks", 0) > 0
        is_blocked = context.get("blocked_reason") is not None
        queue_length = context.get("queue_length", 0)

        if is_blocked:
            return "blocked_needs_resolution"

        if status == "working" and has_task:
            # Check if task is complete (progress near 100%)
            task = context.get("current_task", {})
            progress_str = task.get("progress", "0%") if task else "0%"
            progress = float(progress_str.rstrip("%")) / 100
            if progress >= 0.99:
                return "task_complete"
            return "has_task_working"

        if status == "idle":
            if available:
                return "idle_with_available"
            if queue_length > 0:
                return "idle_with_queue"
            return "no_work"

        if context.get("unread_messages", 0) > 0:
            return "has_messages"

        return "complex_situation"

    def _rule_based_decide(self, context: dict, situation: str) -> dict:
        """
        Rule-based decision making for simple situations.
        Fast and deterministic.
        """
        if situation == "has_task_working":
            return {"action": "continue", "reason": "working on task"}

        if situation == "task_complete":
            return {"action": "complete_task", "reason": "task finished"}

        if situation == "no_work":
            return {"action": "go_idle", "reason": "no tasks available"}

        if situation == "idle_with_available":
            return {"action": "start_task", "reason": "task available"}

        if situation == "idle_with_queue":
            return {"action": "start_task", "reason": "task in queue"}

        if situation == "blocked_needs_resolution":
            # Simple rule: if blocked too long, try to switch tasks
            blocked_duration = context.get("blocked_duration")
            if blocked_duration and float(blocked_duration.rstrip("h")) > 2.0:
                if context.get("available_tasks", 0) > 0:
                    return {"action": "start_task", "reason": "switching due to long block"}
            return {"action": "continue", "reason": "waiting for blocker resolution"}

        if situation == "has_messages":
            return {"action": "continue", "reason": "processing messages"}

        # Default
        return {"action": "go_idle", "reason": "default fallback"}

    def _tier2_decide(self, context: dict, situation: str) -> dict:
        """
        Complex tier 2 decision using larger model.
        Used for prioritization, communication, blocker resolution.
        """
        ollama = self._get_ollama()
        if not ollama:
            return self._rule_based_decide(context, situation)

        prompt = self._build_tier2_prompt(context, situation)

        try:
            response = ollama.generate(
                model=self.tier2_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 100,
                    "top_p": 0.95,
                },
            )
            return self._parse_tier2_response(response["response"])
        except Exception as e:
            print(f"Tier 2 LLM error: {e}")
            return self._rule_based_decide(context, situation)

    def _build_tier2_prompt(self, context: dict, situation: str) -> str:
        """Build detailed prompt for tier 2."""
        if situation == "blocked_needs_resolution":
            return f"""You are {context.get('agent_name')}, a {context.get('agent_role')}.
You have been blocked for {context.get('blocked_duration', 'unknown')}.
Reason: {context.get('blocked_reason')}

Recent actions: {context.get('recent_actions', [])}

Decide what to do:
1. Keep waiting (if recently blocked)
2. Escalate to manager
3. Try a workaround
4. Switch to different task

Respond with JSON: {{"action": "...", "reason": "..."}}"""

        if situation == "has_messages":
            return f"""You are {context.get('agent_name')}.
You have unread messages.

Decide how to respond.
Respond with JSON: {{"action": "send_message", "recipient": "...", "content": "..."}}
Or: {{"action": "continue", "reason": "..."}}"""

        # Default complex prompt
        return f"""You are {context.get('agent_name')}, a {context.get('agent_role')}.
Current status: {context.get('agent_status')}
Task: {context.get('current_task')}
Queue: {context.get('queue_preview')}

What should you do next?
Respond with JSON: {{"action": "...", "reason": "..."}}"""

    def _parse_tier2_response(self, response: str) -> dict:
        """Parse tier 2 response (JSON)."""
        response = response.strip()

        # Try to extract JSON
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback parsing
        if "continue" in response.lower():
            return {"action": "continue", "reason": "parsed_from_text"}
        if "blocked" in response.lower():
            return {"action": "report_blocked", "reason": "parsed_from_text"}

        return {"action": "continue", "reason": "parse_fallback"}

    def get_stats(self) -> dict:
        """Get LLM usage statistics."""
        return {
            "cache_hit_rate": self.cache.hit_rate if self.cache else 0,
            "cache_size": len(self.cache.cache) if self.cache else 0,
            "tier1_calls": self.tier1_calls,
            "tier2_calls": self.tier2_calls,
            "rule_calls": self.rule_calls,
            "fallback_calls": self.fallback_calls,
            "use_llm": self.use_llm,
            "deterministic_mode": self.deterministic_mode,
        }

    # -------------------------------------------------------------------------
    # Feature 005: Enhanced operation selection methods
    # -------------------------------------------------------------------------

    def select_decision_tier(self, context: DecisionContext) -> int:
        """
        Select which tier to use based on decision complexity.

        Args:
            context: DecisionContext with current state

        Returns:
            1 for tier 1 (fast), 2 for tier 2 (complex)
        """
        if self.deterministic_mode and not self.use_llm:
            return 0  # Use rules only

        complexity = context.complexity_score

        if complexity < 0.3:
            return 1  # Simple decision, use tier 1
        else:
            return 2  # Complex decision, use tier 2

    def consult_llm_for_task(
        self,
        agent: "EngineerAgent",
        context: DecisionContext,
    ) -> List[TaskRecommendation]:
        """
        Consult LLM for task prioritization.

        Returns ranked list of task recommendations.

        Args:
            agent: The agent requesting advice
            context: Decision context with available tasks

        Returns:
            List of TaskRecommendation sorted by rank
        """
        if not context.available_tasks:
            return []

        # Use rule-based fallback in deterministic mode
        if self.deterministic_mode or not self.use_llm:
            return self.rule_based_fallback(context)

        tier = self.select_decision_tier(context)

        if tier == 1:
            return self._tier1_task_selection(context)
        else:
            return self._tier2_task_selection(context)

    def _tier1_task_selection(self, context: DecisionContext) -> List[TaskRecommendation]:
        """Simple tier 1 task selection - prioritize by type match."""
        recommendations = []

        for i, task in enumerate(context.available_tasks):
            # Simple heuristic: prefer tasks matching specialization
            confidence = 0.7
            reasoning = "Available task"

            if context.agent_role and task.get("type", "").startswith(context.agent_role.split("_")[0]):
                confidence = 0.9
                reasoning = "Matches agent role"

            recommendations.append(
                TaskRecommendation(
                    task_id=task.get("id", i),
                    task_name=task.get("name", f"Task {i}"),
                    rank=i + 1,
                    reasoning=reasoning,
                    confidence=confidence,
                )
            )

        self.tier1_calls += 1
        return recommendations

    def _tier2_task_selection(self, context: DecisionContext) -> List[TaskRecommendation]:
        """Complex tier 2 task selection using LLM."""
        prompt = self._build_task_prioritization_prompt(context)

        ollama = self._get_ollama()
        if not ollama:
            self.fallback_calls += 1
            return self.rule_based_fallback(context)

        try:
            response = ollama.generate(
                model=self.tier2_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 200,
                },
            )
            self.tier2_calls += 1
            return self._parse_task_recommendations(response["response"], context)
        except Exception as e:
            print(f"Tier 2 task selection error: {e}")
            self.fallback_calls += 1
            return self.rule_based_fallback(context)

    def _build_task_prioritization_prompt(self, context: DecisionContext) -> str:
        """Build prompt for task prioritization."""
        tasks_desc = "\n".join(
            f"- {t.get('name', 'Unknown')}: type={t.get('type', 'unknown')}, "
            f"complexity={t.get('complexity', 'medium')}"
            for t in context.available_tasks
        )

        blocked_desc = ", ".join(context.blocked_resources) if context.blocked_resources else "none"

        return f"""You are {context.agent_name}, a {context.agent_role} ({context.agent_skill_level} level).

Available tasks:
{tasks_desc}

Blocked resources: {blocked_desc}

Rank these tasks by priority. Consider:
1. Tasks matching your skill level
2. Tasks not requiring blocked resources
3. Urgency and dependencies

Respond with JSON array:
[{{"task_name": "...", "rank": 1, "reasoning": "..."}}, ...]"""

    def _build_blocked_resource_prompt(self, context: DecisionContext) -> str:
        """Build prompt for blocked resource strategy."""
        return f"""You are {context.agent_name}, a {context.agent_role}.
You have been blocked for {context.blocked_duration_hours:.1f} hours.

Blocked resources: {', '.join(context.blocked_resources)}
Available alternative tasks: {len(context.available_tasks)}

What should you do?
1. Keep waiting (resource will be free soon)
2. Switch to alternative task
3. Escalate to manager

Respond with JSON: {{"action": "...", "reasoning": "..."}}"""

    def _parse_task_recommendations(
        self,
        response: str,
        context: DecisionContext,
    ) -> List[TaskRecommendation]:
        """Parse LLM response into task recommendations."""
        try:
            # Try to extract JSON array
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)

                recommendations = []
                for item in parsed:
                    # Find matching task
                    task_name = item.get("task_name", "")
                    task_id = 0
                    for t in context.available_tasks:
                        if t.get("name", "") == task_name:
                            task_id = t.get("id", 0)
                            break

                    recommendations.append(
                        TaskRecommendation(
                            task_id=task_id,
                            task_name=task_name,
                            rank=item.get("rank", len(recommendations) + 1),
                            reasoning=item.get("reasoning", "LLM recommendation"),
                            confidence=0.8,
                        )
                    )

                return sorted(recommendations, key=lambda r: r.rank)
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback
        return self.rule_based_fallback(context)

    def rule_based_fallback(self, context: DecisionContext) -> List[TaskRecommendation]:
        """
        Rule-based task selection fallback.

        Used when LLM is unavailable or in deterministic mode.
        """
        recommendations = []

        for i, task in enumerate(context.available_tasks):
            # Score based on simple heuristics
            score = 100 - i  # Base score by position

            # Boost for matching role
            task_type = task.get("type", "")
            if "designer" in context.agent_role.lower() and "design" in task_type:
                score += 20
            if "engineer" in context.agent_role.lower() and "simulation" in task_type:
                score += 20

            # Penalize if resource is blocked
            resource = task.get("resource")
            if resource and resource in context.blocked_resources:
                score -= 50

            # Prefer simpler tasks for junior
            complexity = task.get("complexity", "medium")
            if context.agent_skill_level == "junior":
                if complexity == "simple":
                    score += 10
                elif complexity == "complex":
                    score -= 10

            recommendations.append(
                TaskRecommendation(
                    task_id=task.get("id", i),
                    task_name=task.get("name", f"Task {i}"),
                    rank=0,  # Will be set after sorting
                    reasoning="Rule-based selection",
                    confidence=0.7,
                )
            )
            recommendations[-1]._score = score

        # Sort by score and assign ranks
        recommendations.sort(key=lambda r: getattr(r, "_score", 0), reverse=True)
        for i, rec in enumerate(recommendations):
            rec.rank = i + 1
            delattr(rec, "_score")

        self.rule_calls += 1
        return recommendations
