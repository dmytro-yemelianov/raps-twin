"""
LLM-based decision making with tiered routing.
Uses Ollama for local inference.
"""

import json
import hashlib
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import EngineerAgent


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
    ):
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.cache = DecisionCache() if use_cache else None
        self.use_llm = use_llm

        # LLM client (lazy loaded)
        self._ollama = None

        # Stats
        self.tier1_calls = 0
        self.tier2_calls = 0
        self.rule_calls = 0

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
            "use_llm": self.use_llm,
        }
