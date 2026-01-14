"""Workflow engine for deliverable state transitions."""

from datetime import datetime
from typing import Dict, Optional
from .environment import Deliverable


class WorkflowEngine:
    """Manages state transitions and approval chains for deliverables."""

    DESIGN_WORKFLOW = {
        "draft": {
            "transitions": ["submit_for_review"],
            "allowed_roles": ["designer", "engineer"],
        },
        "in_review": {
            "transitions": ["approve", "reject", "request_changes"],
            "allowed_roles": ["reviewer", "lead_engineer"],
            "sla_hours": 24,
        },
        "changes_requested": {
            "transitions": ["resubmit"],
            "allowed_roles": ["designer", "engineer"],
        },
        "approved": {
            "transitions": ["release"],
            "allowed_roles": ["release_engineer", "plm_admin"],
        },
        "released": {
            "transitions": [],  # Terminal state
            "triggers": ["erp_sync", "notification"],
        },
    }

    def __init__(self):
        """Initialize workflow engine."""
        self.state_entered_at: Dict[str, datetime] = {}

    def can_transition(
        self, deliverable: Deliverable, agent_role: str, transition: str
    ) -> bool:
        """
        Check if agent can perform this transition.

        Args:
            deliverable: Deliverable to transition
            agent_role: Role of the agent
            transition: Transition name

        Returns:
            True if transition is allowed
        """
        current_state_config = self.DESIGN_WORKFLOW.get(deliverable.current_state)
        if not current_state_config:
            return False

        return (
            transition in current_state_config["transitions"]
            and agent_role in current_state_config.get("allowed_roles", [])
        )

    def transition(
        self, deliverable: Deliverable, new_state: str, agent_role: str
    ) -> bool:
        """
        Perform state transition.

        Args:
            deliverable: Deliverable to transition
            new_state: Target state
            agent_role: Role of the agent performing transition

        Returns:
            True if transition succeeded
        """
        # Find transition name
        current_config = self.DESIGN_WORKFLOW.get(deliverable.current_state)
        if not current_config:
            return False

        # Check if transition exists
        transition_name = None
        for transition in current_config["transitions"]:
            # Map transition to state (simplified)
            if transition == "submit_for_review" and new_state == "in_review":
                transition_name = transition
            elif transition == "approve" and new_state == "approved":
                transition_name = transition
            elif transition == "reject" and new_state == "changes_requested":
                transition_name = transition
            elif transition == "request_changes" and new_state == "changes_requested":
                transition_name = transition
            elif transition == "resubmit" and new_state == "in_review":
                transition_name = transition
            elif transition == "release" and new_state == "released":
                transition_name = transition

        if not transition_name:
            return False

        if not self.can_transition(deliverable, agent_role, transition_name):
            return False

        # Perform transition
        old_state = deliverable.current_state
        deliverable.current_state = new_state
        self.state_entered_at[deliverable.id] = datetime.now()

        # Handle state-specific logic
        if new_state == "in_review":
            deliverable.review_cycles += 1

        return True

    def get_sla_status(self, deliverable: Deliverable) -> Optional[Dict]:
        """
        Calculate SLA compliance for deliverable.

        Args:
            deliverable: Deliverable to check

        Returns:
            SLA status dict or None
        """
        current_config = self.DESIGN_WORKFLOW.get(deliverable.current_state)
        if not current_config or "sla_hours" not in current_config:
            return None

        entered_at = self.state_entered_at.get(deliverable.id)
        if not entered_at:
            return None

        elapsed_hours = (datetime.now() - entered_at).total_seconds() / 3600
        sla_hours = current_config["sla_hours"]

        return {
            "sla_hours": sla_hours,
            "elapsed_hours": elapsed_hours,
            "compliant": elapsed_hours < sla_hours,
        }
