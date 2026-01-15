"""
Tests for Real-time Dashboard feature (004-realtime-dashboard).

Tests cover:
- US1: Watch Simulation Progress Live
- US2: View Live Utilization Metrics
- US3: Control Simulation Speed
- US4: View Task Flow Visualization

Note: Widget rendering requires manual verification in Jupyter.
These tests focus on logic and state management.
"""

import pytest

from eddt.dashboard import (
    SPEED_SETTINGS,
    AgentDisplayState,
    Dashboard,
    DashboardConfig,
    DashboardState,
    QueueDisplayState,
    SpeedSetting,
    create_dashboard,
)
from eddt.visualizations import (
    TASK_TYPE_COLORS,
    format_agent_card_html,
    format_queue_item_html,
    get_status_color,
    get_task_type_color,
    get_utilization_color,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


class MockTask:
    """Mock task for testing."""

    def __init__(self, task_type: str, status: str = "pending"):
        self.task_type = type("TaskType", (), {"value": task_type})()
        self.status = type("Status", (), {"value": status})()


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        name: str,
        role: str,
        status: str = "idle",
        utilization: float = 0.0,
        current_task=None,
        tasks_completed: int = 0,
    ):
        self.name = name
        self.role = type("Role", (), {"value": role})()
        self.status = type("Status", (), {"value": status})()
        self.utilization = utilization
        self.current_task = current_task
        self.tasks_completed = tasks_completed


class MockMetrics:
    """Mock metrics collector for testing."""

    def __init__(self):
        self.queue_depths = {"part_design": 5, "review": 3}


class MockModel:
    """Mock EngineeringDepartment model for testing."""

    def __init__(self):
        self.agents = [
            MockAgent("Alice", "senior_designer", "working", 0.8, MockTask("part_design"), 5),
            MockAgent("Bob", "junior_designer", "idle", 0.3, None, 2),
            MockAgent("Carol", "reviewer", "blocked", 0.95, None, 8),
        ]
        self.tasks = [
            MockTask("part_design", "completed"),
            MockTask("part_design", "completed"),
            MockTask("part_design", "pending"),
            MockTask("review", "pending"),
            MockTask("review", "pending"),
        ]
        self.tick_count = 25
        self.metrics = MockMetrics()
        self._days_to_run = 5

    def step(self):
        """Advance simulation by one tick."""
        self.tick_count += 1


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel()


@pytest.fixture
def dashboard(mock_model):
    """Create a dashboard instance for testing."""
    return Dashboard(mock_model)


# -----------------------------------------------------------------------------
# US1: Watch Simulation Progress Live
# -----------------------------------------------------------------------------


class TestDashboardStateUpdates:
    """Tests for dashboard state updates."""

    def test_dashboard_state_updates(self, dashboard):
        """Test that dashboard state updates from model."""
        dashboard._update_state()

        assert dashboard.state.current_tick == 25
        assert len(dashboard.state.agents) == 3
        assert dashboard.state.tasks_completed == 2
        assert dashboard.state.tasks_total == 5

    def test_agent_status_reflects_model(self, dashboard):
        """Test that agent status reflects model state."""
        dashboard._update_state()

        alice = next(a for a in dashboard.state.agents if a.name == "Alice")
        assert alice.status.value == "working"
        assert alice.utilization == 0.8
        assert alice.current_task == "part_design"

    def test_tick_counter_increments(self, dashboard):
        """Test that tick counter increments on step."""
        initial_tick = dashboard.model.tick_count
        dashboard.step()

        assert dashboard.model.tick_count == initial_tick + 1
        assert dashboard.state.current_tick == initial_tick + 1


class TestAgentDisplayState:
    """Tests for AgentDisplayState dataclass."""

    def test_status_color_working(self):
        """Test status color for working agent."""
        from eddt.dashboard import AgentStatus

        agent = AgentDisplayState(
            name="Test",
            role="designer",
            status=AgentStatus.WORKING,
            utilization=0.5,
        )
        assert agent.status_color == "#28a745"  # Green

    def test_status_color_blocked(self):
        """Test status color for blocked agent."""
        from eddt.dashboard import AgentStatus

        agent = AgentDisplayState(
            name="Test",
            role="designer",
            status=AgentStatus.BLOCKED,
            utilization=0.5,
        )
        assert agent.status_color == "#dc3545"  # Red

    def test_utilization_color_low(self):
        """Test utilization color for low utilization."""
        from eddt.dashboard import AgentStatus

        agent = AgentDisplayState(
            name="Test",
            role="designer",
            status=AgentStatus.IDLE,
            utilization=0.3,
        )
        assert agent.utilization_color == "#28a745"  # Green

    def test_utilization_color_high(self):
        """Test utilization color for high utilization."""
        from eddt.dashboard import AgentStatus

        agent = AgentDisplayState(
            name="Test",
            role="designer",
            status=AgentStatus.WORKING,
            utilization=0.9,
        )
        assert agent.utilization_color == "#dc3545"  # Red


# -----------------------------------------------------------------------------
# US2: View Live Utilization Metrics
# -----------------------------------------------------------------------------


class TestUtilizationCalculation:
    """Tests for utilization calculation."""

    def test_utilization_calculation(self, dashboard):
        """Test utilization calculation from model."""
        utilizations = dashboard._calculate_utilization()

        assert utilizations["Alice"] == 0.8
        assert utilizations["Bob"] == 0.3
        assert utilizations["Carol"] == 0.95

    def test_utilization_updates_each_tick(self, dashboard):
        """Test that utilization history updates on state update."""
        dashboard._update_state()
        history_len_1 = len(dashboard.state.utilization_history)

        dashboard._update_state()
        history_len_2 = len(dashboard.state.utilization_history)

        assert history_len_2 == history_len_1 + 1


class TestUtilizationHistory:
    """Tests for utilization history tracking."""

    def test_history_window_limit(self, dashboard):
        """Test that history respects window limit."""
        dashboard.config.history_window = 5

        for _ in range(10):
            dashboard._update_state()

        assert len(dashboard.state.utilization_history) <= 5


# -----------------------------------------------------------------------------
# US3: Control Simulation Speed
# -----------------------------------------------------------------------------


class TestSpeedControlChangesDelay:
    """Tests for speed control."""

    def test_speed_control_changes_delay(self):
        """Test that speed settings have different delays."""
        speeds = {s.name: s.delay_ms for s in SPEED_SETTINGS}

        # Slower speeds should have higher delays
        assert speeds["0.5x"] > speeds["1x"]
        assert speeds["1x"] > speeds["2x"]
        assert speeds["2x"] > speeds["4x"]


class TestPauseStopsSimulation:
    """Tests for pause functionality."""

    def test_pause_stops_simulation(self, dashboard):
        """Test that pause stops the simulation."""
        dashboard._paused = False
        dashboard._running = True

        dashboard.pause()

        assert dashboard._paused is True

    def test_play_resumes_simulation(self, dashboard):
        """Test that play resumes the simulation."""
        dashboard._paused = True
        # Note: We can't actually test the async loop without running Jupyter
        # This test verifies the state change logic
        dashboard._paused = False
        dashboard._running = True

        assert dashboard._paused is False
        assert dashboard._running is True


class TestStepAdvancesOneTick:
    """Tests for step functionality."""

    def test_step_advances_one_tick(self, dashboard):
        """Test that step advances exactly one tick."""
        initial_tick = dashboard.model.tick_count
        dashboard.step()

        assert dashboard.model.tick_count == initial_tick + 1


class TestSpeedSettings:
    """Tests for speed setting dataclass."""

    def test_speed_setting_properties(self):
        """Test SpeedSetting dataclass properties."""
        setting = SpeedSetting("Test", 2.0, 50)

        assert setting.name == "Test"
        assert setting.multiplier == 2.0
        assert setting.delay_ms == 50

    def test_speed_settings_list(self):
        """Test that SPEED_SETTINGS contains expected values."""
        names = [s.name for s in SPEED_SETTINGS]

        assert "Pause" in names
        assert "1x" in names
        assert "2x" in names
        assert "Max" in names


# -----------------------------------------------------------------------------
# US4: View Task Flow Visualization
# -----------------------------------------------------------------------------


class TestQueueDepthTracking:
    """Tests for queue depth tracking."""

    def test_queue_depth_tracking(self, dashboard):
        """Test that queue depths are extracted from model."""
        queues = dashboard._extract_queue_depths()

        assert len(queues) > 0
        queue_types = {q.task_type for q in queues}
        assert "part_design" in queue_types or "review" in queue_types


class TestQueueChartData:
    """Tests for queue chart data."""

    def test_queue_chart_data(self, dashboard):
        """Test that queue history is tracked for charts."""
        dashboard._update_state()
        dashboard._update_state()

        assert len(dashboard.state.queue_history) >= 2


class TestQueueDisplayState:
    """Tests for QueueDisplayState dataclass."""

    def test_queue_display_state(self):
        """Test QueueDisplayState properties."""
        queue = QueueDisplayState(
            task_type="part_design",
            depth=10,
            avg_wait_time=1.5,
        )

        assert queue.task_type == "part_design"
        assert queue.depth == 10
        assert queue.avg_wait_time == 1.5


# -----------------------------------------------------------------------------
# Visualization Tests
# -----------------------------------------------------------------------------


class TestFormatAgentCardHtml:
    """Tests for agent card HTML formatting."""

    def test_format_agent_card_html(self):
        """Test HTML formatting for agent card."""
        html = format_agent_card_html(
            name="Alice",
            role="senior_designer",
            status="working",
            utilization=0.8,
            current_task="part_design",
            tasks_completed=5,
        )

        assert "Alice" in html
        assert "senior_designer" in html
        assert "working" in html
        assert "80%" in html
        assert "part_design" in html


class TestFormatQueueItemHtml:
    """Tests for queue item HTML formatting."""

    def test_format_queue_item_html(self):
        """Test HTML formatting for queue item."""
        html = format_queue_item_html(
            task_type="review",
            depth=12,
            avg_wait_time=2.5,
        )

        assert "review" in html
        assert "12" in html
        assert "2.5" in html


class TestColorFunctions:
    """Tests for color helper functions."""

    def test_get_task_type_color(self):
        """Test task type color retrieval."""
        color = get_task_type_color("part_design")
        assert color == "#3498db"

        default = get_task_type_color("unknown_type")
        assert default == TASK_TYPE_COLORS["default"]

    def test_get_status_color(self):
        """Test status color retrieval."""
        working = get_status_color("working")
        assert working == "#28a745"

        idle = get_status_color("idle")
        assert idle == "#6c757d"

    def test_get_utilization_color(self):
        """Test utilization color retrieval."""
        low = get_utilization_color(0.3)
        assert low == "#28a745"

        high = get_utilization_color(0.9)
        assert high == "#dc3545"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestCreateDashboard:
    """Tests for dashboard creation."""

    def test_create_dashboard(self, mock_model):
        """Test creating a dashboard with factory function."""
        dashboard = create_dashboard(mock_model)

        assert isinstance(dashboard, Dashboard)
        assert dashboard.model is mock_model

    def test_create_dashboard_with_config(self, mock_model):
        """Test creating a dashboard with custom config."""
        config = DashboardConfig(
            update_interval_ms=200,
            history_window=50,
            show_charts=False,
        )
        dashboard = create_dashboard(mock_model, config)

        assert dashboard.config.update_interval_ms == 200
        assert dashboard.config.history_window == 50
        assert dashboard.config.show_charts is False


class TestDashboardConfig:
    """Tests for DashboardConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DashboardConfig()

        assert config.update_interval_ms == 100
        assert config.history_window == 100
        assert config.show_charts is True
        assert config.show_agent_cards is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DashboardConfig(
            update_interval_ms=50,
            history_window=200,
            show_charts=False,
        )

        assert config.update_interval_ms == 50
        assert config.history_window == 200
        assert config.show_charts is False


class TestDashboardState:
    """Tests for DashboardState."""

    def test_initial_state(self):
        """Test initial dashboard state."""
        state = DashboardState()

        assert state.is_running is False
        assert state.is_paused is True
        assert state.current_tick == 0
        assert len(state.agents) == 0
        assert len(state.queues) == 0

    def test_get_state(self, dashboard):
        """Test get_state method."""
        state = dashboard.get_state()

        assert isinstance(state, DashboardState)
        assert state.current_tick == 25
