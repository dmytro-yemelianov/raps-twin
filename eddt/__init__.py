"""
EDDT: Engineering Department Digital Twin
Mesa/SimPy implementation for transparent, debuggable simulation.
"""

from .model import EngineeringDepartment, run_simulation
from .agents import EngineerAgent, AgentStatus, EngineerRole
from .tasks import Task, TaskStatus, TaskType
from .resources import ToolResources
from .llm import LLMDecisionMaker
from .metrics import MetricsCollector
from .comparison import (
    compare_scenarios,
    validate_scenario_configs,
    export_comparison_csv,
    export_comparison_json,
    get_comparison_summary_table,
    ScenarioMetrics,
    ScenarioResult,
    ComparisonMetric,
    ComparisonSet,
)
from .bottleneck import (
    analyze_bottlenecks,
    detect_engineer_bottlenecks,
    detect_queue_bottlenecks,
    generate_recommendations,
    get_bottleneck_time_series,
    get_utilization_timeseries,
    export_bottleneck_report_csv,
    format_bottleneck_report,
    BottleneckConfig,
    BottleneckReport,
    EngineerBottleneck,
    QueueBottleneck,
    BottleneckRecommendation,
    TimeSeriesPoint,
    UtilizationTimeSeries,
)
from .whatif import (
    run_whatif_experiment,
    ask_whatif,
    parse_modification,
    validate_modifications,
    apply_modifications,
    format_experiment_result,
    Modification,
    ModificationError,
    MetricDelta,
    ExperimentComparison,
    WhatIfExperiment,
)
from .dashboard import (
    Dashboard,
    DashboardConfig,
    DashboardState,
    AgentDisplayState,
    QueueDisplayState,
    SpeedSetting,
    SPEED_SETTINGS,
    create_dashboard,
    run_with_dashboard,
)
from .visualizations import (
    format_agent_card_html,
    format_queue_item_html,
    format_summary_card_html,
    create_utilization_chart,
    create_queue_chart,
    create_status_pie_chart,
    get_task_type_color,
    get_status_color,
    get_utilization_color,
    TASK_TYPE_COLORS,
    STATUS_COLORS,
    UTILIZATION_COLORS,
)

__version__ = "0.1.0"
__all__ = [
    "EngineeringDepartment",
    "run_simulation",
    "EngineerAgent",
    "AgentStatus",
    "EngineerRole",
    "Task",
    "TaskStatus",
    "TaskType",
    "ToolResources",
    "LLMDecisionMaker",
    "MetricsCollector",
    # Comparison module
    "compare_scenarios",
    "validate_scenario_configs",
    "export_comparison_csv",
    "export_comparison_json",
    "get_comparison_summary_table",
    "ScenarioMetrics",
    "ScenarioResult",
    "ComparisonMetric",
    "ComparisonSet",
    # Bottleneck module
    "analyze_bottlenecks",
    "detect_engineer_bottlenecks",
    "detect_queue_bottlenecks",
    "generate_recommendations",
    "get_bottleneck_time_series",
    "get_utilization_timeseries",
    "export_bottleneck_report_csv",
    "format_bottleneck_report",
    "BottleneckConfig",
    "BottleneckReport",
    "EngineerBottleneck",
    "QueueBottleneck",
    "BottleneckRecommendation",
    "TimeSeriesPoint",
    "UtilizationTimeSeries",
    # What-If module
    "run_whatif_experiment",
    "ask_whatif",
    "parse_modification",
    "validate_modifications",
    "apply_modifications",
    "format_experiment_result",
    "Modification",
    "ModificationError",
    "MetricDelta",
    "ExperimentComparison",
    "WhatIfExperiment",
    # Dashboard module
    "Dashboard",
    "DashboardConfig",
    "DashboardState",
    "AgentDisplayState",
    "QueueDisplayState",
    "SpeedSetting",
    "SPEED_SETTINGS",
    "create_dashboard",
    "run_with_dashboard",
    # Visualizations module
    "format_agent_card_html",
    "format_queue_item_html",
    "format_summary_card_html",
    "create_utilization_chart",
    "create_queue_chart",
    "create_status_pie_chart",
    "get_task_type_color",
    "get_status_color",
    "get_utilization_color",
    "TASK_TYPE_COLORS",
    "STATUS_COLORS",
    "UTILIZATION_COLORS",
]
