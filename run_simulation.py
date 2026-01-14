#!/usr/bin/env python3
"""
run_simulation.py - Quick way to run EDDT simulations.
"""

from eddt.model import EngineeringDepartment


def main():
    print("=" * 60)
    print("EDDT: Engineering Department Digital Twin")
    print("=" * 60)

    # Create model with default config
    print("\nInitializing simulation...")
    model = EngineeringDepartment()

    # Run for 5 simulated days
    print("\nRunning simulation (5 days)...")
    results = model.run(days=5)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    summary = results["summary"]
    print(f"  Simulated days: {summary['simulated_days']:.1f}")
    print(f"  Total ticks: {summary['total_ticks']}")
    print(f"  Tasks completed: {summary['tasks_completed']} / {summary['tasks_total']}")
    print(f"  Completion rate: {summary['completion_rate']:.1%}")

    # Agent utilization
    print("\nAgent Utilization:")
    for agent in model.agents:
        status_icon = {
            "working": "[W]",
            "idle": "[I]",
            "blocked": "[B]",
        }.get(agent.status.value, "[?]")
        print(f"  {status_icon} {agent.name}: {agent.utilization:.1%}")

    # LLM stats
    print("\nLLM Stats:")
    llm_stats = results["llm_stats"]
    print(f"  Cache hit rate: {llm_stats['cache_hit_rate']:.1%}")
    print(f"  Rule-based calls: {llm_stats['rule_calls']}")

    # Bottlenecks
    bottlenecks = results["bottlenecks"]
    if bottlenecks:
        print("\nBottlenecks Identified:")
        for b in bottlenecks[:5]:
            print(f"  - [{b.get('severity', 'medium')}] {b['description']}")

    # Try to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        if len(results["model"]) > 0:
            df = results["model"]

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))

            # Utilization over time
            axes[0].plot(df["tick"], df["avg_utilization"], label="Avg Utilization")
            axes[0].set_xlabel("Tick")
            axes[0].set_ylabel("Utilization")
            axes[0].set_title("Team Utilization Over Time")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Tasks over time
            axes[1].plot(df["tick"], df["tasks_completed"], label="Completed")
            axes[1].plot(df["tick"], df["tasks_in_progress"], label="In Progress")
            axes[1].set_xlabel("Tick")
            axes[1].set_ylabel("Tasks")
            axes[1].set_title("Task Progress Over Time")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("simulation_results.png", dpi=150)
            print("\nSaved plot to simulation_results.png")
    except ImportError:
        print("\n(Install matplotlib for visualization: pip install matplotlib)")
    except Exception as e:
        print(f"\nCould not create plot: {e}")

    return results


if __name__ == "__main__":
    main()
