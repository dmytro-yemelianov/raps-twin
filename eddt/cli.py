"""
Command-line interface for EDDT simulation.
"""

import argparse
from pathlib import Path

from .model import EngineeringDepartment


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="EDDT: Engineering Department Digital Twin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eddt --days 5                    Run 5-day simulation with defaults
  eddt --config scenario.yaml      Run with custom config
  eddt --days 10 --seed 123        Run with specific random seed
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=5,
        help="Number of days to simulate (default: 5)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for results (CSV)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for complex decisions (requires Ollama)",
    )

    args = parser.parse_args()

    # Build config overrides
    config_overrides = None
    if args.use_llm:
        config_overrides = {"llm": {"use_llm": True}}

    print("=" * 60)
    print("EDDT: Engineering Department Digital Twin")
    print("=" * 60)

    # Create model
    print("\nInitializing simulation...")

    if args.config:
        model = EngineeringDepartment(
            config_path=args.config,
            random_seed=args.seed,
        )
    else:
        config = EngineeringDepartment(random_seed=args.seed)._default_config()
        if config_overrides:
            for key, value in config_overrides.items():
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
        model = EngineeringDepartment(config=config, random_seed=args.seed)

    # Run simulation
    print(f"\nRunning simulation ({args.days} days)...")
    results = model.run(days=args.days, verbose=not args.quiet)

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
    llm_stats = results["llm_stats"]
    print(f"\nLLM Stats:")
    print(f"  Cache hit rate: {llm_stats['cache_hit_rate']:.1%}")
    print(f"  Rule-based calls: {llm_stats['rule_calls']}")
    if llm_stats["use_llm"]:
        print(f"  LLM tier2 calls: {llm_stats['tier2_calls']}")

    # Bottlenecks
    bottlenecks = results["bottlenecks"]
    if bottlenecks:
        print("\nBottlenecks Identified:")
        for b in bottlenecks[:5]:
            print(f"  - [{b.get('severity', 'medium')}] {b['description']}")

    # Save output
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == ".csv":
            results["model"].to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
