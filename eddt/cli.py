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

Comparison Examples:
  eddt --compare baseline.yaml add_designer.yaml --days 5
  eddt --compare a.yaml b.yaml c.yaml --labels "A" "B" "C" --export output/
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

    # Bottleneck analysis arguments
    parser.add_argument(
        "--bottleneck",
        action="store_true",
        help="Enable bottleneck analysis after simulation",
    )
    parser.add_argument(
        "--util-threshold",
        type=float,
        default=0.85,
        help="Utilization threshold for bottleneck detection (default: 0.85)",
    )
    parser.add_argument(
        "--wait-threshold",
        type=float,
        default=2.0,
        help="Wait time threshold in hours for queue bottlenecks (default: 2.0)",
    )

    # What-if analysis arguments
    parser.add_argument(
        "--whatif",
        nargs="+",
        metavar="MOD",
        help="What-if analysis: specify modifications (e.g., '+1 senior_designer', '-50%% review')",
    )

    # Comparison mode arguments
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="CONFIG",
        help="Compare 2-5 scenario configs (enables comparison mode)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="Labels for comparison scenarios (must match number of configs)",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="DIR",
        help="Directory to export comparison results (CSV/JSON)",
    )

    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        return run_comparison_mode(args)

    # Handle what-if mode
    if args.whatif:
        return run_whatif_mode(args)

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

    # Bottleneck analysis
    if args.bottleneck:
        from .bottleneck import (
            BottleneckConfig,
            analyze_bottlenecks,
            export_bottleneck_report_csv,
            format_bottleneck_report,
        )

        print("\n" + "=" * 60)
        print("BOTTLENECK ANALYSIS")
        print("=" * 60)

        bottleneck_config = BottleneckConfig(
            utilization_threshold=args.util_threshold,
            wait_time_threshold_hours=args.wait_threshold,
        )
        report = analyze_bottlenecks(model, config=bottleneck_config)
        print(format_bottleneck_report(report))

        if args.export:
            export_dir = Path(args.export)
            export_dir.mkdir(parents=True, exist_ok=True)
            files = export_bottleneck_report_csv(report, str(export_dir))
            print(f"\nBottleneck report exported to {export_dir}/")

    # Save output
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == ".csv":
            results["model"].to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")

    return results


def run_whatif_mode(args):
    """Run EDDT in what-if analysis mode."""
    from .whatif import format_experiment_result, run_whatif_experiment

    print("=" * 60)
    print("EDDT: What-If Analysis Mode")
    print("=" * 60)

    if not args.config:
        print("\nError: --whatif requires --config to specify baseline scenario")
        return None

    try:
        experiment = run_whatif_experiment(
            baseline_config_path=args.config,
            modifications=args.whatif,
            days=args.days,
            random_seed=args.seed,
            verbose=not args.quiet,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        return None

    # Print formatted results
    if args.quiet:
        print("\n" + format_experiment_result(experiment))

    # Export if requested
    if args.export:
        output_dir = Path(args.export)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export comparison as JSON
        import json

        result_dict = {
            "baseline_config": args.config,
            "modifications": [
                {
                    "type": m.target_type,
                    "operation": m.operation,
                    "target": m.target,
                    "value": m.value,
                }
                for m in experiment.modifications
            ],
            "comparison": {
                "summary": experiment.comparison.summary,
                "improved": experiment.comparison.improved,
                "degraded": experiment.comparison.degraded,
                "unchanged": experiment.comparison.unchanged,
                "metrics": [
                    {
                        "name": m.name,
                        "baseline": m.baseline_value,
                        "modified": m.modified_value,
                        "delta": m.delta,
                        "delta_percent": m.delta_percent,
                        "direction": m.direction,
                    }
                    for m in experiment.comparison.metrics
                ],
            },
        }

        json_path = output_dir / "whatif_result.json"
        with open(json_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"\nResults exported to {output_dir}/")
        print(f"  - whatif_result.json")

    return experiment


def run_comparison_mode(args):
    """Run EDDT in comparison mode."""
    from .comparison import (
        compare_scenarios,
        export_comparison_csv,
        export_comparison_json,
        get_comparison_summary_table,
        validate_scenario_configs,
    )

    print("=" * 60)
    print("EDDT: Scenario Comparison Mode")
    print("=" * 60)

    # Validate configs first
    errors = validate_scenario_configs(args.compare)
    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  - {error}")
        return None

    # Run comparison
    try:
        result = compare_scenarios(
            config_paths=args.compare,
            labels=args.labels,
            days=args.days,
            random_seed=args.seed,
            verbose=not args.quiet,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        return None

    # Print summary table
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(get_comparison_summary_table(result))

    # Export if requested
    if args.export:
        output_dir = Path(args.export)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_files = export_comparison_csv(result, str(output_dir))
        json_file = export_comparison_json(result, str(output_dir / "comparison.json"))

        print(f"\nResults exported to {output_dir}/")
        for f in csv_files + [json_file]:
            print(f"  - {Path(f).name}")

    return result


if __name__ == "__main__":
    main()
