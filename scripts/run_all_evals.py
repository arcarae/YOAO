#!/usr/bin/env python3
"""Unified Evaluation Runner for RedBlackBench

Runs all evaluation tables in sequence or individually.

Usage:
    # Run all evaluations
    python scripts/run_all_evals.py --all

    # Run specific table
    python scripts/run_all_evals.py --table 2
    python scripts/run_all_evals.py --table 3

    # Run with custom output directory
    python scripts/run_all_evals.py --all --output-dir results/my_eval

    # Run with multiple games for statistical significance
    python scripts/run_all_evals.py --all --games 3

    # Resume interrupted evaluation
    python scripts/run_all_evals.py --all --resume

    # Dry run (show what would be executed)
    python scripts/run_all_evals.py --all --dry-run
"""

import argparse
import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_table2(output_dir: Path, games: int, resume: bool, dry_run: bool) -> bool:
    """Run Table 2: Robustness to Opponent Strategies."""
    print("\n" + "="*70)
    print("TABLE 2: Robustness to Opponent Strategies")
    print("="*70 + "\n")

    cmd = [
        sys.executable,
        "scripts/eval_robustness.py",
        "--output-dir", str(output_dir / "table2_robustness"),
        "--games-per-combo", str(games),
    ]
    if resume:
        cmd.append("--resume")

    print(f"Command: {' '.join(cmd)}\n")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_table3(output_dir: Path, games: int, resume: bool, trained_model: str, dry_run: bool) -> bool:
    """Run Table 3: Meta-Alignment with Mixed Teams."""
    print("\n" + "="*70)
    print("TABLE 3: Meta-Alignment with Mixed Teams")
    print("="*70 + "\n")

    cmd = [
        sys.executable,
        "scripts/eval_meta_alignment.py",
        "--output-dir", str(output_dir / "table3_meta_alignment"),
        "--games-per-combo", str(games),
        "--trained-model", trained_model,
    ]
    if resume:
        cmd.append("--resume")

    print(f"Command: {' '.join(cmd)}\n")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


def generate_combined_report(output_dir: Path):
    """Generate a combined markdown report from all evaluation results."""
    report_path = output_dir / "combined_report.md"

    with open(report_path, 'w') as f:
        f.write("# RedBlackBench Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        # Table 2
        table2_summary = output_dir / "table2_robustness" / "summary_table.md"
        if table2_summary.exists():
            f.write("## Table 2: Robustness to Opponent Strategies\n\n")
            with open(table2_summary, 'r') as t2:
                content = t2.read()
                # Skip the header if it exists
                if content.startswith("# "):
                    content = "\n".join(content.split("\n")[1:])
                f.write(content)
            f.write("\n\n---\n\n")
        else:
            f.write("## Table 2: Robustness to Opponent Strategies\n\n")
            f.write("*Not yet evaluated*\n\n---\n\n")

        # Table 3
        table3_summary = output_dir / "table3_meta_alignment" / "summary_table.md"
        if table3_summary.exists():
            f.write("## Table 3: Meta-Alignment with Mixed Teams\n\n")
            with open(table3_summary, 'r') as t3:
                content = t3.read()
                if content.startswith("# "):
                    content = "\n".join(content.split("\n")[1:])
                f.write(content)
            f.write("\n\n")
        else:
            f.write("## Table 3: Meta-Alignment with Mixed Teams\n\n")
            f.write("*Not yet evaluated*\n\n")

    print(f"\nCombined report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation runner for RedBlackBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all evaluations
    python scripts/run_all_evals.py --all

    # Run only Table 2 (robustness)
    python scripts/run_all_evals.py --table 2

    # Run only Table 3 (meta-alignment)
    python scripts/run_all_evals.py --table 3

    # Run with 3 games per combination for significance
    python scripts/run_all_evals.py --all --games 3

    # Resume interrupted run
    python scripts/run_all_evals.py --all --resume

    # Preview commands without running
    python scripts/run_all_evals.py --all --dry-run
        """
    )
    parser.add_argument("--all", action="store_true", help="Run all evaluation tables")
    parser.add_argument("--table", type=int, choices=[2, 3], help="Run specific table (2 or 3)")
    parser.add_argument("--output-dir", type=str, default="results/evaluations",
                        help="Base output directory")
    parser.add_argument("--games", type=int, default=1,
                        help="Games per combination (for statistical significance)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--trained-model", type=str, default="qwen/qwen3-14b",
                        help="Trained model for Table 3 (replace with your SFT model)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing")
    args = parser.parse_args()

    if not args.all and args.table is None:
        parser.print_help()
        print("\nError: Specify --all or --table N")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("REDBLACKBENCH EVALUATION SUITE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Games per combo: {args.games}")
    print(f"Resume mode: {args.resume}")
    print(f"Trained model: {args.trained_model}")
    print("="*70)

    success = True

    # Run Table 2
    if args.all or args.table == 2:
        if not run_table2(output_dir, args.games, args.resume, args.dry_run):
            print("\n❌ Table 2 evaluation failed!")
            success = False
        else:
            print("\n✓ Table 2 complete")

    # Run Table 3
    if args.all or args.table == 3:
        if not run_table3(output_dir, args.games, args.resume, args.trained_model, args.dry_run):
            print("\n❌ Table 3 evaluation failed!")
            success = False
        else:
            print("\n✓ Table 3 complete")

    # Generate combined report
    if not args.dry_run and (args.all or args.table):
        generate_combined_report(output_dir)

    print("\n" + "="*70)
    if success:
        print("EVALUATION COMPLETE")
    else:
        print("EVALUATION COMPLETED WITH ERRORS")
    print("="*70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
