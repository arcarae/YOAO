#!/usr/bin/env python3
"""Master Orchestrator: Run All Remaining Experiments

Dual-pool concurrent execution:
  - vLLM pool: Scale Effect baselines (N=5/10/15) + N=5 scenario gaps
  - OpenRouter pool: Heterogeneous Llama/Mistral baselines + Mistral gaps

Features:
  - Auto-dedup: scans existing results, skips completed games
  - Per-game logging + master log + real-time summary.json
  - vLLM health checks with auto-pause/resume
  - Consecutive failure detection with alerts
  - Crash-safe: restart script to continue from where it left off

Usage:
    python3.10 scripts/run_remaining_all.py \
        --vllm-url http://localhost:8000/v1 \
        --openrouter-key $OPENROUTER_API_KEY \
        --log-dir logs/remaining_$(date +%%Y%%m%%d)

    # Dry run (show plan, don't execute)
    python3.10 scripts/run_remaining_all.py --dry-run

    # vLLM-only (skip OpenRouter experiments)
    python3.10 scripts/run_remaining_all.py --vllm-only
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from redblackbench.game.coordinator import GameCoordinator
from redblackbench.game.config import GameConfig
from redblackbench.game.scoring import Choice
from redblackbench.teams.team import Team
from redblackbench.scenarios import get_scenario
from redblackbench.trajectory import TrajectoryCollector
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.cli import create_provider
from redblackbench.strategies import create_scripted_team


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

GAME_CONFIG = GameConfig(num_rounds=10, multipliers={5: 3, 8: 5, 10: 10})
OPPONENT_STRATEGY = "always_defect"
RUNS_PER_CONDITION = 5

SEED_PERCENTAGES = [0, 20, 40, 60, 80, 100]
SEED_PERCENTAGES_HETERO = [0, 20, 40, 60, 80]  # Skip 100% for heterogeneous

EVAL_SCENARIOS = ["baseline", "trade_war", "gpu_contention"]

AGENT_NAMES = [
    "Dr. Sarah Chen", "Marcus Webb", "Dr. Priya Sharma",
    "James O'Connor", "Dr. Elena Vasquez", "Dr. Amir Hassan",
    "Lisa Park", "Dr. Thomas Müller", "Fatima Al-Rashid",
    "Dr. Kenji Tanaka", "Rachel Torres", "Dr. Oluwaseun Adeyemi",
    "Viktor Petrov", "Dr. Maya Gupta", "Samuel Okafor",
]

# Models (override with YOAO_SFT_MODEL / YOAO_BASE_MODEL)
from redblackbench import defaults

TRAINED_MODEL = defaults.SFT_MODEL
UNTRAINED_QWEN_BASE = defaults.BASE_MODEL
UNTRAINED_LLAMA = "meta-llama/llama-3.1-8b-instruct"
UNTRAINED_MISTRAL = "mistralai/mistral-small-3.1-24b-instruct"

# Concurrency defaults
DEFAULT_VLLM_CONCURRENCY = 6
DEFAULT_OPENROUTER_CONCURRENCY = 2

# Health check / failure thresholds
HEALTH_CHECK_INTERVAL = 60  # seconds
MAX_CONSECUTIVE_FAILURES = 3
FAILURE_PAUSE_SECONDS = 30


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Job:
    """A single game to run."""
    job_id: str
    experiment: str          # "scale_n5_baseline", "hetero_llama_baseline", etc.
    pool: str                # "vllm" or "openrouter"
    team_size: int
    seed_pct: int
    scenario: str
    run_index: int
    output_dir: str
    trained_model: str
    untrained_model: str
    untrained_provider_type: str  # "vllm" or "openrouter"
    num_trained: int = 0
    num_untrained: int = 0

    def __post_init__(self):
        self.num_trained = round(self.team_size * self.seed_pct / 100)
        self.num_untrained = self.team_size - self.num_trained

    @property
    def dedup_key(self) -> Tuple:
        return (self.team_size, self.seed_pct, self.scenario, self.run_index)


@dataclass
class RunStats:
    """Live statistics."""
    total_jobs: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    in_progress: int = 0
    consecutive_failures: int = 0
    start_time: float = 0.0
    pool_stats: Dict[str, Dict] = field(default_factory=lambda: {
        "vllm": {"active": 0, "completed": 0, "failed": 0, "paused": False},
        "openrouter": {"active": 0, "completed": 0, "failed": 0, "paused": False},
    })
    failed_jobs: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up master logger + per-game log directory."""
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "games").mkdir(exist_ok=True)

    logger = logging.getLogger("master")
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # File handler (DEBUG)
    fh = logging.FileHandler(log_dir / "master.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    return logger


def get_game_logger(log_dir: Path, job_id: str) -> logging.Logger:
    """Create a per-game logger."""
    game_logger = logging.getLogger(f"game.{job_id}")
    game_logger.setLevel(logging.DEBUG)
    # Clear existing handlers to avoid duplicates on retry
    game_logger.handlers.clear()

    fh = logging.FileHandler(log_dir / "games" / f"{job_id}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    game_logger.addHandler(fh)
    return game_logger


# ═══════════════════════════════════════════════════════════════════════
# Summary / progress persistence
# ═══════════════════════════════════════════════════════════════════════

def save_summary(log_dir: Path, stats: RunStats, jobs: List[Job]):
    """Write real-time summary.json for monitoring script."""
    elapsed = time.time() - stats.start_time if stats.start_time else 0
    remaining = stats.total_jobs - stats.completed - stats.failed - stats.skipped

    if stats.completed > 0 and elapsed > 0:
        avg_time = elapsed / stats.completed
        eta_seconds = remaining * avg_time
    else:
        eta_seconds = -1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed),
        "eta_seconds": round(eta_seconds) if eta_seconds > 0 else None,
        "total": stats.total_jobs,
        "completed": stats.completed,
        "failed": stats.failed,
        "skipped": stats.skipped,
        "remaining": remaining,
        "in_progress": stats.in_progress,
        "progress_pct": round((stats.completed + stats.skipped) / max(stats.total_jobs, 1) * 100, 1),
        "pools": stats.pool_stats,
        "consecutive_failures": stats.consecutive_failures,
        "recent_failures": stats.failed_jobs[-5:] if stats.failed_jobs else [],
    }

    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# Result scanning & dedup
# ═══════════════════════════════════════════════════════════════════════

def scan_existing_results(results_base: Path, experiment_dirs: Dict[str, str]) -> Dict[str, Set[Tuple]]:
    """Scan result directories and return completed (team_size, seed_pct, scenario, run_index) sets per experiment."""
    completed = defaultdict(set)

    for exp_name, dir_name in experiment_dirs.items():
        result_dir = results_base / dir_name
        progress_file = result_dir / "progress.json"

        if progress_file.exists():
            try:
                with open(progress_file) as f:
                    data = json.load(f)
                for game in data.get("games", []):
                    key = (game["team_size"], game["seed_pct"], game["scenario"], game["run_index"])
                    # Validate it's not placeholder data
                    if game.get("cooperation_rate", 0) > 0 or game.get("team_a_score", 0) != 0:
                        completed[exp_name].add(key)
                    elif game.get("cooperation_rate", 0) == 0 and game.get("team_a_score", 0) == 0:
                        # Could be legitimate 0% cooperation, check if trajectory exists
                        traj = game.get("trajectory_path", "")
                        if traj and Path(traj).exists() and Path(traj).stat().st_size > 100:
                            completed[exp_name].add(key)
            except Exception as e:
                print(f"  Warning: Could not read {progress_file}: {e}")

    return completed


# ═══════════════════════════════════════════════════════════════════════
# Job generation
# ═══════════════════════════════════════════════════════════════════════

def generate_jobs(
    results_base: Path,
    vllm_url: str,
) -> Tuple[List[Job], Dict[str, str]]:
    """Generate all remaining jobs, skipping already-completed ones."""

    # ── Output directories (new dirs for baseline re-runs) ──
    experiment_dirs = {
        # P1: Baseline re-runs (format_history bug fix) → new directories
        "scale_n5_baseline":     "eval_scale_n5_baseline_fixed",
        "scale_n10_baseline":    "eval_scale_n10_baseline_fixed",
        "scale_n15_baseline":    "eval_scale_n15_baseline_fixed",
        "hetero_llama_baseline": "eval_hetero_llama_baseline_fixed",
        "hetero_mistral_baseline": "eval_hetero_mistral_baseline_fixed",

        # P2: Missing scenario data (append to existing or new)
        "scale_n5_scenarios":    "eval_scale_n5_scenarios_backfill",
        "hetero_mistral_gaps":   "eval_hetero_mistral_gaps",
    }

    # Scan existing results
    completed_by_exp = scan_existing_results(results_base, experiment_dirs)

    jobs = []

    def add_job(exp_name, pool, team_size, seed_pct, scenario, run_index,
                trained_model, untrained_model, untrained_provider_type):
        key = (team_size, seed_pct, scenario, run_index)
        if key in completed_by_exp.get(exp_name, set()):
            return False  # Skip

        job_id = f"{exp_name}_s{seed_pct}_r{run_index}_{scenario}"
        jobs.append(Job(
            job_id=job_id,
            experiment=exp_name,
            pool=pool,
            team_size=team_size,
            seed_pct=seed_pct,
            scenario=scenario,
            run_index=run_index,
            output_dir=str(results_base / experiment_dirs[exp_name]),
            trained_model=trained_model,
            untrained_model=untrained_model,
            untrained_provider_type=untrained_provider_type,
        ))
        return True

    # ── P1: Scale Effect Baselines (vLLM only, baseline scenario only) ──
    for N, exp_name in [(5, "scale_n5_baseline"), (10, "scale_n10_baseline"), (15, "scale_n15_baseline")]:
        for pct in SEED_PERCENTAGES:
            for run in range(RUNS_PER_CONDITION):
                add_job(exp_name, "vllm", N, pct, "baseline", run,
                        TRAINED_MODEL, UNTRAINED_QWEN_BASE, "vllm")

    # ── P1: Heterogeneous Baselines (vLLM + OpenRouter, baseline only) ──
    # Llama
    for pct in SEED_PERCENTAGES_HETERO:
        for run in range(RUNS_PER_CONDITION):
            add_job("hetero_llama_baseline", "openrouter", 5, pct, "baseline", run,
                    TRAINED_MODEL, UNTRAINED_LLAMA, "openrouter")

    # Mistral
    for pct in SEED_PERCENTAGES_HETERO:
        for run in range(RUNS_PER_CONDITION):
            add_job("hetero_mistral_baseline", "openrouter", 5, pct, "baseline", run,
                    TRAINED_MODEL, UNTRAINED_MISTRAL, "openrouter")

    # ── P2: Scale N=5 scenario gaps ──
    # Missing from status: baseline s20 r1, baseline s100 r1-2, trade_war s40 r1
    # But baselines are being re-run above, so only scenario gaps:
    n5_scenario_gaps = [
        ("trade_war", 40, 1),
    ]
    for scenario, pct, run in n5_scenario_gaps:
        add_job("scale_n5_scenarios", "vllm", 5, pct, scenario, run,
                TRAINED_MODEL, UNTRAINED_QWEN_BASE, "vllm")

    # ── P2: Heterogeneous Mistral non-baseline gaps ──
    # From EXPERIMENT_STATUS.md: seed=40,60,80 have gaps across trade_war & gpu_contention
    mistral_gaps = {
        # (scenario, seed_pct): list of missing run_indices
        # trade_war: seed=40 (4/5 missing → runs 1,2,3,4), seed=60 (5/5), seed=80 (5/5)
        ("trade_war", 40): [1, 2, 3, 4],
        ("trade_war", 60): [0, 1, 2, 3, 4],
        ("trade_war", 80): [0, 1, 2, 3, 4],
        # gpu_contention: seed=40 (1/5 missing → run TBD), seed=60 (5/5), seed=80 (2/5 missing)
        ("gpu_contention", 40): [1],  # Only 1 missing
        ("gpu_contention", 60): [0, 1, 2, 3, 4],
        ("gpu_contention", 80): [0, 1],  # 2 missing (3 exist)
    }
    for (scenario, pct), runs in mistral_gaps.items():
        for run in runs:
            add_job("hetero_mistral_gaps", "openrouter", 5, pct, scenario, run,
                    TRAINED_MODEL, UNTRAINED_MISTRAL, "openrouter")

    return jobs, experiment_dirs


# ═══════════════════════════════════════════════════════════════════════
# Game execution
# ═══════════════════════════════════════════════════════════════════════

async def run_single_game(
    job: Job,
    vllm_url: str,
    openrouter_key: Optional[str],
    game_logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Run a single game and return result dict."""

    game_logger.info(f"Starting: {job.job_id}")
    game_logger.info(f"  team_size={job.team_size}, seed_pct={job.seed_pct}%, "
                     f"scenario={job.scenario}, run={job.run_index}")
    game_logger.info(f"  trained={job.trained_model}, untrained={job.untrained_model} ({job.untrained_provider_type})")

    start_time = time.time()

    try:
        # Scenario setup
        if job.scenario == "baseline":
            prompt_template = None
        else:
            scenario = get_scenario(job.scenario)
            if not scenario:
                game_logger.error(f"Unknown scenario: {job.scenario}")
                return None
            prompt_template = scenario.to_prompt_template()

        # Create Team A agents
        agents_a = []

        # Trained agents (SFT via vLLM)
        for i in range(job.num_trained):
            provider = create_provider({
                "type": "vllm",
                "model": job.trained_model,
                "base_url": vllm_url,
                "temperature": 0.7,
                "max_tokens": 512,
            })
            agent = LLMAgent(
                agent_id=AGENT_NAMES[i % len(AGENT_NAMES)],
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
                enable_thinking=False,
            )
            agents_a.append(agent)

        # Untrained agents
        for i in range(job.num_untrained):
            idx = job.num_trained + i
            if job.untrained_provider_type == "vllm":
                provider = create_provider({
                    "type": "vllm",
                    "model": job.untrained_model,
                    "base_url": vllm_url,
                    "temperature": 0.7,
                    "max_tokens": 512,
                })
            else:
                provider = create_provider({
                    "type": "openrouter",
                    "model": job.untrained_model,
                    "api_key": openrouter_key,
                    "temperature": 0.7,
                    "include_reasoning": False,
                })
            agent = LLMAgent(
                agent_id=AGENT_NAMES[idx % len(AGENT_NAMES)],
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
                enable_thinking=False,
            )
            agents_a.append(agent)

        team_a = Team(name="Team A", agents=agents_a)
        team_b = create_scripted_team(
            strategy_id=OPPONENT_STRATEGY,
            team_name=f"Team B ({OPPONENT_STRATEGY})",
        )

        # Trajectory
        output_dir = Path(job.output_dir)
        traj_dir = output_dir / "trajectories" / f"N{job.team_size}" / job.scenario
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"seed{job.seed_pct}pct_run{job.run_index}.json"

        collector = TrajectoryCollector()
        team_a_desc = f"mixed:{job.num_trained}xSFT+{job.num_untrained}x{job.untrained_model.split('/')[-1]}"

        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=GAME_CONFIG,
            trajectory_collector=collector,
            trajectory_save_path=str(traj_path),
            team_a_model=team_a_desc,
            team_b_model=f"scripted:{OPPONENT_STRATEGY}",
        )

        # Run with dynamic timeout (longer for large teams and Mistral)
        timeout = 3600 + (job.team_size - 5) * 900
        if "mistral" in job.untrained_model.lower():
            timeout = max(timeout, 7200)
        game_logger.info(f"  Running game (timeout={timeout}s)...")
        game_state = await asyncio.wait_for(coordinator.play_game(), timeout=timeout)

        # Extract results
        total_rounds = len(game_state.history)
        black_count = sum(1 for r in game_state.history if r.team_a_choice == Choice.BLACK)
        cooperation_rate = black_count / total_rounds if total_rounds > 0 else 0.0

        elapsed = time.time() - start_time
        game_logger.info(f"  Completed in {elapsed:.1f}s: coop={cooperation_rate*100:.0f}%, "
                         f"score={game_state.team_a_total} vs {game_state.team_b_total}")

        return {
            "team_size": job.team_size,
            "seed_pct": job.seed_pct,
            "num_trained": job.num_trained,
            "num_untrained": job.num_untrained,
            "scenario": job.scenario,
            "run_index": job.run_index,
            "cooperation_rate": cooperation_rate,
            "total_rounds": total_rounds,
            "team_a_score": game_state.team_a_total,
            "team_b_score": game_state.team_b_total,
            "trained_model": job.trained_model,
            "untrained_model": job.untrained_model,
            "experiment": job.experiment,
            "trajectory_path": str(traj_path),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        game_logger.error(f"  TIMEOUT after {elapsed:.0f}s")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        game_logger.error(f"  ERROR after {elapsed:.0f}s: {e}")
        game_logger.error(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════════════
# Progress persistence per experiment
# ═══════════════════════════════════════════════════════════════════════

def save_experiment_progress(output_dir: str, result: Dict[str, Any]):
    """Append a game result to the experiment's progress.json."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_file = output_path / "progress.json"

    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
    else:
        data = {"games": [], "config": {}}

    # Dedup: replace existing entry with same key, or append
    key = (result["team_size"], result["seed_pct"], result["scenario"], result["run_index"])
    deduped = [g for g in data["games"]
               if (g["team_size"], g["seed_pct"], g["scenario"], g["run_index"]) != key]
    deduped.append(result)
    data["games"] = deduped
    data["last_updated"] = datetime.now().isoformat()

    with open(progress_file, "w") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# vLLM health check
# ═══════════════════════════════════════════════════════════════════════

async def check_vllm_health(vllm_url: str, logger: logging.Logger) -> bool:
    """Check if vLLM server is responsive using curl (more reliable than httpx async)."""
    try:
        import subprocess
        base = vllm_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        models_url = f"{base}/v1/models"

        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "60", "--max-time", "90", models_url],
            capture_output=True, text=True, timeout=95,
        )
        if result.returncode == 0 and '"object"' in result.stdout:
            return True
        else:
            logger.warning(f"vLLM health check: curl exit={result.returncode}")
            return False
    except Exception as e:
        logger.warning(f"vLLM health check failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# Pool worker
# ═══════════════════════════════════════════════════════════════════════

async def pool_worker(
    pool_name: str,
    jobs: List[Job],
    semaphore: asyncio.Semaphore,
    vllm_url: str,
    openrouter_key: Optional[str],
    stats: RunStats,
    log_dir: Path,
    logger: logging.Logger,
):
    """Process jobs in a pool with concurrency control."""

    async def run_job(job: Job):
        async with semaphore:
            stats.in_progress += 1
            stats.pool_stats[pool_name]["active"] += 1

            game_logger = get_game_logger(log_dir, job.job_id)
            logger.info(f"[{pool_name}] START {job.job_id} "
                        f"(N={job.team_size} s{job.seed_pct}% {job.scenario} r{job.run_index})")

            try:
                result = await run_single_game(job, vllm_url, openrouter_key, game_logger)

                if result:
                    stats.completed += 1
                    stats.pool_stats[pool_name]["completed"] += 1
                    stats.consecutive_failures = 0

                    save_experiment_progress(job.output_dir, result)

                    coop = result["cooperation_rate"] * 100
                    elapsed = result["elapsed_seconds"]
                    logger.info(f"[{pool_name}] DONE  {job.job_id} → coop={coop:.0f}% ({elapsed:.0f}s) "
                                f"[{stats.completed}/{stats.total_jobs - stats.skipped}]")
                else:
                    stats.failed += 1
                    stats.pool_stats[pool_name]["failed"] += 1
                    stats.consecutive_failures += 1
                    stats.failed_jobs.append({
                        "job_id": job.job_id,
                        "experiment": job.experiment,
                        "timestamp": datetime.now().isoformat(),
                    })
                    logger.warning(f"[{pool_name}] FAIL  {job.job_id} "
                                   f"(consecutive: {stats.consecutive_failures})")

                    # Pause on consecutive failures
                    if stats.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error(
                            f"⚠️  {stats.consecutive_failures} consecutive failures in {pool_name}! "
                            f"Pausing {FAILURE_PAUSE_SECONDS}s. Run: claude 来诊断问题"
                        )
                        stats.pool_stats[pool_name]["paused"] = True
                        await asyncio.sleep(FAILURE_PAUSE_SECONDS)

                        # For vLLM pool, check health before resuming
                        if pool_name == "vllm":
                            healthy = await check_vllm_health(vllm_url, logger)
                            while not healthy:
                                logger.error("vLLM still unhealthy, waiting 30s...")
                                await asyncio.sleep(30)
                                healthy = await check_vllm_health(vllm_url, logger)
                            logger.info("vLLM recovered, resuming")

                        stats.pool_stats[pool_name]["paused"] = False
                        stats.consecutive_failures = 0

            finally:
                stats.in_progress -= 1
                stats.pool_stats[pool_name]["active"] -= 1
                save_summary(log_dir, stats, jobs)

    # Launch all jobs concurrently (semaphore controls actual parallelism)
    tasks = [asyncio.create_task(run_job(job)) for job in jobs]
    await asyncio.gather(*tasks, return_exceptions=True)


# ═══════════════════════════════════════════════════════════════════════
# Health check background task
# ═══════════════════════════════════════════════════════════════════════

async def health_check_loop(vllm_url: str, log_dir: Path, logger: logging.Logger, stop_event: asyncio.Event):
    """Periodically log vLLM health status."""
    health_log = log_dir / "vllm_health.log"
    while not stop_event.is_set():
        healthy = await check_vllm_health(vllm_url, logger)
        status = "OK" if healthy else "FAIL"
        with open(health_log, "a") as f:
            f.write(f"{datetime.now().isoformat()} {status}\n")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=HEALTH_CHECK_INTERVAL)
        except asyncio.TimeoutError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Retry failed jobs
# ═══════════════════════════════════════════════════════════════════════

async def retry_failed_jobs(
    failed_job_ids: Set[str],
    all_jobs: List[Job],
    vllm_url: str,
    openrouter_key: Optional[str],
    stats: RunStats,
    log_dir: Path,
    logger: logging.Logger,
):
    """Retry previously failed jobs one at a time."""
    retry_jobs = [j for j in all_jobs if j.job_id in failed_job_ids]
    if not retry_jobs:
        return

    logger.info(f"\n{'='*70}")
    logger.info(f"RETRY PHASE: {len(retry_jobs)} failed jobs")
    logger.info(f"{'='*70}")

    for job in retry_jobs:
        game_logger = get_game_logger(log_dir, f"{job.job_id}_retry")
        logger.info(f"[retry] {job.job_id}")

        result = await run_single_game(job, vllm_url, openrouter_key, game_logger)
        if result:
            stats.completed += 1
            stats.failed -= 1
            save_experiment_progress(job.output_dir, result)
            logger.info(f"[retry] DONE {job.job_id} → coop={result['cooperation_rate']*100:.0f}%")
        else:
            logger.warning(f"[retry] STILL FAILING {job.job_id}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def async_main(args):
    results_base = PROJECT_ROOT / "results"
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)

    logger.info("=" * 70)
    logger.info("REDBLACKBENCH — REMAINING EXPERIMENTS ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info(f"vLLM URL:              {args.vllm_url}")
    logger.info(f"OpenRouter key:        {'SET' if args.openrouter_key else 'NOT SET'}")
    logger.info(f"vLLM concurrency:      {args.vllm_concurrency}")
    logger.info(f"OpenRouter concurrency: {args.openrouter_concurrency}")
    logger.info(f"Log dir:               {log_dir}")

    # ── Phase 0: Pre-checks ──
    logger.info("\n--- Phase 0: Pre-checks ---")

    # Check vLLM
    vllm_ok = await check_vllm_health(args.vllm_url, logger)
    if vllm_ok:
        logger.info("✓ vLLM server is healthy")
    else:
        logger.warning("✗ vLLM health check failed (may be under load)")
        logger.warning("  Proceeding anyway — games have their own timeouts")

    # Check OpenRouter key
    openrouter_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        logger.info("✓ OpenRouter API key available")
    else:
        logger.warning("✗ OpenRouter API key not set")
        if not args.vllm_only:
            logger.warning("  Cannot run heterogeneous experiments. Use --vllm-only to skip them.")

    # ── Generate jobs ──
    logger.info("\n--- Generating job list ---")
    all_jobs, experiment_dirs = generate_jobs(results_base, args.vllm_url)

    vllm_jobs = sorted([j for j in all_jobs if j.pool == "vllm"],
                       key=lambda j: -j.team_size)  # N=15 first, then N=10, then N=5
    openrouter_jobs = [j for j in all_jobs if j.pool == "openrouter"]

    if args.vllm_only:
        openrouter_jobs = []
    if args.openrouter_only:
        vllm_jobs = []

    active_jobs = vllm_jobs + openrouter_jobs

    # ── Print plan ──
    logger.info(f"\nJob Summary:")
    logger.info(f"  vLLM pool:       {len(vllm_jobs)} games")
    logger.info(f"  OpenRouter pool: {len(openrouter_jobs)} games")
    logger.info(f"  Total:           {len(active_jobs)} games")

    # Group by experiment for display
    by_exp = defaultdict(int)
    for j in active_jobs:
        by_exp[j.experiment] += 1
    logger.info("\n  Breakdown:")
    for exp, count in sorted(by_exp.items()):
        logger.info(f"    {exp}: {count} games")

    if args.dry_run:
        logger.info("\n--- DRY RUN — not executing ---")
        logger.info("\nAll jobs:")
        for j in active_jobs:
            logger.info(f"  {j.job_id} | pool={j.pool} | N={j.team_size} "
                        f"s{j.seed_pct}% {j.scenario} r{j.run_index}")
        return

    if not active_jobs:
        logger.info("\n✓ All experiments already completed! Nothing to do.")
        return

    # ── Phase 1: Execute ──
    logger.info(f"\n--- Phase 1: Executing {len(active_jobs)} games ---")

    stats = RunStats(
        total_jobs=len(active_jobs),
        start_time=time.time(),
    )
    save_summary(log_dir, stats, active_jobs)

    # Start health check loop
    stop_health = asyncio.Event()
    health_task = asyncio.create_task(
        health_check_loop(args.vllm_url, log_dir, logger, stop_health)
    )

    # Create semaphores for each pool
    vllm_sem = asyncio.Semaphore(args.vllm_concurrency)
    openrouter_sem = asyncio.Semaphore(args.openrouter_concurrency)

    # Run both pools concurrently
    pool_tasks = []
    if vllm_jobs:
        pool_tasks.append(
            pool_worker("vllm", vllm_jobs, vllm_sem, args.vllm_url,
                        openrouter_key, stats, log_dir, logger)
        )
    if openrouter_jobs:
        pool_tasks.append(
            pool_worker("openrouter", openrouter_jobs, openrouter_sem, args.vllm_url,
                        openrouter_key, stats, log_dir, logger)
        )

    await asyncio.gather(*pool_tasks, return_exceptions=True)

    # ── Phase 2: Retry failures ──
    if stats.failed_jobs:
        failed_ids = {f["job_id"] for f in stats.failed_jobs}
        await retry_failed_jobs(
            failed_ids, active_jobs, args.vllm_url, openrouter_key,
            stats, log_dir, logger,
        )

    # Stop health check
    stop_health.set()
    await health_task

    # ── Phase 3: Final report ──
    elapsed = time.time() - stats.start_time
    logger.info(f"\n{'='*70}")
    logger.info("FINAL REPORT")
    logger.info(f"{'='*70}")
    logger.info(f"Total time:  {elapsed/60:.1f} minutes")
    logger.info(f"Completed:   {stats.completed}")
    logger.info(f"Failed:      {stats.failed}")
    logger.info(f"Skipped:     {stats.skipped}")

    if stats.failed_jobs:
        logger.warning("\nStill-failing jobs:")
        for fj in stats.failed_jobs:
            logger.warning(f"  {fj['job_id']}")
        # Save failed jobs list
        with open(log_dir / "failed_jobs.json", "w") as f:
            json.dump(stats.failed_jobs, f, indent=2)
        logger.warning(f"\nSaved to {log_dir / 'failed_jobs.json'}")
        logger.warning("Run: claude 来诊断和修复失败的实验")

    # Save final summary
    save_summary(log_dir, stats, active_jobs)

    # Verify completeness
    logger.info("\n--- Completeness Check ---")
    _, _ = generate_jobs(results_base, args.vllm_url)
    remaining_jobs, _ = generate_jobs(results_base, args.vllm_url)
    if remaining_jobs:
        logger.warning(f"Still {len(remaining_jobs)} incomplete games. Re-run script to continue.")
    else:
        logger.info("✓ All experiments complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run all remaining RedBlackBench experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--openrouter-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--vllm-concurrency", type=int, default=DEFAULT_VLLM_CONCURRENCY,
                        help=f"Max concurrent games in vLLM pool (default: {DEFAULT_VLLM_CONCURRENCY})")
    parser.add_argument("--openrouter-concurrency", type=int, default=DEFAULT_OPENROUTER_CONCURRENCY,
                        help=f"Max concurrent games in OpenRouter pool (default: {DEFAULT_OPENROUTER_CONCURRENCY})")
    parser.add_argument("--log-dir", type=str,
                        default=f"logs/remaining_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Log output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    parser.add_argument("--vllm-only", action="store_true",
                        help="Only run vLLM experiments (skip OpenRouter)")
    parser.add_argument("--openrouter-only", action="store_true",
                        help="Only run OpenRouter experiments (skip vLLM)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
