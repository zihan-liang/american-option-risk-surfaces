"""Non-evidence HD3 throughput smoke used to choose sequential vs concurrent execution."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hd_pipeline.runtime import atomic_json
import hd3_classical_vs_structured_high_dimensional_pinn as hd3
from hd2_hvi_objective_stabilization import curriculum


def parse_compute_memory_gib(text: str, pids: set[int]) -> float:
    """Parse ``nvidia-smi --query-compute-apps=pid,used_memory`` output."""
    total_mib = 0.0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid not in pids:
            continue
        memory_token = parts[1].lower().replace("mib", "").strip()
        try:
            total_mib += float(memory_token)
        except ValueError:
            continue
    return total_mib / 1024.0


def _interpolate(value10: float, value50: float, d: int) -> float:
    if d <= 10:
        return float(value10)
    if d >= 50:
        return float(value50)
    alpha = (float(d) - 10.0) / 40.0
    return float(value10) + alpha * (float(value50) - float(value10))


def project_canonical_seconds(
    benchmark: dict[str, Any],
    *,
    canonical_steps: int,
    parallel_enabled: bool,
) -> float:
    """Project 10/20/50D canonical training from measured 200-step smokes."""
    smoke_steps = int(benchmark["smoke_steps"])
    if smoke_steps <= 0:
        raise ValueError("smoke_steps must be positive")
    sequential = benchmark["sequential"]
    fc10 = float(sequential["d10_FC_HVI"]["wall_seconds"])
    ds10 = float(sequential["d10_DEEPSETS_HVI"]["wall_seconds"])
    fc50 = float(sequential["d50_FC_HVI"]["wall_seconds"])
    ds50 = float(sequential["d50_DEEPSETS_HVI"]["wall_seconds"])
    fc20 = _interpolate(fc10, fc50, 20)
    ds20 = _interpolate(ds10, ds50, 20)
    if parallel_enabled:
        concurrent10 = float(benchmark["concurrent_d10"]["wall_seconds"])
        pair10_sum = max(fc10 + ds10, 1e-12)
        parallel_ratio = concurrent10 / pair10_sum
        smoke_total = concurrent10 + (fc20 + ds20) * parallel_ratio + fc50 + ds50
    else:
        smoke_total = fc10 + ds10 + fc20 + ds20 + fc50 + ds50
    return smoke_total * (float(canonical_steps) / float(smoke_steps))


def run_smoke(
    model_id: str,
    *,
    d: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run a fresh, disposable 200-step training trace and return timing/memory only."""
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:HD3 parallelism benchmark")
    cfg = hd3.load_config()
    if model_id not in cfg["models"]:
        raise ValueError("unknown HD3 model")
    if d not in (10, 50):
        raise ValueError("parallelism smoke dimensions are frozen to d=10 and d=50")
    if steps != int(cfg["parallelism_smoke_steps"]):
        raise ValueError("parallelism smoke step count is frozen")
    objective, _ = hd3.resolved_objective()
    device = torch.device("cuda:0")
    oracle = hd3.load_oracle(d)
    training_points = min(int(cfg["supervised_train_points"]), 4096)
    x_train, tau_train, y_train, _ = hd3._fixed_labeled_data(
        d,
        cfg,
        oracle,
        split="train",
        n=training_points,
        device=device,
    )
    model = hd3._initialize_model(model_id, d, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(hd3.paired_batch_seed(d, cfg, model_id))
    physics_engine = torch.quasirandom.SobolEngine(
        d + 2,
        scramble=True,
        seed=int(cfg["sobol_seeds"]["physics"]) + d,
    )
    warm_steps = int(round(steps * float(objective["warm_fraction"])))
    mode = hd3.objective_mode(objective)
    probe_count = hd3.objective_probe_count(objective, cfg)
    relobralo = hd3._make_relobralo(objective) if mode == "relobralo" else None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps):
        schedule_final = 1.0 if mode == "relobralo" else float(objective["final_physics_weight"])
        weight, epsilon = curriculum(
            step,
            steps,
            warm_steps,
            schedule_final,
            epsilon_initial=float(cfg["fb_epsilon_initial"]),
            epsilon_final=float(cfg["fb_epsilon_final"]),
        )
        if mode == "relobralo" and step >= warm_steps:
            weight = 1.0
        index = torch.randint(0, len(x_train), (batch_size,), generator=generator).to(device)
        prediction = model(x_train[index], tau_train[index])
        supervised = (prediction - y_train[index]).square().mean()
        physics = torch.zeros((), device=device)
        if step >= warm_steps and weight > 0.0:
            unit = physics_engine.draw(batch_size).to(device=device, dtype=torch.float32)
            xp, tp = hd3._transform_physics_unit(unit, d, cfg)
            physics = hd3._physics_loss(
                model,
                xp,
                tp,
                d,
                cfg,
                epsilon=epsilon,
                probe_count=probe_count,
            )
        hd3._optimizer_step(
            model=model,
            optimizer=optimizer,
            supervised_loss=supervised,
            physics_loss=physics,
            weight=weight,
            mode=mode,
            objective=objective,
            relobralo=relobralo,
        )
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    if not math.isfinite(wall) or wall <= 0.0 or not math.isfinite(peak):
        raise RuntimeError("nonfinite_parallelism_smoke")
    return {
        "model_id": model_id,
        "dimension": d,
        "steps": steps,
        "batch_size": batch_size,
        "wall_seconds": wall,
        "seconds_per_step": wall / steps,
        "peak_cuda_gib": peak,
        "optimizer_mode": mode,
        "hutchinson_probe_count": probe_count,
        "non_evidence": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["FC_HVI", "DEEPSETS_HVI"], required=True)
    parser.add_argument("--dimension", type=int, choices=[10, 50], required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_smoke(
        args.model,
        d=args.dimension,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    if args.output is not None:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
