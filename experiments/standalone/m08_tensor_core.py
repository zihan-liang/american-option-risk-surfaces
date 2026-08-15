"""Core tensor decompositions and frozen M08 configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class PODRepresentation:
    shape: tuple[int, ...]
    state_modes: int
    mean: np.ndarray
    basis: np.ndarray
    coefficients: np.ndarray

    @property
    def stored_scalars(self) -> int:
        return int(self.mean.size + self.basis.size + self.coefficients.size)


@dataclass(frozen=True)
class PrimalDualRepresentation:
    primal_basis: np.ndarray
    dual_basis: np.ndarray
    primal_coefficients: np.ndarray
    dual_coefficients: np.ndarray
    selected_regimes: tuple[int, ...]

    @property
    def stored_scalars(self) -> int:
        return int(
            self.primal_basis.size
            + self.dual_basis.size
            + self.primal_coefficients.size
            + self.dual_coefficients.size
        )


@dataclass(frozen=True)
class TTCompletionRepresentation:
    cores: tuple[np.ndarray, ...]
    observed_regime_indices: tuple[int, ...]
    seed: int

    @property
    def stored_scalars(self) -> int:
        return tt_storage(self.cores)


def _validate_tensor(tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(tensor, dtype=float)
    if arr.ndim < 2:
        raise ValueError("tensor must have at least two modes")
    if not np.isfinite(arr).all():
        raise ValueError("tensor must be finite")
    return arr


def tt_svd(tensor: np.ndarray, max_rank: int) -> tuple[np.ndarray, ...]:
    """Deterministic TT-SVD with a hard maximum internal rank."""
    arr = _validate_tensor(tensor)
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    shape = arr.shape
    cores: list[np.ndarray] = []
    unfolding = arr.copy()
    r_prev = 1
    for mode, n_mode in enumerate(shape[:-1]):
        matrix = unfolding.reshape(r_prev * n_mode, -1)
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        rank = min(int(max_rank), len(s))
        u = u[:, :rank]
        s = s[:rank]
        vt = vt[:rank, :]
        cores.append(u.reshape(r_prev, n_mode, rank))
        unfolding = s[:, None] * vt
        r_prev = rank
    cores.append(unfolding.reshape(r_prev, shape[-1], 1))
    return tuple(cores)


def tt_reconstruct(cores: Iterable[np.ndarray]) -> np.ndarray:
    cores = tuple(np.asarray(core, dtype=float) for core in cores)
    if not cores:
        raise ValueError("at least one TT core is required")
    if cores[0].shape[0] != 1 or cores[-1].shape[-1] != 1:
        raise ValueError("TT boundary ranks must equal one")
    result = cores[0][0]
    for previous, core in zip(cores[:-1], cores[1:]):
        if previous.shape[-1] != core.shape[0]:
            raise ValueError("adjacent TT ranks do not match")
        result = np.tensordot(result, core, axes=([-1], [0]))
    return np.squeeze(result, axis=-1)


def tt_storage(cores: Iterable[np.ndarray]) -> int:
    return int(sum(np.asarray(core).size for core in cores))


def tt_rank_vector(cores: Iterable[np.ndarray]) -> list[int]:
    cores = tuple(np.asarray(core) for core in cores)
    return [int(core.shape[-1]) for core in cores[:-1]]


def pod_compress(tensor: np.ndarray, *, state_modes: int, rank: int) -> PODRepresentation:
    arr = _validate_tensor(tensor)
    if not 1 <= state_modes < arr.ndim:
        raise ValueError("state_modes must split the tensor modes")
    state_size = int(np.prod(arr.shape[:state_modes]))
    regime_size = int(np.prod(arr.shape[state_modes:]))
    matrix = arr.reshape(state_size, regime_size)
    mean = matrix.mean(axis=1, keepdims=True)
    centered = matrix - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    use_rank = min(int(rank), len(s))
    if use_rank < 1:
        raise ValueError("rank must be positive")
    basis = u[:, :use_rank]
    coefficients = s[:use_rank, None] * vt[:use_rank, :]
    return PODRepresentation(arr.shape, state_modes, mean[:, 0], basis, coefficients)


def pod_reconstruct(rep: PODRepresentation) -> np.ndarray:
    matrix = rep.mean[:, None] + rep.basis @ rep.coefficients
    return matrix.reshape(rep.shape)


def _orthonormal_append(basis: list[np.ndarray], vector: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    residual = np.asarray(vector, dtype=float).copy()
    for column in basis:
        residual -= column * float(column @ residual)
    norm = float(np.linalg.norm(residual))
    if norm <= tol:
        # Deterministic fallback for rank-deficient snapshot sets.
        for index in range(residual.size):
            candidate = np.zeros_like(residual)
            candidate[index] = 1.0
            for column in basis:
                candidate -= column * float(column @ candidate)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > tol:
                return candidate / candidate_norm
        raise ValueError("cannot extend reduced basis")
    return residual / norm


def primal_dual_greedy_basis(
    primal_matrix: np.ndarray,
    dual_matrix: np.ndarray,
    *,
    rank: int,
) -> PrimalDualRepresentation:
    primal = np.asarray(primal_matrix, dtype=float)
    dual = np.asarray(dual_matrix, dtype=float)
    if primal.ndim != 2 or dual.ndim != 2 or primal.shape != dual.shape:
        raise ValueError("primal and dual matrices must be matching feature-by-regime matrices")
    if not np.isfinite(primal).all() or not np.isfinite(dual).all():
        raise ValueError("primal and dual snapshots must be finite")
    use_rank = min(int(rank), primal.shape[1], primal.shape[0])
    if use_rank < 1:
        raise ValueError("rank must be positive")

    p_basis: list[np.ndarray] = []
    d_basis: list[np.ndarray] = []
    selected: list[int] = []
    p_scale = max(float(np.linalg.norm(primal)), 1e-15)
    d_scale = max(float(np.linalg.norm(dual)), 1e-15)

    for _ in range(use_rank):
        if p_basis:
            p_stack = np.column_stack(p_basis)
            p_res = primal - p_stack @ (p_stack.T @ primal)
        else:
            p_res = primal.copy()
        if d_basis:
            d_stack = np.column_stack(d_basis)
            d_res = dual - d_stack @ (d_stack.T @ dual)
        else:
            d_res = dual.copy()
        score = np.maximum(
            np.linalg.norm(p_res, axis=0) / p_scale,
            np.linalg.norm(d_res, axis=0) / d_scale,
        )
        if selected:
            score[np.asarray(selected, dtype=int)] = -np.inf
        regime = int(np.argmax(score))
        selected.append(regime)
        p_basis.append(_orthonormal_append(p_basis, primal[:, regime]))
        d_basis.append(_orthonormal_append(d_basis, dual[:, regime]))

    p_stack = np.column_stack(p_basis)
    d_stack = np.column_stack(d_basis)
    return PrimalDualRepresentation(
        primal_basis=p_stack,
        dual_basis=d_stack,
        primal_coefficients=p_stack.T @ primal,
        dual_coefficients=d_stack.T @ dual,
        selected_regimes=tuple(selected),
    )


def primal_dual_reconstruct(rep: PrimalDualRepresentation) -> tuple[np.ndarray, np.ndarray]:
    return (
        rep.primal_basis @ rep.primal_coefficients,
        rep.dual_basis @ rep.dual_coefficients,
    )


def select_observed_regimes(train_regime_indices: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    train = np.asarray(train_regime_indices)
    if train.ndim != 1:
        raise ValueError("observations must select complete flattened regimes, not cells")
    if train.size == 0:
        raise ValueError("training regimes cannot be empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    count = max(1, int(round(train.size * fraction)))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(train.astype(int), size=count, replace=False))


def _flat_to_mode_indices(flat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.column_stack(np.unravel_index(np.asarray(flat, dtype=int), shape))


def _tt_batch_predict_torch(cores: list[Any], indices: Any) -> Any:
    import torch

    batch = indices.shape[0]
    first = cores[0][:, indices[:, 0], :].permute(1, 0, 2)
    product = first
    for mode, core in enumerate(cores[1:], start=1):
        selected = core[:, indices[:, mode], :].permute(1, 0, 2)
        product = torch.bmm(product, selected)
    return product.reshape(batch)


def fit_tt_completion(
    tensor: np.ndarray,
    *,
    state_modes: int,
    rank: int,
    observed_regime_indices: np.ndarray,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[TTCompletionRepresentation, list[dict[str, float | int]]]:
    """Fit TT cores to complete observed regime surfaces with CPU-float64 Adam.

    This is a genuine tensor-train completion approximation. It is deliberately
    named ``tt_completion`` rather than TT-cross.
    """
    import torch

    arr = _validate_tensor(tensor)
    if not 1 <= state_modes < arr.ndim:
        raise ValueError("state_modes must split state/time and parameter modes")
    observed = np.asarray(observed_regime_indices, dtype=int)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observed_regime_indices must be a nonempty 1-D array")
    state_shape = arr.shape[:state_modes]
    regime_shape = arr.shape[state_modes:]
    n_regimes = int(np.prod(regime_shape))
    if np.any(observed < 0) or np.any(observed >= n_regimes):
        raise ValueError("observed regime index out of range")
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("invalid optimization settings")

    # Mean-impute unobserved complete regimes for a stable TT-SVD initialization.
    matrix = arr.reshape(int(np.prod(state_shape)), n_regimes)
    observed_mean = matrix[:, observed].mean(axis=1, keepdims=True)
    imputed = np.repeat(observed_mean, n_regimes, axis=1)
    imputed[:, observed] = matrix[:, observed]
    init_cores = tt_svd(imputed.reshape(arr.shape), max_rank=rank)

    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))
    params = [torch.nn.Parameter(torch.tensor(core, dtype=torch.float64)) for core in init_cores]
    optimizer = torch.optim.Adam(params, lr=float(learning_rate))

    state_size = int(np.prod(state_shape))
    history: list[dict[str, float | int]] = []

    def sample_batch() -> tuple[Any, Any]:
        state_flat = np_rng.integers(0, state_size, size=batch_size)
        regime_flat = np_rng.choice(observed, size=batch_size, replace=True)
        state_idx = _flat_to_mode_indices(state_flat, state_shape)
        regime_idx = _flat_to_mode_indices(regime_flat, regime_shape)
        indices_np = np.concatenate([state_idx, regime_idx], axis=1)
        values_np = arr[tuple(indices_np[:, mode] for mode in range(arr.ndim))]
        return (
            torch.tensor(indices_np, dtype=torch.long),
            torch.tensor(values_np, dtype=torch.float64),
        )

    # A fixed audit batch makes the history comparable across optimization steps.
    audit_indices, audit_values = sample_batch()
    with torch.no_grad():
        initial_loss = torch.mean((_tt_batch_predict_torch(params, audit_indices) - audit_values) ** 2)
    initial_loss_value = float(initial_loss.cpu().item())
    best_loss = initial_loss_value
    best_cores = tuple(parameter.detach().cpu().numpy().copy() for parameter in params)
    history.append({"step": 0, "loss": initial_loss_value, "best_loss": best_loss})

    for step in range(1, int(steps) + 1):
        indices, values = sample_batch()
        optimizer.zero_grad(set_to_none=True)
        prediction = _tt_batch_predict_torch(params, indices)
        loss = torch.mean((prediction - values) ** 2)
        loss.backward()
        optimizer.step()
        if step == steps or step % max(1, steps // 10) == 0:
            with torch.no_grad():
                audit_loss = torch.mean((_tt_batch_predict_torch(params, audit_indices) - audit_values) ** 2)
            audit_loss_value = float(audit_loss.cpu().item())
            if audit_loss_value < best_loss:
                best_loss = audit_loss_value
                best_cores = tuple(parameter.detach().cpu().numpy().copy() for parameter in params)
            history.append({"step": step, "loss": audit_loss_value, "best_loss": best_loss})

    return TTCompletionRepresentation(best_cores, tuple(int(x) for x in observed), int(seed)), history


def configuration_passes(
    compression_ratio: float,
    degradation_upper_pct: Mapping[str, float],
    *,
    ordinary_limit_pct: float,
    residual_limit_pct: float,
) -> bool:
    if not np.isfinite(compression_ratio) or compression_ratio < 10.0:
        return False
    ordinary = ("price_rmse", "delta_rmse", "gamma_rmse", "boundary_rmse", "obstacle_violation")
    residual = ("dual_violation", "complementarity", "continuation_residual")
    for key in ordinary:
        value = float(degradation_upper_pct[key])
        if not np.isfinite(value) or value > ordinary_limit_pct:
            return False
    for key in residual:
        value = float(degradation_upper_pct[key])
        if not np.isfinite(value) or value > residual_limit_pct:
            return False
    return True


def classify_status(evidence_mode: str, *, any_configuration_passed: bool) -> str:
    if evidence_mode == "smoke":
        return "diagnostic_only"
    if evidence_mode != "canonical":
        raise ValueError("evidence_mode must be smoke or canonical")
    return "candidate_supported" if any_configuration_passed else "failed_metric_gate"


def default_test_config() -> dict[str, Any]:
    return {
        "method_id": "M08",
        "phase": "screen",
        "evidence_mode": "canonical",
        "output_root": "results/standalone/M08",
        "budget_seconds": 3600,
        "base_solver": {
            "option_type": "put", "k": 1.0, "t": 2.0, "s_max": 4.0,
            "m": 128, "n": 64, "omega": 1.2, "tol": 1e-10,
            "max_iter": 20000, "american": True, "time_scheme": "cn",
            "lcp_solver": "psor",
        },
        "refined_solver": {"m": 256, "n": 128},
        "parameter_grid": {
            "sigma": {"min": 0.10, "max": 0.60, "count": 9},
            "r": {"min": 0.00, "max": 0.10, "count": 7},
            "q": {"min": 0.00, "max": 0.08, "count": 5},
        },
        "split": {"train_fraction": 0.70, "validation_fraction": 0.15, "seed": 808},
        "tt_completion_seeds": [101, 203, 307, 409, 503],
        "completion": {"steps": 800, "batch_size": 4096, "learning_rate": 0.01},
        "timing": {"warmups": 5, "repetitions": 30},
        "bootstrap": {"repetitions": 500, "seed": 808},
        "diagnostics": {
            "contact_tol": 1e-8, "ordinary_floor": 1e-12,
            "obstacle_floor": 1e-10, "residual_floor": 1e-10,
        },
        "configurations": [
            {"id": "tt_svd_r4", "family": "tt_svd", "rank": 4},
            {"id": "tt_svd_r8", "family": "tt_svd", "rank": 8},
            {"id": "tt_completion_r4_f10", "family": "tt_completion", "rank": 4, "observed_fraction": 0.10},
            {"id": "tt_completion_r8_f25", "family": "tt_completion", "rank": 8, "observed_fraction": 0.25},
            {"id": "pod_r4", "family": "pod", "rank": 4},
            {"id": "pod_r8", "family": "pod", "rank": 8},
            {"id": "primal_dual_rb_r4", "family": "primal_dual_rb", "rank": 4},
            {"id": "primal_dual_rb_r8", "family": "primal_dual_rb", "rank": 8},
        ],
        "gates": {
            "compression_ratio_min": 10.0,
            "ordinary_degradation_pct_max": 2.0,
            "residual_degradation_pct_max": 5.0,
        },
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("method_id") != "M08" or config.get("phase") != "screen":
        raise ValueError("M08 runner requires method_id=M08 and phase=screen")
    configurations = list(config.get("configurations", []))
    if len(configurations) != 8:
        raise ValueError("M08 requires exactly eight predeclared configurations")
    ids = [item.get("id") for item in configurations]
    if len(set(ids)) != 8:
        raise ValueError("M08 configuration ids must be unique")
    families = [item.get("family") for item in configurations]
    required = {"tt_svd", "tt_completion", "pod", "primal_dual_rb"}
    if set(families) != required or any(families.count(name) != 2 for name in required):
        raise ValueError("M08 requires exactly two configurations per family")
    seeds = list(config.get("tt_completion_seeds", []))
    if seeds != [101, 203, 307, 409, 503]:
        raise ValueError("M08 requires exactly five TT-completion campaign seeds")
    if int(config.get("budget_seconds", 0)) != 3600:
        raise ValueError("M08 canonical budget must be exactly 3600 seconds")
    gates = config.get("gates", {})
    if float(gates.get("compression_ratio_min", 0)) != 10.0:
        raise ValueError("M08 compression gate must be 10x")
    if float(gates.get("ordinary_degradation_pct_max", 0)) != 2.0:
        raise ValueError("M08 ordinary diagnostic gate must be 2%")
    if float(gates.get("residual_degradation_pct_max", 0)) != 5.0:
        raise ValueError("M08 residual diagnostic gate must be 5%")

