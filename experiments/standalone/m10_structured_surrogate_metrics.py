"""Shared reconstructed-price diagnostics for M10."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig, fd_operator_matrix, time_boundary_values
from solvers.greeks import delta_gamma, greek_diagnostic_masks


def enforce_known_constraints(values: np.ndarray, result: Any, cfg: SolverConfig) -> np.ndarray:
    out=np.maximum(np.asarray(values,dtype=float).copy(), result.payoff[None,:]); out[0,:]=result.payoff
    for i,tau in enumerate(result.taus):
        left,right=time_boundary_values(cfg,float(tau)); out[i,0]=left; out[i,-1]=right
    return out


def cn_residual_fields(values: np.ndarray, cfg: SolverConfig, result: Any):
    spots,taus=result.spots,result.taus; ds=float(spots[1]-spots[0]); interior=spots[1:-1]
    diffusion=.5*cfg.sigma**2*interior**2; drift=(cfg.r-cfg.q)*interior
    lower=diffusion/ds**2-drift/(2*ds); upper=diffusion/ds**2+drift/(2*ds)
    lmat=fd_operator_matrix(cfg,spots); eye=np.eye(cfg.m-1); residual=np.empty((cfg.n,cfg.m-1))
    gap=values[1:,1:-1]-result.payoff[1:-1][None,:]
    def source(tau):
        out=np.zeros(cfg.m-1); left,right=time_boundary_values(cfg,float(tau)); out[0]=lower[0]*left; out[-1]=upper[-1]*right; return out
    for step in range(cfg.n):
        dt=float(taus[step+1]-taus[step]); a=eye-.5*dt*lmat; b=eye+.5*dt*lmat
        rhs=b@values[step,1:-1]+.5*dt*(source(taus[step])+source(taus[step+1])); residual[step]=a@values[step+1,1:-1]-rhs
    return residual,gap


def _rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,dtype=float)-np.asarray(b,dtype=float))**2)))


def surface_metrics(candidate: np.ndarray, surface: Mapping[str, Any], *, contact_tol: float):
    result=surface["result"]; cfg=surface["cfg"]; ref=result.values
    ref_b=boundary_curve(result.spots,ref,result.payoff,cfg.option_type,tol=contact_tol)
    cand_b=boundary_curve(result.spots,candidate,result.payoff,cfg.option_type,tol=contact_tol)
    rd,rg=delta_gamma(ref,result.spots); cd,cg=delta_gamma(candidate,result.spots)
    strict=np.asarray(greek_diagnostic_masks(result.spots,ref_b,cfg.k)["strict"],dtype=bool)
    time_mask=result.taus>0; ref_valid=np.isfinite(ref_b)&time_mask; points=int(np.count_nonzero(ref_valid))
    if points:
        cb=cand_b[ref_valid]; missing=~np.isfinite(cb); err=np.where(missing,float(cfg.s_max),cb-ref_b[ref_valid])
        boundary_rmse=float(np.sqrt(np.mean(err**2))); missing_fraction=float(np.mean(missing))
    else: boundary_rmse=0.; missing_fraction=0.
    spurious=(~np.isfinite(ref_b))&np.isfinite(cand_b)&time_mask
    residual,gap=cn_residual_fields(candidate,cfg,result); ref_gap=ref[1:,1:-1]-result.payoff[1:-1][None,:]
    continuation=ref_gap>contact_tol; masked=float(np.sqrt(np.mean(residual[continuation]**2))) if np.any(continuation) else 0.
    metrics={"price_rmse":_rmse(candidate,ref),"delta_rmse":_rmse(cd[strict],rd[strict]) if np.any(strict) else 0.,
             "gamma_rmse":_rmse(cg[strict],rg[strict]) if np.any(strict) else 0.,"boundary_rmse":boundary_rmse,
             "obstacle_violation":float(np.max(np.maximum(-gap,0.))),"masked_residual":masked,
             "dual_violation":float(np.max(np.maximum(-residual,0.))),"complementarity":float(np.max(np.abs(gap*residual))),
             "reference_boundary_points":points,"boundary_missing_fraction":missing_fraction,
             "boundary_spurious_fraction":float(np.count_nonzero(spurious)/max(int(np.count_nonzero(time_mask)),1))}
    if any(isinstance(v,float) and not np.isfinite(v) for v in metrics.values()): raise RuntimeError("non-finite M10 diagnostic")
    return metrics
