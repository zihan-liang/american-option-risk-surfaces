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
        return int(self.primal_basis.size+self.dual_basis.size+self.primal_coefficients.size+self.dual_coefficients.size)

@dataclass(frozen=True)
class TTCompletionRepresentation:
    cores: tuple[np.ndarray, ...]
    observed_regime_indices: tuple[int, ...]
    seed: int
    @property
    def stored_scalars(self) -> int:
        return tt_storage(self.cores)

def _validate_tensor(tensor: np.ndarray) -> np.ndarray:
    arr=np.asarray(tensor,dtype=float)
    if arr.ndim<2: raise ValueError("tensor must have at least two modes")
    if not np.isfinite(arr).all(): raise ValueError("tensor must be finite")
    return arr

def tt_svd(tensor: np.ndarray,max_rank:int)->tuple[np.ndarray,...]:
    arr=_validate_tensor(tensor)
    if max_rank<1: raise ValueError("max_rank must be positive")
    cores=[]; unfolding=arr.copy(); r_prev=1
    for n_mode in arr.shape[:-1]:
        matrix=unfolding.reshape(r_prev*n_mode,-1)
        u,s,vt=np.linalg.svd(matrix,full_matrices=False); rank=min(int(max_rank),len(s))
        cores.append(u[:,:rank].reshape(r_prev,n_mode,rank)); unfolding=s[:rank,None]*vt[:rank,:]; r_prev=rank
    cores.append(unfolding.reshape(r_prev,arr.shape[-1],1)); return tuple(cores)

def tt_reconstruct(cores:Iterable[np.ndarray])->np.ndarray:
    cores=tuple(np.asarray(c,dtype=float) for c in cores)
    if not cores: raise ValueError("at least one TT core is required")
    if cores[0].shape[0]!=1 or cores[-1].shape[-1]!=1: raise ValueError("TT boundary ranks must equal one")
    result=cores[0][0]
    for previous,core in zip(cores[:-1],cores[1:]):
        if previous.shape[-1]!=core.shape[0]: raise ValueError("adjacent TT ranks do not match")
        result=np.tensordot(result,core,axes=([-1],[0]))
    return np.squeeze(result,axis=-1)

def tt_storage(cores:Iterable[np.ndarray])->int: return int(sum(np.asarray(c).size for c in cores))
def tt_rank_vector(cores:Iterable[np.ndarray])->list[int]:
    cores=tuple(np.asarray(c) for c in cores); return [int(c.shape[-1]) for c in cores[:-1]]

def pod_compress(tensor:np.ndarray,*,state_modes:int,rank:int)->PODRepresentation:
    arr=_validate_tensor(tensor)
    if not 1<=state_modes<arr.ndim: raise ValueError("state_modes must split the tensor modes")
    matrix=arr.reshape(int(np.prod(arr.shape[:state_modes])),int(np.prod(arr.shape[state_modes:])))
    mean=matrix.mean(axis=1,keepdims=True); centered=matrix-mean; u,s,vt=np.linalg.svd(centered,full_matrices=False)
    r=min(int(rank),len(s))
    if r<1: raise ValueError("rank must be positive")
    return PODRepresentation(arr.shape,state_modes,mean[:,0],u[:,:r],s[:r,None]*vt[:r,:])
def pod_reconstruct(rep:PODRepresentation)->np.ndarray: return (rep.mean[:,None]+rep.basis@rep.coefficients).reshape(rep.shape)

def _orthonormal_append(basis:list[np.ndarray],vector:np.ndarray,tol:float=1e-12)->np.ndarray:
    residual=np.asarray(vector,dtype=float).copy()
    for column in basis: residual-=column*float(column@residual)
    norm=float(np.linalg.norm(residual))
    if norm>tol: return residual/norm
    for index in range(residual.size):
        candidate=np.zeros_like(residual); candidate[index]=1.0
        for column in basis: candidate-=column*float(column@candidate)
        cn=float(np.linalg.norm(candidate))
        if cn>tol: return candidate/cn
    raise ValueError("cannot extend reduced basis")

def primal_dual_greedy_basis(primal_matrix:np.ndarray,dual_matrix:np.ndarray,*,rank:int)->PrimalDualRepresentation:
    primal=np.asarray(primal_matrix,dtype=float); dual=np.asarray(dual_matrix,dtype=float)
    if primal.ndim!=2 or dual.ndim!=2 or primal.shape[1]!=dual.shape[1]: raise ValueError("primal and dual matrices must share regime columns")
    if not np.isfinite(primal).all() or not np.isfinite(dual).all(): raise ValueError("primal and dual snapshots must be finite")
    use_rank=min(int(rank),primal.shape[1],primal.shape[0],dual.shape[0])
    if use_rank<1: raise ValueError("rank must be positive")
    pb=[]; db=[]; selected=[]; ps=max(float(np.linalg.norm(primal)),1e-15); ds=max(float(np.linalg.norm(dual)),1e-15)
    for _ in range(use_rank):
        pres=primal-(np.column_stack(pb)@(np.column_stack(pb).T@primal) if pb else 0)
        dres=dual-(np.column_stack(db)@(np.column_stack(db).T@dual) if db else 0)
        score=np.maximum(np.linalg.norm(pres,axis=0)/ps,np.linalg.norm(dres,axis=0)/ds)
        if selected: score[np.asarray(selected,dtype=int)]=-np.inf
        regime=int(np.argmax(score)); selected.append(regime)
        pb.append(_orthonormal_append(pb,primal[:,regime])); db.append(_orthonormal_append(db,dual[:,regime]))
    p=np.column_stack(pb); d=np.column_stack(db)
    return PrimalDualRepresentation(p,d,p.T@primal,d.T@dual,tuple(selected))
def primal_dual_reconstruct(rep:PrimalDualRepresentation)->tuple[np.ndarray,np.ndarray]: return rep.primal_basis@rep.primal_coefficients,rep.dual_basis@rep.dual_coefficients

def select_observed_regimes(train_regime_indices:np.ndarray,*,fraction:float,seed:int)->np.ndarray:
    train=np.asarray(train_regime_indices)
    if train.ndim!=1: raise ValueError("observations must select complete flattened regimes, not cells")
    if train.size==0: raise ValueError("training regimes cannot be empty")
    if not 0.0<fraction<=1.0: raise ValueError("fraction must lie in (0,1]")
    count=max(1,int(round(train.size*fraction))); rng=np.random.default_rng(int(seed)); return np.sort(rng.choice(train.astype(int),size=count,replace=False))

def _flat_to_mode_indices(flat:np.ndarray,shape:tuple[int,...])->np.ndarray: return np.column_stack(np.unravel_index(np.asarray(flat,dtype=int),shape))
def _tt_batch_predict_torch(cores:list[Any],indices:Any)->Any:
    import torch
    product=cores[0][:,indices[:,0],:].permute(1,0,2)
    for mode,core in enumerate(cores[1:],start=1): product=torch.bmm(product,core[:,indices[:,mode],:].permute(1,0,2))
    return product.reshape(indices.shape[0])

def fit_tt_completion(tensor:np.ndarray,*,state_modes:int,rank:int,observed_regime_indices:np.ndarray,seed:int,steps:int,batch_size:int,learning_rate:float)->tuple[TTCompletionRepresentation,list[dict[str,float|int]]]:
    import torch
    arr=_validate_tensor(tensor)
    if not 1<=state_modes<arr.ndim: raise ValueError("state_modes must split state/time and parameter modes")
    observed=np.asarray(observed_regime_indices,dtype=int); state_shape=arr.shape[:state_modes]; regime_shape=arr.shape[state_modes:]; n_regimes=int(np.prod(regime_shape))
    if observed.ndim!=1 or observed.size==0: raise ValueError("observed_regime_indices must be a nonempty 1-D array")
    if np.any(observed<0) or np.any(observed>=n_regimes): raise ValueError("observed regime index out of range")
    if steps<1 or batch_size<1 or learning_rate<=0: raise ValueError("invalid optimization settings")
    matrix=arr.reshape(int(np.prod(state_shape)),n_regimes); observed_mean=matrix[:,observed].mean(axis=1,keepdims=True); imputed=np.repeat(observed_mean,n_regimes,axis=1); imputed[:,observed]=matrix[:,observed]
    init=tt_svd(imputed.reshape(arr.shape),max_rank=rank); torch.manual_seed(int(seed)); rng=np.random.default_rng(int(seed)); params=[torch.nn.Parameter(torch.tensor(c,dtype=torch.float64)) for c in init]; optimizer=torch.optim.Adam(params,lr=float(learning_rate)); state_size=int(np.prod(state_shape))
    def sample_batch():
        sf=rng.integers(0,state_size,size=batch_size); rf=rng.choice(observed,size=batch_size,replace=True); idx=np.concatenate([_flat_to_mode_indices(sf,state_shape),_flat_to_mode_indices(rf,regime_shape)],axis=1); vals=arr[tuple(idx[:,m] for m in range(arr.ndim))]; return torch.tensor(idx,dtype=torch.long),torch.tensor(vals,dtype=torch.float64)
    audit_idx,audit_vals=sample_batch()
    with torch.no_grad(): initial=float(torch.mean((_tt_batch_predict_torch(params,audit_idx)-audit_vals)**2).cpu().item())
    best=initial; best_cores=tuple(p.detach().cpu().numpy().copy() for p in params); history=[{"step":0,"loss":initial,"best_loss":best}]
    for step in range(1,int(steps)+1):
        idx,vals=sample_batch(); optimizer.zero_grad(set_to_none=True); loss=torch.mean((_tt_batch_predict_torch(params,idx)-vals)**2); loss.backward(); optimizer.step()
        if step==steps or step%max(1,steps//10)==0:
            with torch.no_grad(): audit=float(torch.mean((_tt_batch_predict_torch(params,audit_idx)-audit_vals)**2).cpu().item())
            if audit<best: best=audit; best_cores=tuple(p.detach().cpu().numpy().copy() for p in params)
            history.append({"step":step,"loss":audit,"best_loss":best})
    return TTCompletionRepresentation(best_cores,tuple(int(x) for x in observed),int(seed)),history

def configuration_passes(compression_ratio:float,degradation_upper_pct:Mapping[str,float],*,ordinary_limit_pct:float,residual_limit_pct:float)->bool:
    if not np.isfinite(compression_ratio) or compression_ratio<10.0:return False
    for key in ("price_rmse","delta_rmse","gamma_rmse","boundary_rmse","obstacle_violation"):
        value=float(degradation_upper_pct[key]);
        if not np.isfinite(value) or value>ordinary_limit_pct:return False
    for key in ("dual_violation","complementarity","continuation_residual"):
        value=float(degradation_upper_pct[key]);
        if not np.isfinite(value) or value>residual_limit_pct:return False
    return True

def classify_status(evidence_mode:str,*,any_configuration_passed:bool)->str:
    if evidence_mode=="smoke":return "diagnostic_only"
    if evidence_mode!="canonical":raise ValueError("evidence_mode must be smoke or canonical")
    return "candidate_supported" if any_configuration_passed else "failed_metric_gate"

def validate_config(config:Mapping[str,Any])->None:
    if config.get("method_id")!="M08" or config.get("phase")!="screen":raise ValueError("M08 runner requires method_id=M08 and phase=screen")
    items=list(config.get("configurations",[]));
    if len(items)!=8 or len({x.get('id') for x in items})!=8:raise ValueError("M08 requires exactly eight unique predeclared configurations")
    families=[x.get("family") for x in items]; required={"tt_svd","tt_completion","pod","primal_dual_rb"}
    if set(families)!=required or any(families.count(name)!=2 for name in required):raise ValueError("M08 requires exactly two configurations per family")
    if list(config.get("tt_completion_seeds",[]))!=[101,203,307,409,503]:raise ValueError("M08 requires exactly five TT-completion campaign seeds")
    if int(config.get("budget_seconds",0))!=3600:raise ValueError("M08 canonical budget must be exactly 3600 seconds")
    timing=config.get("timing",{})
    if int(timing.get("warmups",-1))!=5 or int(timing.get("repetitions",-1))!=30:raise ValueError("M08 canonical reconstruction timing requires five warm-ups and 30 repetitions")
    gates=config.get("gates",{})
    if float(gates.get("compression_ratio_min",0))!=10.0 or float(gates.get("ordinary_degradation_pct_max",0))!=2.0 or float(gates.get("residual_degradation_pct_max",0))!=5.0:raise ValueError("M08 packet gates must be exactly 10x/2%/5%")