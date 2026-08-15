#!/usr/bin/env python3
"""M08 tensor and low-rank screen for CN/PSOR American-put risk surfaces."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,subprocess,time
from pathlib import Path
from typing import Any,Callable,Mapping
import numpy as np
from data.splits import three_way_regime_split
from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig,fd_operator_matrix,solve_option,time_boundary_values
from solvers.greeks import delta_gamma
from m08_tensor_core import (TTCompletionRepresentation,classify_status,configuration_passes,fit_tt_completion,pod_compress,pod_reconstruct,primal_dual_greedy_basis,primal_dual_reconstruct,select_observed_regimes,tt_rank_vector,tt_reconstruct,tt_svd,tt_storage,validate_config)

ORDINARY_METRICS=("price_rmse","delta_rmse","gamma_rmse","boundary_rmse","obstacle_violation")
RESIDUAL_METRICS=("dual_violation","complementarity","continuation_residual")
ALL_GATE_METRICS=ORDINARY_METRICS+RESIDUAL_METRICS

def _git(*args:str)->str:
    try:return subprocess.check_output(["git",*args],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "unavailable"
def _json(path:Path,payload:Any)->None:path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
def _sha256(path:Path)->str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)

def _linspace(spec:Mapping[str,Any])->np.ndarray:return np.linspace(float(spec["min"]),float(spec["max"]),int(spec["count"]),dtype=float)
def build_regimes(config:Mapping[str,Any])->tuple[list[dict[str,Any]],tuple[np.ndarray,np.ndarray,np.ndarray]]:
    grid=config["parameter_grid"]; sigmas,rates,dividends=(_linspace(grid[n]) for n in ("sigma","r","q")); regimes=[]; flat=0
    for i_sigma,sigma in enumerate(sigmas):
        for i_r,rate in enumerate(rates):
            for i_q,dividend in enumerate(dividends):
                regimes.append({"id":f"regime_{flat:04d}","flat_index":flat,"i_sigma":i_sigma,"i_r":i_r,"i_q":i_q,"sigma":float(sigma),"r":float(rate),"q":float(dividend)});flat+=1
    return regimes,(sigmas,rates,dividends)
def solver_config(base:Mapping[str,Any],regime:Mapping[str,Any])->SolverConfig:
    kwargs=dict(base);kwargs.update({"sigma":float(regime["sigma"]),"r":float(regime["r"]),"q":float(regime["q"])});return SolverConfig(**kwargs)
def build_value_tensor(base:Mapping[str,Any],regimes:list[dict[str,Any]],parameter_shape:tuple[int,int,int])->tuple[np.ndarray,dict[int,Any],float]:
    started=time.perf_counter();surfaces={};tensor=None
    for regime in regimes:
        result=solve_option(solver_config(base,regime))
        if not np.isfinite(result.values).all() or not all(result.lcp_converged):raise RuntimeError(f"CN/PSOR failed for {regime['id']}")
        if tensor is None:tensor=np.empty(result.values.shape+parameter_shape,dtype=float)
        tensor[:,:,regime["i_sigma"],regime["i_r"],regime["i_q"]]=result.values;surfaces[int(regime["flat_index"])]=result
    assert tensor is not None;return tensor,surfaces,float(time.perf_counter()-started)

def _operator_coefficients(cfg:SolverConfig,spots:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    ds=float(spots[1]-spots[0]);interior=spots[1:-1];diffusion=.5*cfg.sigma**2*interior**2;drift=(cfg.r-cfg.q)*interior
    return diffusion/ds**2-drift/(2*ds),diffusion/ds**2+drift/(2*ds)
def cn_residual_fields(values:np.ndarray,cfg:SolverConfig,spots:np.ndarray,taus:np.ndarray,payoff:np.ndarray)->dict[str,np.ndarray]:
    if values.shape!=(cfg.n+1,cfg.m+1):raise ValueError("candidate surface shape does not match solver config")
    l_mat=fd_operator_matrix(cfg,spots);lower,upper=_operator_coefficients(cfg,spots);eye=np.eye(cfg.m-1);obstacle=payoff[1:-1];residual=np.empty((cfg.n,cfg.m-1));gap=values[1:,1:-1]-obstacle[None,:]
    def source(tau:float)->np.ndarray:
        out=np.zeros(cfg.m-1);left,right=time_boundary_values(cfg,tau);out[0]=lower[0]*left;out[-1]=upper[-1]*right;return out
    for step in range(cfg.n):
        dt=float(taus[step+1]-taus[step]);a=eye-.5*dt*l_mat;b=eye+.5*dt*l_mat;rhs=b@values[step,1:-1]+.5*dt*(source(float(taus[step]))+source(float(taus[step+1])));residual[step]=a@values[step+1,1:-1]-rhs
    return {"residual":residual,"gap":gap}
def build_dual_tensor(tensor_shape:tuple[int,...],base_surfaces:Mapping[int,Any],regimes:list[dict[str,Any]],base_cfg:Mapping[str,Any])->np.ndarray:
    dual=np.zeros(tensor_shape,dtype=float)
    for regime in regimes:
        result=base_surfaces[int(regime["flat_index"])];cfg=solver_config(base_cfg,regime);fields=cn_residual_fields(result.values,cfg,result.spots,result.taus,result.payoff);snapshot=np.zeros_like(result.values);snapshot[1:,1:-1]=np.maximum(fields["residual"],0.0);dual[:,:,regime["i_sigma"],regime["i_r"],regime["i_q"]]=snapshot
    return dual

def interpolate_surface(values:np.ndarray,spots:np.ndarray,taus:np.ndarray,target_spots:np.ndarray,target_taus:np.ndarray)->np.ndarray:
    by_spot=np.vstack([np.interp(target_spots,spots,row) for row in values]);out=np.empty((len(target_taus),len(target_spots)))
    for j in range(len(target_spots)):out[:,j]=np.interp(target_taus,taus,by_spot[:,j])
    return out
def _interpolate_boundary(values:np.ndarray,taus:np.ndarray,target_taus:np.ndarray)->np.ndarray:
    mask=np.isfinite(values)
    if np.count_nonzero(mask)<2:return np.full_like(target_taus,np.nan,dtype=float)
    return np.interp(target_taus,taus[mask],values[mask],left=np.nan,right=np.nan)
def _rmse(a:np.ndarray,b:np.ndarray)->float:return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def _shape_diagnostics(values:np.ndarray,spots:np.ndarray)->dict[str,float]:
    first=np.diff(values,axis=1)/np.diff(spots)[None,:];second=np.diff(first,axis=1)/((spots[2:]-spots[:-2])/2.0)[None,:]
    return {"spot_monotonicity_violation_fraction":float(np.mean(first>1e-10)),"spot_convexity_violation_fraction":float(np.mean(second<-1e-10))}
def surface_diagnostics(candidate:np.ndarray,base_result:Any,refined_result:Any,cfg:SolverConfig,*,contact_tol:float)->tuple[dict[str,float],dict[str,float]]:
    refined_on_base=interpolate_surface(refined_result.values,refined_result.spots,refined_result.taus,base_result.spots,base_result.taus);base=base_result.values;base_delta,base_gamma=delta_gamma(base,base_result.spots);cand_delta,cand_gamma=delta_gamma(candidate,base_result.spots);ref_delta,ref_gamma=delta_gamma(refined_on_base,base_result.spots);interior=(slice(None),slice(1,-1))
    base_b=boundary_curve(base_result.spots,base,base_result.payoff,"put",tol=contact_tol);cand_b=boundary_curve(base_result.spots,candidate,base_result.payoff,"put",tol=contact_tol);ref_native=boundary_curve(refined_result.spots,refined_result.values,refined_result.payoff,"put",tol=contact_tol);ref_b=_interpolate_boundary(ref_native,refined_result.taus,base_result.taus);reference_mask=np.isfinite(base_b)&np.isfinite(ref_b)&(base_result.taus>0)
    if np.any(reference_mask):
        base_boundary=_rmse(base_b[reference_mask],ref_b[reference_mask]);candidate_available=np.isfinite(cand_b[reference_mask]);cand_boundary=_rmse(cand_b[reference_mask],ref_b[reference_mask]) if np.all(candidate_available) else float("inf")
    else:base_boundary=cand_boundary=0.0
    base_fields=cn_residual_fields(base,cfg,base_result.spots,base_result.taus,base_result.payoff);cand_fields=cn_residual_fields(candidate,cfg,base_result.spots,base_result.taus,base_result.payoff);reference_cont=base_fields["gap"]>contact_tol
    def lcp(fields:dict[str,np.ndarray])->tuple[float,float,float,float]:
        gap,res=fields["gap"],fields["residual"];obs=float(np.max(np.maximum(-gap,0)));dual=float(np.max(np.maximum(-res,0)));comp=float(np.max(np.abs(gap*res)));cont=float(np.sqrt(np.mean(res[reference_cont]**2))) if np.any(reference_cont) else 0.0;return obs,dual,comp,cont
    bl=lcp(base_fields);cl=lcp(cand_fields)
    baseline={"price_rmse":_rmse(base,refined_on_base),"delta_rmse":_rmse(base_delta[interior],ref_delta[interior]),"gamma_rmse":_rmse(base_gamma[interior],ref_gamma[interior]),"boundary_rmse":base_boundary,"obstacle_violation":bl[0],"dual_violation":bl[1],"complementarity":bl[2],"continuation_residual":bl[3]}
    metrics={"price_rmse":_rmse(candidate,refined_on_base),"delta_rmse":_rmse(cand_delta[interior],ref_delta[interior]),"gamma_rmse":_rmse(cand_gamma[interior],ref_gamma[interior]),"boundary_rmse":cand_boundary,"obstacle_violation":cl[0],"dual_violation":cl[1],"complementarity":cl[2],"continuation_residual":cl[3],**_shape_diagnostics(candidate,base_result.spots),"boundary_reference_points":int(np.count_nonzero(reference_mask))}
    return metrics,baseline

def _bootstrap_upper(candidate:np.ndarray,baseline:np.ndarray,*,floor:float,repetitions:int,seed:int)->tuple[float,float]:
    candidate=np.asarray(candidate,dtype=float);baseline=np.asarray(baseline,dtype=float)
    if candidate.shape!=baseline.shape or candidate.ndim!=1 or candidate.size==0:raise ValueError("bootstrap inputs must be paired nonempty regime vectors")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():return float("inf"),float("inf")
    paired=100.0*(candidate-baseline)/np.maximum(np.abs(baseline),float(floor));point=float(np.mean(paired));rng=np.random.default_rng(int(seed));means=np.empty(int(repetitions))
    for i in range(int(repetitions)):
        sample=rng.integers(0,paired.size,size=paired.size);means[i]=float(np.mean(paired[sample]))
    return point,float(np.quantile(means,.975))
def _candidate_surface(tensor:np.ndarray,regime:Mapping[str,Any])->np.ndarray:return tensor[:,:,int(regime["i_sigma"]),int(regime["i_r"]),int(regime["i_q"])]
def build_refined_cache(base_cfg:Mapping[str,Any],refined_cfg:Mapping[str,Any],regimes:list[dict[str,Any]])->dict[int,Any]:
    merged=dict(base_cfg);merged.update(refined_cfg);out={}
    for regime in regimes:
        result=solve_option(solver_config(merged,regime))
        if not np.isfinite(result.values).all() or not all(result.lcp_converged):raise RuntimeError(f"refined CN/PSOR failed for {regime['id']}")
        out[int(regime["flat_index"])]=result
    return out
def evaluate_partition(reconstructed:np.ndarray,regimes:list[dict[str,Any]],base_surfaces:Mapping[int,Any],refined_cache:Mapping[int,Any],base_cfg:Mapping[str,Any],diagnostics_cfg:Mapping[str,Any],bootstrap_cfg:Mapping[str,Any])->dict[str,float]:
    cand={name:[] for name in ALL_GATE_METRICS};base={name:[] for name in ALL_GATE_METRICS};extras=[]
    for regime in regimes:
        index=int(regime["flat_index"]);metrics,baseline=surface_diagnostics(_candidate_surface(reconstructed,regime),base_surfaces[index],refined_cache[index],solver_config(base_cfg,regime),contact_tol=float(diagnostics_cfg["contact_tol"]));extras.append(metrics)
        for name in ALL_GATE_METRICS:cand[name].append(float(metrics[name]));base[name].append(float(baseline[name]))
    floors={name:float(diagnostics_cfg["ordinary_floor"]) for name in ("price_rmse","delta_rmse","gamma_rmse","boundary_rmse")};floors["obstacle_violation"]=float(diagnostics_cfg["obstacle_floor"])
    for name in RESIDUAL_METRICS:floors[name]=float(diagnostics_cfg["residual_floor"])
    summary={}
    for offset,name in enumerate(ALL_GATE_METRICS):
        ca=np.asarray(cand[name]);ba=np.asarray(base[name]);point,upper=_bootstrap_upper(ca,ba,floor=floors[name],repetitions=int(bootstrap_cfg["repetitions"]),seed=int(bootstrap_cfg["seed"])+offset);summary[name]=float(np.mean(ca)) if np.isfinite(ca).all() else float("inf");summary[f"baseline_{name}"]=float(np.mean(ba));summary[f"degradation_pct_{name}"]=point;summary[f"degradation_upper_pct_{name}"]=upper
    summary["spot_monotonicity_violation_fraction"]=float(np.mean([x["spot_monotonicity_violation_fraction"] for x in extras]));summary["spot_convexity_violation_fraction"]=float(np.mean([x["spot_convexity_violation_fraction"] for x in extras]));summary["boundary_reference_points"]=int(sum(x["boundary_reference_points"] for x in extras));return summary
def _gate_upper(summary:Mapping[str,float])->dict[str,float]:return {name:float(summary[f"degradation_upper_pct_{name}"]) for name in ALL_GATE_METRICS}
def _gate_ratio(summary:Mapping[str,float],compression_ratio:float,gates:Mapping[str,Any])->float:
    ratios=[float(gates["compression_ratio_min"])/max(float(compression_ratio),1e-15)]
    for name in ORDINARY_METRICS:ratios.append(max(0.0,float(summary[f"degradation_upper_pct_{name}"]))/float(gates["ordinary_degradation_pct_max"]))
    for name in RESIDUAL_METRICS:ratios.append(max(0.0,float(summary[f"degradation_upper_pct_{name}"]))/float(gates["residual_degradation_pct_max"]))
    return float(max(ratios))
def _time_reconstruction(fn:Callable[[],np.ndarray],timing:Mapping[str,Any])->tuple[np.ndarray,float,float]:
    for _ in range(int(timing["warmups"])):out=fn()
    values=[]
    for _ in range(int(timing["repetitions"])):
        started=time.perf_counter();out=fn();values.append(time.perf_counter()-started)
    return out,float(np.mean(values)),float(np.std(values))

def fit_deterministic(item:Mapping[str,Any],tensor:np.ndarray,dual_tensor:np.ndarray,timing:Mapping[str,Any])->tuple[np.ndarray,dict[str,Any]]:
    family=str(item["family"]);rank=int(item["rank"]);started=time.perf_counter()
    if family=="tt_svd":rep=tt_svd(tensor,rank);fit_seconds=time.perf_counter()-started;reconstructed,mean,std=_time_reconstruction(lambda:tt_reconstruct(rep),timing);stored=tt_storage(rep);meta={"tt_ranks":tt_rank_vector(rep)}
    elif family=="pod":rep=pod_compress(tensor,state_modes=2,rank=rank);fit_seconds=time.perf_counter()-started;reconstructed,mean,std=_time_reconstruction(lambda:pod_reconstruct(rep),timing);stored=rep.stored_scalars;meta={}
    elif family=="primal_dual_rb":
        primal=tensor.reshape(tensor.shape[0]*tensor.shape[1],-1);dual=dual_tensor.reshape(dual_tensor.shape[0]*dual_tensor.shape[1],-1);rep=primal_dual_greedy_basis(primal,dual,rank=rank);fit_seconds=time.perf_counter()-started
        def rebuild():return primal_dual_reconstruct(rep)[0].reshape(tensor.shape)
        reconstructed,mean,std=_time_reconstruction(rebuild,timing);stored=rep.stored_scalars;meta={"selected_regimes":list(rep.selected_regimes),"claim_scope":"primal_dual_compression_baseline_not_reduced_vi_solver"}
    else:raise ValueError("deterministic fitter received learned completion family")
    return reconstructed,{"family":family,"rank":rank,"stored_scalars":int(stored),"compression_ratio":float(tensor.size/max(int(stored),1)),"fit_seconds":float(fit_seconds),"reconstruction_seconds_mean":mean,"reconstruction_seconds_std":std,"oracle_complete_tensor_access":True,**meta}
def fit_completion(item:Mapping[str,Any],tensor:np.ndarray,train_indices:np.ndarray,seed:int,completion_cfg:Mapping[str,Any],timing:Mapping[str,Any])->tuple[np.ndarray,dict[str,Any]]:
    observed=select_observed_regimes(train_indices,fraction=float(item["observed_fraction"]),seed=seed);started=time.perf_counter();rep,history=fit_tt_completion(tensor,state_modes=2,rank=int(item["rank"]),observed_regime_indices=observed,seed=seed,steps=int(completion_cfg["steps"]),batch_size=int(completion_cfg["batch_size"]),learning_rate=float(completion_cfg["learning_rate"]));fit_seconds=time.perf_counter()-started;reconstructed,mean,std=_time_reconstruction(lambda:tt_reconstruct(rep.cores),timing)
    return reconstructed,{"family":"tt_completion","rank":int(item["rank"]),"stored_scalars":rep.stored_scalars,"compression_ratio":float(tensor.size/max(rep.stored_scalars,1)),"fit_seconds":float(fit_seconds),"reconstruction_seconds_mean":mean,"reconstruction_seconds_std":std,"oracle_complete_tensor_access":False,"seed":int(seed),"observed_regime_count":len(rep.observed_regime_indices),"observed_fraction":float(item["observed_fraction"]),"tt_ranks":tt_rank_vector(rep.cores),"initial_audit_loss":float(history[0]["loss"]),"best_audit_loss":float(history[-1]["best_loss"])}
def _row(partition:str,item:Mapping[str,Any],seed:Any,meta:Mapping[str,Any],summary:Mapping[str,Any],gates:Mapping[str,Any])->dict[str,Any]:
    return {"partition":partition,"config_id":item["id"],"family":item["family"],"seed":"" if seed is None else seed,"rank":item["rank"],"stored_scalars":meta["stored_scalars"],"compression_ratio":meta["compression_ratio"],"fit_seconds":meta["fit_seconds"],"reconstruction_seconds_mean":meta["reconstruction_seconds_mean"],"reconstruction_seconds_std":meta["reconstruction_seconds_std"],"oracle_complete_tensor_access":meta["oracle_complete_tensor_access"],"gate_ratio":_gate_ratio(summary,float(meta["compression_ratio"]),gates),**summary}
def _smoke_config(config:dict[str,Any])->dict[str,Any]:
    c=json.loads(json.dumps(config));c["evidence_mode"]="smoke";c["base_solver"].update({"m":20,"n":10,"t":.5});c["refined_solver"]={"m":30,"n":15};c["parameter_grid"]={"sigma":{"min":.15,"max":.35,"count":3},"r":{"min":.02,"max":.08,"count":3},"q":{"min":0.0,"max":.04,"count":2}};c["completion"].update({"steps":5,"batch_size":64});c["bootstrap"]["repetitions"]=30;c["timing"]={"warmups":5,"repetitions":30};return c

def run(config:dict[str,Any])->dict[str,Any]:
    validate_config(config);started=time.perf_counter();regimes,axes=build_regimes(config);split=three_way_regime_split(regimes,train_fraction=float(config["split"]["train_fraction"]),validation_fraction=float(config["split"]["validation_fraction"]),seed=int(config["split"]["seed"]));train_indices=np.asarray([x["flat_index"] for x in split["train"]],dtype=int);tensor,base_surfaces,label_seconds=build_value_tensor(config["base_solver"],regimes,tuple(len(a) for a in axes));dual_tensor=build_dual_tensor(tensor.shape,base_surfaces,regimes,config["base_solver"]);refined_regimes=split["validation"]+split["test"];refined_started=time.perf_counter();refined_cache=build_refined_cache(config["base_solver"],config["refined_solver"],refined_regimes);refined_seconds=float(time.perf_counter()-refined_started)
    if time.perf_counter()-started>float(config["budget_seconds"]):raise TimeoutError("M08 screen budget exceeded during oracle generation")
    validation_rows=[];artifacts={};config_scores={}
    for item in config["configurations"]:
        rows=[]
        if item["family"]=="tt_completion":
            for seed in config["tt_completion_seeds"]:
                reconstructed,meta=fit_completion(item,tensor,train_indices,int(seed),config["completion"],config["timing"]);summary=evaluate_partition(reconstructed,split["validation"],base_surfaces,refined_cache,config["base_solver"],config["diagnostics"],config["bootstrap"]);row=_row("validation",item,seed,meta,summary,config["gates"]);rows.append(row);validation_rows.append(row);artifacts[(item["id"],int(seed))]=(reconstructed,meta)
        else:
            reconstructed,meta=fit_deterministic(item,tensor,dual_tensor,config["timing"]);summary=evaluate_partition(reconstructed,split["validation"],base_surfaces,refined_cache,config["base_solver"],config["diagnostics"],config["bootstrap"]);row=_row("validation",item,None,meta,summary,config["gates"]);rows.append(row);validation_rows.append(row);artifacts[(item["id"],None)]=(reconstructed,meta)
        config_scores[item["id"]]=max(float(row["gate_ratio"]) for row in rows)
        if time.perf_counter()-started>float(config["budget_seconds"]):raise TimeoutError("M08 screen budget exceeded during validation")
    frozen={}
    for family in ("tt_svd","tt_completion","pod","primal_dual_rb"):
        candidates=[item for item in config["configurations"] if item["family"]==family];winner=min(candidates,key=lambda x:(config_scores[x["id"]],int(x["rank"]),str(x["id"])));frozen[family]=winner["id"]
    test_rows=[];passing=[]
    for family,config_id in frozen.items():
        item=next(x for x in config["configurations"] if x["id"]==config_id);replicate_pass=[];seeds=config["tt_completion_seeds"] if family=="tt_completion" else [None]
        for seed in seeds:
            reconstructed,meta=artifacts[(config_id,seed)];summary=evaluate_partition(reconstructed,split["test"],base_surfaces,refined_cache,config["base_solver"],config["diagnostics"],config["bootstrap"]);passed=configuration_passes(float(meta["compression_ratio"]),_gate_upper(summary),ordinary_limit_pct=float(config["gates"]["ordinary_degradation_pct_max"]),residual_limit_pct=float(config["gates"]["residual_degradation_pct_max"]));replicate_pass.append(bool(passed));row=_row("test",item,seed,meta,summary,config["gates"]);row["gate_passed"]=passed;test_rows.append(row)
        if all(replicate_pass):passing.append(config_id)
    elapsed=float(time.perf_counter()-started)
    if elapsed>float(config["budget_seconds"]):raise TimeoutError("M08 canonical screen exceeded one-hour budget")
    return {"rows":validation_rows+test_rows,"frozen":frozen,"passing":passing,"status":classify_status(str(config["evidence_mode"]),any_configuration_passed=bool(passing)),"total_seconds":elapsed,"label_seconds":label_seconds,"refined_seconds":refined_seconds,"split_ids":{name:[x["id"] for x in values] for name,values in split.items()},"axes":{"sigma":axes[0].tolist(),"r":axes[1].tolist(),"q":axes[2].tolist()}}

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--smoke",action="store_true");args=parser.parse_args();config_path=Path(args.config);config=json.loads(config_path.read_text());config=_smoke_config(config) if args.smoke else config;validate_config(config);output=Path(config["output_root"]);output.mkdir(parents=True,exist_ok=True)
    try:
        result=run(config);status=result["status"];reason="At least one frozen low-rank configuration passed the 10x/2%/5% gate." if result["passing"] else "No frozen low-rank configuration passed compression and all shared diagnostic gates."
    except TimeoutError as exc:
        result={"rows":[],"frozen":{},"passing":[],"total_seconds":float(config["budget_seconds"]),"label_seconds":0.0,"refined_seconds":0.0,"split_ids":{},"axes":{}};status="resource_blocked";reason=str(exc)
    limits=["Classical tensor/model-reduction experiment only; no quantum hardware or QTT PDE solver.","TT-SVD, POD, and primal-dual RB are oracle compressibility baselines with complete-tensor access.","tt_completion alone restricts observed fitting data to complete training regimes and is run with all five campaign seeds.","primal_dual_rb uses CN LCP residual snapshots as the dual basis but is a compression baseline, not a reduced variational-inequality solver.","No post-reconstruction obstacle repair is applied before diagnostics."]
    manifest={"method_id":"M08","phase":"screen","evidence_mode":config["evidence_mode"],"resolved_config":config,"source_config":str(config_path),"branch":_git("branch","--show-current"),"implementation_sha":_git("rev-parse","HEAD"),"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"seeds":config["tt_completion_seeds"],"splits":result["split_ids"],"parameter_axes":result["axes"],"blind_holdout_accessed":False,"scope_limits":limits}
    decision={"method_id":"M08","status":status,"claim_status":"candidate_supported" if status=="candidate_supported" else "diagnostic_only","reason":reason,"frozen_validation_winners":result["frozen"],"passing_test_configurations":result["passing"],"numerical_gate_passed":bool(result["passing"]),"blind_holdout_accessed":False,"total_seconds":result["total_seconds"],"label_generation_seconds":result["label_seconds"],"refined_oracle_seconds":result["refined_seconds"],"limitations":limits}
    _json(output/"config_manifest.json",manifest);_csv(output/"metrics.csv",result["rows"] if result["rows"] else [{"partition":"none","status":status}]);_json(output/"decision.json",decision);_json(output/"artifact_hashes.json",{"files":{name:_sha256(output/name) for name in ("config_manifest.json","metrics.csv","decision.json")}});print(json.dumps(decision,indent=2,sort_keys=True))
if __name__=="__main__":main()
