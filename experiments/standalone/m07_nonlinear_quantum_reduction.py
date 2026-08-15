#!/usr/bin/env python3
"""M07 classical LP falsification audit for finite-horizon American stopping."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,subprocess,time
from pathlib import Path
from typing import Any,Mapping,NamedTuple
import numpy as np, scipy
from scipy import sparse
from scipy.optimize import linprog

class MarkovProblem(NamedTuple):
    n_states:int; n_times:int; states:np.ndarray; spots:np.ndarray; payoff:np.ndarray
    transition:sparse.csr_matrix; initial_distribution:np.ndarray; beta:float; dt:float; model:dict[str,float]
class LPProblem(NamedTuple):
    formulation:str; matrix:sparse.csr_matrix; rhs:np.ndarray; objective:np.ndarray
    n_variables:int; n_constraints:int; n_nonzeros:int; constraint_kind:str
class LPSolution(NamedTuple):
    formulation:str; price:float; vector:np.ndarray; success:bool; status:int; message:str

def _initial_distribution(x,x0):
    if not x[0] <= x0 <= x[-1]: raise ValueError("x0 must lie inside the log-price domain")
    w=np.zeros(len(x)); j=int(np.searchsorted(x,x0))
    if j<len(x) and np.isclose(x[j],x0,atol=1e-14,rtol=0): w[j]=1.; return w
    if j==0 or j==len(x): raise ValueError("cannot interpolate x0")
    a=(x0-x[j-1])/(x[j]-x[j-1]); w[j-1],w[j]=1-a,a; return w

def build_markov_chain(n_states:int,n_times:int,model:Mapping[str,Any])->MarkovProblem:
    if n_states<2 or n_times<1: raise ValueError("invalid grid")
    keys=("k","t","r","q","sigma","x_min","x_max","x0"); m={k:float(model[k]) for k in keys}
    if not all(np.isfinite(list(m.values()))): raise ValueError("model parameters must be finite")
    if m["k"]<=0 or m["t"]<=0 or m["sigma"]<=0 or m["x_max"]<=m["x_min"]: raise ValueError("invalid model")
    x=np.linspace(m["x_min"],m["x_max"],n_states); dx=x[1]-x[0]; dt=m["t"]/n_times
    mu=m["r"]-m["q"]-.5*m["sigma"]**2; v=m["sigma"]**2*dt/dx**2; d=mu*dt/dx
    pd,ps,pu=.5*(v-d),1-v,.5*(v+d)
    if min(pd,ps,pu)<-1e-14: raise ValueError("Markov local-moment probabilities are negative")
    pd,ps,pu=max(pd,0.),max(ps,0.),max(pu,0.)
    P=sparse.lil_matrix((n_states,n_states))
    for i in range(n_states):
        if i==0: P[i,i],P[i,i+1]=ps+pd,pu
        elif i==n_states-1: P[i,i-1],P[i,i]=pd,ps+pu
        else: P[i,i-1],P[i,i],P[i,i+1]=pd,ps,pu
    P=P.tocsr(); rows=np.asarray(P.sum(axis=1)).ravel()
    if not np.isfinite(P.data).all() or np.min(P.data)<-1e-14 or not np.allclose(rows,1,atol=1e-13): raise RuntimeError("transition is not stochastic")
    s=m["k"]*np.exp(x); g=np.maximum(m["k"]-s,0.); w=_initial_distribution(x,m["x0"])
    return MarkovProblem(n_states,n_times,x,s,g,P,w,float(np.exp(-m["r"]*dt)),dt,m)

def backward_dynamic_programming(p):
    V=np.empty((p.n_times+1,p.n_states)); V[-1]=p.payoff
    for t in range(p.n_times-1,-1,-1): V[t]=np.maximum(p.payoff,p.beta*(p.transition@V[t+1]))
    return V

def initial_price(p,V): return float(p.initial_distribution@np.asarray(V)[0])

def assemble_optimal_stopping_lp(p):
    S,T=p.n_states,p.n_times; nv=S*T; A=sparse.lil_matrix((2*S*T,nv)); b=np.zeros(2*S*T); row=0
    for t in range(T):
        for i in range(S): A[row,t*S+i]=-1; b[row]=-p.payoff[i]; row+=1
    P=p.transition.tocsr(); terminal=p.beta*(P@p.payoff)
    for t in range(T):
        for i in range(S):
            A[row,t*S+i]=-1
            if t==T-1: b[row]=-terminal[i]
            else:
                for k in range(P.indptr[i],P.indptr[i+1]): A[row,(t+1)*S+P.indices[k]]=p.beta*P.data[k]
            row+=1
    A=A.tocsr(); c=np.full(nv,1/nv)
    return LPProblem("optimal_stopping_lp",A,b,c,nv,2*S*T,int(A.nnz),"ub")

def solve_optimal_stopping_lp(p,method="highs",assembled=None):
    a=assembled or assemble_optimal_stopping_lp(p); r=linprog(a.objective,A_ub=a.matrix,b_ub=a.rhs,bounds=(0,None),method=method)
    x=np.asarray(r.x if r.x is not None else np.full(a.n_variables,np.nan)); price=float(p.initial_distribution@x[:p.n_states]) if r.success else float("nan")
    return LPSolution(a.formulation,price,x,bool(r.success),int(r.status),str(r.message))

def assemble_occupation_measure_lp(p):
    S,T=p.n_states,p.n_times; nc=S*T; nv=nc+S*(T+1); A=sparse.lil_matrix((S*(T+1),nv)); b=np.zeros(S*(T+1)); row=0
    ci=lambda t,i:t*S+i; zi=lambda t,i:nc+t*S+i
    for i in range(S): A[row,ci(0,i)]=A[row,zi(0,i)]=1; b[row]=p.initial_distribution[i]; row+=1
    PT=p.transition.T.tocsr()
    for t in range(1,T):
        for i in range(S):
            A[row,ci(t,i)]=A[row,zi(t,i)]=1
            for k in range(PT.indptr[i],PT.indptr[i+1]): A[row,ci(t-1,PT.indices[k])]=-PT.data[k]
            row+=1
    for i in range(S):
        A[row,zi(T,i)]=1
        for k in range(PT.indptr[i],PT.indptr[i+1]): A[row,ci(T-1,PT.indices[k])]=-PT.data[k]
        row+=1
    c=np.zeros(nv)
    for t in range(T+1): c[nc+t*S:nc+(t+1)*S]=-(p.beta**t)*p.payoff
    A=A.tocsr(); return LPProblem("occupation_measure_lp",A,b,c,nv,S*(T+1),int(A.nnz),"eq")

def solve_occupation_measure_lp(p,method="highs",assembled=None):
    a=assembled or assemble_occupation_measure_lp(p); r=linprog(a.objective,A_eq=a.matrix,b_eq=a.rhs,bounds=(0,None),method=method)
    x=np.asarray(r.x if r.x is not None else np.full(a.n_variables,np.nan)); price=float(-r.fun) if r.success else float("nan")
    return LPSolution(a.formulation,price,x,bool(r.success),int(r.status),str(r.message))

def dimension_accounting(formulation,n_states,n_times,lp_problem):
    if formulation!=lp_problem.formulation: raise ValueError("formulation mismatch")
    total=lp_problem.n_variables+lp_problem.n_constraints
    return {"formulation":formulation,"lp_variables":lp_problem.n_variables,"lp_constraints":lp_problem.n_constraints,"lp_nonzeros":lp_problem.n_nonzeros,"direct_lcp_unknowns":n_states,"direct_horizon_value_nodes":n_states*n_times,"dimension_growth_ratio":total/n_states,"horizon_growth_ratio":total/(n_states*n_times)}

def classify_status(evidence_mode,*,equivalence_passed,dimension_passed):
    if evidence_mode not in {"smoke","canonical"}: raise ValueError("evidence_mode must be smoke or canonical")
    return "diagnostic_only" if evidence_mode=="smoke" or (equivalence_passed and dimension_passed) else "failed_metric_gate"

def _time(fn,warmups,repetitions):
    for _ in range(warmups): result=fn()
    d=[]
    for _ in range(repetitions):
        t=time.perf_counter(); result=fn(); d.append(time.perf_counter()-t)
    a=np.asarray(d); return result,float(a.mean()),float(a.std()),float(a.min())

def _git(*args):
    try:return subprocess.check_output(["git",*args],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "unavailable"
def _json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
def _csv(path,rows):
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)

def _validate(c):
    if c.get("method_id")!="M07" or c.get("phase")!="screen": raise ValueError("M07 runner requires method_id=M07 and phase=screen")
    mode=str(c.get("evidence_mode","smoke")); pairs=[tuple(map(int,x)) for x in c["grid_pairs"]]; warm=int(c["timing"]["warmups"]); reps=int(c["timing"]["repetitions"])
    if mode=="canonical":
        if warm!=5 or reps!=30: raise ValueError("canonical M07 requires five warm-ups and 30 repetitions")
        if pairs!=[(4,4),(8,8),(16,16),(32,32)]: raise ValueError("canonical M07 requires frozen grid pairs")
    return mode,pairs,warm,reps

def run_screen(config:Mapping[str,Any],*,provenance=None):
    mode,pairs,warm,reps=_validate(config); started=time.perf_counter(); rows=[]; blocked=False
    model=config["model"]; ref=int(config.get("refinement_factor",2)); method=config.get("lp",{}).get("method","highs"); atol=float(config["gates"]["lp_value_atol"]); limit=float(config["gates"]["dimension_growth_ratio_max"]); budget=float(config.get("budget_seconds",14400))
    for S,T in pairs:
        ta=time.perf_counter(); p=build_markov_chain(S,T,model); trans_assembly=time.perf_counter()-ta
        V,dm,ds,dmin=_time(lambda:backward_dynamic_programming(p),warm,reps); direct=initial_price(p,V)
        fine=build_markov_chain(ref*S,ref*T,model); refinement=abs(direct-initial_price(fine,backward_dynamic_programming(fine))); tol=max(atol,refinement)
        for formulation in ("optimal_stopping_lp","occupation_measure_lp"):
            ta=time.perf_counter(); assembled=assemble_optimal_stopping_lp(p) if formulation.startswith("optimal") else assemble_occupation_measure_lp(p); assembly=time.perf_counter()-ta
            solver=(lambda:solve_optimal_stopping_lp(p,method,assembled)) if formulation.startswith("optimal") else (lambda:solve_occupation_measure_lp(p,method,assembled))
            solved,lm,ls,lmin=_time(solver,warm,reps)
            if not solved.success or not np.isfinite(solved.price): raise RuntimeError(f"{formulation} failed: {solved.message}")
            err=abs(solved.price-direct); counts=dimension_accounting(formulation,S,T,assembled)
            rows.append({"formulation":formulation,"n_states":S,"n_times":T,"direct_price":direct,"lp_price":solved.price,"price_abs_error":err,"direct_refinement_error":refinement,"equivalence_tolerance":tol,"equivalence_passed":err<=tol,**counts,"transition_nonzeros":int(p.transition.nnz),"transition_assembly_seconds":trans_assembly,"lp_assembly_seconds":assembly,"direct_solve_seconds_mean":dm,"direct_solve_seconds_std":ds,"direct_solve_seconds_min":dmin,"lp_solve_seconds_mean":lm,"lp_solve_seconds_std":ls,"lp_solve_seconds_min":lmin,"lp_over_direct_solve_time_ratio":lm/max(dm,np.finfo(float).tiny),"solver_status":solved.status})
        if time.perf_counter()-started>budget: blocked=True; break
    equivalence=bool(rows) and all(r["equivalence_passed"] for r in rows); forms=("optimal_stopping_lp","occupation_measure_lp")
    form_gate={f:len([r for r in rows if r["formulation"]==f])==len(pairs) and all(r["dimension_growth_ratio"]<=limit for r in rows if r["formulation"]==f) for f in forms}; dimension=any(form_gate.values())
    mx={f:max((r["dimension_growth_ratio"] for r in rows if r["formulation"]==f),default=float("inf")) for f in forms}; best=min(forms,key=mx.get)
    status="resource_blocked" if blocked else classify_status(mode,equivalence_passed=equivalence,dimension_passed=dimension)
    reason=("M07 exceeded the four-hour screening budget." if blocked else "At least one LP value failed the frozen equivalence tolerance." if not equivalence else "Both LP formulations matched the direct value but expanded variables plus constraints relative to direct LCP unknowns." if not dimension else "At least one LP matched without exceeding the dimension limit; evidence remains classical and diagnostic.")
    prov=dict(provenance or {"branch":_git("branch","--show-current") or "detached","commit":_git("rev-parse","HEAD")}); out=Path(config["output_root"]); out.mkdir(parents=True,exist_ok=True)
    manifest={"method_id":"M07","method_name":"Nonlinear Quantum Reduction","method_scope":"classical finite-state optimal-stopping LP falsification experiment; not a quantum run","branch":prov.get("branch","unavailable"),"implementation_commit":prov.get("commit","unavailable"),"environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},"data_hashes":{},"seeds":[],"split_identifiers":[],"config":json.loads(json.dumps(dict(config))),"blind_holdout_accessed":False,"method_scope_limitation":"No quantum circuit, QPU, quantum LP backend, or speedup claim is executed or supported."}
    decision={"method_id":"M07","status":status,"claim_status":"diagnostic_only","primary_metric":"best_formulation_max_dimension_growth_ratio","primary_metric_value":mx[best],"best_formulation":best,"numerical_gate_passed":bool(equivalence and dimension and not blocked),"equivalence_gate_passed":equivalence,"dimension_gate_passed":dimension,"formulation_dimension_gate_passed":form_gate,"reason":reason,"limitations":["Classical Markov-chain LP audit only; no quantum solver was run.","Primary denominator is simultaneous direct-LCP unknowns N_S; secondary N_S*N_T denominator is also reported.","Finite log-price boundaries reflect outgoing probability mass."],"blind_holdout_accessed":False,"total_seconds":time.perf_counter()-started}
    _json(out/"config_manifest.json",manifest); _csv(out/"metrics.csv",rows); _json(out/"decision.json",decision); hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!="artifact_hashes.json"}; _json(out/"artifact_hashes.json",{"algorithm":"sha256","files":hashes})
    return {**decision,"metrics":rows,"output_root":str(out)}

def main():
    a=argparse.ArgumentParser(); a.add_argument("--config",required=True); a.add_argument("--output-root"); args=a.parse_args(); c=json.loads(Path(args.config).read_text());
    if args.output_root:c["output_root"]=args.output_root
    print(json.dumps(run_screen(c),indent=2,sort_keys=True,allow_nan=False))
if __name__=="__main__": main()
