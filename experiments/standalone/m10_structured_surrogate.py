#!/usr/bin/env python3
"""M10 structured-surrogate representation ablation."""
from __future__ import annotations

import argparse, copy, csv, hashlib, json, platform, subprocess, time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

import m10_structured_surrogate_core as core
from m10_structured_surrogate_data import (
    FEATURE_NAMES, assert_complete_regime_rows, build_regimes, build_surface_cache,
    concatenate_surfaces, fit_feature_normalizer, normalize_features, solve_surface, split_regimes,
)
from m10_structured_surrogate_metrics import surface_metrics
from m10_structured_surrogate_train import (
    build_seeded_model, model_parameter_count, predict_surface, time_inference, train_model,
)


def _git(*args: str) -> str:
    try: return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return "unavailable"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def evaluate_model(model, cache, regime_ids, mean, std, model_spec, diagnostics, *, seed, arm, split_name):
    rows=[]; started=time.perf_counter(); scales=diagnostics["representation_scales"]
    for rid in regime_ids:
        surface=cache[str(rid)]
        candidate=predict_surface(model,surface,mean,std,model_spec,arm=arm,scales=scales,recompute_anchor=False)
        rows.append({"row_type":"surface_metric","seed":int(seed),"arm":arm,"split":split_name,"regime_id":str(rid),
                     **surface_metrics(candidate,surface,contact_tol=float(diagnostics["contact_tol"]))})
    return rows,float(time.perf_counter()-started)


def _scenario_vector(rows, arm, metric):
    selected=[r for r in rows if r.get("row_type")=="surface_metric" and r.get("split")=="test" and r.get("arm")==arm]
    ids=sorted({str(r["regime_id"]) for r in selected}); values=[]
    for rid in ids:
        paired=[float(r[metric]) for r in selected if str(r["regime_id"])==rid]
        if not paired: raise RuntimeError(f"missing M10 {metric} for {arm}/{rid}")
        values.append(float(np.mean(paired)))
    out=np.asarray(values,dtype=float)
    if not np.isfinite(out).all(): raise RuntimeError(f"non-finite M10 vector {arm}/{metric}")
    return ids,out


def candidate_summaries(rows, runtime):
    boot=runtime["bootstrap"]; diag=runtime["diagnostics"]; gates=runtime["gates"]
    base_ids,base_price=_scenario_vector(rows,"direct_price","price_rmse")
    raw_p={}; price_ci={}
    for i,arm in enumerate(core.STRUCTURED_ARMS):
        ids,cand=_scenario_vector(rows,arm,"price_rmse")
        if ids!=base_ids: raise RuntimeError("M10 price pairing mismatch")
        price_ci[arm]=core.bootstrap_relative_change(base_price,cand,repetitions=int(boot["repetitions"]),seed=int(boot["seed"])+100+i,
                                                     improvement=True,floor=float(diag["ordinary_floor"]),confidence=float(boot["confidence"]))
        raw_p[arm]=core.paired_signflip_pvalue(base_price,cand,alternative="candidate_lower",seed=int(boot["seed"])+200+i)
    holm=core.holm_adjust(raw_p); summaries={}; summary_rows=[]; passing=[]
    for i,arm in enumerate(core.STRUCTURED_ARMS):
        degradation={}; payload={}
        for j,metric in enumerate(core.ORDINARY_DIAGNOSTICS+core.RESIDUAL_DIAGNOSTICS):
            ids_b,b=_scenario_vector(rows,"direct_price",metric); ids_c,c=_scenario_vector(rows,arm,metric)
            if ids_b!=ids_c or ids_b!=base_ids: raise RuntimeError(f"M10 {metric} pairing mismatch")
            floor=float(diag["residual_floor"] if metric in core.RESIDUAL_DIAGNOSTICS else diag["ordinary_floor"])
            point,low,high=core.bootstrap_relative_change(b,c,repetitions=int(boot["repetitions"]),seed=int(boot["seed"])+1000+10*i+j,
                                                         improvement=False,floor=floor,confidence=float(boot["confidence"]))
            degradation[metric]=high; payload[metric]={"point_pct":point,"ci_low_pct":low,"ci_high_pct":high}
        obstacle=max(float(r["obstacle_violation"]) for r in rows if r.get("row_type")=="surface_metric" and r.get("split")=="test" and r.get("arm")==arm)
        ci=price_ci[arm]
        passed=core.candidate_passes(price_improvement_pct=ci[0],price_improvement_ci_low_pct=ci[1],holm_p_value=holm[arm],
                                     degradation_ci_upper_pct=degradation,obstacle_violation=obstacle,gates=gates)
        if passed: passing.append(arm)
        summaries[arm]={"price_improvement_pct":ci[0],"price_improvement_ci_low_pct":ci[1],"price_improvement_ci_high_pct":ci[2],
                        "paired_p_value":raw_p[arm],"holm_p_value":holm[arm],"diagnostic_degradation":payload,
                        "max_obstacle_violation":obstacle,"gate_passed":bool(passed)}
        summary_rows.append({"row_type":"candidate_summary","arm":arm,"price_improvement_pct":ci[0],"price_improvement_ci_low_pct":ci[1],
                             "price_improvement_ci_high_pct":ci[2],"paired_p_value":raw_p[arm],"holm_p_value":holm[arm],
                             "max_obstacle_violation":obstacle,"gate_passed":bool(passed),
                             **{f"{k}_degradation_ci_upper_pct":v for k,v in degradation.items()}})
    selected=max(passing,key=lambda a:summaries[a]["price_improvement_pct"]) if passing else None
    return summaries,summary_rows,selected


def run(config_path: str|Path, *, mode: str|None=None):
    config_path=Path(config_path); config=json.loads(config_path.read_text()); core.validate_config(config)
    evidence_mode=str(mode or config["evidence_mode"])
    if evidence_mode not in {"smoke","canonical"}: raise ValueError("mode must be smoke or canonical")
    runtime=copy.deepcopy(config)
    if evidence_mode=="smoke":
        smoke=runtime["smoke"]; runtime["training_seeds"]=list(smoke["training_seeds"]); runtime["model"]["epochs"]=int(smoke["epochs"])
        runtime["timing"]["warmups"]=int(smoke["timing_warmups"]); runtime["timing"]["repetitions"]=int(smoke["timing_repetitions"])
    out_dir=Path(runtime["output_root"]); out_dir.mkdir(parents=True,exist_ok=True); started_total=time.perf_counter()
    regimes=build_regimes(runtime)
    if evidence_mode=="smoke": regimes=regimes[:int(runtime["smoke"]["max_regimes"])]
    partitions=split_regimes(runtime,regimes)
    if min(len(partitions[k]) for k in ("train","validation","test"))<1: raise RuntimeError("M10 requires nonempty regime partitions")
    cache,surface_rows=build_surface_cache(runtime["solver"],partitions)
    ids={k:[str(r["id"]) for r in partitions[k]] for k in ("train","validation","test")}; nodes=(int(runtime["solver"]["m"])+1)*(int(runtime["solver"]["n"])+1)
    tx_raw,ty,te,tkeys=concatenate_surfaces(cache,ids["train"]); vx_raw,vy,ve,vkeys=concatenate_surfaces(cache,ids["validation"]); _,_,_,qkeys=concatenate_surfaces(cache,ids["test"])
    assert_complete_regime_rows(tkeys,nodes,ids["train"]); assert_complete_regime_rows(vkeys,nodes,ids["validation"]); assert_complete_regime_rows(qkeys,nodes,ids["test"])
    mean,std=fit_feature_normalizer(tx_raw); scales=core.fit_representation_scales(ty,te,floor=float(runtime["diagnostics"]["output_scale_floor"]))
    ttau=tx_raw[:,1].copy(); vtau=vx_raw[:,1].copy(); tx=normalize_features(tx_raw,mean,std); vx=normalize_features(vx_raw,mean,std)
    diagnostics=dict(runtime["diagnostics"]); diagnostics["representation_scales"]=scales; spec=runtime["model"]
    trained={}; history=[]; metrics=[]; training_rows=[]; counts=set()
    for seed in map(int,runtime["training_seeds"]):
        for arm in core.ARMS:
            model=build_seeded_model(seed=seed,hidden=int(spec["hidden"]),depth=int(spec["depth"])); count=model_parameter_count(model); counts.add(count)
            model,h,train_s=train_model(model,tx,ty,te,ttau,vx,vy,ve,vtau,spec,seed=seed,arm=arm,scales=scales); trained[(seed,arm)]=model; history.extend(h)
            rows,val_s=evaluate_model(model,cache,ids["validation"],mean,std,spec,diagnostics,seed=seed,arm=arm,split_name="validation"); metrics.extend(rows)
            training_rows.append({"row_type":"training_summary","seed":seed,"arm":arm,"epochs":int(spec["epochs"]),"training_seconds":train_s,"validation_eval_seconds":val_s,"parameter_count":count})
    if len(counts)!=1: raise RuntimeError("M10 arm parameter counts differ")
    for seed in map(int,runtime["training_seeds"]):
        for arm in core.ARMS:
            rows,_=evaluate_model(trained[(seed,arm)],cache,ids["test"],mean,std,spec,diagnostics,seed=seed,arm=arm,split_name="test"); metrics.extend(rows)
    summaries,summary_rows,selected=candidate_summaries(metrics,runtime); passed=selected is not None
    timing_rows=[]; sample=cache[ids["test"][0]]
    for seed in map(int,runtime["training_seeds"]):
        for arm in core.ARMS:
            m,s=time_inference(trained[(seed,arm)],sample,mean,std,spec,arm=arm,scales=scales,timing=runtime["timing"])
            timing_rows.append({"row_type":"timing","seed":seed,"arm":arm,"inference_seconds_mean":m,"inference_seconds_std":s,
                                "warmups":int(runtime["timing"]["warmups"]),"repetitions":int(runtime["timing"]["repetitions"])})
    total=float(time.perf_counter()-started_total); exceeded=evidence_mode=="canonical" and total>float(runtime["budget_seconds"])
    status=core.classify_status(evidence_mode,passed=passed,budget_exceeded=exceeded)
    manifest={"method_id":"M10","phase":"screen","evidence_mode":evidence_mode,"branch":_git("branch","--show-current"),"implementation_sha":_git("rev-parse","HEAD"),
              "python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"torch":__import__("torch").__version__,
              "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),"feature_names":list(FEATURE_NAMES),
              "feature_normalizer":{"mean":mean.tolist(),"std":std.tolist(),"fit_split":"train"},"representation_scales":{**scales,"fit_split":"train"},
              "arms":list(core.ARMS),"model_parameter_count":int(next(iter(counts))),"training_seeds":list(runtime["training_seeds"]),"split_regime_keys":ids,
              "blind_holdout_accessed":False,"scope_limitation":"Classical supervised 1D American-put output-parameterization ablation; identical MLP/data/optimizer/epochs; reconstructed American price metrics only."}
    _write_json(out_dir/"config_manifest.json",manifest); _write_csv(out_dir/"surface_generation.csv",surface_rows); _write_csv(out_dir/"metrics.csv",metrics+training_rows+summary_rows+timing_rows); _write_csv(out_dir/"training_history.csv",history)
    decision={"method_id":"M10","status":status,"claim_status":"candidate_supported" if status=="candidate_supported" else "diagnostic_only",
              "primary_metric":"price_rmse_improvement_vs_direct_pct","selected_candidate":selected if status=="candidate_supported" else None,
              "candidate_summaries":summaries,"numerical_gate_passed":bool(status=="candidate_supported" and passed),"blind_holdout_accessed":False,
              "total_seconds":total,"budget_seconds":float(runtime["budget_seconds"]),
              "reason":"Canonical M10 exceeded the one-hour screen budget." if exceeded else ("At least one structured representation passed all frozen gates." if status=="candidate_supported" else "No structured representation passed all frozen gates."),
              "limitations":["Output parameterization only; fixed coordinate MLP and training schedule.","Normalization and output scales fit on train regimes only; validation does not tune or early-stop.","Analytical European anchor is recomputed in inference timing.","Complete test regimes are the bootstrap unit after averaging paired seed results."]}
    _write_json(out_dir/"decision.json",decision); artifacts=[out_dir/x for x in ("config_manifest.json","surface_generation.csv","metrics.csv","training_history.csv","decision.json")]
    _write_json(out_dir/"artifact_hashes.json",{p.name:_sha256(p) for p in artifacts}); return decision


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/standalone/m10_structured_surrogate.json"); p.add_argument("--mode",choices=("smoke","canonical"),default=None); a=p.parse_args()
    print(json.dumps(run(a.config,mode=a.mode),indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__": main()
