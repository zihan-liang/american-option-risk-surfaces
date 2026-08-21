"""Reproducible post-processing for consolidated experiment data.

This module never trains models or mutates accepted source artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SHARED_DOMAIN = (0.5, 1.5, 0.01, 1.0)
HD_DOMAIN = (0.2, 1.5, 0.01, 1.0)
DATA_SUFFIXES = {".csv", ".json", ".npz", ".npy", ".txt", ".md", ".pt", ".pth"}

DIRK_METRIC_COLUMNS = ["method_id","dimension","seed","regime_id","internal_cnpsor_price_rmse","dirk_price_rmse_hd_domain","dirk_price_rmse_shared_domain","reference_type","metric_scope","s_min","s_max","tau_min","tau_max","grid_points","interpolation_method","source_repo","source_branch","source_commit","reference_source","reference_commit","reference_artifact_sha256"]
TABLE_HEADERS = {
    "source_artifact_inventory.csv": ["repo","representative_path","extension","size_bytes","git_blob_sha","content_sha256","artifact_type","experiment_family","method_id","regime_id","seed","dimension","duplicate_group","canonical_source"],
    "provenance.csv": ["repo","branch","branch_commit","path","git_blob_sha","content_sha256","duplicate_group"],
    "source_records.csv": ["repo","branch","source_commit","path","git_blob_sha","record_kind","record_index","field","value","value_type"],
    "array_artifacts.csv": ["repo","branch","source_commit","path","git_blob_sha","array_name","shape","dtype","csv_gz_path","content_sha256"],
    "experiment_runs.csv": ["repo","branch","source_commit","experiment_family","experiment_id","method_id","variant","dimension","seed","status","diagnostic_only","candidate_supported","reference_type","scoring_protocol"],
    "price_metrics.csv": ["repo","branch","source_commit","experiment_id","method_id","dimension","seed","metric_name","metric_value","reference_type","metric_scope","regime_id"],
    "greek_metrics.csv": ["repo","branch","source_commit","experiment_id","method_id","dimension","seed","metric_name","metric_value","greek_definition","greek_source","reference_type"],
    "boundary_metrics.csv": ["repo","branch","source_commit","experiment_id","method_id","dimension","seed","threshold","reference_count","predicted_count","matched_count","coverage","found_rate","conditional_rmse","full_rmse","exercise_f1","boundary_tube_f1","missing_reason","boundary_extraction_method"],
    "free_boundary_points.csv": ["repo","branch","source_commit","method_id","dimension","seed","regime_id","tau_over_T","boundary_over_K","boundary_found","threshold","missing_reason"],
    "price_slices.csv": ["repo","branch","source_commit","method_id","dimension","seed","regime_id","S_over_K","tau_over_T","V_over_K","payoff_over_K","premium_over_K","reference_type"],
    "structural_metrics.csv": ["repo","branch","source_commit","method_id","dimension","seed","metric_name","metric_value"],
    "compute_metrics.csv": ["repo","branch","source_commit","method_id","dimension","seed","training_seconds","evaluation_seconds","solver_seconds","oracle_seconds","screening_seconds","confirmation_seconds","wall_clock_seconds","hardware","protocol"],
    "classical_solver_metrics.csv": ["repo","branch","source_commit","method_id","regime_id","metric_name","metric_value","reference_type"],
    "binomial_convergence.csv": ["repo","branch","source_commit","steps","S_over_K","tau_over_T","V_over_K","comparison_metric","comparison_value"],
    "ruixi_arm_metrics.csv": ["repo","branch","source_commit","arm","method_id","regime_id","seed","metric_name","metric_value","reference_type"],
    "hd1_metrics.csv": ["repo","branch","source_commit","model_id","dimension","seed","metric_name","metric_value","gate_name","gate_pass"],
    "hd2_candidate_metrics.csv": ["repo","branch","source_commit","candidate_id","dimension","seed","metric_name","metric_value","gate_name","gate_pass","selection_status"],
    "hd3_metrics.csv": ["repo","branch","source_commit","model_id","dimension","seed","metric_name","metric_value","reference_type","regime_id"],
    "campaign_method_results.csv": ["repo","branch","source_commit","method_id","method_name","run_status","metric_name","metric_value","gate_name","gate_threshold","gate_pass","final_decision"],
    "experiment_gates.csv": ["repo","branch","source_commit","method_id","experiment_id","gate","metric","operator","threshold","actual","passed","decision"],
    "environment_runs.csv": ["repo","branch","source_commit","experiment_id","method_id","seed","os","python","pytorch","cuda","gpu","cpu","ram","device"],
    "surface_artifacts.csv": ["repo","branch","source_commit","path","method_id","dimension","seed","regime_id","shape","s_min","s_max","tau_min","tau_max","fields","point_count","content_sha256"],
    "dirk_common_reference_metrics.csv": DIRK_METRIC_COLUMNS,
}


def price_rmse(predicted: Sequence[float], reference: Sequence[float]) -> float:
    if len(predicted) != len(reference): raise ValueError("predicted and reference lengths differ")
    if not predicted: raise ValueError("RMSE requires at least one point")
    total=0.0
    for pred,ref in zip(predicted,reference):
        p=float(pred); r=float(ref)
        if not (math.isfinite(p) and math.isfinite(r)): raise ValueError("RMSE inputs must be finite")
        total+=(p-r)**2
    return math.sqrt(total/len(predicted))


def rectilinear_interpolate(source_s, source_tau, source_values, target_s, target_tau):
    import numpy as np
    s=np.asarray(source_s,dtype=float); tau=np.asarray(source_tau,dtype=float); values=np.asarray(source_values,dtype=float); ts=np.asarray(target_s,dtype=float); tt=np.asarray(target_tau,dtype=float)
    if s.ndim!=1 or tau.ndim!=1 or ts.ndim!=1 or tt.ndim!=1: raise ValueError("coordinate arrays must be one-dimensional")
    if values.shape!=(tau.size,s.size): raise ValueError("surface must have shape (len(tau), len(S))")
    if s.size<2 or tau.size<2 or not np.all(np.diff(s)>0) or not np.all(np.diff(tau)>0): raise ValueError("source coordinates must be strictly increasing")
    if not np.all(np.isfinite(values)): raise ValueError("surface values must be finite")
    eps=1e-12
    if ts.size and (ts.min()<s[0]-eps or ts.max()>s[-1]+eps): raise ValueError("target S grid requires extrapolation")
    if tt.size and (tt.min()<tau[0]-eps or tt.max()>tau[-1]+eps): raise ValueError("target tau grid requires extrapolation")
    along_s=np.vstack([np.interp(ts,s,row) for row in values]); result=np.empty((tt.size,ts.size),dtype=float)
    for j in range(ts.size): result[:,j]=np.interp(tt,tau,along_s[:,j])
    return result


def flatten_json_scalars(value, prefix: str="") -> Iterator[tuple[str,object]]:
    if isinstance(value,dict):
        for key in sorted(value): yield from flatten_json_scalars(value[key],f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value,list):
        for index,item in enumerate(value): yield from flatten_json_scalars(item,f"{prefix}[{index}]")
    else: yield prefix,value


def classify_artifact_path(path: str) -> str:
    lower=path.lower(); suffix=Path(lower).suffix
    if suffix in {".pt",".pth"}: return "checkpoint"
    if suffix in {".npz",".npy"}: return "numeric_array"
    if suffix==".csv":
        if "metric" in lower or "summary" in lower or "diagnostic" in lower: return "metrics_csv"
        if "boundary" in lower: return "boundary_csv"
        if "price_slice" in lower: return "price_slice_csv"
        return "data_csv"
    if suffix==".json": return "metadata_json"
    if suffix in {".txt",".md"}: return "text_evidence"
    return "other"


def _git(repo: Path, *args: str, binary: bool=False):
    result=subprocess.run(["git",*args],cwd=repo,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return result.stdout if binary else result.stdout.decode("utf-8",errors="replace")


def _path_hints(path: str) -> dict[str,object]:
    lower=path.lower(); dim=None; seed=None; method=""; regime=""; family=""
    md=re.search(r"(?:^|[/_-])d(?:im)?(10|20|50|100|[1-9])(?:[/_.-]|$)",lower)
    if md: dim=int(md.group(1))
    ms=re.search(r"seed[_-]?(\d+)",lower)
    if ms: seed=int(ms.group(1))
    mr=re.search(r"put_t\d+_s\d+_r\d+_q\d+",lower)
    if mr: regime=mr.group(0)
    for candidate in ("deepsets_hvi","fc_hvi","arm_c","arm_d","cn_psor","projected_lu","policy_iteration"):
        if candidate in lower: method=candidate.upper(); break
    if "hd1" in lower: family="HD1"
    elif "hd2" in lower: family="HD2"
    elif "hd3" in lower: family="HD3"
    else:
        mm=re.search(r"(?:^|/)m(\d{2})(?:[/_.-]|$)",lower)
        if mm: family=f"M{mm.group(1)}"
    return {"dimension":dim,"seed":seed,"method_id":method,"regime_id":regime,"experiment_family":family}


def inventory_repository(repo_path: Path|str, repo_name: str) -> dict[str,object]:
    repo=Path(repo_path)
    remote_refs=[r.strip() for r in _git(repo,"for-each-ref","--format=%(refname:short)","refs/remotes/origin").splitlines() if r.strip() and not r.strip().endswith("/HEAD")]
    refs=remote_refs or [r.strip() for r in _git(repo,"for-each-ref","--format=%(refname:short)","refs/heads").splitlines() if r.strip()]
    provenance=[]; unique={}; content_hashes={}
    for ref in sorted(refs):
        branch=ref[len("origin/"):] if ref.startswith("origin/") else ref
        branch_commit=_git(repo,"rev-parse",ref).strip()
        listing=_git(repo,"ls-tree","-r","-l",ref)
        for line in listing.splitlines():
            if "\t" not in line: continue
            meta,path=line.split("\t",1)
            parts=meta.split()
            if len(parts)<4 or parts[1]!="blob": continue
            blob_sha=parts[2]; size=int(parts[3]) if parts[3].isdigit() else None
            suffix=Path(path).suffix.lower()
            if suffix not in DATA_SUFFIXES or path.startswith("results/consolidated_csv/"): continue
            if blob_sha not in content_hashes: content_hashes[blob_sha]=hashlib.sha256(_git(repo,"cat-file","blob",blob_sha,binary=True)).hexdigest()
            sha256=content_hashes[blob_sha]; duplicate_group=sha256[:16]
            occurrence={"repo":repo_name,"branch":branch,"branch_commit":branch_commit,"path":path,"git_blob_sha":blob_sha,"content_sha256":sha256,"duplicate_group":duplicate_group}
            provenance.append(occurrence)
            if blob_sha not in unique:
                hints=_path_hints(path)
                unique[blob_sha]={"repo":repo_name,"representative_path":path,"extension":suffix,"size_bytes":size,"git_blob_sha":blob_sha,"content_sha256":sha256,"artifact_type":classify_artifact_path(path),**hints,"duplicate_group":duplicate_group,"canonical_source":f"{repo_name}@{branch}:{path}"}
    return {"branch_count":len(refs),"branches":[r[len("origin/"):] if r.startswith("origin/") else r for r in sorted(refs)],"artifacts":list(unique.values()),"provenance":provenance}


def normalize_surface_arrays(arrays):
    import numpy as np
    keys=set(arrays.keys() if hasattr(arrays,"keys") else arrays.files)
    def pick(candidates):
        for key in candidates:
            if key in keys: return np.asarray(arrays[key],dtype=float)
        raise KeyError(f"none of {candidates} found; available={sorted(keys)}")
    s=pick(["S_over_K","G_over_K","moneyness_grid","s_over_k","g_over_k","S","G","s","g"])
    tau=pick(["tau_over_T","normalized_time_grid","tau","tau_grid","time_grid"])
    v=pick(["V_over_K","value_over_k","V","value","price","values"])

    if s.ndim==1 and tau.ndim==1 and v.ndim==1 and s.size==tau.size==v.size:
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(tau)) and np.all(np.isfinite(v))):
            raise ValueError("flattened surface contains non-finite values")
        s_axis=np.unique(s); tau_axis=np.unique(tau)
        if s_axis.size*tau_axis.size!=v.size:
            raise ValueError("flattened pointwise surface is not a complete rectangular mesh")
        grid=np.full((tau_axis.size,s_axis.size),np.nan,dtype=float)
        s_index=np.searchsorted(s_axis,s); tau_index=np.searchsorted(tau_axis,tau)
        for si,ti,value in zip(s_index,tau_index,v):
            if not np.isnan(grid[ti,si]):
                raise ValueError("flattened pointwise surface contains duplicate coordinates")
            grid[ti,si]=value
        if np.isnan(grid).any():
            raise ValueError("flattened pointwise surface has missing grid points")
        s=s_axis; tau=tau_axis; v=grid
    else:
        if s.ndim==2:
            if np.allclose(s,s[0:1,:]): s=s[0,:]
            elif np.allclose(s,s[:,0:1]): s=s[:,0]
            else: raise ValueError("cannot reduce S mesh to one dimension")
        if tau.ndim==2:
            if np.allclose(tau,tau[:,0:1]): tau=tau[:,0]
            elif np.allclose(tau,tau[0:1,:]): tau=tau[0,:]
            else: raise ValueError("cannot reduce tau mesh to one dimension")
        s=np.ravel(s); tau=np.ravel(tau)
        if v.shape==(s.size,tau.size): v=v.T
        elif v.shape!=(tau.size,s.size): raise ValueError(f"value shape {v.shape} incompatible with S={s.size}, tau={tau.size}")

    if s.size>1 and s[0]>s[-1]: s=s[::-1]; v=v[:,::-1]
    if tau.size>1 and tau[0]>tau[-1]: tau=tau[::-1]; v=v[::-1,:]
    if not np.all(np.diff(s)>0) or not np.all(np.diff(tau)>0): raise ValueError("surface coordinates must be strictly increasing")
    if not np.all(np.isfinite(v)): raise ValueError("surface contains non-finite values")
    return s,tau,v


def _write_header(path: Path, header: Iterable[str]) -> None:
    with path.open("w",newline="",encoding="utf-8") as handle: csv.writer(handle).writerow(list(header))


def write_empty_output_contract(output_dir: Path|str) -> None:
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    for filename,header in TABLE_HEADERS.items(): _write_header(output/filename,header)
    manifest={"schema_version":1,"headline_cross_model_metric":"dirk_price_rmse_shared_domain","headline_hd_scaling_metric":"dirk_price_rmse_hd_domain","shared_domain":list(SHARED_DOMAIN),"hd_domain":list(HD_DOMAIN),"generated_tables":sorted(TABLE_HEADERS)}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (output/"README.md").write_text("# Consolidated experiment CSV data\n\nGenerated by post-processing accepted immutable experiment artifacts. Empty specialized tables mean no supporting source artifact has yet been parsed; they must never be interpreted as zero-valued results.\n",encoding="utf-8")
