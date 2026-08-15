import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "standalone" / "m07_nonlinear_quantum_reduction.py"

def _module():
    spec = importlib.util.spec_from_file_location("m07_standalone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def _model():
    return {"k":1.0,"t":1.0,"r":0.05,"q":0.0,"sigma":0.2,"x_min":-1.5,"x_max":1.5,"x0":0.0}

def test_markov_chain_is_finite_nonnegative_stochastic_and_initial_mass_is_one():
    m=_module(); p=m.build_markov_chain(8,8,_model()); dense=p.transition.toarray()
    assert np.isfinite(dense).all(); assert np.min(dense)>=-1e-15; assert np.allclose(dense.sum(axis=1),1,atol=1e-13)
    assert np.isclose(p.initial_distribution.sum(),1); assert np.min(p.initial_distribution)>=0

def test_optimal_stopping_lp_matches_direct_dp_and_reports_actual_sparse_counts():
    m=_module(); p=m.build_markov_chain(4,4,_model()); direct=m.initial_price(p,m.backward_dynamic_programming(p)); a=m.assemble_optimal_stopping_lp(p); s=m.solve_optimal_stopping_lp(p)
    assert s.success; assert abs(s.price-direct)<1e-9; assert a.n_variables==16; assert a.n_constraints==32; assert a.n_nonzeros==a.matrix.nnz

def test_occupation_measure_lp_matches_direct_dp_and_conserves_flow():
    m=_module(); p=m.build_markov_chain(4,4,_model()); direct=m.initial_price(p,m.backward_dynamic_programming(p)); a=m.assemble_occupation_measure_lp(p); s=m.solve_occupation_measure_lp(p)
    assert s.success; assert abs(s.price-direct)<1e-9; assert a.n_variables==(2*4+1)*4; assert a.n_constraints==(4+1)*4; assert a.n_nonzeros==a.matrix.nnz
    assert np.max(np.abs(a.matrix@s.vector-a.rhs))<1e-9

def test_dimension_accounting_uses_packet_denominator_and_secondary_horizon_denominator():
    m=_module(); p=m.build_markov_chain(4,4,_model()); os=m.dimension_accounting("optimal_stopping_lp",4,4,m.assemble_optimal_stopping_lp(p)); om=m.dimension_accounting("occupation_measure_lp",4,4,m.assemble_occupation_measure_lp(p))
    assert os["dimension_growth_ratio"]==pytest.approx(12); assert os["horizon_growth_ratio"]==pytest.approx(3); assert om["dimension_growth_ratio"]==pytest.approx(14); assert om["horizon_growth_ratio"]==pytest.approx(3.5)

def test_status_classification_keeps_smoke_diagnostic_and_fails_canonical_gate():
    m=_module(); assert m.classify_status("smoke",equivalence_passed=False,dimension_passed=False)=="diagnostic_only"; assert m.classify_status("canonical",equivalence_passed=True,dimension_passed=False)=="failed_metric_gate"; assert m.classify_status("canonical",equivalence_passed=False,dimension_passed=True)=="failed_metric_gate"; assert m.classify_status("canonical",equivalence_passed=True,dimension_passed=True)=="diagnostic_only"

def test_canonical_requires_five_warmups_and_thirty_repetitions(tmp_path):
    m=_module(); c={"method_id":"M07","phase":"screen","evidence_mode":"canonical","output_root":str(tmp_path/"M07"),"model":_model(),"grid_pairs":[[4,4],[8,8],[16,16],[32,32]],"refinement_factor":2,"timing":{"warmups":0,"repetitions":1},"lp":{"method":"highs"},"gates":{"lp_value_atol":1e-10,"dimension_growth_ratio_max":1.0}}
    with pytest.raises(ValueError,match="five warm-ups and 30 repetitions"): m.run_screen(c,provenance={"branch":"test","commit":"test"})

def test_smoke_screen_writes_hashed_nonpromotion_evidence(tmp_path):
    m=_module(); out=tmp_path/"M07"; c={"method_id":"M07","phase":"screen","evidence_mode":"smoke","output_root":str(out),"model":_model(),"grid_pairs":[[4,4],[8,8]],"refinement_factor":2,"timing":{"warmups":0,"repetitions":1},"lp":{"method":"highs"},"gates":{"lp_value_atol":1e-10,"dimension_growth_ratio_max":1.0}}
    r=m.run_screen(c,provenance={"branch":"test","commit":"test"}); assert r["status"]=="diagnostic_only"; assert r["claim_status"]=="diagnostic_only"; assert r["blind_holdout_accessed"] is False; assert len(r["metrics"])==4
    assert {x["formulation"] for x in r["metrics"]}=={"occupation_measure_lp","optimal_stopping_lp"}; assert all(np.isfinite(float(x["price_abs_error"])) for x in r["metrics"]); assert all(x["dimension_growth_ratio"]>1 for x in r["metrics"])
    expected={"artifact_hashes.json","config_manifest.json","decision.json","metrics.csv"}; assert {p.name for p in out.iterdir()}==expected
    hashes=json.loads((out/"artifact_hashes.json").read_text())["files"]
    for name,digest in hashes.items(): assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
