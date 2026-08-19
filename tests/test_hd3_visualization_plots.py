from __future__ import annotations

import csv
from pathlib import Path

from hd3_visualization_export import build_figure_metrics, generate_all_figures, write_rows_csv


def _write_synthetic_csvs(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True); metrics=[]; structural=[]
    for d in (10,20,50,100):
        run_class="stress" if d==100 else "canonical"
        for j,model_id in enumerate(("FC_HVI","DEEPSETS_HVI")):
            metrics.append({"dimension":d,"model_id":model_id,"run_class":run_class,"normalized_price_rmse":.001*(1+j)*(1+d/100),"delta_rmse":.002*(1+j),"trusted_pde_p95":.05*(1+j),"training_seconds":20+d/2+j})
            structural.append({"dimension":d,"model_id":model_id,"run_class":run_class,"permutation_consistency_rmse":(1e-3 if j==0 else 1e-8)*(1+d/100),"transverse_invariance_rmse":1e-4*(1+j)*(1+d/100)})
    write_rows_csv(data_dir/"_dimension_summary.csv",metrics); write_rows_csv(data_dir/"_structural_diagnostics.csv",structural)
    build_figure_metrics(dimension_summary_path=data_dir/"_dimension_summary.csv",structural_path=data_dir/"_structural_diagnostics.csv",output_path=data_dir/"figure_metrics.csv")
    slices=[]
    for d in (10,20,50,100):
        for s in (.5,1.,1.5):
            ref=max(1-s,0)+.02; slices.append({"dimension":d,"run_class":"stress" if d==100 else "canonical","tau":1.,"S":s,"cnpsor":ref,"crr":ref+1e-5,"fc_hvi":ref+1e-3,"deepsets_hvi":ref-5e-4,"fc_abs_error":1e-3,"deepsets_abs_error":5e-4,"crr_depth":1000,"reference_agreement_passed":True})
    write_rows_csv(data_dir/"diagonal_slices.csv",slices)
    write_rows_csv(data_dir/"reference_solver_agreement.csv",[{"dimension":d,"depth":1000,"selected_depth":True,"reference_agreement_passed":True} for d in (10,20,50,100)])
    for d in (10,20,50):
        rows=[]
        for tau in (.1,1.):
            for s in (.5,1.5): rows.append({"dimension":d,"tau":tau,"S":s,"cnpsor":.1,"fc_hvi":.101,"deepsets_hvi":.0995,"fc_abs_error":.001*(1+d/100),"deepsets_abs_error":.0005*(1+d/100)})
        write_rows_csv(data_dir/f"error_surface_{d}d.csv",rows)
    boundaries=[]
    for d in (10,20,50):
        for tau in (.1,1.): boundaries.append({"dimension":d,"run_class":"canonical","tau":tau,"cnpsor_boundary":.8,"fc_hvi_boundary":.75,"deepsets_hvi_boundary":.78,"exercise_gap_tolerance":1e-5})
    write_rows_csv(data_dir/"exercise_boundaries.csv",boundaries)


def test_build_figure_metrics_merges_scaling_and_structural_sources(tmp_path: Path):
    data=tmp_path/"data"; _write_synthetic_csvs(data); rows=list(csv.DictReader((data/"figure_metrics.csv").open()))
    assert len(rows)==8 and {"normalized_price_rmse","delta_rmse","trusted_pde_p95","training_seconds","permutation_consistency_rmse","transverse_invariance_rmse"} <= set(rows[0])
    assert {r["run_class"] for r in rows if r["dimension"]=="100"}=={"stress"}


def test_plotting_smoke_creates_all_declared_pngs(tmp_path: Path):
    data=tmp_path/"results"/"hd3_visualization"; figures=tmp_path/"figures"/"hd3"; _write_synthetic_csvs(data); generate_all_figures(data_dir=data,figure_dir=figures)
    expected={"hd3_scaling_summary.png","hd3_price_slices.png","hd3_error_heatmaps.png","hd3_structural_diagnostics.png","hd3_exercise_boundaries.png","hd3_100d_stress_slice.png"}
    assert expected=={p.name for p in figures.glob("*.png")}; assert all((figures/n).stat().st_size>0 for n in expected)
