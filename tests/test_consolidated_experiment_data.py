import csv
import gzip
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path


class ConsolidatedExperimentDataContract(unittest.TestCase):
    def test_public_api_and_rmse_definition(self):
        from analysis.consolidated_experiment_data import price_rmse
        self.assertAlmostEqual(price_rmse([1.0, 2.0], [1.0, 4.0]), math.sqrt(2.0))
        with self.assertRaises(ValueError): price_rmse([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError): price_rmse([], [])

    def test_output_contract_has_required_tables_and_metric_columns(self):
        from analysis.consolidated_experiment_data import write_empty_output_contract
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); write_empty_output_contract(out)
            required = {"source_artifact_inventory.csv","provenance.csv","source_records.csv","array_artifacts.csv","experiment_runs.csv","price_metrics.csv","greek_metrics.csv","boundary_metrics.csv","free_boundary_points.csv","price_slices.csv","structural_metrics.csv","compute_metrics.csv","classical_solver_metrics.csv","binomial_convergence.csv","ruixi_arm_metrics.csv","hd1_metrics.csv","hd2_candidate_metrics.csv","hd3_metrics.csv","campaign_method_results.csv","experiment_gates.csv","environment_runs.csv","surface_artifacts.csv","dirk_common_reference_metrics.csv","manifest.json","README.md"}
            self.assertTrue(required.issubset({p.name for p in out.iterdir()}))
            with (out / "source_records.csv").open(newline="") as fh: source_header = next(csv.reader(fh))
            for name in ("repo","branch","path","record_kind","record_index","field","value","value_type"): self.assertIn(name, source_header)
            with (out / "array_artifacts.csv").open(newline="") as fh: array_header = next(csv.reader(fh))
            for name in ("repo","branch","path","array_name","shape","dtype","csv_gz_path","content_sha256"): self.assertIn(name, array_header)
            with (out / "dirk_common_reference_metrics.csv").open(newline="") as fh: header = next(csv.reader(fh))
            for name in ("internal_cnpsor_price_rmse","dirk_price_rmse_hd_domain","dirk_price_rmse_shared_domain","reference_type","metric_scope","s_min","s_max","tau_min","tau_max","grid_points","interpolation_method","source_repo","source_branch","source_commit","reference_source","reference_commit","reference_artifact_sha256"): self.assertIn(name, header)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["headline_cross_model_metric"], "dirk_price_rmse_shared_domain")
            self.assertEqual(manifest["headline_hd_scaling_metric"], "dirk_price_rmse_hd_domain")

    def test_shared_domain_is_frozen(self):
        from analysis.consolidated_experiment_data import SHARED_DOMAIN, HD_DOMAIN
        self.assertEqual(SHARED_DOMAIN, (0.5, 1.5, 0.01, 1.0)); self.assertEqual(HD_DOMAIN, (0.2, 1.5, 0.01, 1.0))

    def test_rectilinear_interpolation_is_exact_for_affine_surface_and_refuses_extrapolation(self):
        import numpy as np
        from analysis.consolidated_experiment_data import rectilinear_interpolate
        s=np.array([0.,1.,2.]); tau=np.array([0.,1.]); values=tau[:,None]*3+s[None,:]*2+5
        target_s=np.array([.25,1.5]); target_tau=np.array([.2,.8])
        got=rectilinear_interpolate(s,tau,values,target_s,target_tau); expected=target_tau[:,None]*3+target_s[None,:]*2+5
        np.testing.assert_allclose(got,expected,rtol=0,atol=1e-12)
        with self.assertRaises(ValueError): rectilinear_interpolate(s,tau,values,np.array([-.1]),target_tau)
        with self.assertRaises(ValueError): rectilinear_interpolate(s,tau,values,target_s,np.array([1.1]))

    def test_json_flattening_and_artifact_classification_are_lossless_for_scalars(self):
        from analysis.consolidated_experiment_data import classify_artifact_path, flatten_json_scalars
        flattened=dict(flatten_json_scalars({"a":1,"b":{"c":True,"d":[2.5,"x"]},"e":None}))
        self.assertEqual(flattened["a"],1); self.assertEqual(flattened["b.c"],True); self.assertEqual(flattened["b.d[0]"],2.5); self.assertEqual(flattened["b.d[1]"],"x"); self.assertIsNone(flattened["e"])
        self.assertEqual(classify_artifact_path("results/x/run_metrics.csv"),"metrics_csv"); self.assertEqual(classify_artifact_path("results/x/surface.npz"),"numeric_array"); self.assertEqual(classify_artifact_path("results/x/checkpoints/model.pt"),"checkpoint")

    def test_branch_inventory_deduplicates_blobs_but_preserves_occurrences(self):
        from analysis.consolidated_experiment_data import inventory_repository
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td)/"repo"; repo.mkdir()
            subprocess.run(["git","init","-b","main"],cwd=repo,check=True,stdout=subprocess.DEVNULL)
            subprocess.run(["git","config","user.email","test@example.com"],cwd=repo,check=True); subprocess.run(["git","config","user.name","Test"],cwd=repo,check=True)
            (repo/"results").mkdir(); (repo/"results"/"a.json").write_text('{"x":1}\n')
            subprocess.run(["git","add","."],cwd=repo,check=True); subprocess.run(["git","commit","-m","a"],cwd=repo,check=True,stdout=subprocess.DEVNULL)
            subprocess.run(["git","switch","-c","extra"],cwd=repo,check=True,stdout=subprocess.DEVNULL)
            (repo/"results"/"b.csv").write_text("v\n2\n")
            subprocess.run(["git","add","."],cwd=repo,check=True); subprocess.run(["git","commit","-m","b"],cwd=repo,check=True,stdout=subprocess.DEVNULL)
            result=inventory_repository(repo,"example/repo")
            self.assertEqual(result["branch_count"],2)
            self.assertEqual(len(result["artifacts"]),2)
            self.assertEqual(len(result["provenance"]),3)
            a=[r for r in result["provenance"] if r["path"]=="results/a.json"]
            self.assertEqual(len(a),2); self.assertEqual(len({r["git_blob_sha"] for r in a}),1)

    def test_surface_normalization_orients_value_grid_as_tau_by_s(self):
        import numpy as np
        from analysis.consolidated_experiment_data import normalize_surface_arrays
        s=np.array([.5,1.,1.5]); tau=np.array([.1,.5]); v=(tau[:,None]+s[None,:]).T
        ns,nt,nv=normalize_surface_arrays({"S_over_K":s,"tau_over_T":tau,"V":v})
        np.testing.assert_allclose(ns,s); np.testing.assert_allclose(nt,tau); np.testing.assert_allclose(nv,tau[:,None]+s[None,:])

    def test_surface_normalization_reconstructs_flattened_pointwise_mesh(self):
        import numpy as np
        from analysis.consolidated_experiment_data import normalize_surface_arrays
        s_axis=np.array([.2,.5,1.0,1.5]); tau_axis=np.array([.01,.5,1.0])
        ss,tt=np.meshgrid(s_axis,tau_axis,indexing="xy")
        vv=2.0*ss+3.0*tt
        ns,nt,nv=normalize_surface_arrays({"G_over_K":ss.ravel(),"tau_over_T":tt.ravel(),"V":vv.ravel()})
        np.testing.assert_allclose(ns,s_axis); np.testing.assert_allclose(nt,tau_axis); np.testing.assert_allclose(nv,2.0*s_axis[None,:]+3.0*tau_axis[:,None])

    def test_oversized_csv_is_losslessly_chunked_and_catalogued(self):
        from analysis.build_consolidated_experiment_data import _chunk_oversized_csvs
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            source=out/"big.csv"
            original=[{"id":str(i),"payload":"x"*(40+i)} for i in range(7)]
            with source.open("w",newline="",encoding="utf-8") as fh:
                writer=csv.DictWriter(fh,fieldnames=["id","payload"]); writer.writeheader(); writer.writerows(original)
            catalog=_chunk_oversized_csvs(out,max_bytes=100,max_rows_per_chunk=2)
            self.assertFalse(source.exists())
            self.assertTrue((out/"table_chunks.csv").is_file())
            self.assertEqual(sum(int(row["row_count"]) for row in catalog if row["logical_table"]=="big.csv"),len(original))
            rebuilt=[]
            for row in catalog:
                if row["logical_table"]!="big.csv": continue
                chunk=out/row["chunk_path"]
                self.assertLess(chunk.stat().st_size,100_000_000)
                self.assertEqual(hashlib.sha256(chunk.read_bytes()).hexdigest(),row["sha256"])
                with gzip.open(chunk,"rt",newline="",encoding="utf-8") as fh:
                    rebuilt.extend(csv.DictReader(fh))
            self.assertEqual(rebuilt,original)


if __name__ == "__main__": unittest.main()
