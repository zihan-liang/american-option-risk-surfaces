import importlib.util
import inspect
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch
import uuid

import yaml

from plotting import report_visuals, surfaces


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage4 = _load_module(
    "stage4_convergence_visual_polish",
    "experiments/03_convergence_domain_runtime.py",
)
enrichment = _load_module(
    "report_visualization_enrichment_polish",
    "experiments/19_report_visualization_enrichment.py",
)


class VisualLayoutPolishTest(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path("tmp") / f"test_visual_layout_polish_{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_surface_helpers_accept_explicit_professional_labels(self):
        curve_parameters = inspect.signature(surfaces.save_curve).parameters
        surface_parameters = inspect.signature(surfaces.save_price_surface).parameters

        self.assertIn("xlabel", curve_parameters)
        self.assertIn("xlabel", surface_parameters)
        self.assertIn("ylabel", surface_parameters)
        self.assertIn("colorbar_label", surface_parameters)

    def test_display_labels_use_aliases_and_readable_regime_parameters(self):
        aliases = {
            "positive_premium_mlp": "Positive-premium MLP",
            "price_over_K": r"$U/K$",
        }

        self.assertEqual(
            report_visuals.format_display_label("positive_premium_mlp", aliases),
            "Positive-premium MLP",
        )
        self.assertEqual(
            report_visuals.format_display_label("price_over_K", aliases),
            r"$U/K$",
        )
        self.assertEqual(
            report_visuals.format_display_label(
                "call_T0p25_sig0p4_r0p05_q0p06",
                {},
            ),
            r"Call: $T=0.25$, $\sigma=0.4$, $r=0.05$, $q=0.06$",
        )

    def test_bar_helper_uses_horizontal_orientation(self):
        figure = MagicMock()
        axis = MagicMock()
        with (
            patch.object(report_visuals.plt, "subplots", return_value=(figure, axis)),
            patch.object(report_visuals, "_save"),
        ):
            report_visuals._bar(
                "ignored.pdf",
                ["Long label A", "Long label B"],
                [1.0, 2.0],
                "Readable comparison",
                "RMSE",
                orientation="horizontal",
            )

        axis.barh.assert_called_once()
        axis.bar.assert_not_called()

    def test_report_visuals_use_300_dpi_export_policy(self):
        self.assertEqual(report_visuals.DEFAULT_REPORT_DPI, 300)
        report_visuals.apply_report_style()
        self.assertEqual(report_visuals.plt.rcParams["figure.dpi"], 300)
        self.assertEqual(report_visuals.plt.rcParams["savefig.dpi"], 300)

    def test_save_uses_explicit_300_dpi(self):
        figure = MagicMock()
        with (
            patch.object(report_visuals.Path, "mkdir"),
            patch.object(report_visuals.plt, "close"),
        ):
            report_visuals._save(figure, self.temp_root / "ignored.png")

        figure.savefig.assert_called_once()
        self.assertEqual(
            figure.savefig.call_args.kwargs["dpi"],
            report_visuals.DEFAULT_REPORT_DPI,
        )

    def test_enrichment_groups_consecutive_figures_without_forced_page_breaks(self):
        root = self.temp_root
        config = root / "config.yaml"
        output_dir = root / "figures"
        block_dir = root / "blocks"
        config.write_text(
            yaml.safe_dump(
                {
                    "output_dir": str(output_dir),
                    "latex_block_dir": str(block_dir),
                    "reports": [
                        {
                            "path": "reports/final_report.tex",
                            "figures": [
                                {
                                    "figure_id": "first",
                                    "kind": "workflow",
                                    "title": "First",
                                    "caption": "First panel.",
                                    "panel_group": "pair",
                                },
                                {
                                    "figure_id": "second",
                                    "kind": "workflow",
                                    "title": "Second",
                                    "caption": "Second panel.",
                                    "panel_group": "pair",
                                },
                                {
                                    "figure_id": "third",
                                    "kind": "payoff",
                                    "title": "Third",
                                    "caption": "Full-width figure.",
                                },
                            ],
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        enrichment.run(str(config))
        block = (block_dir / "final_report_visuals.tex").read_text(encoding="utf-8")

        self.assertEqual(block.count(r"\begin{subfigure}"), 2)
        self.assertEqual(block.count(r"\begin{figure}[!htbp]"), 2)
        self.assertNotIn(r"\clearpage", block)
        self.assertIn(r"\FloatBarrier", block)

    def test_stage4_figures_only_uses_existing_artifacts(self):
        figure_root = self.temp_root / "figures"
        with patch.object(stage4, "solve_option", side_effect=AssertionError("solver called")):
            summary = stage4.regenerate_figures(
                output_root="results/experiment_03_convergence_domain_runtime",
                figure_root=figure_root,
            )

        expected = figure_root / "put" / "price_error_vs_grid.png"
        self.assertTrue(expected.exists())
        self.assertEqual(summary["mode"], "figures_only")
        self.assertEqual(summary["cases"], 2)


if __name__ == "__main__":
    unittest.main()
