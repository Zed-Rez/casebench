"""Offline tests for the CLI parser and report rendering (no API)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casebench import cli, runner  # noqa: E402


class CliParser(unittest.TestCase):
    def test_run_flags(self):
        args = cli.build_parser().parse_args(
            ["run", "--models", "anthropic:claude-haiku-4-5", "--judges",
             "anthropic:claude-opus-4-8", "anthropic:claude-sonnet-4-6",
             "--gen-samples", "5", "--judge-passes", "3", "--limit", "4"]
        )
        self.assertEqual(args.command, "run")
        self.assertEqual(args.gen_samples, 5)
        self.assertEqual(args.judge_passes, 3)
        self.assertEqual(len(args.judges), 2)

    def test_subcommands_exist(self):
        for cmd in ("cases", "ablate", "validate"):
            args = cli.build_parser().parse_args([cmd])
            self.assertEqual(args.command, cmd)


class ReportRendering(unittest.TestCase):
    REPORT = {
        "judge_protocol": "v2.0",
        "config": {"judges": ["claude-opus-4-8"], "gen_samples": 3, "judge_passes": 1,
                   "n_cases": 6, "n_ideas": 5},
        "quality_leaderboard": [
            {"model": "m1", "quality_score": 62.0, "quality_ci": [55.0, 69.0],
             "divergence_score": 30.0, "divergence_ci": [22.0, 38.0], "composite_score": 52.4,
             "mean_feasibility": 2.1, "mean_impact": 2.0, "mean_originality": 1.8,
             "mean_divergence": 0.9, "convergence_rate": 0.6, "invalid_rate": 0.05,
             "n_cases": 6, "n_ideas": 90},
            {"model": "m2", "quality_score": 58.0, "quality_ci": [50.0, 66.0],
             "divergence_score": 40.0, "divergence_ci": [32.0, 48.0], "composite_score": 52.6,
             "mean_feasibility": 2.0, "mean_impact": 1.9, "mean_originality": 1.7,
             "mean_divergence": 1.2, "convergence_rate": 0.45, "invalid_rate": 0.0,
             "n_cases": 6, "n_ideas": 90},
        ],
        "diversity_leaderboard": [
            {"model": "m2", "diversity_score": 40.0, "convergence_rate": 0.45},
            {"model": "m1", "diversity_score": 30.0, "convergence_rate": 0.6},
        ],
        "significance_quality": [
            {"a": "m1", "b": "m2", "mean_diff_quality": 4.0, "ci": [-3.0, 11.0],
             "significant": False, "p": 0.3, "p_floor": 0.03, "paired_diff_sd": 5.0}
        ],
        "interjudge_reliability": {"note": "single judge"},
        "novelty_validation": {"n": 60, "spearman_rho": 0.42, "interpretation": "ok"},
        "board_decorrelation": {"n_ideas": 60, "headline_originality_vs_diversity_pearson": 0.18,
                                "disclosed_originality_vs_refdivergence_pearson": 0.74,
                                "quality_vs_refdivergence_pearson": 0.12,
                                "realized_axis_std": {"feasibility": 0.5, "impact": 0.46,
                                                      "originality": 0.7, "divergence": 0.82}},
        "self_preference_audit": {"candidate_judge_overlap": [], "warning": None, "per_candidate": []},
        "judge_vs_human_gold": {"n": 0, "note": "no gold set"},
        "contradiction_rate": 0.02,
        "power": {"paired_diff_sd": 5.0, "minimum_detectable_effect_95_80": 5.7,
                  "basis": "paired per-case difference SD", "caveat": "coarse at small n", "note": "..."},
    }

    def test_renders_both_boards_and_ties(self):
        md = runner.render_leaderboard_md(self.REPORT)
        self.assertIn("Idea Quality", md)
        self.assertIn("Portfolio Diversity", md)
        self.assertIn("not significant", md)            # overlapping/insig => tie wording
        self.assertIn("minimum detectable effect", md.lower())
        self.assertIn("originality-vs-**diversity**", md)  # orthogonality surfaced
        self.assertIn("Realized axis discrimination", md)  # impact-variance transparency


if __name__ == "__main__":
    unittest.main(verbosity=2)
