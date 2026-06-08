"""Offline unit tests for CASE-Bench scoring, stats, novelty, and validation.

No API calls — these lock in the invariants the v1 review flagged as blockers:
  * a model dominating on the intrinsic axes cannot rank below one that doesn't
    (the v1 novelty multiplier could invert the ranking — C1),
  * incoherent (feasibility=0) ideas contribute zero (C2),
  * divergence/novelty does NOT enter the primary quality score (C1/C8).

Run: .venv/bin/python -m pytest tests/test_scoring.py
 or: .venv/bin/python tests/test_scoring.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casebench import novelty_baseline, scoring, stats, validation  # noqa: E402
from casebench.generate import Idea  # noqa: E402
from casebench.judge import Verdict  # noqa: E402


def V(index, feas, impact, orig, div, ref="r1", also=None, risk="some risk"):
    return Verdict(
        index=index, feasibility=feas, impact=impact, originality=orig, divergence=div,
        nearest_reference_id=ref, also_covers=also or [], failure_risk=risk,
        rationale="", judge_model="test",
    )


class ScoringInvariants(unittest.TestCase):
    def test_dominance_cannot_invert(self):
        # A is >= B on every intrinsic axis, idea-for-idea.
        a = [V(i, 3, 3, 3, 0) for i in range(5)]          # great ideas, all "known" (div 0)
        b = [V(i, 2, 1, 1, 3) for i in range(5)]          # weaker ideas, all "novel" (div 3)
        qa = scoring.score_case("c", "A", a).quality_score
        qb = scoring.score_case("c", "B", b).quality_score
        self.assertGreater(qa, qb, "high-merit known ideas must beat low-merit novel ideas")

    def test_divergence_does_not_affect_quality(self):
        base = [V(i, 2, 2, 2, 0) for i in range(5)]
        novel = [V(i, 2, 2, 2, 3) for i in range(5)]
        self.assertAlmostEqual(
            scoring.score_case("c", "m", base).quality_score,
            scoring.score_case("c", "m", novel).quality_score,
            msg="novelty/divergence must not leak into the primary quality score",
        )
        # but it MUST move the divergence board
        self.assertGreater(
            scoring.score_case("c", "m", novel).divergence_score,
            scoring.score_case("c", "m", base).divergence_score,
        )

    def test_incoherent_idea_is_gated_to_zero(self):
        junk = V(0, 0, 3, 3, 3)  # incoherent but "impactful/original/novel"
        self.assertEqual(scoring.idea_value(junk), 0.0)
        # a sound idea must outscore a quota of exotic junk
        sound = [V(i, 3, 3, 2, 0) for i in range(5)]
        junky = [V(i, 0, 3, 3, 3) for i in range(5)]
        self.assertGreater(
            scoring.score_case("c", "sound", sound).quality_score,
            scoring.score_case("c", "junk", junky).quality_score,
        )

    def test_feasibility_gate_levels(self):
        self.assertEqual(scoring.idea_value(V(0, 0, 3, 3, 0)), 0.0)
        self.assertEqual(scoring.idea_value(V(0, 1, 3, 3, 0)), 0.5)  # half gate
        self.assertEqual(scoring.idea_value(V(0, 2, 3, 3, 0)), 1.0)  # full
        self.assertEqual(scoring.idea_value(V(0, 3, 3, 3, 0)), 1.0)

    def test_under_delivery_cannot_inflate(self):
        # The under-delivery exploit (Round 2 blocker): returning only the best
        # idea must NOT beat that same idea inside the full requested slate.
        great = V(0, 3, 3, 3, 1)
        one_of_five = scoring.score_case("c", "m", [great], n_expected=5).quality_score
        full_slate = scoring.score_case(
            "c", "m", [V(i, 3, 3, 3, 1) for i in range(5)], n_expected=5).quality_score
        self.assertLess(one_of_five, full_slate)
        # and padding the slate with zeros for missing ideas actually bites:
        no_slate = scoring.score_case("c", "m", [great]).quality_score
        self.assertLess(one_of_five, no_slate)

    def test_divergence_is_feasibility_gated(self):
        # An incoherent (feasibility 0) but mechanism-distinct idea earns NO
        # divergence credit (Round 2 major: divergence loophole).
        self.assertEqual(scoring.idea_divergence_value(V(0, 0, 3, 3, 3)), 0.0)
        self.assertGreater(scoring.idea_divergence_value(V(0, 2, 3, 3, 3)), 0.0)

    def test_float_consolidation_not_pinned_to_grid(self):
        # Panel disagreement must survive (Round 2 blocker: mean-then-round pinned
        # peak_value at 0.667). A 3-vs-2 impact split should yield 2.5, not 2.
        merged = stats.consolidate_idea([V(0, 3, 3, 3, 1), V(0, 3, 2, 3, 1)])
        self.assertAlmostEqual(merged.impact, 2.5)
        # and that lifts idea_value above the all-2 grid point
        self.assertGreater(scoring.idea_value(merged), scoring.idea_value(V(0, 2, 2, 2, 1)))

    def test_topk_rewards_having_strong_ideas(self):
        # padding mediocre novel ideas should not beat a few excellent ones
        few_great = [V(0, 3, 3, 3, 0), V(1, 3, 3, 3, 0)] + [V(i, 1, 1, 1, 3) for i in range(2, 5)]
        all_mediocre = [V(i, 2, 1, 1, 3) for i in range(5)]
        self.assertGreater(
            scoring.score_case("c", "great", few_great).quality_score,
            scoring.score_case("c", "mid", all_mediocre).quality_score,
        )


class DiversityTests(unittest.TestCase):
    def test_coverage_rewards_distinct_mechanisms(self):
        clones = [V(i, 2, 2, 2, 0, ref="r3") for i in range(5)]   # 5 ideas, same mechanism
        spread = [V(0, 2, 2, 2, 0, ref="r1"), V(1, 2, 2, 2, 0, ref="r3"),
                  V(2, 2, 2, 2, 0, ref="r7"), V(3, 2, 2, 2, 3, ref=None),
                  V(4, 2, 2, 2, 3, ref=None)]
        self.assertLess(scoring.slate_coverage(clones, 5), scoring.slate_coverage(spread, 5))
        self.assertAlmostEqual(scoring.slate_coverage(clones, 5), 0.2)

    def test_intra_slate_dissimilarity(self):
        same = [Idea("A", "discount slow hours"), Idea("B", "discount slow hours")]
        diff = [Idea("A", "discount slow hours to pull traffic"),
                Idea("B", "rent idle floor space as paid desks for remote workers")]
        self.assertLess(novelty_baseline.intra_slate_dissimilarity(same),
                        novelty_baseline.intra_slate_dissimilarity(diff))


class StatsTests(unittest.TestCase):
    def test_bootstrap_ci_contains_mean(self):
        vals = [40, 55, 60, 50, 70, 45, 65, 52]
        lo, hi = stats.bootstrap_ci(vals)
        mean = sum(vals) / len(vals)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_identical_models_not_significant(self):
        v = [50, 60, 55, 45, 65]
        res = stats.paired_diff_test(v, v)
        self.assertFalse(res["significant"])
        self.assertEqual(res["mean_diff"], 0.0)

    def test_clear_difference_is_significant(self):
        a = [80, 82, 79, 85, 81, 83, 80, 84]
        b = [50, 52, 49, 55, 51, 53, 50, 54]
        res = stats.paired_diff_test(a, b)
        self.assertTrue(res["significant"])

    def test_spearman_monotonic(self):
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_consolidate_averages(self):
        vs = [V(0, 3, 3, 2, 1, ref="r1"), V(0, 1, 1, 2, 3, ref="r1"), V(0, 2, 2, 2, 2, ref="r2")]
        c = stats.consolidate_idea(vs)
        self.assertAlmostEqual(c.feasibility, 2.0)  # mean(3,1,2), float (not rounded)
        self.assertAlmostEqual(c.impact, 2.0)
        self.assertEqual(c.nearest_reference_id, "r1")  # majority

    def test_paired_diff_sd_smaller_than_pooled(self):
        a = [80, 82, 79, 85, 81, 83]
        b = [50, 52, 49, 55, 51, 53]  # nearly constant +30 offset
        paired = stats.paired_diff_sd(a, b)
        pooled = stats.statistics.pstdev(a + b)
        self.assertLess(paired, pooled)  # paired removes the between-model effect

    def test_interjudge_reliability_perfect(self):
        pairs = [(V(i, 2, 2, 2, 1), V(i, 2, 2, 2, 1)) for i in range(6)]
        rel = stats.interjudge_reliability(pairs)
        self.assertEqual(rel["axes"]["impact"]["exact_agreement"], 1.0)


class NoveltyBaselineTests(unittest.TestCase):
    CASE = {
        "reference_answers": [
            {"id": "r1", "mechanism": "loyalty discount", "idea": "give members a discount during slow hours", "scope_note": "price cuts to pull traffic"},
            {"id": "r2", "mechanism": "co-working space", "idea": "rent idle floor space as paid desks", "scope_note": "monetise unused space"},
        ]
    }

    def test_near_reference_low_divergence(self):
        ideas = [Idea("Discount", "give members a discount during slow hours to pull traffic")]
        d = novelty_baseline.lexical_divergence(self.CASE, ideas)[0]
        far = novelty_baseline.lexical_divergence(
            self.CASE, [Idea("Quantum", "deploy a satellite uplink for orbital telemetry physics")]
        )[0]
        self.assertLess(d, far, "an idea echoing a reference should be lexically closer")


class ValidationTests(unittest.TestCase):
    def test_contradiction_flags(self):
        bad = [
            V(0, 3, 3, 3, 0),               # original=3 but divergence=0
            V(1, 0, 3, 2, 3),               # infeasible but impactful
            V(2, 2, 3, 2, 2, risk=""),      # impact=3 with no failure risk
        ]
        flags = validation.audit_contradictions(bad)
        types = {f["type"] for f in flags}
        self.assertIn("original_yet_identical", types)
        self.assertIn("impactful_yet_infeasible", types)
        self.assertIn("impact3_no_risk", types)

    def test_clean_verdicts_no_flags(self):
        good = [V(0, 2, 2, 2, 2), V(1, 3, 2, 1, 1)]
        self.assertEqual(validation.audit_contradictions(good), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
