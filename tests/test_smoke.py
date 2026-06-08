"""Minimal end-to-end smoke test for CASE-Bench v2 (one real pass).

Runs the full pipeline — generate -> panel-judge(1) -> consolidate -> score —
for a single cheap model on a single case against the live Anthropic API, and
asserts the result is well-formed under the v2 schema. Intentionally a single
pass; the offline invariants live in test_scoring.py.

    ANTHROPIC_API_KEY=... .venv/bin/python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casebench import generate, judge, runner, scoring, validation  # noqa: E402

CHEAP_MODEL = "anthropic:claude-haiku-4-5"


@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set")
class SmokeTest(unittest.TestCase):
    def test_single_pass_pipeline(self):
        case = runner.load_cases()[0]

        g = generate.generate_for_case(CHEAP_MODEL, case, n=generate.IDEAS_PER_CASE, temperature=1.0)
        self.assertEqual(len(g.ideas), generate.IDEAS_PER_CASE)

        verdicts = judge.judge_case(case, g.ideas, judge_model=judge.DEFAULT_JUDGE_MODEL)
        self.assertEqual(len(verdicts), len(g.ideas))
        for v in verdicts:
            for axis in ("feasibility", "impact", "originality", "divergence"):
                self.assertIn(getattr(v, axis), (0, 1, 2, 3))

        cs = scoring.score_case(case["id"], CHEAP_MODEL, verdicts)
        self.assertGreaterEqual(cs.quality_score, 0.0)
        self.assertLessEqual(cs.quality_score, 100.0)
        self.assertGreaterEqual(cs.divergence_score, 0.0)

        # divergence must not leak into quality: zero out divergence, quality unchanged
        zeroed = [judge.Verdict(**{**v.__dict__, "divergence": 0}) for v in verdicts]
        self.assertAlmostEqual(
            cs.quality_score, scoring.score_case(case["id"], CHEAP_MODEL, zeroed).quality_score,
            msg="quality score must be independent of divergence",
        )

        flags = validation.audit_contradictions(verdicts)  # may be empty; must not raise
        self.assertIsInstance(flags, list)

        print(f"\nSMOKE OK: {CHEAP_MODEL} on '{case['id']}' -> Quality {cs.quality_score:.1f}, "
              f"Divergence {cs.divergence_score:.1f}, feas {cs.mean_feasibility:.2f}, "
              f"impact {cs.mean_impact:.2f}, orig {cs.mean_originality:.2f}, "
              f"invalid {cs.n_invalid}, contradictions {len(flags)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
