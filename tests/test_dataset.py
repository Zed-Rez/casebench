"""Dataset-integrity tests for CASE-Bench (offline).

Production guards on data/cases.json: schema, reference-set granularity and
distinctness, and a prompt-leakage heuristic (the prompt must not name a
reference's core mechanism, or the novelty signal is primed). These run in CI
without any API key.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casebench import runner  # noqa: E402

ALLOWED_ARCHETYPES = {
    "margin-erosion", "demand-trough", "cold-start-liquidity", "churn-retention",
    "process-throughput", "adoption-diffusion", "trust-compliance", "waste-efficiency",
    "monetization-pricing", "capacity-utilization",
}
MIN_REFERENCES = 12
MIN_CASES = 12

_STOP = set(
    "the a an and or of to in for on with by at from as is are be service program model "
    "based into your you their our that this it its".split()
)


def _content_words(phrase: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in _STOP and len(w) > 3}


class DatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = runner.load_cases()

    def test_enough_cases(self):
        self.assertGreaterEqual(len(self.cases), MIN_CASES)

    def test_unique_case_ids(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), "case ids must be unique")

    def test_required_fields_present(self):
        for c in self.cases:
            for k in ("id", "title", "domain", "archetype", "context", "prompt", "reference_answers"):
                self.assertIn(k, c, f"{c.get('id')} missing {k}")
            self.assertTrue(c["prompt"].strip())
            self.assertTrue(c["context"].strip())

    def test_archetypes_valid(self):
        for c in self.cases:
            self.assertIn(c["archetype"], ALLOWED_ARCHETYPES, f"{c['id']} bad archetype {c['archetype']}")

    def test_reference_sets_well_formed(self):
        for c in self.cases:
            refs = c["reference_answers"]
            self.assertGreaterEqual(len(refs), MIN_REFERENCES, f"{c['id']} has too few references")
            rids = [r["id"] for r in refs]
            self.assertEqual(len(rids), len(set(rids)), f"{c['id']} duplicate ref ids")
            mechs = []
            for r in refs:
                for k in ("id", "mechanism", "idea", "scope_note"):
                    self.assertIn(k, r, f"{c['id']}/{r.get('id')} missing {k}")
                    self.assertTrue(str(r[k]).strip(), f"{c['id']}/{r.get('id')} empty {k}")
                mechs.append(r["mechanism"].strip().lower())
            self.assertEqual(len(mechs), len(set(mechs)),
                             f"{c['id']} has duplicate reference mechanisms")

    def test_archetype_diversity(self):
        # No single archetype should dominate the set (calibration / coverage).
        from collections import Counter
        counts = Counter(c["archetype"] for c in self.cases)
        self.assertLessEqual(max(counts.values()), len(self.cases) // 2 + 1,
                             f"archetype over-represented: {counts}")

    def test_prompts_do_not_leak_mechanisms(self):
        """Heuristic: a multi-word reference mechanism should not appear whole in the prompt.

        Requires >=2 shared content words: a single generic overlap (e.g. the
        mechanism 'Mix shift' vs. the problem statement's 'day shift') is noise,
        not a leaked solution.
        """
        leaks = []
        for c in self.cases:
            prompt_words = _content_words(c["prompt"])
            for r in c["reference_answers"]:
                mech_words = _content_words(r["mechanism"])
                if len(mech_words) >= 2 and mech_words.issubset(prompt_words):
                    leaks.append((c["id"], r["id"], r["mechanism"]))
        self.assertEqual(leaks, [], f"prompt leaks reference mechanisms: {leaks[:5]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
