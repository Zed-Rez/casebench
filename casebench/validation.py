"""Validity checks for CASE-Bench verdicts.

Two kinds of check:

1. **Internal-contradiction audit** — flags verdicts whose numbers contradict
   each other or the rationale (e.g. an idea rated maximally original yet scored
   as identical to a reference, or rated impactful while judged infeasible).
   These are cheap, automatic sanity checks on the judge's self-consistency.

2. **Judge-vs-human gold agreement** — loads a small hand-labelled gold set
   (``data/gold/*.json``) and reports how well the LLM judge agrees with human
   ratings on the same ideas. The seed gold set shipped with the repo is small
   and illustrative; the loader + metrics are the durable contribution, and the
   protocol for scaling it to a proper annotator pool is documented in the
   README. Construct validity is *exercised* here (illustratively), not asserted —
   the shipped single-rater seed is a harness smoke test, not validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .judge import Verdict
from .stats import AXES, _pearson

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"


def idea_fingerprint(case_id: str, idea_text: str) -> str:
    """Stable id for a specific generated idea, so gold labels can target it."""
    return hashlib.sha1(f"{case_id}|{idea_text}".encode()).hexdigest()[:16]


# --- internal contradictions ----------------------------------------------


def audit_contradictions(verdicts: list[Verdict]) -> list[dict]:
    """Return a list of contradiction flags (empty == clean)."""
    flags = []
    for v in verdicts:
        if v.originality >= 3 and v.divergence == 0:
            flags.append(
                {"index": v.index, "type": "original_yet_identical",
                 "detail": "originality=3 but divergence=0 (called a paraphrase of a reference)"}
            )
        if v.feasibility == 0 and v.impact >= 2:
            flags.append(
                {"index": v.index, "type": "impactful_yet_infeasible",
                 "detail": "feasibility=0 (incoherent) but impact>=2"}
            )
        if v.impact >= 3 and not v.failure_risk.strip():
            flags.append(
                {"index": v.index, "type": "impact3_no_risk",
                 "detail": "impact=3 requires a named failure risk; none given"}
            )
        if v.divergence == 0 and v.nearest_reference_id is None:
            flags.append(
                {"index": v.index, "type": "identical_to_nothing",
                 "detail": "divergence=0 (same as a reference) but nearest_reference_id is null"}
            )
        if v.divergence == 3 and v.also_covers:
            flags.append(
                {"index": v.index, "type": "distinct_yet_covers",
                 "detail": "divergence=3 (distinct mechanism) but also_covers lists references"}
            )
    return flags


def contradiction_rate(verdicts: list[Verdict]) -> float:
    if not verdicts:
        return 0.0
    flagged = {f["index"] for f in audit_contradictions(verdicts)}
    return len(flagged) / len(verdicts)


# --- human gold ------------------------------------------------------------


def load_gold(gold_dir: Path = GOLD_DIR) -> dict[str, dict]:
    """Load gold labels keyed by idea fingerprint.

    Each gold file is ``{"case_id": ..., "labels": [{"idea_text": ...,
    "feasibility": .., "impact": .., "originality": .., "divergence": ..}]}``.
    """
    gold: dict[str, dict] = {}
    if not gold_dir.exists():
        return gold
    for path in sorted(gold_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        cid = blob["case_id"]
        for label in blob.get("labels", []):
            fp = idea_fingerprint(cid, label["idea_text"])
            gold[fp] = label
    return gold


def judge_vs_gold(
    case_id: str,
    ideas_texts: list[str],
    verdicts: list[Verdict],
    gold: dict[str, dict],
) -> list[tuple[dict, Verdict]]:
    """Match verdicts to gold labels for the same ideas. Returns matched pairs."""
    pairs = []
    for v in verdicts:
        # Match by the verdict's own index, not positional zip: if the judge
        # omitted a verdict the consolidated list shifts and a zip would pair
        # fingerprints with the wrong idea text.
        if 0 <= v.index < len(ideas_texts):
            fp = idea_fingerprint(case_id, ideas_texts[v.index])
            if fp in gold:
                pairs.append((gold[fp], v))
    return pairs


def gold_agreement(pairs: list[tuple[dict, Verdict]]) -> dict:
    """Agreement between human gold and judge over matched ideas."""
    if not pairs:
        return {"n": 0, "note": "no gold labels matched the evaluated ideas"}
    out: dict = {"n": len(pairs), "axes": {}}
    for axis in AXES:
        h = [float(g[axis]) for g, _ in pairs if axis in g]
        m = [float(getattr(v, axis)) for g, v in pairs if axis in g]
        if not h:
            continue
        exact = sum(1 for a, b in zip(h, m) if a == b) / len(h)
        within1 = sum(1 for a, b in zip(h, m) if abs(a - b) <= 1) / len(h)
        r = _pearson(h, m)
        out["axes"][axis] = {
            "exact_agreement": round(exact, 3),
            "within1_agreement": round(within1, 3),
            "pearson_r": round(r, 3) if r is not None else None,
        }
    return out
