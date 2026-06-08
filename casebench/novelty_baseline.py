"""Reference-free novelty baseline for CASE-Bench.

A purely lexical (TF-IDF cosine) measure of how far a candidate idea sits from
the nearest reference answer, computed offline with no model and no embedding
API. It is deliberately crude — it captures surface/term overlap, not deep
semantics — and exists for one purpose: to provide an *independent* signal that
the judge's ``divergence`` rating can be validated against. If the judge's
divergence and this lexical divergence are positively rank-correlated, the
judge's novelty calls are at least not arbitrary; if they are uncorrelated, that
is itself a finding worth surfacing.

(A sentence-embedding distance would be a stronger baseline; this avoids adding a
dependency or an embedding key. The limitation is documented in the README.)
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .generate import Idea

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an and or of to in for on with by at from as is are be this that it "
    "its their our your you we they he she them his her into over under per via "
    "not no can could would should will may might more most less least than then "
    "so such these those each any all both also out up down off about across".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2]


def _tfidf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}
    vecs = []
    for d in docs:
        tf = Counter(d)
        length = len(d) or 1
        vecs.append({t: (c / length) * idf[t] for t, c in tf.items()})
    return vecs


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def lexical_divergence(case: dict, ideas: list[Idea]) -> list[float]:
    """Per-idea lexical divergence in [0, 1] (1 = far from every reference).

    Vectors are built over the joint corpus of references + ideas so IDF is
    shared; each idea's score is ``1 - max cosine to any reference``.
    """
    ref_texts = [
        f"{r.get('mechanism','')} {r['idea']} {r.get('scope_note','')}"
        for r in case["reference_answers"]
    ]
    idea_texts = [idea.as_text() for idea in ideas]
    docs = [_tokens(t) for t in ref_texts + idea_texts]
    vecs = _tfidf_vectors(docs)
    n_ref = len(ref_texts)
    ref_vecs = vecs[:n_ref]
    idea_vecs = vecs[n_ref:]

    out = []
    for iv in idea_vecs:
        max_sim = max((_cosine(iv, rv) for rv in ref_vecs), default=0.0)
        out.append(1.0 - max_sim)
    return out


def intra_slate_dissimilarity(ideas: list[Idea]) -> float:
    """Mean pairwise lexical dissimilarity (1 - cosine) AMONG a model's own ideas.

    A *portfolio* property: high = the model proposed genuinely different ideas;
    low = five variations on one theme. Reference-free, and — unlike per-idea
    divergence-from-references — not just a restatement of per-idea originality.
    """
    if len(ideas) < 2:
        return 0.0
    docs = [_tokens(i.as_text()) for i in ideas]
    vecs = _tfidf_vectors(docs)
    sims = []
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            sims.append(1.0 - _cosine(vecs[a], vecs[b]))
    return sum(sims) / len(sims) if sims else 0.0


def divergence_validation(
    judge_divergence_0_3: list[int],
    lexical_div_0_1: list[float],
) -> dict:
    """Rank-correlate judge divergence (0-3) with lexical divergence (0-1)."""
    from .stats import spearman

    rho = spearman([float(x) for x in judge_divergence_0_3], list(lexical_div_0_1))
    return {
        "n": len(judge_divergence_0_3),
        "spearman_rho": round(rho, 3) if rho is not None else None,
        "interpretation": (
            "judge novelty calls track an independent lexical signal"
            if (rho or 0) >= 0.3
            else "weak/no agreement with lexical baseline — inspect"
        ),
    }
