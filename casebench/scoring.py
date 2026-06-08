"""Scoring logic for CASE-Bench (v2).

v1 collapsed everything into one number dominated by a binary 0.5/1.0 novelty
multiplier, which (a) ranked the model the judge rated *worse* on quality and
creativity *above* the better one, and (b) let incoherent-but-exotic ideas
outscore sound ones. v2 fixes this by separating two questions that were
conflated:

PRIMARY — "idea quality" (decision: which model produces better solutions?)
    Built only from the intrinsic, feasibility-gated axes. Novelty/divergence is
    deliberately NOT in this score, so a model that dominates on feasibility,
    impact, and originality can never rank below one that doesn't.

        feas_gate(f)  = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0}   # gates out junk
        value_i       = feas_gate(feasibility) * (impact/3 + originality/3) / 2
        quality_score = 100 * (0.6 * mean(value) + 0.4 * mean(top-3 value))

    The top-k term rewards *having* strong ideas and dampens the incentive to pad
    the quota with mediocre filler.

DIAGNOSTIC — "divergence" (decision: which model explores beyond the playbook?)
    Reported separately and explicitly labelled as NOT a quality measure.

        divergence_score = 100 * mean(divergence / 3)
        convergence_rate = fraction of ideas at divergence <= 1 (a known move)

COMPOSITE — a clearly-secondary convenience: 0.7*quality + 0.3*divergence,
    additive and bounded (never a multiplier).

All sub-metrics are reported beside every headline so a single number never hides
the story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import Verdict

TOPK = 3
MEAN_WEIGHT = 0.6
PEAK_WEIGHT = 0.4
COMPOSITE_QUALITY_WEIGHT = 0.7
COMPOSITE_DIVERGENCE_WEIGHT = 0.3


def feas_gate(f: float) -> float:
    """Feasibility multiplier, piecewise-linear so it works on *consolidated*
    float scores (panel averages), not just integer judge outputs.

    Anchored at the integer rubric points: gate(0)=0, gate(1)=0.5, gate(>=2)=1.0.
    Incoherent ideas (feasibility 0) are gated to zero on every board.
    """
    f = max(0.0, min(3.0, f))
    if f <= 1.0:
        return 0.5 * f                 # 0 -> 0.0, 1 -> 0.5
    if f <= 2.0:
        return 0.5 + 0.5 * (f - 1.0)   # 1 -> 0.5, 2 -> 1.0
    return 1.0                         # >= 2 -> 1.0


def _norm(score_0_3: float) -> float:
    return max(0.0, min(3.0, score_0_3)) / 3.0


def idea_value(v: Verdict) -> float:
    """Feasibility-gated intrinsic merit of one idea, in [0, 1]. No novelty term."""
    return feas_gate(v.feasibility) * (_norm(v.impact) + _norm(v.originality)) / 2.0


def idea_divergence_value(v: Verdict) -> float:
    """Divergence credit in [0, 1], also feasibility-gated so an incoherent idea
    earns NO divergence credit (closes the 'exotic-but-infeasible' loophole on the
    divergence board, not just on the quality board)."""
    if feas_gate(v.feasibility) <= 0.0:
        return 0.0
    return _norm(v.divergence)


def slate_coverage(verdicts: list[Verdict], slate_size: int) -> float:
    """Fraction of the slate that lands on *distinct* mechanisms — a portfolio
    breadth measure. Ideas that restate a known move collapse onto that move's
    bucket; genuinely-divergent ideas each count as their own exploration. A slate
    of five clones of one mechanism scores ~1/n; one spanning five mechanisms ~1.
    """
    if slate_size <= 0:
        return 0.0
    buckets: set[str] = set()
    for i, v in enumerate(verdicts):
        if v.divergence <= 1 and v.nearest_reference_id:
            buckets.add(v.nearest_reference_id)      # a known move — bucket by mechanism
        else:
            buckets.add(f"novel#{i}")                # a distinct exploration
    return len(buckets) / slate_size


def slate_diversity_score(coverage: float, intra_dissimilarity: float) -> float:
    """Portfolio-exploration diagnostic in [0, 100]: half mechanism-coverage,
    half internal (idea-to-idea) dissimilarity. Distinct from per-idea
    originality, which feeds the Quality board."""
    return 100.0 * 0.5 * (coverage + intra_dissimilarity)


def _topk_mean(values: list[float], k: int = TOPK) -> float:
    if not values:
        return 0.0
    top = sorted(values, reverse=True)[: min(k, len(values))]
    return sum(top) / len(top)


@dataclass
class CaseScore:
    case_id: str
    model: str
    quality_score: float          # PRIMARY (0-100)
    divergence_score: float       # DIAGNOSTIC (0-100)
    composite_score: float        # secondary convenience (0-100)
    n_ideas: int                  # ideas actually returned
    slate_size: int               # denominator used for scoring (requested quota)
    n_missing: int                # quota slots the model did not fill
    n_invalid: int                # feasibility ~ 0
    n_known_moves: int            # divergence <= 1
    mean_value: float             # 0-1 (over the slate, missing = 0)
    peak_value: float             # 0-1, best single idea
    convergence_rate: float       # fraction of known moves (over returned ideas)
    mean_feasibility: float
    mean_impact: float
    mean_originality: float
    mean_divergence: float
    headroom: float               # 1 - mean_value, a saturation diagnostic
    verdicts: list[Verdict] = field(default_factory=list)


def score_case(
    case_id: str, model: str, verdicts: list[Verdict], n_expected: int | None = None
) -> CaseScore:
    n_actual = len(verdicts)
    # Score against a FIXED slate = the requested quota, padding unfilled slots
    # with value 0. Otherwise a model can inflate its mean by returning only its
    # single best idea (the under-delivery exploit).
    slate = max(n_expected or n_actual, n_actual)
    if slate == 0:
        # Keyword args so the field alignment can't silently drift.
        return CaseScore(
            case_id=case_id, model=model, quality_score=0.0, divergence_score=0.0,
            composite_score=0.0, n_ideas=0, slate_size=0, n_missing=0, n_invalid=0,
            n_known_moves=0, mean_value=0.0, peak_value=0.0, convergence_rate=0.0,
            mean_feasibility=0.0, mean_impact=0.0, mean_originality=0.0,
            mean_divergence=0.0, headroom=1.0, verdicts=[])

    values = [idea_value(v) for v in verdicts] + [0.0] * (slate - n_actual)
    div_values = [idea_divergence_value(v) for v in verdicts] + [0.0] * (slate - n_actual)

    mean_value = sum(values) / slate
    quality = 100.0 * (MEAN_WEIGHT * mean_value + PEAK_WEIGHT * _topk_mean(values))
    divergence = 100.0 * sum(div_values) / slate
    composite = COMPOSITE_QUALITY_WEIGHT * quality + COMPOSITE_DIVERGENCE_WEIGHT * divergence

    den = n_actual or 1
    return CaseScore(
        case_id=case_id,
        model=model,
        quality_score=quality,
        divergence_score=divergence,
        composite_score=composite,
        n_ideas=n_actual,
        slate_size=slate,
        n_missing=slate - n_actual,
        n_invalid=sum(1 for v in verdicts if v.is_invalid),
        n_known_moves=sum(1 for v in verdicts if v.is_known_move),
        mean_value=mean_value,
        peak_value=max(values),
        convergence_rate=sum(1 for v in verdicts if v.is_known_move) / den,
        mean_feasibility=sum(v.feasibility for v in verdicts) / den,
        mean_impact=sum(v.impact for v in verdicts) / den,
        mean_originality=sum(v.originality for v in verdicts) / den,
        mean_divergence=sum(v.divergence for v in verdicts) / den,
        headroom=1.0 - mean_value,
        verdicts=verdicts,
    )


@dataclass
class ModelScore:
    model: str
    quality_score: float          # mean across cases — primary leaderboard number
    divergence_score: float
    composite_score: float
    n_cases: int
    n_ideas: int
    convergence_rate: float
    invalid_rate: float
    mean_feasibility: float
    mean_impact: float
    mean_originality: float
    mean_divergence: float
    # populated by the runner with bootstrap CIs / significance:
    quality_ci: tuple[float, float] | None = None
    divergence_ci: tuple[float, float] | None = None
    per_case: list[CaseScore] = field(default_factory=list)


def _idea_weighted(case_scores: list[CaseScore], attr: str, total_ideas: int) -> float:
    if total_ideas == 0:
        return 0.0
    return sum(getattr(c, attr) * c.n_ideas for c in case_scores) / total_ideas


def aggregate_model(model: str, case_scores: list[CaseScore]) -> ModelScore:
    n_cases = len(case_scores)
    if n_cases == 0:
        return ModelScore(model, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_ideas = sum(c.n_ideas for c in case_scores)
    total_invalid = sum(c.n_invalid for c in case_scores)
    return ModelScore(
        model=model,
        quality_score=sum(c.quality_score for c in case_scores) / n_cases,
        divergence_score=sum(c.divergence_score for c in case_scores) / n_cases,
        composite_score=sum(c.composite_score for c in case_scores) / n_cases,
        n_cases=n_cases,
        n_ideas=total_ideas,
        convergence_rate=_idea_weighted(case_scores, "convergence_rate", total_ideas),
        invalid_rate=(total_invalid / total_ideas) if total_ideas else 0.0,
        mean_feasibility=_idea_weighted(case_scores, "mean_feasibility", total_ideas),
        mean_impact=_idea_weighted(case_scores, "mean_impact", total_ideas),
        mean_originality=_idea_weighted(case_scores, "mean_originality", total_ideas),
        mean_divergence=_idea_weighted(case_scores, "mean_divergence", total_ideas),
        per_case=case_scores,
    )
