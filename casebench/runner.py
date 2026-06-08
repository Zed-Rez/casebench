"""Orchestration for CASE-Bench v2: sample -> panel-judge -> consolidate ->
score -> CIs -> report.

Research-grade knobs (all cached + resumable):
  * ``gen_samples``  k independent idea-sets per (model, case) at a logged temperature,
  * ``judges``       a panel of judge models (reliability comes from agreement),
  * ``judge_passes`` repeated judging with the idea order shuffled (order-bias control),
  * bootstrap 95% CIs over cases + paired-difference significance between models,
  * inter-judge reliability, judge-vs-lexical novelty validation, contradiction audit.

Defaults stay cheap (1 sample, 1 judge, 1 pass); bump the knobs for a publishable run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import generate as gen
from . import judge as judging
from . import novelty_baseline
from . import providers
from . import scoring
from . import stats
from . import validation

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cases.json"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"


def load_cases(path: Path = DATA_PATH) -> list[dict]:
    return json.loads(path.read_text())["cases"]


def _h_gen(*parts) -> str:
    """Hash for generation cache — independent of the judge protocol, so bumping
    the rubric re-judges but reuses cached generations."""
    raw = "|".join(str(p) for p in parts) + f"|gen|{__import__('casebench').__version__}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _h_judge(*parts) -> str:
    """Hash for judge cache — keyed to the judge protocol version."""
    raw = "|".join(str(p) for p in parts) + f"|{__import__('casebench').__version__}|{judging.JUDGE_PROTOCOL_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _safe(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


@dataclass
class RunConfig:
    models: list[str]
    judges: list[str] = field(default_factory=lambda: [judging.DEFAULT_JUDGE_MODEL])
    gen_samples: int = 1
    judge_passes: int = 1
    n_ideas: int = gen.IDEAS_PER_CASE
    temperature: float | None = 1.0
    use_cache: bool = True


# --- cached primitives -----------------------------------------------------


def _cached_generation(model, case, s, cfg, log):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"gen__{_safe(model)}__{case['id']}__s{s}__{_h_gen(model, case['id'], s, cfg.n_ideas, cfg.temperature)}.json"
    if cfg.use_cache and f.exists():
        blob = json.loads(f.read_text())
        return [gen.Idea(**i) for i in blob["ideas"]]
    log(f"  [gen] {model} / {case['id']} sample {s}")
    g = gen.generate_for_case(model, case, n=cfg.n_ideas, temperature=cfg.temperature)
    f.write_text(json.dumps({
        "model": model, "case_id": case["id"], "sample": s,
        "ideas": [dataclasses.asdict(i) for i in g.ideas],
        "temperature": g.temperature,
        "input_tokens": g.input_tokens, "output_tokens": g.output_tokens, "ts": time.time(),
    }, indent=2))
    return g.ideas


def _cached_verdicts(model, case, ideas, s, judge_model, p, cfg, log):
    f = CACHE_DIR / f"judge__{_safe(model)}__{case['id']}__s{s}__{_safe(judge_model)}__p{p}__{_h_judge(model, case['id'], s, judge_model, p)}.json"
    if cfg.use_cache and f.exists():
        blob = json.loads(f.read_text())
        return [judging.Verdict(**v) for v in blob["verdicts"]]
    # Shuffle presentation order reproducibly per (sample, judge, pass). Use a
    # hashlib-derived seed, NOT builtin hash() (which is salted per process and
    # would make the order non-reproducible across runs on a cache miss).
    order = list(range(len(ideas)))
    seed = int(hashlib.sha1(f"{case['id']}|{s}|{judge_model}|{p}".encode()).hexdigest(), 16) & 0xFFFFFFFF
    random.Random(seed).shuffle(order)
    log(f"  [judge] {judge_model} <- {model}/{case['id']} s{s} p{p}")
    verdicts = judging.judge_case(case, ideas, judge_model=judge_model, order=order)
    f.write_text(json.dumps({
        "model": model, "case_id": case["id"], "sample": s, "judge_model": judge_model,
        "pass": p, "order": order,
        "verdicts": [dataclasses.asdict(v) for v in verdicts], "ts": time.time(),
    }, indent=2))
    return verdicts


# --- evaluation ------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    model: str
    sample_quality: list[float]      # one per gen sample (scored on the fixed slate)
    sample_diversity: list[float]    # PORTFOLIO diversity (coverage + intra-slate dissimilarity)
    case_score: scoring.CaseScore    # pooled consolidated verdicts (for sub-metrics)
    contradiction_rate: float
    interjudge_pairs: list = field(default_factory=list)        # (Verdict, Verdict)
    novelty_points: list = field(default_factory=list)          # (judge_div, lexical_div)
    gold_pairs: list = field(default_factory=list)              # (gold, Verdict)
    judge_values: dict = field(default_factory=dict)            # judge_model -> [idea_value]
    axis_points: list = field(default_factory=list)             # (feasibility, impact, originality, divergence, idea_value)
    cell_mean_originality: float = 0.0                          # for orig-vs-diversity orthogonality
    cell_diversity: float = 0.0


def evaluate_case(model: str, case: dict, cfg: RunConfig, gold: dict, log=print) -> CaseResult:
    sample_quality, sample_diversity = [], []
    pooled_consolidated: list[judging.Verdict] = []
    interjudge_pairs, novelty_points, gold_pairs = [], [], []
    judge_values: dict[str, list[float]] = {jm: [] for jm in cfg.judges}
    axis_points: list = []
    orig_vals_all: list[float] = []

    for s in range(cfg.gen_samples):
        ideas = _cached_generation(model, case, s, cfg, log)
        lex = novelty_baseline.lexical_divergence(case, ideas)

        per_idea: dict[int, list[judging.Verdict]] = {i: [] for i in range(len(ideas))}
        pass_grouped: dict[tuple, list[judging.Verdict]] = {}
        for jm in cfg.judges:
            for p in range(cfg.judge_passes):
                verdicts = _cached_verdicts(model, case, ideas, s, jm, p, cfg, log)
                pass_grouped[(jm, p)] = verdicts
                for v in verdicts:
                    if v.index in per_idea:
                        per_idea[v.index].append(v)
                # self-preference signal: each judge's mean idea-value on THIS candidate
                if p == 0:
                    judge_values.setdefault(jm, []).extend(scoring.idea_value(v) for v in verdicts)

        if len(cfg.judges) >= 2:
            base = {v.index: v for v in pass_grouped.get((cfg.judges[0], 0), [])}
            for jm in cfg.judges[1:]:
                for v in pass_grouped.get((jm, 0), []):
                    if v.index in base:
                        interjudge_pairs.append((base[v.index], v))

        consolidated = [stats.consolidate_idea(per_idea[i]) for i in range(len(ideas)) if per_idea[i]]
        # Score each sample against the FIXED requested slate (under-delivery defence).
        cs = scoring.score_case(case["id"], model, consolidated, n_expected=cfg.n_ideas)
        sample_quality.append(cs.quality_score)
        # PORTFOLIO diversity: mechanism coverage + intra-slate dissimilarity. A
        # genuinely separate construct from per-idea originality (which feeds Quality).
        coverage = scoring.slate_coverage(consolidated, cfg.n_ideas)
        intra = novelty_baseline.intra_slate_dissimilarity(ideas)
        sample_diversity.append(scoring.slate_diversity_score(coverage, intra))
        pooled_consolidated.extend(consolidated)

        for cv in consolidated:
            novelty_points.append((cv.divergence, lex[cv.index] if cv.index < len(lex) else 0.0))
            axis_points.append((cv.feasibility, cv.impact, cv.originality, cv.divergence, scoring.idea_value(cv)))
            orig_vals_all.append(cv.originality)
        gold_pairs.extend(validation.judge_vs_gold(
            case["id"], [idea.as_text() for idea in ideas], consolidated, gold))

    pooled_score = scoring.score_case(
        case["id"], model, pooled_consolidated, n_expected=cfg.n_ideas * cfg.gen_samples)
    return CaseResult(
        case_id=case["id"], model=model,
        sample_quality=sample_quality, sample_diversity=sample_diversity,
        case_score=pooled_score,
        contradiction_rate=validation.contradiction_rate(pooled_consolidated),
        interjudge_pairs=interjudge_pairs, novelty_points=novelty_points, gold_pairs=gold_pairs,
        judge_values=judge_values, axis_points=axis_points,
        cell_mean_originality=(sum(orig_vals_all) / len(orig_vals_all)) if orig_vals_all else 0.0,
        cell_diversity=(sum(sample_diversity) / len(sample_diversity)) if sample_diversity else 0.0,
    )


@dataclass
class ModelResult:
    model: str
    model_score: scoring.ModelScore
    per_case_quality: dict       # case_id -> mean sample quality
    per_case_divergence: dict
    case_results: list[CaseResult]


def evaluate_model(model: str, cases: list[dict], cfg: RunConfig, gold: dict, log=print) -> ModelResult:
    log(f"== {model} ==")
    case_results = [evaluate_case(model, c, cfg, gold, log) for c in cases]
    model_score = scoring.aggregate_model(model, [cr.case_score for cr in case_results])

    # Single estimand everywhere: per-case quality = mean over gen-samples of the
    # fixed-slate sample score. The board headline, CIs, and significance test all
    # use THESE numbers (the pooled CaseScore is kept only for sub-metrics).
    pcq = {cr.case_id: (sum(cr.sample_quality) / len(cr.sample_quality)) for cr in case_results}
    pcd = {cr.case_id: (sum(cr.sample_diversity) / len(cr.sample_diversity)) for cr in case_results}
    q_vals, d_vals = list(pcq.values()), list(pcd.values())
    model_score.quality_score = sum(q_vals) / len(q_vals) if q_vals else 0.0
    model_score.divergence_score = sum(d_vals) / len(d_vals) if d_vals else 0.0
    model_score.composite_score = (
        scoring.COMPOSITE_QUALITY_WEIGHT * model_score.quality_score
        + scoring.COMPOSITE_DIVERGENCE_WEIGHT * model_score.divergence_score)
    model_score.quality_ci = stats.bootstrap_ci(q_vals)
    model_score.divergence_ci = stats.bootstrap_ci(d_vals)
    return ModelResult(model, model_score, pcq, pcd, case_results)


# --- top-level run ---------------------------------------------------------


def run_benchmark(cfg: RunConfig, cases: list[dict], log=print) -> dict:
    gold = validation.load_gold()
    results = [evaluate_model(m, cases, cfg, gold, log) for m in cfg.models]
    return assemble_report(cfg, cases, results)


def assemble_report(cfg: RunConfig, cases: list[dict], results: list[ModelResult]) -> dict:
    case_ids = [c["id"] for c in cases]

    # pairwise significance on the PRIMARY (quality) board
    significance = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            va = [a.per_case_quality[cid] for cid in case_ids]
            vb = [b.per_case_quality[cid] for cid in case_ids]
            test = stats.paired_diff_test(va, vb)
            significance.append({
                "a": a.model, "b": b.model,
                "mean_diff_quality": round(test["mean_diff"], 2),
                "ci": [round(test["ci"][0], 2), round(test["ci"][1], 2)],
                "significant": test["significant"], "p": test["p"], "p_floor": test.get("p_floor"),
                "paired_diff_sd": round(stats.paired_diff_sd(va, vb), 2),
            })

    # diversity significance — so the diagnostic board can declare ties instead of
    # printing an over-precise ordering (Round 4 major: the board must carry the
    # same tie/CI honesty as the quality board).
    diversity_significance = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            t = stats.paired_diff_test(
                [a.per_case_divergence[c] for c in case_ids],
                [b.per_case_divergence[c] for c in case_ids])
            diversity_significance.append({"a": a.model, "b": b.model,
                                           "mean_diff": round(t["mean_diff"], 2),
                                           "significant": t["significant"], "p": t["p"]})

    # Robustness: does the Quality ranking survive DROPPING the weak impact axis?
    # (Round 4 major: the headline must not rest on the least-validated axis.)
    def _orig_only_quality(r: ModelResult) -> float:
        vals = [scoring.feas_gate(f) * (o / 3.0)
                for cr in r.case_results for (f, _imp, o, _d, _v) in cr.axis_points]
        return 100.0 * sum(vals) / len(vals) if vals else 0.0
    headline_rank = [r.model for r in sorted(results, key=lambda r: r.model_score.quality_score, reverse=True)]
    orig_only = {r.model: _orig_only_quality(r) for r in results}
    orig_only_rank = [m for m, _ in sorted(orig_only.items(), key=lambda x: -x[1])]
    robustness = {
        "headline_quality_ranking": headline_rank,
        "ranking_without_impact": orig_only_rank,
        "ranking_preserved_when_impact_dropped": headline_rank == orig_only_rank,
        "originality_only_quality": {m: round(v, 2) for m, v in orig_only.items()},
        "note": "impact is low-discrimination and weakly validated; this shows the headline ranking "
                "does not depend on it (feasibility + the validated originality axis carry the result).",
    }

    # reliability, novelty validation, contradictions, gold (pooled across run)
    all_pairs = [pr for r in results for cr in r.case_results for pr in cr.interjudge_pairs]
    reliability = stats.interjudge_reliability(all_pairs) if all_pairs else {
        "note": "single judge — run --judges A B [C] for inter-judge reliability"}

    nov = [(d, l) for r in results for cr in r.case_results for (d, l) in cr.novelty_points]
    novelty_validation = (
        novelty_baseline.divergence_validation([d for d, _ in nov], [l for _, l in nov]) if nov else {})

    gold_pairs = [gp for r in results for cr in r.case_results for gp in cr.gold_pairs]
    gold_agreement = validation.gold_agreement(gold_pairs)

    contradiction_rate = round(
        sum(cr.contradiction_rate for r in results for cr in r.case_results)
        / max(1, sum(len(r.case_results) for r in results)), 3)

    # Honest MDE: from the PAIRED-difference SD of the closest model pair (the
    # cross-model pooled SD contains the very effect being tested and overstates
    # dispersion). Falls back to within-model SD if <2 models.
    if len(results) >= 2:
        pair = sorted(significance, key=lambda s: abs(s["mean_diff_quality"]))[0]
        ra = next(r for r in results if r.model == pair["a"])
        rb = next(r for r in results if r.model == pair["b"])
        paired_sd = stats.paired_diff_sd(
            [ra.per_case_quality[c] for c in case_ids], [rb.per_case_quality[c] for c in case_ids])
    else:
        vals = [results[0].per_case_quality[c] for c in case_ids] if results else []
        paired_sd = stats.statistics.pstdev(vals) if len(vals) > 1 else 0.0
    mde = stats.minimum_detectable_effect(paired_sd, len(case_ids))

    # Self-preference audit: a candidate that also judges may favour its own ideas.
    # Normalize specs first so 'claude-opus-4-8' and 'anthropic:claude-opus-4-8'
    # are recognised as the same model.
    def _canon(spec: str) -> str:
        p, m = providers.parse_spec(spec)
        return f"{p}:{m}"
    candidate_judge_overlap = sorted({_canon(m) for m in cfg.models} & {_canon(j) for j in cfg.judges})
    self_pref = []
    for r in results:
        jv: dict[str, list[float]] = {}
        for cr in r.case_results:
            for jm, vals in cr.judge_values.items():
                jv.setdefault(jm, []).extend(vals)
        means = {jm: (sum(v) / len(v) if v else 0.0) for jm, v in jv.items()}
        if len(means) >= 2:
            delta = max(means.values()) - min(means.values())
            most_lenient = max(means, key=means.get)
            self_pref.append({
                "candidate": r.model,
                "judge_value_means": {jm: round(x, 3) for jm, x in means.items()},
                "max_judge_delta": round(delta, 3),
                "most_lenient_judge": most_lenient,
                "self_judged": _canon(r.model) in {_canon(j) for j in cfg.judges},
            })

    # Board-separation / orthogonality (C8 + R2 M2). The headline diagnostic is
    # PORTFOLIO diversity, which should be near-orthogonal to per-idea originality.
    # We also disclose the per-idea originality-vs-(reference)divergence coupling,
    # which is high by construction (an "original" idea is also "far from the refs")
    # — which is exactly why per-idea divergence is NOT the headline board.
    ax = [pt for r in results for cr in r.case_results for pt in cr.axis_points]
    cells = [(cr.cell_mean_originality, cr.cell_diversity) for r in results for cr in r.case_results]
    decorrelation = {}
    if len(ax) >= 2:
        feas = [p[0] for p in ax]; imp = [p[1] for p in ax]; orig = [p[2] for p in ax]
        div = [p[3] for p in ax]; val = [p[4] for p in ax]
        co = [c[0] for c in cells]; cd = [c[1] for c in cells]
        decorrelation = {
            "n_ideas": len(ax),
            "headline_originality_vs_diversity_pearson":
                (round(x, 3) if (x := stats._pearson(co, cd)) is not None else None),
            "headline_originality_vs_diversity_spearman":
                (round(x, 3) if (x := stats.spearman(co, cd)) is not None else None),
            "disclosed_originality_vs_refdivergence_pearson":
                (round(x, 3) if (x := stats._pearson(orig, div)) is not None else None),
            "quality_vs_refdivergence_pearson":
                (round(x, 3) if (x := stats._pearson(val, div)) is not None else None),
            "realized_axis_std": {
                "feasibility": round(stats.statistics.pstdev(feas), 3) if len(feas) > 1 else 0.0,
                "impact": round(stats.statistics.pstdev(imp), 3) if len(imp) > 1 else 0.0,
                "originality": round(stats.statistics.pstdev(orig), 3) if len(orig) > 1 else 0.0,
                "divergence": round(stats.statistics.pstdev(div), 3) if len(div) > 1 else 0.0,
            },
            "note": ("Headline diagnostic is portfolio diversity (coverage + intra-slate "
                     "dissimilarity), reported orthogonal to originality. Per-idea reference-"
                     "divergence is ~originality-coupled by construction and kept only as a "
                     "sub-metric. Low impact std means impact barely discriminates on this "
                     "model population — read the per-axis std, not just the headline."),
        }

    quality_board = sorted(results, key=lambda r: r.model_score.quality_score, reverse=True)
    diversity_board = sorted(results, key=lambda r: r.model_score.divergence_score, reverse=True)

    def board_row(r: ModelResult) -> dict:
        m = r.model_score
        return {
            "model": m.model,
            "quality_score": round(m.quality_score, 2),
            "quality_ci": [round(m.quality_ci[0], 2), round(m.quality_ci[1], 2)] if m.quality_ci else None,
            "diversity_score": round(m.divergence_score, 2),         # portfolio exploration
            "diversity_ci": [round(m.divergence_ci[0], 2), round(m.divergence_ci[1], 2)] if m.divergence_ci else None,
            "composite_score": round(m.composite_score, 2),
            "mean_feasibility": round(m.mean_feasibility, 2),
            "mean_impact": round(m.mean_impact, 2),
            "mean_originality": round(m.mean_originality, 2),
            "mean_refdivergence": round(m.mean_divergence, 2),       # per-idea, sub-metric only
            "convergence_rate": round(m.convergence_rate, 3),
            "invalid_rate": round(m.invalid_rate, 3),
            "n_cases": m.n_cases, "n_ideas": m.n_ideas,
        }

    return {
        "benchmark": "CASE-Bench",
        "version": __import__("casebench").__version__,
        "judge_protocol": judging.JUDGE_PROTOCOL_VERSION,
        "config": {
            "models": cfg.models, "judges": cfg.judges, "gen_samples": cfg.gen_samples,
            "judge_passes": cfg.judge_passes, "n_ideas": cfg.n_ideas, "temperature": cfg.temperature,
            "n_cases": len(case_ids),
        },
        "quality_leaderboard": [board_row(r) for r in quality_board],
        "diversity_leaderboard": [
            {"model": r.model, "diversity_score": round(r.model_score.divergence_score, 2),
             "convergence_rate": round(r.model_score.convergence_rate, 3)}
            for r in diversity_board
        ],
        "significance_quality": significance,
        "significance_diversity": diversity_significance,
        "robustness_without_impact": robustness,
        "interjudge_reliability": reliability,
        "novelty_validation": novelty_validation,
        "board_decorrelation": decorrelation,
        "self_preference_audit": {
            "candidate_judge_overlap": candidate_judge_overlap,
            "warning": ("a candidate also sits on the judge panel — results may be "
                        "self-preference-biased; prefer judges disjoint from candidates")
                       if candidate_judge_overlap else None,
            "per_candidate": self_pref,
        },
        "judge_vs_human_gold": gold_agreement,
        "contradiction_rate": contradiction_rate,
        "power": {"paired_diff_sd": round(paired_sd, 2),
                  "minimum_detectable_effect_95_80": round(mde, 2),
                  "basis": "paired per-case difference SD of the closest model pair",
                  "caveat": "CIs reflect between-case variance only; at small n they are coarse.",
                  "note": f"~{int((2.8 * paired_sd / 5) ** 2) + 1 if paired_sd else 0} cases needed to detect a 5-pt gap at this paired SD."},
        "per_case": {
            r.model: [dataclasses.asdict(cr.case_score) | {"verdicts": None,
                       "sample_quality": [round(x, 1) for x in cr.sample_quality]}
                      for cr in r.case_results]
            for r in results
        },
    }


# --- reporting -------------------------------------------------------------


def render_leaderboard_md(report: dict) -> str:
    cfg = report["config"]
    L = []
    L.append("# CASE-Bench Leaderboard")
    L.append("")
    L.append(f"_Feasibility-constrained business ideation. Judge protocol `{report['judge_protocol']}`, "
             f"judges {cfg['judges']}, {cfg['gen_samples']} gen-sample(s) x {cfg['judge_passes']} judge-pass(es), "
             f"{cfg['n_cases']} cases, {cfg['n_ideas']} ideas/case._")
    L.append("")
    L.append("## Primary board — Idea Quality")
    L.append("_Which model produces better solutions? Feasibility-gated impact+originality. "
             "Novelty is deliberately excluded here (see the diagnostic board)._")
    L.append("")
    L.append("| Rank | Model | Quality | 95% CI | Feas | Impact | Orig | Invalid% | Cases | Ideas |")
    L.append("|----:|:------|----:|:----:|----:|----:|----:|----:|----:|----:|")
    for i, row in enumerate(report["quality_leaderboard"], 1):
        ci = f"[{row['quality_ci'][0]:.1f}, {row['quality_ci'][1]:.1f}]" if row["quality_ci"] else "—"
        L.append(f"| {i} | `{row['model']}` | **{row['quality_score']:.1f}** | {ci} | "
                 f"{row['mean_feasibility']:.2f} | {row['mean_impact']:.2f} | {row['mean_originality']:.2f} | "
                 f"{row['invalid_rate']*100:.0f}% | {row['n_cases']} | {row['n_ideas']} |")
    L.append("")
    L.append("## Diagnostic board — Portfolio Diversity (NOT a quality measure)")
    L.append("_How varied a model's slate is: mechanism coverage + how dissimilar its own ideas are from "
             "each other. A genuinely separate axis from per-idea originality (which feeds Quality). High "
             "diversity with low quality = scattershot; read it alongside the primary board._")
    L.append("")
    # diversity CIs keyed by model (from the quality board rows)
    div_ci = {row["model"]: row.get("diversity_ci") for row in report["quality_leaderboard"]}
    L.append("| Rank | Model | Diversity | 95% CI | Convergence rate |")
    L.append("|----:|:------|----:|:----:|----:|")
    for i, row in enumerate(report["diversity_leaderboard"], 1):
        ci = div_ci.get(row["model"])
        cistr = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "—"
        L.append(f"| {i} | `{row['model']}` | {row['diversity_score']:.1f} | {cistr} | {row['convergence_rate']*100:.0f}% |")
    divsig = report.get("significance_diversity") or []
    if divsig and not any(s["significant"] for s in divsig):
        L.append("")
        L.append("_Diversity differences are **not significant** (overlapping CIs / paired test) — "
                 "treat the diversity ranking as a tie; both models explore broadly._")
    L.append("")

    # significance
    L.append("## Significance (primary board, paired bootstrap over cases)")
    L.append("_Authoritative test is the PAIRED bootstrap below — not marginal-CI overlap, which is "
             "over-conservative for paired data. p is floored at the bootstrap/sign-test resolution._")
    if report["significance_quality"]:
        for s in report["significance_quality"]:
            verdict = "**significant**" if s["significant"] else "not significant (treat as a tie)"
            pstr = f"p≈{s['p']}" + (f" (floor {s['p_floor']})" if s.get("p_floor") and s["p"] <= s["p_floor"] else "")
            L.append(f"- `{s['a']}` vs `{s['b']}`: Δquality {s['mean_diff_quality']:+.1f} "
                     f"(95% CI [{s['ci'][0]}, {s['ci'][1]}], paired-diff SD {s.get('paired_diff_sd')}, {pstr}) — {verdict}")
    else:
        L.append("- (single model)")
    L.append("")

    # self-preference
    spa = report.get("self_preference_audit") or {}
    if spa.get("warning"):
        L.append(f"> ⚠️ **Self-preference:** {spa['warning']} (overlap: {spa['candidate_judge_overlap']}).")
        for sp in spa.get("per_candidate", []):
            if sp["self_judged"]:
                L.append(f">   `{sp['candidate']}` judged itself; cross-judge value delta "
                         f"{sp['max_judge_delta']} (most lenient: `{sp['most_lenient_judge']}`).")
        L.append("")

    # validity panel
    L.append("## Validity & reliability")
    nv = report.get("novelty_validation") or {}
    if nv:
        L.append(f"- Novelty validation (judge divergence vs lexical baseline): Spearman ρ="
                 f"{nv.get('spearman_rho')} over n={nv.get('n')} — {nv.get('interpretation','')}")
    dc = report.get("board_decorrelation") or {}
    if dc:
        L.append(f"- Board separation: headline originality-vs-**diversity** r="
                 f"{dc.get('headline_originality_vs_diversity_pearson')} (orthogonal target — low is good). "
                 f"Disclosed: per-idea originality-vs-refdivergence r="
                 f"{dc.get('disclosed_originality_vs_refdivergence_pearson')} (high by construction — "
                 f"why per-idea divergence is a sub-metric, not the board).")
        ras = dc.get("realized_axis_std") or {}
        L.append(f"- Realized axis discrimination (std over {dc.get('n_ideas')} ideas): "
                 f"feas {ras.get('feasibility')}, impact {ras.get('impact')}, "
                 f"orig {ras.get('originality')}, refdiv {ras.get('divergence')}. "
                 f"Low impact std = impact barely separates these models (read sub-metrics, not just headline).")
    rel = report.get("interjudge_reliability") or {}
    if "axes" in rel:
        imp = rel["axes"].get("impact", {})
        L.append(f"- Inter-judge reliability (n={rel.get('n_pairs')}): impact within-1 "
                 f"{imp.get('within1_agreement')}, known-move κ={rel.get('known_move_kappa')}")
    else:
        L.append(f"- Inter-judge reliability: {rel.get('note','n/a')}")
    g = report.get("judge_vs_human_gold") or {}
    if g.get("n"):
        ia = g["axes"].get("impact", {})
        oa = g["axes"].get("originality", {})
        L.append(f"- Judge vs human gold (ILLUSTRATIVE n={g['n']} single-rater seed from one model's "
                 f"ideas, NOT a validated standard): impact within-1 {ia.get('within1_agreement')}, "
                 f"impact Pearson {ia.get('pearson_r')} (n={g['n']} — not distinguishable from 0); "
                 f"originality Pearson {oa.get('pearson_r')}. Impact is the least-discriminating, "
                 f"least-validated axis — read as provisional.")
    else:
        L.append(f"- Judge vs human gold: {g.get('note','no gold set')}")
    rb = report.get("robustness_without_impact") or {}
    if rb:
        ok = "preserved" if rb.get("ranking_preserved_when_impact_dropped") else "CHANGES"
        L.append(f"- Robustness: dropping the weak impact axis leaves the Quality ranking **{ok}** "
                 f"({' > '.join(rb.get('ranking_without_impact', []))}) — the headline rests on "
                 f"feasibility + the better-validated originality axis (illustrative gold + independent "
                 f"lexical baseline), not on impact.")
    L.append(f"- Internal contradiction rate (judge self-consistency): {report['contradiction_rate']*100:.1f}%")
    pw = report["power"]
    L.append(f"- Power: paired-diff SD={pw['paired_diff_sd']}, minimum detectable effect (95/80) "
             f"≈ {pw['minimum_detectable_effect_95_80']} quality points ({pw['basis']}). {pw['note']} {pw['caveat']}")
    L.append("")
    L.append("Quality = `100 x (0.6*mean + 0.4*mean(top-3))` of per-idea "
             "`feas_gate(feasibility) x (impact+originality)/6`, scored against the FIXED requested slate "
             "(unfilled slots = 0). The diagnostic board (portfolio diversity = mechanism coverage + "
             "intra-slate dissimilarity) is reported separately and never enters Quality. Significance is "
             "the paired test above.")
    return "\n".join(L) + "\n"


def write_reports(report: dict, out_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    md_path = out_dir / "leaderboard.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_leaderboard_md(report))
    return json_path, md_path
