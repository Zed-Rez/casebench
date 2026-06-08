"""Statistics for CASE-Bench: confidence intervals, significance, reliability.

Pure standard-library implementations (no numpy/scipy) so the benchmark stays
dependency-light. Covers:

- Bootstrap CIs over cases (the dominant variance source).
- Paired-difference bootstrap to decide whether two models actually differ.
- Consolidation of a panel / repeated passes into one verdict per idea.
- Inter-judge reliability (within-1 agreement, Pearson r, Cohen's kappa on the
  known-move flag).

Randomness uses a seeded ``random.Random`` so reported CIs are reproducible.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import replace

from .judge import Verdict

AXES = ("feasibility", "impact", "originality", "divergence")


# --- consolidation ---------------------------------------------------------


def consolidate_idea(verdicts: list[Verdict]) -> Verdict:
    """Merge several verdicts for the SAME idea into one (panel/pass average).

    0-3 axes are averaged in **float space and NOT rounded** — rounding back to
    int destroyed every split decision (a 3-vs-2 became 2), pinning peak scores
    at the grid and collapsing the scale. The scorer's gate and norms are all
    float-safe. The nearest reference and known-move status are by majority.
    """
    if not verdicts:
        raise ValueError("no verdicts to consolidate")
    if len(verdicts) == 1:
        return verdicts[0]

    def avg(attr: str) -> float:
        return sum(getattr(v, attr) for v in verdicts) / len(verdicts)

    refs = Counter(v.nearest_reference_id for v in verdicts)
    nearest = refs.most_common(1)[0][0]
    also: set[str] = set()
    for v in verdicts:
        also.update(v.also_covers)

    base = verdicts[0]
    return replace(
        base,
        feasibility=avg("feasibility"),
        impact=avg("impact"),
        originality=avg("originality"),
        divergence=avg("divergence"),
        nearest_reference_id=nearest,
        also_covers=sorted(also),
        judge_model="+".join(sorted({v.judge_model for v in verdicts if v.judge_model})),
    )


# --- bootstrap CIs ---------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    per_case_values: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> tuple[float, float]:
    """95% bootstrap CI for the mean of per-case values (resample cases)."""
    import random

    n = len(per_case_values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (per_case_values[0], per_case_values[0])
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [per_case_values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def paired_diff_test(
    values_a: list[float],
    values_b: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> dict:
    """Paired bootstrap of (a - b) over matched cases.

    Returns the mean difference, its 95% CI, and whether it is significant
    (0 outside the CI). Also returns a paired t-style p-value approximation.
    """
    import random

    assert len(values_a) == len(values_b), "paired test needs matched cases"
    n = len(values_a)
    diffs = [a - b for a, b in zip(values_a, values_b)]
    mean_diff = sum(diffs) / n if n else 0.0

    if n < 2:
        return {"mean_diff": mean_diff, "ci": (mean_diff, mean_diff), "significant": False, "p": 1.0}

    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    ci = (_percentile(boot, alpha / 2), _percentile(boot, 1 - alpha / 2))
    significant = not (ci[0] <= 0.0 <= ci[1])

    # Two-sided p approximation from the bootstrap distribution, floored at the
    # bootstrap resolution (1/n_boot) — reporting p=0.0 is never honest. Also
    # floored at the exact paired sign-test minimum 2 / 2**n (you cannot beat that
    # with n matched cases however lopsided the bootstrap looks).
    share_le0 = sum(1 for b in boot if b <= 0) / len(boot)
    p = 2 * min(share_le0, 1 - share_le0)
    p_floor = max(1.0 / n_boot, 2.0 / (2 ** n))
    p = max(p, p_floor)
    return {"mean_diff": mean_diff, "ci": ci, "significant": significant,
            "p": round(p, 4), "p_floor": round(p_floor, 4)}


def ci_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    return not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])


# --- minimum detectable effect (power) -------------------------------------


def minimum_detectable_effect(per_case_sd: float, n_cases: int) -> float:
    """Rough MDE at ~80% power, two-sided 0.05 (z ~ 2.8 combined)."""
    if n_cases <= 0:
        return float("inf")
    return 2.8 * per_case_sd / math.sqrt(n_cases)


def paired_diff_sd(values_a: list[float], values_b: list[float]) -> float:
    """SD of the matched per-case differences — the correct dispersion for a
    paired MDE. (Pooling both models' scores into one SD inflates it with the
    very between-model effect being tested.)"""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    return statistics.pstdev(diffs) if len(diffs) > 1 else 0.0


# --- inter-judge reliability ----------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float | None:
    """Cohen's kappa for two raters' binary labels."""
    n = len(labels_a)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def interjudge_reliability(paired_verdicts: list[tuple[Verdict, Verdict]]) -> dict:
    """Reliability between two judges over the same (case, idea) pairs.

    For each 0-3 axis: exact-agreement and within-1-agreement rates, plus Pearson
    r. Plus Cohen's kappa on the binary known-move flag (divergence <= 1).
    """
    if not paired_verdicts:
        return {}
    out: dict = {"n_pairs": len(paired_verdicts), "axes": {}}
    for axis in AXES:
        a = [getattr(x, axis) for x, _ in paired_verdicts]
        b = [getattr(y, axis) for _, y in paired_verdicts]
        exact = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        within1 = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)
        out["axes"][axis] = {
            "exact_agreement": round(exact, 3),
            "within1_agreement": round(within1, 3),
            "pearson_r": (round(r, 3) if (r := _pearson([float(x) for x in a], [float(y) for y in b])) is not None else None),
        }
    ka = [1 if x.is_known_move else 0 for x, _ in paired_verdicts]
    kb = [1 if y.is_known_move else 0 for _, y in paired_verdicts]
    kappa = cohen_kappa(ka, kb)
    out["known_move_kappa"] = round(kappa, 3) if kappa is not None else None
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (used for novelty-baseline validation)."""
    n = len(xs)
    if n < 2:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return _pearson(ranks(xs), ranks(ys))
