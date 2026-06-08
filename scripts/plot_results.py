"""Generate the CASE-Bench results figure directly from results/results.json.

Dot-and-whisker (forest-plot) panels with 95% bootstrap CIs — Quality (primary)
and Portfolio Diversity (diagnostic) — so the figure is a faithful render of the
numbers the benchmark computed, not a hand-drawn picture.

    pip install matplotlib
    python scripts/plot_results.py            # -> results/quality_ci.svg (+ .png)

Reads results/results.json (produced by `python -m casebench.cli run`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results.json"

INK = "#2b2b2b"
ACCENT = "#7a1f1f"      # restrained terracotta-ish accent
MUTED = "#8a8580"
GRID = "#e6e1d8"
BG = "#faf8f3"          # warm off-white, matches the site


def _serif():
    for name in ("EB Garamond", "Georgia", "DejaVu Serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "serif"


def short(model: str) -> str:
    return model.split(":")[-1].replace("claude-", "")


def main() -> int:
    if not RESULTS.exists():
        print(f"no results at {RESULTS} — run `python -m casebench.cli run` first", file=sys.stderr)
        return 2
    r = json.loads(RESULTS.read_text())
    rows = r["quality_leaderboard"]
    cfg = r["config"]
    sig = (r.get("significance_quality") or [{}])[0]

    plt.rcParams.update({
        "font.family": _serif(), "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    })

    panels = [
        ("Idea Quality  (0–100, higher = better solutions)", "quality_score", "quality_ci", ACCENT),
        ("Portfolio Diversity  (diagnostic, not a quality measure)", "diversity_score", "diversity_ci", MUTED),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.3), gridspec_kw={"hspace": 0.65})

    for ax, (title, skey, cikey, color) in zip(axes, panels):
        ordered = sorted(rows, key=lambda x: x[skey])
        ys = range(len(ordered))
        for y, row in zip(ys, ordered):
            score = row[skey]
            ci = row.get(cikey) or [score, score]
            ax.plot([ci[0], ci[1]], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot([ci[0], ci[1]], [y, y], "|", color=color, ms=10, mew=2.2, zorder=3)
            ax.plot(score, y, "o", color=color, ms=8, zorder=4,
                    markeredgecolor=BG, markeredgewidth=1.2)
            ax.annotate(f"{score:.1f}", (score, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=10.5, color=INK)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([short(row["model"]) for row in ordered], fontsize=11)
        ax.set_ylim(-0.6, len(ordered) - 0.4)
        ax.set_title(title, fontsize=11.5, loc="left", pad=8, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)

    if sig:
        verdict = "significant" if sig.get("significant") else "n.s. (tie)"
        axes[0].annotate(
            f"Δ {abs(sig.get('mean_diff_quality', 0)):.1f}  ·  paired bootstrap p≈{sig.get('p')}  ·  {verdict}",
            xy=(0, -0.42), xycoords="axes fraction", fontsize=9.5, color=ACCENT)
    axes[1].annotate("overlapping CIs → ranking is a tie", xy=(0, -0.42),
                     xycoords="axes fraction", fontsize=9.5, color=MUTED)

    fig.suptitle(
        f"CASE-Bench — {cfg['n_cases']} cases × {cfg['gen_samples']} samples, "
        f"disjoint Opus judge panel, 95% bootstrap CIs",
        fontsize=12.5, x=0.012, ha="left", y=0.99, color=INK)

    out_svg = ROOT / "results" / "quality_ci.svg"
    out_png = ROOT / "results" / "quality_ci.png"
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_svg}\nwrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
