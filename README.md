# CASE-Bench

**A benchmark for feasibility-constrained business ideation by LLMs.**

CASE = **C**ase-based **A**nswer-**S**et **E**valuation.

LLMs are prone to two failure modes when asked for ideas: bland textbook answers,
and impressive-sounding proposals that collapse on contact with reality.
CASE-Bench measures the capability that matters in practice — generating
solutions to a real business problem that are **feasible**, **high-impact**, and
**original** — and separately reports how far a model's ideas **diverge** from
the known playbook, without letting divergence masquerade as quality.

Each case pairs a business problem (problem + hard constraints only — no solution
hints) with a curated set of **reference answers**, each tagged with its *core
mechanism* and a *scope note*. The candidate is shown the problem (not the
references), and a fixed LLM judge rates every idea on four independent axes.

> **What this is not.** An earlier version pitched reference-overlap as a
> training-contamination probe. That claim was wrong by construction (hand-
> authored, withheld, paraphrase-dodgeable references can't detect memorization)
> and has been removed. Reference overlap here measures **convergence with
> conventional solutions** — a divergence diagnostic, nothing more. A genuine
> contamination probe requires a private, rotating, never-published planted set
> (scaffold in `data/gold/README.md` notes the protocol); the creativity judge
> does not double as one.

## How it works

```
  case (problem + constraints) ──▶ candidate model ──▶ k idea-sets (JSON)
                                                          │
   references (+ scope notes) ──▶  JUDGE PANEL (structured output, p passes)
                                                          │
        per idea: feasibility · impact · originality · divergence + nearest-ref
                                                          │
                          consolidate → score → bootstrap CIs
```

### Four axes (all 0–3, judged independently)

| Axis | Question | Role |
|---|---|---|
| **feasibility** | Coherent and executable in context? (0 = incoherent/impossible) | **gate** |
| **impact** | If executed, does it move the needle? (3 requires a named failure risk) | quality |
| **originality** | Inventive / non-obvious *on its own merits* (independent of references) | quality |
| **divergence** | Graded distance from nearest reference (0 = same mechanism … 3 = distinct) | **diagnostic only** |

### Two boards (the key design fix)

The v1 score was dominated by a binary 0.5/1.0 novelty multiplier that **inverted
the ranking** (it ranked the model the judge rated *worse* on quality and
creativity *above* the better one) and let incoherent-but-exotic ideas outscore
sound ones. v2 separates the two questions:

**Primary — Idea Quality** (decision: *which model produces better solutions?*)

```
feas_gate(f)  = piecewise-linear, 0→0.0, 1→0.5, ≥2→1.0    # gates out junk; float-safe
value_i       = feas_gate(feasibility) · (impact/3 + originality/3) / 2
quality_score = 100 · (0.6 · mean(value) + 0.4 · mean(top-3 value))   # over the FIXED slate
```

Novelty/divergence is **deliberately excluded** here, so a model that dominates
on feasibility, impact, and originality can never rank below one that doesn't
(enforced by a unit test). The top-3 term rewards *having* strong ideas; the
score is computed over a **fixed slate equal to the requested quota** (unfilled
slots = 0), so under-delivering (returning only your best idea) can't inflate the
mean. The feasibility gate is **float-safe**, so panel averages aren't rounded
back onto the rubric grid (which would otherwise pin peak scores and flatten the
scale).

**Diagnostic — Portfolio Diversity** (decision: *does the model explore a spread
of approaches, or five variations on one idea?*), reported separately and
labelled **not** a quality measure:

```
diversity = 100 · ½ · ( mechanism_coverage + intra_slate_dissimilarity )
```

where `mechanism_coverage` = fraction of the slate landing on *distinct* reference
mechanisms (5 clones of one move ≈ 1/5; five distinct mechanisms ≈ 1) and
`intra_slate_dissimilarity` = mean pairwise lexical distance among the model's own
ideas. This is a **portfolio** property, deliberately chosen because in our canonical
run per-idea "distance from the references" was ~0.75-correlated with per-idea
originality (the same construct twice). Portfolio diversity is near-orthogonal to
originality instead (the report prints both correlations so you can check). Per-
idea reference-divergence is kept only as a sub-metric, and is feasibility-gated
so an incoherent idea earns no credit on any board.

A clearly-secondary additive composite (`0.7·quality + 0.3·diversity`) is also
reported for convenience — never as a multiplier.

### Statistical rigor

- **Sampling:** `--gen-samples k` independent idea-sets per (model, case) at a
  logged temperature (dropped automatically for models that reject it).
- **Judge panel + passes:** `--judges A B` and `--judge-passes p`; verdicts are
  consolidated, and idea order is shuffled each pass to control order bias.
- **Confidence intervals:** bootstrap 95% CIs over cases for every model (board
  headline, CIs, and the significance test all use the *same* per-case estimand).
- **Significance:** paired-difference bootstrap between models — the authoritative
  test (marginal-CI overlap is over-conservative for paired data); p is **floored
  at the bootstrap/sign-test resolution** (never reported as 0).
- **Power:** the MDE is built on the **paired-difference SD** (not a cross-model
  pooled SD, which would be inflated by the very effect being tested).
- **Reliability:** with ≥2 judges, inter-judge within-1 agreement, Pearson r, and
  Cohen's κ on the known-move flag.
- **Self-preference audit:** the runner flags any candidate that also sits on the
  judge panel and reports each candidate's cross-judge leniency delta. The
  **canonical board uses judges disjoint from every candidate** (Opus panel
  judging Haiku/Sonnet candidates).
- **Validity:** the judge's `divergence` is rank-correlated against an
  independent reference-free **lexical** novelty baseline; an internal-
  contradiction audit flags self-inconsistent verdicts; and a judge-vs-human gold
  harness (`data/gold/`) reports agreement against hand labels.

## Related work

CASE-Bench draws on a long creativity-assessment literature and differs
deliberately:

- **Torrance Tests / Alternative Uses Task (AUT):** divergent-thinking tasks
  scoring fluency/flexibility/originality on open prompts. CASE-Bench is
  *convergent-and-constrained*: ideas must solve a specific, constrained business
  problem, and **feasibility gates the score** — pure divergence is reported but
  never counted as quality.
- **Divergent Association Task (DAT):** semantic-distance novelty from word
  choice. We borrow the idea as an *automatic validation baseline* (lexical
  divergence) rather than as the metric itself.
- **Consensual Assessment Technique (CAT):** expert panels rate creativity.
  CASE-Bench operationalizes the panel as a multi-judge LLM ensemble with
  reported agreement, and ships a human-gold harness to validate it.
- **Novelty search / quality-diversity (QD):** rewards behavioural novelty. We
  keep novelty and quality on *separate* axes for the same reason QD separates
  them — novelty alone is not fitness.

**The contribution** is the combination: a *feasibility-constrained, judge-
scored, reference-anchored* ideation benchmark with novelty separated from
quality, statistical CIs, and built-in judge-validity instrumentation.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # get a key at console.anthropic.com
# OpenRouter (gpt-*-mini, 7B/70B open models): export OPENROUTER_API_KEY=... ; pip install requests
```

## Usage

```bash
.venv/bin/python -m casebench.cli cases                       # list cases

# cheap default (1 sample / 1 judge / 1 pass)
.venv/bin/python -m casebench.cli run

# research-grade — judges DISJOINT from candidates (no self-preference)
.venv/bin/python -m casebench.cli run \
    --models anthropic:claude-haiku-4-5 anthropic:claude-sonnet-4-6 \
             openrouter:openai/gpt-4o-mini openrouter:meta-llama/llama-3.1-70b-instruct \
    --judges anthropic:claude-opus-4-8 anthropic:claude-opus-4-7 \
    --gen-samples 5 --judge-passes 3

.venv/bin/python -m casebench.cli run --limit 1                      # 1-case dry run (cheap)
.venv/bin/python -m casebench.cli ablate --cases bookstore-decline   # batch vs isolated judging
.venv/bin/python -m casebench.cli validate                          # audit cached verdicts
```

Outputs: `results/leaderboard.md`, `results/results.json` (full report incl. CIs,
significance, reliability, novelty validation, power), `results/cache/` (resumable).

## Dataset

16 cases (`data/cases.json`) across retail, operations, healthcare, food-service,
energy, culture, fitness, logistics, B2B-SaaS, fintech, public-sector,
manufacturing, and nonprofit domains, tagged by `archetype` (demand-trough,
cold-start-liquidity, churn-retention, process-throughput, adoption-diffusion,
trust-compliance, waste-efficiency, monetization-pricing, capacity-utilization),
with several explicitly **non-US** contexts. Each case has
16–18 reference mechanisms at consistent granularity with scope notes. Prompts
are audited to state the problem and constraints **without hinting at any
reference mechanism**.

To add a case, append an object with `id`, `title`, `domain`, `archetype`,
`context`, `prompt`, and `reference_answers` (`{id, mechanism, idea, scope_note}`).

## Limitations

- **The judge is a model.** Creativity scoring is subjective; the panel +
  agreement metrics control *consistency*, not ground truth. The judge-vs-gold
  harness exists to measure this, but the shipped gold set is a small
  illustrative seed — **not** a validated standard (see `data/gold/README.md`).
- **Self-preference.** If a candidate also sits on the judge panel it may favour
  its own ideas; the runner now *detects and flags* this and reports a per-model
  cross-judge leniency delta. The canonical board avoids it entirely (disjoint
  judges). Same-family judges (e.g. two Opus versions) are more correlated than
  cross-family ones, which inflates agreement somewhat — read κ with that in mind.
- **Frontier compression.** On two strong, similar models (as in our canonical run)
  the judge uses a narrow band (impact clusters at 2; the feasibility-0 gate rarely
  fires because both models propose feasible ideas), so separation rests mostly on
  originality. Reported figures are from that run; re-running resamples. Add a
  deliberately weaker generator to exercise the gate and widen the spread, and
  read the per-axis sub-metrics, not just the headline.
- **Impact axis is provisional.** Empirically impact is the least-discriminating
  axis (std ≈ 0.43) and has only weak/illustrative human-gold support, so it is
  reported with a caveat. The report includes a **robustness check** confirming the
  Quality ranking is preserved when impact is dropped — i.e. the headline rests on
  feasibility and the better-validated originality axis, not on impact.
- **Diversity ties.** On the current two frontier models the portfolio-diversity
  scores are a statistical tie (overlapping CIs); the board reports this rather
  than printing a spurious ordering. Diversity discriminates more across a wider
  ability range.
- **Reference sets are finite,** not a complete solution frontier — a model can
  earn divergence credit for a real-world-common idea the authors didn't list.
  The lexical baseline and graded (not binary) divergence reduce, but don't
  eliminate, this.
- **Lexical novelty baseline is shallow** (term overlap, not semantics); a
  sentence-embedding baseline would be stronger and is the natural next step.
- **Scale.** 16 cases is enough to demonstrate the methodology, not to settle
  close model comparisons — the printed MDE tells you how many cases a given gap
  needs. Treat overlapping-CI results as ties.

## Project layout

```
casebench/
  providers.py        # provider abstraction (anthropic, openrouter), temperature-aware
  generate.py         # candidate prompt (no reward-hack coaching) + idea parsing
  judge.py            # 4-axis judge, graded divergence, pinned protocol version, batch/isolated
  scoring.py          # feasibility-gated quality + separate divergence board
  stats.py            # bootstrap CIs, paired significance, consolidation, reliability
  novelty_baseline.py # reference-free lexical divergence (judge validation)
  validation.py       # contradiction audit + judge-vs-human-gold agreement
  runner.py           # orchestration, sampling, panel, CIs, reporting
  cli.py              # run / cases / ablate / validate
data/cases.json       # 16 cases
data/gold/            # human-gold harness + protocol
tests/                # offline invariants (test_scoring, test_dataset, test_cli) + live smoke (test_smoke)
results/              # leaderboard, results.json, cache
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 30 offline + 1 live smoke
```

The offline tests run with no API key (the live smoke self-skips). They lock in
the core invariants — no ranking inversion, junk-gating, under-delivery defence,
float consolidation, divergence orthogonality, and dataset integrity.

## License

MIT — see [LICENSE](LICENSE). The case set and gold seeds under `data/` are
released under the same terms.
