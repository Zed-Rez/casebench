"""Command-line interface for CASE-Bench (v2).

    python -m casebench.cli cases
    python -m casebench.cli run                       # cheap default (1 sample/1 judge/1 pass)
    python -m casebench.cli run --gen-samples 5 --judge-passes 3 \\
        --judges anthropic:claude-opus-4-8 anthropic:claude-sonnet-4-6   # research-grade
    python -m casebench.cli ablate --cases bookstore-decline            # batch vs isolated judging
"""

from __future__ import annotations

import argparse
import sys

from . import judge as judging
from . import runner

DEFAULT_MODELS = [
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-sonnet-4-6",
]


def _select_cases(args) -> list[dict]:
    cases = runner.load_cases()
    if getattr(args, "cases", None):
        wanted = set(args.cases)
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            print(f"error: unknown case id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            sys.exit(2)
    if getattr(args, "limit", None):
        cases = cases[: args.limit]
    return cases


def _cmd_cases(args) -> int:
    cases = runner.load_cases()
    print(f"{len(cases)} cases:\n")
    for c in cases:
        arche = c.get("archetype", "?")
        print(f"  {c['id']:<26} [{c['domain']}/{arche}] {c['title']}")
    return 0


def _cmd_run(args) -> int:
    cases = _select_cases(args)
    cfg = runner.RunConfig(
        models=args.models or DEFAULT_MODELS,
        judges=args.judges or [judging.DEFAULT_JUDGE_MODEL],
        gen_samples=args.gen_samples,
        judge_passes=args.judge_passes,
        n_ideas=args.ideas,
        temperature=args.temperature,
        use_cache=not args.no_cache,
    )
    print(f"CASE-Bench: {len(cfg.models)} model(s) x {len(cases)} case(s) | "
          f"judges={cfg.judges} samples={cfg.gen_samples} passes={cfg.judge_passes} "
          f"temp={cfg.temperature}\n")
    report = runner.run_benchmark(cfg, cases)
    print("\n" + runner.render_leaderboard_md(report))
    json_path, md_path = runner.write_reports(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def _cmd_ablate(args) -> int:
    """Batch-vs-isolated judging ablation: quantify cross-idea contrast bias."""
    import os
    from . import generate as gen
    from . import stats

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY required", file=sys.stderr)
        return 2
    cases = _select_cases(args)
    model = (args.models or DEFAULT_MODELS)[0]
    jm = (args.judges or [judging.DEFAULT_JUDGE_MODEL])[0]
    print(f"Ablation: {model} judged by {jm}, batch vs isolated, on {len(cases)} case(s)\n")
    deltas = []
    for c in cases:
        g = gen.generate_for_case(model, c, n=args.ideas, temperature=args.temperature)
        batch = {v.index: v for v in judging.judge_case(c, g.ideas, judge_model=jm)}
        for i in range(len(g.ideas)):
            if i not in batch:        # judge omitted this idea in the batch pass
                continue
            iso = judging.judge_idea_isolated(c, g.ideas, i, judge_model=jm)
            for axis in stats.AXES:
                deltas.append(abs(getattr(batch[i], axis) - getattr(iso, axis)))
        print(f"  {c['id']}: judged {len(g.ideas)} ideas batch+isolated")
    mean_abs = sum(deltas) / len(deltas) if deltas else 0.0
    print(f"\nMean |batch - isolated| score delta across axes: {mean_abs:.3f} "
          f"(0 = batch judging introduces no contrast bias)")
    return 0


def _cmd_validate(args) -> int:
    """Re-audit cached verdicts for internal contradictions (no API calls)."""
    import json
    from . import validation

    files = sorted(runner.CACHE_DIR.glob("judge__*.json"))
    if not files:
        print("no cached verdicts found — run the benchmark first", file=sys.stderr)
        return 2
    total, flagged = 0, 0
    by_type: dict[str, int] = {}
    for f in files:
        blob = json.loads(f.read_text())
        verdicts = [judging.Verdict(**v) for v in blob["verdicts"]]
        total += len(verdicts)
        for fl in validation.audit_contradictions(verdicts):
            flagged += 1
            by_type[fl["type"]] = by_type.get(fl["type"], 0) + 1
    print(f"Audited {total} verdicts across {len(files)} cached judge calls.")
    print(f"Contradiction flags: {flagged} ({flagged/max(1,total)*100:.1f}%)")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="casebench", description="CASE-Bench v2: feasibility-constrained ideation benchmark.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("cases", help="list available cases").set_defaults(func=_cmd_cases)

    def add_common(sp):
        sp.add_argument("--models", nargs="+", help="candidate model specs (provider:model_id)")
        sp.add_argument("--judges", nargs="+", help="judge model panel (>=2 enables reliability metrics)")
        sp.add_argument("--cases", nargs="+", help="restrict to these case ids")
        sp.add_argument("--limit", type=int, help="use only the first N cases")
        sp.add_argument("--ideas", type=int, default=5, help="ideas per case (default 5)")
        sp.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (default 1.0; dropped for models that reject it)")

    pr = sub.add_parser("run", help="run the benchmark")
    add_common(pr)
    pr.add_argument("--gen-samples", type=int, default=1, help="independent idea-sets per (model,case)")
    pr.add_argument("--judge-passes", type=int, default=1, help="judge repeats per sample (order shuffled)")
    pr.add_argument("--no-cache", action="store_true", help="ignore and overwrite cache")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("ablate", help="batch-vs-isolated judging ablation")
    add_common(pa)
    pa.set_defaults(func=_cmd_ablate)

    pv = sub.add_parser("validate", help="audit cached verdicts for internal contradictions")
    pv.set_defaults(func=_cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
