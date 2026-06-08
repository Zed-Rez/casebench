"""LLM-as-judge for CASE-Bench (v2 protocol).

Each candidate idea gets four independent ratings plus a reference comparison:

- **feasibility** (0-3): is it coherent and actually executable in the stated
  context? 0 = incoherent / self-contradictory / impossible. This is a GATE:
  an idea the scorer treats as junk must land here, not be hidden behind a high
  novelty score.
- **impact** (0-3): if executed, how much would it move the needle on the
  stated problem? The judge must name a concrete failure risk before awarding 3.
- **originality** (0-3): how inventive / non-obvious the idea is *on its own
  terms*, judged **independently of the reference list** — this is the single
  intrinsic creativity channel, so it is not double-counted against divergence.
- **divergence** (0-3): graded distance from the *nearest* reference answer
  (0 = same core mechanism, 3 = a genuinely distinct mechanism not in the set).
  This replaces v1's binary match and is reported as a separate *diagnostic*,
  never folded into the idea-quality score.

The judge sees the reference set (with per-reference scope notes) and reports the
nearest reference and any others the idea also covers, so divergence is grounded
in explicit mechanism mapping rather than a yes/no guess.

A single fixed judge model with structured outputs keeps verdicts machine-
parseable and consistent; ``runner`` drives a *panel* of judges and repeated
passes (with idea order shuffled) on top of this to estimate reliability.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic

from .generate import Idea

# Bump when the rubric or schema changes; recorded in every result for archival.
JUDGE_PROTOCOL_VERSION = "v2.1"
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

JUDGE_SYSTEM = (
    "You are a rigorous, calibrated benchmark judge scoring proposed business "
    "solutions on feasibility, impact, and originality, and measuring how far "
    "each diverges from a set of known reference solutions. You are strict and "
    "evidence-based: you do not reward confident-sounding nonsense, unverifiable "
    "numbers, or vague slogans. Treat unsupported quantitative claims as "
    "marketing, not evidence."
)

JUDGE_RUBRIC = """\
SCORING RUBRIC — apply each axis independently.

feasibility (0-3): coherence + executability in the stated context.
  0 = incoherent, self-contradictory, factually impossible, or not a real
      solution. ANY internally contradictory or non-executable idea MUST be 0.
  1 = directionally plausible but vague or with serious unresolved obstacles.
  2 = clearly executable with normal effort.
  3 = executable and robust; obstacles are minor and addressable.

impact (0-3): if executed, effect on the stated problem. USE THE FULL RANGE —
  do not default to 2. Most ideas are a 1 or a 2; reserve 3 for the rare standout.
  0 = negligible, irrelevant, or counterproductive.
  1 = marginal — a real but small or peripheral effect; the DEFAULT for ordinary
      ideas that help only a little or touch a secondary lever.
  2 = solid — a meaningful improvement on a primary lever of the stated problem.
  3 = decisive — would substantially move the core metric. Reserve for clear
      standouts. You MUST name one concrete, non-generic failure risk (specific to
      THIS idea, not "execution risk") in the rationale; if you cannot, score 2.

originality (0-3): inventiveness / non-obviousness ON ITS OWN MERITS. Judge this
  independently of the reference list — an idea can be original yet overlap a
  reference, or unoriginal yet absent from the list.
  0 = obvious first move / boilerplate.
  1 = mildly fresh framing.
  2 = clearly non-obvious, a sharp practitioner might miss it.
  3 = strikingly inventive yet sensible.

divergence (0-3): distance from the NEAREST reference answer (mechanism-level).
  0 = same core mechanism as a reference (a paraphrase or restatement).
  1 = a reference's mechanism in a new wrapper or minor combination.
  2 = adjacent to references but a distinct lever.
  3 = a genuinely distinct mechanism not represented in the reference set.
  Map on the CORE MECHANISM using each reference's scope_note, not surface words.
"""

JUDGE_USER_TEMPLATE = """\
BUSINESS CASE: {title}

Operating context: {context}

{prompt}

REFERENCE ANSWERS (known solutions, each with the range of phrasings it covers):
{references}

{rubric}

CANDIDATE IDEAS to evaluate (index : idea):
{candidates}

For EACH candidate idea return a verdict with:
- index: the candidate's index.
- feasibility, impact, originality, divergence: integers 0-3 per the rubric.
- nearest_reference_id: the id (e.g. "r3") of the closest reference, or null only
  if NO reference is even adjacent.
- also_covers: list of other reference ids the idea substantially equals (may be
  empty).
- failure_risk: one concrete way this idea could fail in practice.
- rationale: one concise sentence justifying the scores.

Score each idea on its own merits. Do not inflate. Originality and divergence are
different questions — answer both honestly.
"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "feasibility": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "impact": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "originality": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "divergence": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "nearest_reference_id": {"type": ["string", "null"]},
                    "also_covers": {"type": "array", "items": {"type": "string"}},
                    "failure_risk": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "index",
                    "feasibility",
                    "impact",
                    "originality",
                    "divergence",
                    "nearest_reference_id",
                    "also_covers",
                    "failure_risk",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    index: int
    feasibility: int
    impact: int
    originality: int
    divergence: int
    nearest_reference_id: str | None
    also_covers: list[str]
    failure_risk: str
    rationale: str
    judge_model: str = ""

    @property
    def is_invalid(self) -> bool:
        """Incoherent / not-a-real-solution — gated to zero by the scorer.

        ``<= 0`` (not ``== 0``) so it stays correct for *consolidated* float
        feasibility, where a panel that unanimously says 0 averages to 0.0.
        """
        return self.feasibility <= 0

    @property
    def is_known_move(self) -> bool:
        """Essentially a reference mechanism (divergence 0-1) — a diagnostic."""
        return self.divergence <= 1


_judge_client: anthropic.Anthropic | None = None


def _normalize_judge(model: str) -> str:
    """Judges run on Anthropic; accept either ``claude-...`` or ``anthropic:claude-...``."""
    if ":" in model:
        provider, model_id = model.split(":", 1)
        if provider.strip().lower() != "anthropic":
            raise ValueError(
                f"judge must be an Anthropic model (structured outputs); got {model!r}"
            )
        return model_id.strip()
    return model.strip()


def _client() -> anthropic.Anthropic:
    global _judge_client
    if _judge_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (required for the judge).")
        _judge_client = anthropic.Anthropic()
    return _judge_client


def _format_references(case: dict) -> str:
    lines = []
    for r in case["reference_answers"]:
        mech = r.get("mechanism", "")
        scope = r.get("scope_note", "")
        head = f"- {r['id']}"
        if mech:
            head += f" [{mech}]"
        head += f": {r['idea']}"
        if scope:
            head += f"  (counts as: {scope})"
        lines.append(head)
    return "\n".join(lines)


def _format_candidates(ideas: list[Idea], order: list[int]) -> str:
    # `order` maps display position -> original idea index, so we can shuffle
    # presentation while keeping stable indices in the returned verdicts.
    return "\n".join(f"{orig} : {ideas[orig].as_text()}" for orig in order)


def judge_case(
    case: dict,
    ideas: list[Idea],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    order: list[int] | None = None,
) -> list[Verdict]:
    """Judge all ideas of a case in one call (lets the judge cross-check matches).

    ``order`` controls the presentation order of ideas (for order-bias passes);
    verdicts are always returned sorted by the idea's original index.
    """
    if order is None:
        order = list(range(len(ideas)))

    model_id = _normalize_judge(judge_model)
    user = JUDGE_USER_TEMPLATE.format(
        title=case["title"],
        context=case.get("context", "(not specified)"),
        prompt=case["prompt"],
        references=_format_references(case),
        rubric=JUDGE_RUBRIC,
        candidates=_format_candidates(ideas, order),
    )
    resp = _client().messages.create(
        model=model_id,
        max_tokens=4000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    if not text.strip():
        raise ValueError(f"judge {model_id} returned no text block (refusal or empty response)")
    # Record the spec as given so panels with provider prefixes stay distinct.
    return _parse_verdicts(json.loads(text), case, judge_model)


def judge_idea_isolated(
    case: dict,
    ideas: list[Idea],
    index: int,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Verdict:
    """Judge a single idea with no sibling ideas in context (independence mode).

    Used by the batch-vs-isolated ablation to quantify cross-idea contrast bias.
    """
    verdicts = judge_case(case, [ideas[index]], judge_model=judge_model, order=[0])
    v = verdicts[0]
    v.index = index
    return v


def _parse_verdicts(data: dict, case: dict, judge_model: str) -> list[Verdict]:
    valid_ref_ids = {r["id"] for r in case["reference_answers"]}
    verdicts: list[Verdict] = []
    for v in data["verdicts"]:
        ref = v.get("nearest_reference_id")
        if ref is not None and ref not in valid_ref_ids:
            ref = None
        also = [rid for rid in (v.get("also_covers") or []) if rid in valid_ref_ids]
        verdicts.append(
            Verdict(
                index=int(v["index"]),
                feasibility=int(v["feasibility"]),
                impact=int(v["impact"]),
                originality=int(v["originality"]),
                divergence=int(v["divergence"]),
                nearest_reference_id=ref,
                also_covers=also,
                failure_risk=str(v.get("failure_risk", "")),
                rationale=str(v.get("rationale", "")),
                judge_model=judge_model,
            )
        )
    verdicts.sort(key=lambda x: x.index)
    return verdicts
