"""Candidate-model idea generation for CASE-Bench.

The model is shown a business case (problem + constraints only — the reference
answer set is withheld) and asked for a fixed number of distinct ideas as JSON.

Design note (why the prompt is worded the way it is): an earlier version told the
model "conventional moves will score poorly", which created a reward-hacking
gradient — models learned that exotic, hard-to-pin-down ideas dodge the novelty
penalty even when infeasible. The current prompt asks for solutions that are
*both* genuinely original *and* feasible/effective, and explicitly penalises
vagueness, so the generator is not coached toward the gameable behaviour the
scorer is trying to detect.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import providers

# A fixed idea count keeps scores comparable across models and cases.
IDEAS_PER_CASE = 5

GEN_SYSTEM = (
    "You are an elite innovation strategist and management consultant. You "
    "propose solutions that are simultaneously inventive and genuinely workable: "
    "original enough to give a real edge, yet concrete, feasible, and grounded "
    "enough that an operator could actually execute them. You avoid both bland "
    "boilerplate and impressive-sounding ideas that fall apart on contact with "
    "reality."
)

GEN_USER_TEMPLATE = """\
BUSINESS CASE: {title}

Operating context: {context}

{prompt}

Propose {n} distinct solutions. Each should aim to be BOTH:
  (a) effective and feasible — it would realistically work given the constraints, and
  (b) original — not the first obvious move, but a genuinely good idea.
Be concrete and specific to THIS situation. Do not pad with vague slogans, and do
not propose mechanisms that cannot actually be executed in the stated context.

Return ONLY a JSON array of exactly {n} objects, no prose before or after:
[
  {{"title": "<short name>", "description": "<2-4 sentences: what it is, how it works in this context, and why it would succeed>"}}
]
"""


@dataclass
class Idea:
    title: str
    description: str

    def as_text(self) -> str:
        return f"{self.title}: {self.description}".strip()


def build_prompt(case: dict, n: int = IDEAS_PER_CASE) -> tuple[str, str]:
    user = GEN_USER_TEMPLATE.format(
        title=case["title"],
        context=case.get("context", "(not specified)"),
        prompt=case["prompt"],
        n=n,
    )
    return GEN_SYSTEM, user


def _extract_json_array(text: str) -> list:
    """Pull the first JSON array out of a model completion, tolerantly."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in model output")
    return json.loads(candidate[start : end + 1])


def parse_ideas(text: str) -> list[Idea]:
    raw = _extract_json_array(text)
    ideas: list[Idea] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not (title or desc):
            continue
        ideas.append(Idea(title=title or "(untitled)", description=desc))
    if not ideas:
        raise ValueError("parsed JSON array contained no usable ideas")
    return ideas


@dataclass
class Generation:
    case_id: str
    model: str
    ideas: list[Idea]
    raw_text: str
    input_tokens: int
    output_tokens: int
    temperature: float | None = None


def generate_for_case(
    model: str,
    case: dict,
    n: int = IDEAS_PER_CASE,
    temperature: float | None = None,
) -> Generation:
    system, user = build_prompt(case, n)
    result = providers.generate(model, system, user, max_tokens=4000, temperature=temperature)
    ideas = parse_ideas(result.text)
    return Generation(
        case_id=case["id"],
        model=model,
        ideas=ideas,
        raw_text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        temperature=result.temperature,
    )
