"""CASE-Bench: a feasibility-constrained business-ideation benchmark for LLMs.

CASE = Case-based Answer-Set Evaluation. A fixed LLM judge rates each idea on
feasibility, impact, and originality independently; reference overlap is reported
only as a divergence/diversity *diagnostic*, never as a quality signal. Models are
scored on producing feasible, high-impact, original solutions to business
problems — not on novelty against a reference set.
"""

__version__ = "2.0.0"
