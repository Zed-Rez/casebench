# Human gold labels

This directory holds human ratings used to validate the LLM judge (construct
validity). Each file:

```json
{
  "case_id": "bookstore-decline",
  "annotator": "initials-or-id",
  "labels": [
    {
      "idea_text": "<exact idea title: description as the model emitted it>",
      "feasibility": 2, "impact": 2, "originality": 3, "divergence": 2,
      "note": "optional reasoning"
    }
  ]
}
```

`idea_text` must match a generated idea exactly (it is fingerprinted as
`sha1(case_id|idea_text)[:16]`); the runner matches labels to the judge's
verdicts on the same ideas and reports agreement (`judge_vs_human_gold` in
`results.json`).

## Protocol for a real gold set

1. Run the benchmark; collect the generated ideas from `results/cache/gen__*.json`.
2. Recruit ≥3 annotators with domain familiarity. Give them the same rubric the
   judge uses (`casebench/judge.py` `JUDGE_RUBRIC`) and the case + reference set.
3. Each annotator rates each idea on all four axes, blind to the model and to the
   judge's scores.
4. Adjudicate to a consensus label per idea (median, or discuss disagreements).
5. Report inter-annotator agreement, then judge-vs-consensus agreement.

The `seed_*.json` file here is a small **illustrative** set authored by one rater
to exercise the harness — it is **not** a validated gold standard and must not be
cited as one. Replace it with a real multi-annotator set before making validity
claims.
