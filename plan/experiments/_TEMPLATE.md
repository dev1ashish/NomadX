# YYYY-MM-DD — <Title> {#YYYY-MM-DD--slug}

> **Status:** pre-registered | running | complete
> **Stage / track:** (e.g. plan/14 band-chemistry Stage N)
> **Branch hit:** (A / B / C / Z / —)
> **One-line headline:** <30–80 char summary>
> **Cross-refs:** [research plan section](../14_band_chemistry_research.md#…) · [prior experiment](YYYY-MM-DD_prev.md) · [next experiment](YYYY-MM-DD_next.md)

---

## Pre-registration

> Written **before** the experiment runs. Do not edit retroactively — if a prediction needs revision, add a note in the Verdict section explaining why.

### Method

What we're going to do — script names, inputs, configs, evaluation protocol. Anchor at a research plan section if relevant.

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| … | … | … |

### Branching verdicts

- **(A)** what counts as a clean hit → implications
- **(B)** what counts as partial → implications
- **(C)** what counts as a miss → implications
- **(Z)** explicit failure mode worth pre-naming, if any

### Stage-gate

If the experiment lands at (X), the next stage (Y) is/isn't worth running. Be explicit.

---

## Results

> Written **after** the run. The pre-registration above stays untouched.

### Headline

One paragraph. The number, the verdict, the surprise (if any).

### Detailed results

Tables, breakdowns, per-strain numbers, mechanism analysis. Whatever the experiment actually produced.

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---|---|
| … | … | … | ✅ / ⚠️ / ❌ |

### Implications

What this changes about the plan. Operational decisions taken (or to take). Cross-refs to other experiments that should now be re-run, skipped, or re-prioritized.

---

## Artifacts

- `path/to/script.py`
- `path/to/output/file.csv`
- `path/to/figure.png`
- `data_cache/...` if any
