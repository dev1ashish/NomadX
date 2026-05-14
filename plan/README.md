# Atlas Plan — Index

**Project.** Take-home: classify Raman hyperspectral maps into {STEC, Non-STEC, Salmonella, H₂O} with subclass-aware evaluation. 87 files, ~10K spectra, 4 primary classes / 9 bacterial subclasses + water.

**Start here in every new session:** read `00_status.md` to know where we are, then load the topic file you need.

---

## Files

| # | File | Mutability | When to read |
|---|---|---|---|
| 00 | [status.md](00_status.md) | **mutable** | every session — current phase, what's done, what's next |
| 01 | [data.md](01_data.md) | stable | when touching the data pipeline / parser |
| 02 | [decisions.md](02_decisions.md) | stable | when planning new work that might revisit a settled choice |
| 03 | [architecture.md](03_architecture.md) | stable | when building or modifying a module |
| 04 | [eda_plan.md](04_eda_plan.md) | stable | when extending the EDA notebook |
| 05 | [implementation_order.md](05_implementation_order.md) | stable | when picking the next thing to build |
| 06 | [risks.md](06_risks.md) | stable | when something looks wrong — check if it's a known risk |
| 07 | [findings.md](07_findings.md) | **append** | when interpreting results or after running new experiments |
| 08 | [expectations.md](08_expectations.md) | **append** | before claiming a result is good/bad — check the pre-registered range |
| 09 | [future_work.md](09_future_work.md) | stable | when writing the final README |
| 10 | [decision_log.md](10_decision_log.md) | **append** | when a decision changes and you need provenance |
| 11 | [references.md](11_references.md) | **append** | when interpreting biology / chemistry / domain-shift findings — published papers indexed by tag |

---

## Mutability semantics

- **stable** — content rarely changes. Don't rewrite without thinking. If a stable file changes, also add an entry to `10_decision_log.md` explaining why.
- **append** — new entries get added, old entries stay. Never edit historical entries; if something turns out wrong, append a correction.
- **mutable** — content is regularly rewritten as state changes. No audit trail expected (the audit lives in `10_decision_log.md`).

## Conventions

- Every entry in append-only files is dated `YYYY-MM-DD`. Never use relative dates ("yesterday", "last session").
- When citing a finding, link it: `see [findings.md §batch-effect](07_findings.md#batch-effect)`. Anchor links keep references stable across reorganizations.
- Predictions go in `expectations.md` *before* the experiment runs. When the experiment lands, append the actual measured value next to the prediction — that's how we keep ourselves honest.
- The repo root has a `PLAN.md` that just redirects here. Treat that file as a pointer, not content.
