# Atlas Plan — Index

**Project.** Take-home: classify Raman hyperspectral maps into {STEC, Non-STEC, Salmonella, H₂O} with subclass-aware evaluation. 87 files, ~10K spectra, 4 primary classes / 9 bacterial subclasses + water.

**Start here in every new session:** read `00_status.md` to know where we are, then load the topic file you need.

**Adding new experimental results?** Don't append to `07_findings.md` / `08_expectations.md` anymore — they're capped at the 2026-05-15 historical content. New experiments live as standalone files in [`experiments/`](experiments/README.md). Copy [`experiments/_TEMPLATE.md`](experiments/_TEMPLATE.md), fill it in, add one row to [`experiments/README.md`](experiments/README.md).

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
| 07 | [findings.md](07_findings.md) | **legacy log** (frozen) | historical findings 2026-05-14 → 2026-05-15. New entries go to `experiments/`. |
| 08 | [expectations.md](08_expectations.md) | **legacy log** (frozen) | historical pre-regs 2026-05-14 → 2026-05-15. New entries go to `experiments/`. |
| 09 | [future_work.md](09_future_work.md) | stable | when writing the final README |
| 10 | [decision_log.md](10_decision_log.md) | **append** | when a decision changes and you need provenance |
| 11 | [references.md](11_references.md) | **append** | when interpreting biology / chemistry / domain-shift findings — published papers indexed by tag |
| 12 | [data_gaps_and_external_datasets.md](12_data_gaps_and_external_datasets.md) | mutable | when planning external data integration / data-side research |
| 13 | [methods_research_synthesis.md](13_methods_research_synthesis.md) | mutable | when scoping the methods/representation track (MCR-ALS, SSL, N-PLS, cross-corpus eval) |
| 14 | [band_chemistry_research.md](14_band_chemistry_research.md) | stable | when building chemistry-grounded features, band-aware analyses, or writing about discrimination in spectroscopy vocabulary |
| 15 | [feature_engineering_research.md](15_feature_engineering_research.md) | stable | when extending feature engineering beyond plan/14's band-AUC catalog: spectral processing, biology-specific markers, MCR-ALS/NMF, spatial features. Anchor for the 130-feature implementation track. |
| — | [experiments/](experiments/README.md) | **per-experiment shards** | one file per experiment (pre-reg + method + results + verdict in one place). [`experiments/README.md`](experiments/README.md) is the live index. New work goes here, NOT in 07/08. |

---

## Mutability semantics

- **stable** — content rarely changes. Don't rewrite without thinking. If a stable file changes, also add an entry to `10_decision_log.md` explaining why.
- **append** — new entries get added, old entries stay. Never edit historical entries; if something turns out wrong, append a correction.
- **mutable** — content is regularly rewritten as state changes. No audit trail expected (the audit lives in `10_decision_log.md`).
- **legacy log (frozen)** — was append-only; capped at the date noted at the top of the file. New entries of that type go to `experiments/` instead.
- **per-experiment shards** — each file is the full lifecycle of one experiment. Never re-edit a completed shard except to mark a status flip; create a new shard for follow-ups.

## Conventions

- Every dated entry uses `YYYY-MM-DD`. Never relative dates ("yesterday", "last session").
- When citing a finding, link it: `see [stage 5](experiments/2026-05-17_stage5_band_classifier.md)`. Anchor links keep references stable across reorganizations; the `{#anchor}` IDs from the legacy 07/08 are preserved on each migrated shard's H1 line.
- For per-experiment files: pre-register predictions *before* running, fill in results *after* — both in the same file. That's how we keep ourselves honest.
- The repo root has a `PLAN.md` that just redirects here. Treat that file as a pointer, not content.
