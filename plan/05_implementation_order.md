# 05 — Implementation order

> **Mutability:** stable. Reorder only if a dependency truly demands it.

1. ✅ `atlas/io.py` + `scripts/build_dataset.py` → build cache, verify all 87 files parse.
2. ✅ EDA notebook blocks 1–8 → sanity-check the data.
3. ✅ `atlas/preprocess.py` + `atlas/qc.py` → apply pipeline; cache preprocessed array; Block 9 in notebook.
4. ⏳ **`atlas/splits.py`** → both protocols, serialize fold artifacts as JSON.
5. ⏳ `atlas/models_classical.py` + `atlas/evaluate.py` → run all 6 classical models under Protocol A, then LOSO.
6. ⏳ `atlas/models_cnn.py` + `atlas/train.py` → small variant under Protocol A, then LOSO; medium only if compute permits.
7. ⏳ `atlas/models_transformer.py` → small 1D-Transformer under Protocol A + LOSO. Report alongside CNN.
8. ⏳ Memorization probe → decide whether to enable DANN.
9. ⏳ Finalize plots + README narrative + future-work section.
10. ⏳ `make verify` (smoke test on synthetic data) + CI green.

## Cross-cutting (do as part of whichever step)

- Add an `arPLS boundary artifact` fix (crop start 450 cm⁻¹ instead of 400) — slot into the next time we touch preprocessing or splits.
- Pre-register every experiment's expected range in `08_expectations.md` *before* running, then append the actual result.
