# 09 — Future work (out of scope for this take-home)

> **Mutability:** stable. Items here belong in the final README's "Future Work" section, NOT in the submission scope.
> **Purpose:** show reviewers we know what's possible — and that the things we *did* ship are the right choices for the budget.

---

- **Self-supervised pretraining (Raman foundation model).** Pretrain a transformer with masked-spectrum modeling on a large pool of unlabeled Raman spectra (our ~10K plus public databases like RRUFF, SLOPP, and any large bacterial Raman corpora), then fine-tune on the 4-class task. Standard recipe for closing the gap when the labeled dataset is small. Expected effort: 10–15 working days + GPU cost. Expected gain: 3–7 macro-F1 points based on published Raman SSL papers. Most likely to help the hard STEC vs Non-STEC distinction.
- **External labeled-data augmentation.** Pull labeled Raman bacteria datasets from public sources (NIH SERS-Raman corpora, university group releases) and merge with ours. Risk: instrument-domain shift between corpora.
- **Hierarchical classifier.** Two-stage model: a coarse classifier separates Bacteria vs H₂O, then a fine classifier separates STEC / Non-STEC / Salmonella. Mirrors the natural taxonomy and can let each stage focus on its own decision boundary.
- **Dedicated binary STEC vs Non-STEC submodel.** Train a separate model on just E. coli files for the hardest binary call. Useful if the 4-way model's confusion matrix shows the bulk of errors concentrate there.
- **Active learning loop.** Once deployed, route low-confidence predictions to human review; use their labels as new training data. Closes the small-dataset gap incrementally without a big upfront cost.
- **Cross-instrument generalization study.** All 87 files came from one instrument. A production system needs to work across different Raman setups; a follow-up should explicitly test (and possibly fine-tune for) cross-instrument transfer.
- **Cell-level vs colony-level mapping.** Current data is sub-cellular (1µm steps on individual bacteria). For real-world use, mapping a full colony or food smear is the relevant unit — would need new data collection plus aggregation logic above the file level.
- **Architecturally-diverse ensembling.** First-pass soft-vote, stacking, and confidence-routing ensembles all failed when the base models were too similar (within-DANN-family). Now that we have four meaningfully-different deep architectures with documented complementary per-strain wins (PLS-DA on Salmonella; DANN λ=0.3 on K-12 via adversarial denoising; Patch=5 Transformer on ATCC25922 + O157H7 via narrow-peak attention; 2-channel CNN on O121H19 + O157H7 via explicit edges), re-running soft-vote + stacking + router on this richer base set is a clean follow-up. Note: queued in `plan/00 §what's next` as an in-scope research item, not strictly future work — if it lands during the current research window, the readme will reflect it.
