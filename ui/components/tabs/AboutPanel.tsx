"use client";

/**
 * About / context panel — plain-English guide for reviewers.
 *
 * Six sections:
 *  1. What this is
 *  2. The data
 *  3. The model
 *  4. Glossary (acronyms + jargon decoded)
 *  5. Limitations & difficulties
 *  6. FAQ
 *  7. Deploy this yourself
 */
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDownIcon } from "lucide-react";
import { cn } from "@/lib/cn";

type SectionKey =
  | "what"
  | "data"
  | "model"
  | "glossary"
  | "limits"
  | "faq";

const SECTIONS: { key: SectionKey; label: string }[] = [
  { key: "what", label: "What this is" },
  { key: "data", label: "The data" },
  { key: "model", label: "The model" },
  { key: "glossary", label: "Glossary" },
  { key: "limits", label: "Limitations & difficulties" },
  { key: "faq", label: "FAQ" },
];

export function AboutPanel() {
  return (
    <section className="px-8 lg:px-14 py-10 max-w-4xl">
      {/* Hero */}
      <div className="flex flex-col gap-2 mb-10">
        <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-accent">
          ❔ Project context for reviewers
        </span>
        <h2 className="font-display text-[clamp(2rem,4vw,3.25rem)] leading-[1.05] text-nx-fg">
          About this project
        </h2>
        <p className="text-nx-fg/55 max-w-2xl text-sm">
          Atlas Raman is a four-class bacterial classifier built from 87
          confocal-Raman hyperspectral maps. This page explains the dataset,
          the model, the jargon, the honest limitations, and how to deploy
          everything yourself.
        </p>
      </div>

      {/* TOC */}
      <nav className="flex flex-wrap gap-2 mb-10 pb-6 border-b border-nx-muted/40">
        {SECTIONS.map((s) => (
          <a
            key={s.key}
            href={`#${s.key}`}
            className="font-mono text-[0.65rem] uppercase tracking-[0.16em] text-nx-fg/55 hover:text-nx-accent border border-nx-muted/40 hover:border-nx-accent/60 rounded-sm px-2.5 py-1 transition-colors"
          >
            {s.label}
          </a>
        ))}
      </nav>

      <article className="prose prose-invert max-w-none">
        <WhatSection />
        <DataSection />
        <ModelSection />
        <GlossarySection />
        <LimitsSection />
        <FaqSection />
      </article>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Section components
// ---------------------------------------------------------------------------

function H2({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <h2
      id={id}
      className="font-display text-[1.8rem] text-nx-accent leading-tight mt-12 mb-4 scroll-mt-8"
    >
      {children}
    </h2>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-nx-fg/75 leading-relaxed mb-4 text-[0.95rem]">
      {children}
    </p>
  );
}

function WhatSection() {
  return (
    <section>
      <H2 id="what">What this is</H2>
      <P>
        This is a take-home project for NomadX. Goal: train a classifier that
        tells four kinds of liquid samples apart from their{" "}
        <Tooltip term="Raman">Raman</Tooltip> spectra — three bacterial pathogens
        and a water blank. The dataset is 87 single-cell Raman maps collected
        on a confocal microscope, totaling ~7,000 individual spectra after
        quality control.
      </P>
      <P>
        The four classes are <ClassPill k="STEC" /> (pathogenic <i>E. coli</i>),{" "}
        <ClassPill k="Non-STEC" /> (commensal / lab <i>E. coli</i>),{" "}
        <ClassPill k="Salmonella" /> (
        <i>S. enterica</i>), and <ClassPill k="H2O" /> (water blanks — the
        "what does the substrate alone look like?" control).
      </P>
      <P>
        The honest headline:{" "}
        <strong className="text-nx-fg">
          PLS-DA on the raw spectrum gets 60.3% mean parent-class recall under
          strain-level cross-validation.
        </strong>{" "}
        That's the project record. We tried 1D-CNNs, Transformers, domain-
        adversarial nets, five ensembling schemes, and 259 hand-engineered
        features — nothing beat plain PLS-DA on the raw 987-bin spectrum.
        Reasons covered in <a href="#limits" className="text-nx-accent">Limitations</a>.
      </P>
    </section>
  );
}

function DataSection() {
  return (
    <section>
      <H2 id="data">The data</H2>
      <P>
        87 tab-delimited <code className="font-mono text-nx-accent">.xls</code>{" "}
        files (yes, tab-delimited with an <code>.xls</code> extension — Raman
        instrument quirk). Each file is one bacterial culture, imaged
        pixel-by-pixel on a confocal Raman microscope. Per pixel: a full 2048-bin
        Raman spectrum.
      </P>
      <ul className="my-4 space-y-2 list-none">
        <DataRow label="Files" value="87" />
        <DataRow label="Classes" value="4 (STEC · Non-STEC · Salmonella · H₂O)" />
        <DataRow label="Bacterial strains" value="9 (3 per bacterial class, + H₂O)" />
        <DataRow label="Spectra after QC" value="7,122 of 7,999 (89% retention)" />
        <DataRow label="Wavenumber bins" value="987 (cropped from 2048, 400–1800 + 2800–3050 cm⁻¹)" />
        <DataRow label="Source" value="Single Raman microscope, one lab, Feb-Mar 2026" />
      </ul>
      <P>
        After preprocessing (cosmic-ray removal → arPLS baseline subtraction →
        Savitzky-Golay smoothing → crop to chemistry-informative regions → SNV
        normalization — see the <a href="/preprocessing" className="text-nx-accent">Preprocessing</a> tab
        for the full animation), every spectrum lives on a canonical 987-bin
        wavenumber axis.
      </P>
    </section>
  );
}

function ModelSection() {
  return (
    <section>
      <H2 id="model">The model</H2>
      <P>
        The deployed classifier is a{" "}
        <strong className="text-nx-fg">logistic regression with L2 regularization</strong>,
        trained on 35 hand-picked features selected via mutual information from
        a pool of 259. Those 259 features come from five families:
      </P>
      <ul className="my-4 space-y-2 text-[0.9rem] text-nx-fg/70 list-disc list-inside marker:text-nx-accent">
        <li>
          <strong className="text-nx-fg">Band fits</strong> (166) — pseudo-Voigt
          curve fits at 30 named Raman bands.
        </li>
        <li>
          <strong className="text-nx-fg">Spectral features</strong> (51) —
          wavelet energies, ROI-PCA components, spectral-angle similarities.
        </li>
        <li>
          <strong className="text-nx-fg">MCR-ALS components</strong> (32) —
          per-pixel concentrations of 7 unmixed "pure" sources (substrate +
          biology) — see the <a href="/mcr" className="text-nx-accent">MCR-ALS</a> tab.
        </li>
        <li>
          <strong className="text-nx-fg">Spatial moments</strong> (10) —
          variance / skew / kurtosis of intensity over the pixel grid.
        </li>
        <li>
          <strong className="text-nx-fg">Biology aggregates</strong> (subset) —
          biochemistry-grounded ratios (α-helix score, Trp indole environment,
          cytochrome-c oxidation state, etc.).
        </li>
      </ul>
      <P>
        For full performance numbers — confusion matrix, per-strain recall,
        bootstrap CI, paired McNemar test, and the algorithm bake-off — see
        the{" "}
        <a href="/results" className="text-nx-accent">
          Results
        </a>{" "}
        tab. Reading them out of context is misleading: chance for a 4-class
        problem with 9-strain LOSO is 25%, and the dataset size (87 files)
        sets the realistic ceiling — covered in{" "}
        <a href="#limits" className="text-nx-accent">
          Limitations
        </a>
        .
      </P>
    </section>
  );
}

function GlossarySection() {
  return (
    <section>
      <H2 id="glossary">Glossary</H2>
      <P>Acronyms and jargon used throughout the UI.</P>
      <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-[0.9rem] mt-6">
        {[
          {
            term: "Raman spectroscopy",
            def: "Shine a laser at a sample. Most light scatters back at the same wavelength (boring), but a tiny fraction scatters at shifted wavelengths because it exchanged energy with molecular vibrations. The shift pattern (the \"Raman spectrum\") is a molecular fingerprint.",
          },
          {
            term: "STEC",
            def: "Shiga-toxin-producing E. coli. Foodborne pathogen (O157:H7 etc.). Defined by carrying a single phage-encoded toxin gene — biochemically nearly identical to ordinary E. coli otherwise.",
          },
          {
            term: "Non-STEC",
            def: "Commensal or laboratory E. coli (K-12, ATCC25922, 83972). Same species as STEC, no toxin. Biologically extremely close to STEC.",
          },
          {
            term: "LOSO",
            def: "Leave-One-Strain-Out cross-validation. The model is trained on 2 of 3 strains per bacterial class and tested on the held-out strain. Tests whether the model generalizes to bacteria it has never seen — much harder than leaving out individual files.",
          },
          {
            term: "Cohen's d",
            def: "Standardized effect size for the difference between two distributions: (mean₁ − mean₂) / pooled_std. |d| ≥ 0.8 is \"large\". Used everywhere here to rank features.",
          },
          {
            term: "PLS-DA",
            def: "Partial Least Squares Discriminant Analysis. Project the high-dimensional spectrum onto a few latent variables that maximize class separation, then classify. Surprisingly hard to beat on Raman data.",
          },
          {
            term: "Mutual information (MI)",
            def: "Information-theoretic measure of how much one variable tells you about another. Used to rank features by their statistical relationship to the class label and pick the top 35.",
          },
          {
            term: "MCR-ALS",
            def: "Multivariate Curve Resolution — Alternating Least Squares. Unsupervised matrix factorization that decomposes spectra into K \"pure\" components and their per-pixel concentrations. Sees the 7 dominant signal sources in our data (substrate, lipids, protein, nucleic acids, etc.).",
          },
          {
            term: "SNV",
            def: "Standard Normal Variate — z-score normalization per spectrum (subtract mean, divide by std). Corrects multiplicative scatter from focus and cell-density variation.",
          },
          {
            term: "arPLS",
            def: "Asymmetrically-Reweighted Penalized Least Squares — algorithm that fits and subtracts the slow-varying baseline (fluorescence) under the Raman peaks.",
          },
          {
            term: "Savitzky-Golay",
            def: "A polynomial-fitting smoothing filter. Removes high-frequency CCD noise while preserving Raman peak shape.",
          },
          {
            term: "Stage 15A–F",
            def: "Internal milestone labels for the engineered-feature track (band fits, spectral descriptors, MCR-ALS unmixing, biology aggregates, spatial moments, final production model). Treat these as section numbers; the actual content is what matters.",
          },
          {
            term: "McNemar's test",
            def: "Paired statistical test for two classifiers on the same examples. Counts \"A right, B wrong\" vs \"A wrong, B right\" disagreement pairs. Used here to confirm LogReg-L2 > PLS-DA on the engineered feature set (p = 0.002).",
          },
          {
            term: "Bootstrap CI",
            def: "Resample the test set with replacement 5,000 times, recompute accuracy each time, take the 2.5th / 97.5th percentiles. Quantifies uncertainty without parametric assumptions.",
          },
          {
            term: "Cisek-2013 triple",
            def: "Three Raman bands (1338 / 1454 / 1658 cm⁻¹) that Cisek et al. (2013, Analyst) reported as STEC discriminators in their controlled cell-suspension study. On our 87-file corpus they don't replicate at file level — covered in Limitations.",
          },
          {
            term: "LPS",
            def: "Lipopolysaccharide — the outer-membrane macromolecule that distinguishes bacterial serotypes. Our empirical strongest STEC ↔ Non-STEC discriminator sits in the LPS-chain region (1117 / 1194 cm⁻¹), not the literature triple.",
          },
        ].map((row) => (
          <div key={row.term} className="border-l border-nx-accent/40 pl-3">
            <dt className="font-mono text-[0.78rem] text-nx-accent mb-1">
              {row.term}
            </dt>
            <dd className="text-nx-fg/70 text-[0.85rem] leading-relaxed">
              {row.def}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function LimitsSection() {
  return (
    <section>
      <H2 id="limits">Limitations & difficulties</H2>
      <P>An honest accounting. The model works; here&apos;s what it doesn&apos;t do and why.</P>

      <Limit
        title="STEC ↔ Non-STEC is virulence-defined, not phylogenetic"
        body={
          <>
            The difference between pathogenic and commensal{" "}
            <i>E. coli</i> in our dataset is one Shiga-toxin gene. That gene is
            phage-encoded — neither the toxin protein nor anything downstream of
            it shows up reliably in bulk Raman signal. We&apos;re asking the
            model to find a needle that mostly isn&apos;t in the haystack.
            Expect this boundary to stay thin.
          </>
        }
      />

      <Limit
        title="87 files is a small dataset"
        body={
          <>
            After engineering 259 features, we have 3 features per file. With 9
            held-out strains in LOSO, the fold-mean accuracy is a 9-point
            statistic — confidence intervals are wide and any single number
            should be read with that in mind. The 5,000-iter file-bootstrap CI
            <code className="font-mono text-nx-accent ml-1">[0.345, 0.552]</code>
            {" "}is the honest uncertainty band.
          </>
        }
      />

      <Limit
        title="K-12 is a 100-year laboratory derivative"
        body={
          <>
            K-12 has been propagated in labs since 1922 and has lost large
            chunks of its native genome (notably outer-membrane LPS structure
            genes). On every classical baseline it lands at 0% recall. The DANN
            domain-adversarial model recovers it to ~75% but only via 5-seed
            soft-voting — single seeds are unreliable.
          </>
        }
      />

      <Limit
        title="Mixed-sample contamination causes 10–20% accuracy drop"
        body={
          <>
            Stage 7 stress-tested the model with 25% cross-contamination
            (mixed spectra from one strain blended with another). Accuracy
            drops 10–20 percentage points. Real wet-lab samples won&apos;t be
            single-strain pure — this is a meaningful deployment constraint.
          </>
        }
      />

      <Limit
        title="The published Cisek-2013 STEC triple does not replicate"
        body={
          <>
            Cisek et al. (2013, Analyst) report &gt;95% STEC ↔ non-pathogenic{" "}
            <i>E. coli</i> sensitivity from three discriminative bands (1338,
            1454, 1658 cm⁻¹) using per-batch CV on controlled cell suspensions.
            On Atlas at file level, those three bands give Cohen&apos;s d of
            +0.13 / −0.47 / +0.16 — the middle band is even sign-reversed. Our
            actual strongest signal sits in the LPS region (1117 / 1194 cm⁻¹).
            Cross-protocol generalization across Raman labs is harder than
            cross-strain.
          </>
        }
      />

      <Limit
        title="Within-strain replicates are partially confounded with acquisition batch"
        body={
          <>
            Files within a single strain often share an acquisition date.
            Cross-strain LOSO performance therefore measures a mixture of
            (a) genuine biology generalization, (b) cross-session robustness,
            and (c) instrument drift. Without a re-collection protocol that
            varies session within strain, we can&apos;t cleanly separate these.
            Treat LOSO numbers as a lower bound on biology generalization.
          </>
        }
      />

      <Limit
        title="MCR component 6 ranks impressively but doesn't actually deploy"
        body={
          <>
            On a global MCR-ALS fit, component C6 has Cohen&apos;s
            d = −1.23 STEC ↔ Non-STEC — the project&apos;s strongest engineered
            file-level discriminator. But{" "}
            <strong className="text-nx-fg">0 MCR features survived per-fold MI selection</strong>{" "}
            in Stage 15F. Per-fold refit reorders components in ways that
            destroy the global-fit signal — the d=−1.23 was partly a leakage
            artifact. None of those 35 deployed features are MCR-derived.
          </>
        }
      />
    </section>
  );
}

function FaqSection() {
  return (
    <section>
      <H2 id="faq">FAQ</H2>
      {[
        {
          q: "Why didn't deep learning win?",
          a: "We trained a 1D-CNN (124K params) and a 1D-Transformer; both came in below PLS-DA. With 87 files and a 9-point LOSO statistic, modern deep architectures don't have enough samples to outpace a well-tuned latent-projection method. CNNs DO win on individual strains (K-12, O157:H7 via the DANN variant) but lose on the mean.",
        },
        {
          q: "Could you get a higher number with more data?",
          a: "Almost certainly yes. The cleanest reference point is Tang 2026's WGAN-Transformer on a 10,000-spectrum bacterial dataset — 97% intra-set, 94% on an independent test set. That's the realistic ceiling. The wall here is data, not modeling.",
        },
        {
          q: "Why is PLS-DA so good?",
          a: "PLS-DA projects the 987-bin spectrum onto a low-rank latent space that maximizes class separation. The low-rank projection acts as a regularizer — it ignores high-frequency idiosyncrasies that random forests / XGBoost overfit to. When the test distribution shifts (LOSO), the latent space generalizes better.",
        },
        {
          q: "What does \"file-weighted\" accuracy mean?",
          a: "We compute accuracy per LOSO fold (one fold per held-out strain), then average — weighted by the number of files in each fold rather than treating each fold as equal. The unweighted fold-mean is 0.436; the file-weighted is 0.448. They're both reported in the UI's Results tab.",
        },
        {
          q: "What's the difference between Stage 15F LogReg (0.448) and PLS-DA on raw (0.603)?",
          a: "Same data, different inputs. PLS-DA-on-raw uses the full 987-bin spectrum and projects to latent variables. Stage 15F is a 35-feature LogReg-L2 trained on the engineered features. The two compare different design choices — full spectrum vs. engineered descriptors. The paper headlines PLS-DA-on-raw; the deployed model is LogReg-L2 because it's the right pick when you only have engineered features cached.",
        },
        {
          q: "Why is H₂O easy and STEC ↔ Non-STEC hard?",
          a: "H₂O (water blanks) has no Raman peaks in the biological bands — its spectrum looks completely different from any bacterium. STEC and Non-STEC E. coli are the SAME species, biologically. The bulk cell-wall / ribosome / membrane signal that Raman picks up is essentially identical. We're trying to discriminate them on subtle outer-membrane differences (LPS chain composition).",
        },
        {
          q: "Can this be used in a real food-safety lab?",
          a: "Not in its current state. The 10-20% drop on mixed samples and the small training set mean it's a research prototype. A deployment-ready version would need: cross-site validation, recollection with controlled session/strain variation, mixed-contamination training set augmentation, and re-tuning of the QC + outlier-rejection thresholds.",
        },
        {
          q: "Why all the per-class small multiples and 2D PCA instead of one big scatter?",
          a: "With 87 points and 4 classes, dense small-multiples are easier to read than overlaid scatters. 3D rotation gives wow but loses readability — depth cues are weak. The 2D PC1×PC2 (with PC1×PC3 / PC2×PC3 toggles) covers 69% of variance and lets you see separability instantly.",
        },
        {
          q: "What's the prediction latency in production?",
          a: "Cold start: 3–8 seconds (Modal spins up the Python container + loads ~2 MB of joblib artifacts + parses an .xls + runs preprocessing + 259-feature extraction + LogReg predict). Warm: ~1–2 seconds. The endpoint scales to zero when idle so the first request after ~5 minutes will cold-start again.",
        },
      ].map((item) => (
        <FaqItem key={item.q} q={item.q} a={item.a} />
      ))}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Atomic helpers
// ---------------------------------------------------------------------------

function ClassPill({
  k,
}: {
  k: "STEC" | "Non-STEC" | "Salmonella" | "H2O";
}) {
  const bg = {
    STEC: "bg-class-stec",
    "Non-STEC": "bg-class-nonstec",
    Salmonella: "bg-class-salm",
    H2O: "bg-class-h2o",
  }[k];
  const txt = {
    STEC: "text-class-stec",
    "Non-STEC": "text-class-nonstec",
    Salmonella: "text-class-salm",
    H2O: "text-class-h2o",
  }[k];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-mono text-[0.75rem] font-semibold",
        txt,
      )}
    >
      <span className={cn("inline-block size-1.5 rounded-full", bg)} />
      {k}
    </span>
  );
}

function Tooltip({
  term,
  children,
}: {
  term: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className="text-nx-accent underline decoration-dotted decoration-nx-accent/60 underline-offset-2"
      title={`See "${term}" in the Glossary section below`}
    >
      {children}
    </span>
  );
}

function DataRow({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption?: string;
}) {
  return (
    <li className="flex items-baseline gap-3 border-b border-nx-muted/20 pb-2">
      <span className="font-mono text-[0.65rem] uppercase tracking-[0.16em] text-nx-fg/45 w-44 shrink-0">
        {label}
      </span>
      <span className="font-mono text-[0.85rem] text-nx-fg">{value}</span>
      {caption && (
        <span className="font-mono text-[0.65rem] text-nx-fg/40 ml-auto">
          {caption}
        </span>
      )}
    </li>
  );
}

function Limit({ title, body }: { title: string; body: React.ReactNode }) {
  return (
    <div className="my-5 rounded-md border-l-2 border-nx-accent/60 bg-nx-bg-elev-1/30 pl-4 pr-4 py-3">
      <h4 className="font-display text-[1.05rem] text-nx-fg mb-2">{title}</h4>
      <p className="text-[0.88rem] text-nx-fg/70 leading-relaxed">{body}</p>
    </div>
  );
}

function FaqItem({ q, a }: { q: string; a: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-nx-muted/30 py-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="group w-full flex items-center gap-3 py-2 text-left hover:bg-nx-bg-elev-1/30 transition-colors px-2 rounded-sm"
      >
        <ChevronDownIcon
          className={cn(
            "size-4 text-nx-fg/50 group-hover:text-nx-accent transition-all shrink-0",
            open ? "rotate-0 text-nx-accent" : "-rotate-90",
          )}
          strokeWidth={1.75}
        />
        <span className="font-display text-[1rem] text-nx-fg">{q}</span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <p className="text-[0.88rem] text-nx-fg/70 leading-relaxed pl-7 pr-2 pb-3">
              {a}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

