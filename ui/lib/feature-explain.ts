/**
 * Programmatic plain-English explainer for Atlas Raman feature names.
 *
 * Atlas uses a deterministic naming convention for the 259 engineered features
 * (Stages 15A–E). This module decodes a name like `fit_lipid_1454_height` into
 * a short, readable description without shipping a 259-row lookup table.
 */

const BAND_NAMES: Record<string, string> = {
  aa_1004: "1004 cm⁻¹ Phe ring (aromatic AA)",
  aa_1014: "1014 cm⁻¹ Trp / Phe",
  trp_762: "762 cm⁻¹ Trp ring",
  tyr_831: "831 cm⁻¹ Tyr",
  tyr_855: "855 cm⁻¹ Tyr",
  amide_iii_1242: "1242 cm⁻¹ amide-III β-sheet",
  amide_iii_1245: "1245 cm⁻¹ amide-III",
  amide_i_1658: "1658 cm⁻¹ amide-I α-helix/β-sheet",
  amide_i_1662: "1662 cm⁻¹ amide-I",
  na_720: "720 cm⁻¹ adenine",
  na_786: "786 cm⁻¹ pyrimidine ring",
  na_1335: "1335 cm⁻¹ adenine / guanine",
  na_1338: "1338 cm⁻¹ adenine ring (Cisek-2013 STEC triple, NULL)",
  na_1485: "1485 cm⁻¹ guanine",
  na_1575: "1575 cm⁻¹ adenine / guanine",
  lipid_1080: "1080 cm⁻¹ phospholipid backbone",
  lipid_1451: "1451 cm⁻¹ CH₂ deformation",
  lipid_1454: "1454 cm⁻¹ CH₂ lipid (Cisek-2013 STEC triple, sign-reversed)",
  ch_2850: "2850 cm⁻¹ symmetric CH₂ stretch",
  ch_2900: "2900 cm⁻¹ CH stretch",
  ch_2930: "2930 cm⁻¹ asymmetric CH₂ stretch",
  metabolite_616: "616 cm⁻¹ COO⁻ (Salmonella metabolite)",
  metabolite_925: "925 cm⁻¹ C-C (Salmonella)",
  metabolite_1542: "1542 cm⁻¹ C=C (Salmonella)",
  lps_1050: "1050 cm⁻¹ LPS chain",
  lps_1117: "1117 cm⁻¹ LPS chain (empirical anchor 2)",
  lps_1194: "1194 cm⁻¹ LPS chain (empirical anchor 1, d=+1.03)",
  lps_chain_discrim: "800–1200 cm⁻¹ LPS chain region (continuous)",
  lps_o_antigen_full: "400–900 cm⁻¹ LPS / O-antigen region",
  lps_chain: "LPS chain region",
  aromatic_aa: "aromatic amino acid group",
  protein_amide: "protein amide group",
  nucleic_acid: "nucleic acid group",
  lipid_carbohydrate: "lipid + carbohydrate group",
  metabolite: "Salmonella-metabolite group",
  amide: "amide region (1200–1700 cm⁻¹)",
  ch_stretch: "C-H stretch region (2800–3050 cm⁻¹)",
  silent: "silent region",
};

const FIT_PARAM_LABEL: Record<string, string> = {
  height: "peak height",
  area: "integrated area",
  center: "fitted peak center",
  fwhm: "full width at half max",
  rmse: "fit residual (RMSE)",
  eta: "Lorentzian/Gaussian mix (η)",
};

const ROI_STAT_LABEL: Record<string, string> = {
  mean: "mean intensity",
  std: "intensity std-dev",
  skew: "intensity skewness",
  kurt: "intensity kurtosis",
  entropy: "spectral entropy",
  max: "max intensity",
  p90: "90th-percentile intensity",
};

const MCR_STAT_LABEL: Record<string, string> = {
  mean: "mean concentration",
  std: "concentration std-dev",
  max: "max concentration",
  p90: "90th-percentile concentration",
};

const SPAT_STAT_LABEL: Record<string, string> = {
  var: "pixel-to-pixel variance",
  cv: "coefficient of variation",
  skew: "spatial skewness",
  kurt: "spatial kurtosis",
};

const BIO_NAMES: Record<string, string> = {
  alpha_helix_score: "α-helix protein-2°-structure score (Atlas-strongest biology axis, d=−0.986)",
  beta_sheet_score: "β-sheet protein-2°-structure score",
  trp_indole_env: "Trp indole-ring environment (hydrophobic ↔ exposed)",
  cyt_ox_state: "cytochrome c oxidation state",
  virulence_aa_sig: "virulence amino-acid signature (Trp/Phe ratio; reversed at E.coli↔Salm)",
  lipid_protein_ratio: "lipid-to-protein band ratio",
  na_protein_ratio: "nucleic-acid-to-protein band ratio",
};

function bandLabel(token: string): string {
  return BAND_NAMES[token] ?? token.replace(/_/g, " ");
}

/** Decode an Atlas feature name to a one-line plain-English description. */
export function describeFeature(name: string): string {
  // 1. Bio (Stage 15D)
  if (name.startsWith("bio_")) {
    const key = name.slice(4);
    return BIO_NAMES[key] ?? `Biology aggregate (Stage 15D): ${key.replace(/_/g, " ")}`;
  }

  // 2. MCR-ALS (Stage 15C)
  const mcrMatch = name.match(/^mcr_C(\d+)_(\w+)$/);
  if (mcrMatch) {
    const k = mcrMatch[1];
    const stat = mcrMatch[2];
    return `${MCR_STAT_LABEL[stat] ?? stat} of MCR-ALS component C${k} (Stage 15C, K=7 unmixing)`;
  }
  if (name.startsWith("mcr_residual_norm_")) {
    const stat = name.replace("mcr_residual_norm_", "");
    return `${MCR_STAT_LABEL[stat] ?? stat} of MCR-ALS reconstruction residual`;
  }

  // 3. Spatial (Stage 15E)
  const spatMatch = name.match(/^spat_(\w+?)_(.+)$/);
  if (spatMatch) {
    const stat = spatMatch[1];
    const band = spatMatch[2];
    return `${SPAT_STAT_LABEL[stat] ?? stat} of per-pixel intensity at ${bandLabel(band)}`;
  }

  // 4. Spectral (Stage 15B) — DWT / PCA / SAM
  const dwtMatch = name.match(/^dwt_(\w+?)_L(\d+)$/);
  if (dwtMatch) {
    return `Discrete-wavelet ${dwtMatch[1]} at decomposition level ${dwtMatch[2]} (Daubechies-4)`;
  }
  const pcaMatch = name.match(/^pca_(\w+?)_PC(\d+)$/);
  if (pcaMatch) {
    return `ROI-PCA component ${pcaMatch[2]} over the ${bandLabel(pcaMatch[1])} region`;
  }
  const samMatch = name.match(/^sam_(\w+?)_(\w+)$/);
  if (samMatch) {
    return `Spectral-angle similarity to ${samMatch[2]} template over ${bandLabel(samMatch[1])}`;
  }

  // 5. Derivative AUCs (Stage 15A)
  const d1Match = name.match(/^d1_auc_(.+)$/);
  if (d1Match) {
    return `1st-derivative AUC at ${bandLabel(d1Match[1])}`;
  }
  const d2Match = name.match(/^d2_auc_(.+)$/);
  if (d2Match) {
    return `2nd-derivative AUC at ${bandLabel(d2Match[1])}`;
  }

  // 6. Pseudo-Voigt fits (Stage 15A)
  const fitMatch = name.match(/^fit_(.+?)_(height|area|center|fwhm|rmse|eta)$/);
  if (fitMatch) {
    const band = fitMatch[1];
    const param = fitMatch[2];
    return `Pseudo-Voigt ${FIT_PARAM_LABEL[param]} at ${bandLabel(band)}`;
  }

  // 7. Raw AUC (Stage 15A)
  const aucMatch = name.match(/^auc_(.+)$/);
  if (aucMatch) {
    return `Raw AUC over ${bandLabel(aucMatch[1])}`;
  }

  // 8. Ratios (Stage 15A)
  const ratioMatch = name.match(/^(.+)_over_(.+)$/);
  if (ratioMatch) {
    return `Ratio of ${bandLabel(ratioMatch[1])} to ${bandLabel(ratioMatch[2])}`;
  }

  // 9. ROI moments (Stage 15A/B catch-all)
  const roiMatch = name.match(/^roi_(.+?)_(\w+)$/);
  if (roiMatch) {
    return `${ROI_STAT_LABEL[roiMatch[2]] ?? roiMatch[2]} across the ${bandLabel(roiMatch[1])} region`;
  }

  // 10. EMSC (Extended Multiplicative Scatter Correction) coefficients
  if (name.startsWith("emsc_")) {
    return `EMSC scatter-correction coefficient: ${name.replace("emsc_", "").replace(/_/g, " ")}`;
  }

  return name.replace(/_/g, " ");
}
