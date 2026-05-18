"""Atlas Raman UI — Band-chemistry sidecar builder (W4).

Reads `atlas/band_features.BANDS` (canonical catalog of 30+ named bands across
5 macromolecule groups, plus the 2 LPS regions) and pairs each entry with a
short plain-English chemistry one-liner sourced from `FINAL/PAPER.md` §2.6 +
§4.1-4.2. Hand-encodes the Cisek-2013 falsification panel (1338 / 1454 / 1658
literature claim vs. Atlas Cohen's d) and the LPS-chain empirical anchor.

Emits a single JSON sidecar:
    ui/public/data/bands.json

Schema (matches the React-side `BandPrimer` tab contract):
    {
      "groups": [{key, label, biology, bands: [{name, center, fwhm,
                  chemistry, d_stec_nonstec}]}, ...],
      "cisek_falsification": {headline, bands: [{center, label,
                              literature_claim, atlas_d, verdict}]},
      "anchors": {lps_chain: {region, label, top_band, top_d}}
    }

Plan reference: plan/ui/ULTRAPLAN.md §4 W4.

Usage (one-time, idempotent):
    cd ui && python scripts/build_bands.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports of `atlas/` from the repo root.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from atlas.band_features import BANDS, MACROMOLECULE_GROUPS  # noqa: E402


# ---------------------------------------------------------------------------
# Group metadata (label + biology one-liner, per PAPER §2.6 table line ~234)
# ---------------------------------------------------------------------------

GROUP_META: dict[str, dict[str, str]] = {
    "aromatic_aa": {
        "key": "aromatic_aa",
        "label": "Aromatic AA",
        "biology": "total-protein anchors (Phe / Tyr / Trp side chains)",
    },
    "protein_amide": {
        "key": "protein_amide",
        "label": "Protein amide",
        "biology": "secondary structure (amide-III β-sheet, amide-I α-helix / β / random)",
    },
    "nucleic_acid": {
        "key": "nucleic_acid",
        "label": "Nucleic acid",
        "biology": "DNA / RNA ring breathing + base modes",
    },
    "lipid_carb": {
        "key": "lipid_carb",
        "label": "Lipid + carbohydrate",
        "biology": "membrane composition + LPS chain (per Stage-1 empirical anchor)",
    },
    "metabolite": {
        "key": "metabolite",
        "label": "Metabolite (Salmonella)",
        "biology": "Salmonella-specific markers per Yuan-2024",
    },
}

# Canonical group-key order in `atlas.band_features.MACROMOLECULE_GROUPS`
# uses `lipid_carbohydrate`, but BANDS rows use `lipid_carb`. Normalize.
GROUP_NORMALIZE: dict[str, str] = {
    "aromatic_aa": "aromatic_aa",
    "protein_amide": "protein_amide",
    "nucleic_acid": "nucleic_acid",
    "lipid_carb": "lipid_carb",
    "lipid_carbohydrate": "lipid_carb",
    "metabolite": "metabolite",
}


# ---------------------------------------------------------------------------
# Plain-English chemistry one-liners (per band key)
# Sourced from atlas/band_features.BANDS `note` field + PAPER §2.6 / §4.1-4.2.
# ---------------------------------------------------------------------------

CHEMISTRY: dict[str, str] = {
    # LPS / O-antigen — empirical Stage-1 anchors
    "lps_1050": "Top E. coli vs Salmonella discriminator (3-class ANOVA) — LPS chain region",
    "lps_1117": "Top E. coli STEC↔Non-STEC empirical discriminator — LPS chain",
    "lps_1194": "Top E. coli STEC↔Non-STEC empirical discriminator — LPS chain, d=+1.03",
    # Aromatic AA — protein side chains
    "aa_762":  "Tryptophan (Trp) ring breathing — aromatic AA",
    "aa_831":  "Tyrosine (Tyr) Fermi doublet — aromatic AA",
    "aa_855":  "Tyrosine (Tyr) ring — aromatic AA",
    "aa_1004": "Phenylalanine (Phe) ring breathing — total-protein anchor (sharp, file-stable)",
    "aa_1014": "Phe / Trp ring mode — aromatic AA",
    "aa_1176": "Tyr / Phe in-plane bending — aromatic AA",
    "aa_1212": "Tyr / Phe C-C stretch — aromatic AA",
    "aa_1617": "Tryptophan (Trp) C=C aromatic stretch",
    # Nucleic acids
    "na_720":  "Adenine / Guanine / Thymine ring breathing — DNA/RNA",
    "na_786":  "U, C, T ring breathing — DNA/RNA (overlaps Tyr / aa region)",
    "na_1335": "CH₂ / CH₃ wag — nucleic-acid backbone",
    "na_1338": "CH₂ wag / adenine ring — Cisek-2013 PRIMARY (falsified Stage 1)",
    "na_1362": "Guanine ring mode — DNA/RNA",
    "na_1485": "Guanine ring mode — DNA/RNA",
    "na_1530": "Nucleic-acid base modes",
    "na_1575": "Purine bases (A, G) — DNA/RNA",
    # Protein amide
    "amide_iii_1242": "Amide III β-sheet — protein backbone (C-N stretch + N-H bend)",
    "amide_i_1658":   "Amide I β-sheet / random coil — Cisek-2013 PRIMARY (falsified Stage 1)",
    "amide_i_1662":   "Amide I α-helix — protein secondary structure",
    # Lipid / carbohydrate
    "lipid_1080": "Phospholipid backbone (PO₂⁻ symmetric stretch)",
    "lipid_1451": "CH₂ scissoring / deformation — membrane lipid",
    "lipid_1454": "CH₂ deformation — Cisek-2013 PRIMARY (falsified Stage 1, d=−0.47 SIGN REVERSED)",
    "lipid_1585": "Lipid / carbohydrate / NA overlap region",
    "lipid_2850": "Symmetric CH₂ stretch — lipid (C-H stretch region)",
    "lipid_2930": "Asymmetric CH₂ stretch — lipid (C-H stretch region)",
    # Salmonella metabolites (Yuan 2024)
    "salm_616":  "Carboxylate (COO⁻) wag — Salmonella metabolite marker",
    "salm_925":  "C-C skeletal stretch — Salmonella metabolite marker",
    "salm_1486": "Guanine ring — Salmonella nucleic-acid signature (per Yuan-2024)",
    "salm_1542": "C=C stretch — Salmonella metabolite marker",
}


# ---------------------------------------------------------------------------
# Discrimination d-values (file-level STEC ↔ Non-STEC), per PAPER Table §4.1/§4.2
# Only filled where the paper or atlas literature explicitly reports them.
# `None` means "not measured / not reported", NOT zero.
# ---------------------------------------------------------------------------

D_STEC_NONSTEC: dict[str, float] = {
    # Empirical anchors (§4.2)
    "lps_1050": 0.42,
    "lps_1117": 0.77,
    "lps_1194": 1.03,
    # Literature triple (§4.1)
    "na_1338":     0.13,
    "lipid_1454": -0.47,
    "amide_i_1658": 0.16,
}

# Default Lorentzian / pseudo-Voigt fit window from atlas.band_features
# (DEFAULT_FIT_BANDS uses fit_window=30; FWHM init=12). Use 12 as a sensible
# nominal FWHM for display when the catalog doesn't fix one per band.
DEFAULT_FWHM = 12.0


# ---------------------------------------------------------------------------
# Build payload
# ---------------------------------------------------------------------------

def build_groups() -> list[dict]:
    """Bucket BANDS into the 5 macromolecule groups, preserving catalog order."""
    by_group: dict[str, list[dict]] = {k: [] for k in GROUP_META}
    for band_name, spec in BANDS.items():
        gkey = GROUP_NORMALIZE.get(spec["group"], spec["group"])
        if gkey not in by_group:
            # Unexpected group — skip with a warning rather than crash.
            print(f"  warn: band {band_name!r} has unknown group {spec['group']!r}, skipping")
            continue
        chemistry = CHEMISTRY.get(band_name, spec.get("note", ""))
        by_group[gkey].append({
            "name": band_name,
            "center": float(spec["center"]),
            "fwhm": DEFAULT_FWHM,
            "chemistry": chemistry,
            "d_stec_nonstec": D_STEC_NONSTEC.get(band_name),  # None if unmeasured
        })

    out: list[dict] = []
    for gkey, meta in GROUP_META.items():
        bands = sorted(by_group[gkey], key=lambda b: b["center"])
        out.append({**meta, "bands": bands})
    return out


def build_cisek_falsification() -> dict:
    """Per PAPER §4.1 Table — literature triple + Atlas d + verdict."""
    return {
        "headline": "Cisek-2013 STEC triple — NULL at file level on this corpus",
        "bands": [
            {
                "center": 1338,
                "label": "na_1338",
                "literature_claim": "STEC > Non-STEC (CH₂ wag / adenine)",
                "atlas_d": 0.13,
                "verdict": "null",
            },
            {
                "center": 1454,
                "label": "lipid_1454",
                "literature_claim": "STEC > Non-STEC (CH₂ deformation)",
                "atlas_d": -0.47,
                "verdict": "sign-reversed",
            },
            {
                "center": 1658,
                "label": "amide_i_1658",
                "literature_claim": "STEC > Non-STEC (amide-I)",
                "atlas_d": 0.16,
                "verdict": "null",
            },
        ],
    }


def build_anchors() -> dict:
    """LPS-chain region 800-1200 — Atlas empirical anchor (Stage 1 winner)."""
    return {
        "lps_chain": {
            "region": [800, 1200],
            "label": "LPS chain region — empirical anchor (Stage 1 winner)",
            "top_band": "auc_lps_1194",
            "top_d": 1.03,
        },
    }


def main() -> None:
    out_dir = HERE.parent.parent / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = build_groups()
    total_bands = sum(len(g["bands"]) for g in groups)

    payload = {
        "groups": groups,
        "cisek_falsification": build_cisek_falsification(),
        "anchors": build_anchors(),
    }

    out_path = out_dir / "bands.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"wrote {out_path}")
    print(f"  groups: {len(groups)}")
    for g in groups:
        print(f"    {g['key']:14s} : {len(g['bands']):2d} bands")
    print(f"  total bands: {total_bands}")
    print(f"  cisek_falsification: {len(payload['cisek_falsification']['bands'])} bands")
    print(f"  anchors: {list(payload['anchors'])}")


if __name__ == "__main__":
    main()
