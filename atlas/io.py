"""Parse Atlas Raman .xls/.txt files and cache to parquet/npy.

File format (tab-delimited despite .xls extension):
  ~44 header lines, each `#KEY=\\tVALUE`.
  One wavenumber row: 2 empty cells, then 2048 wn values, then trailing tab.
  N pixel rows: x_um, y_um, then 2048 intensities (comma thousands-separators).

Per file we derive:
  - primary class from top-level folder under Atlas Data/
  - subclass from second-level folder (None for H20)
  - file_id from the filename stem (e.g. R372_100_10000ms_260306)
  - grid dimensions from unique(x_um) x unique(y_um), NOT from #NUMX/#NUMY
    (early-batch headers are unreliable).

Output: every spectrum interpolated onto a single canonical wn axis so all
files share a common feature space downstream.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


CANONICAL_WN = np.linspace(76.0, 3499.0, 2048).astype(np.float32)
N_BINS = 2048

CLASS_MAP = {
    "H20": "H2O",
    "STEC": "STEC",
    "Non STEC": "Non-STEC",
    "Salmonella": "Salmonella",
}

# When a class folder contains strain subfolders, the second-level folder
# is the subclass. H20 has no subclass.
HAS_SUBCLASS = {"STEC", "Non STEC", "Salmonella"}


@dataclass
class FileRecord:
    """Per-file metadata + parsed payload."""

    file_id: str
    file_path: str
    primary_class: str
    subclass: str | None
    n_pixels: int
    grid_nx: int
    grid_ny: int
    header_numx: int | None
    header_numy: int | None
    xsize: float | None
    ysize: float | None
    laser: str | None
    exposure_ms: float | None
    acquisition_date: str | None
    ac_calibration_date: str | None
    wn_start: float
    wn_end: float
    is_complete_scan: bool
    file_sha256: str
    file_mtime: float
    file_size_bytes: int
    fatal_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Not serialized to metadata.parquet — kept transiently while building cache:
    x_um: np.ndarray | None = None
    y_um: np.ndarray | None = None
    intensities: np.ndarray | None = None  # shape (n_pixels, N_BINS)

    @property
    def is_valid(self) -> bool:
        return not self.fatal_errors


# ---------- low-level parsing helpers ----------

_NUM_RE = re.compile(r"[+-]?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")


def _strip_commas(token: str) -> str:
    """`1,034.00` -> `1034.00`. Empty/whitespace passes through."""
    return token.replace(",", "")


def _to_float_safe(token: str) -> float:
    t = _strip_commas(token).strip()
    if not t:
        return np.nan
    return float(t)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_header_lines(lines: list[str]) -> tuple[dict[str, str], int]:
    """Return (header_dict, index_of_first_non_header_line).

    Header lines look like `#KEY=\\tVALUE` (or `#KEY=\\t` for empty values).
    """
    header: dict[str, str] = {}
    i = 0
    for i, raw in enumerate(lines):
        if not raw.startswith("#"):
            return header, i
        # Strip the leading '#', split on first '=' or first tab after the key
        body = raw.rstrip("\n").lstrip("#")
        if "=" in body:
            key, _, rest = body.partition("=")
            # value follows the tab after '='
            value = rest.lstrip("\t").strip()
            header[key.strip()] = value
        # else: skip non-conforming header line
    return header, i + 1


def _classify_path(path: Path, data_root: Path) -> tuple[str, str | None]:
    """Resolve (primary_class, subclass) from path relative to data_root."""
    rel = path.relative_to(data_root)
    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"unexpected layout: {rel}")
    top = parts[0]
    if top not in CLASS_MAP:
        raise ValueError(f"unknown class folder: {top}")
    primary = CLASS_MAP[top]
    if top in HAS_SUBCLASS:
        if len(parts) < 3:
            raise ValueError(f"expected subclass folder under {top}: {rel}")
        sub = parts[1]
        return primary, sub
    return primary, None


# ---------- public API ----------


def discover_files(data_root: Path) -> list[Path]:
    """All .xls and .txt files under data_root, sorted for reproducibility."""
    files: list[Path] = []
    for ext in ("*.xls", "*.txt"):
        files.extend(data_root.rglob(ext))
    return sorted(files)


def parse_file(
    path: Path,
    data_root: Path,
    *,
    canonical_wn: np.ndarray = CANONICAL_WN,
    pixel_cap: int | None = 200,
    rng: np.random.Generator | None = None,
) -> FileRecord:
    """Parse one Atlas file. Always returns a FileRecord; check `.is_valid`."""
    if rng is None:
        rng = np.random.default_rng(seed=0)

    stat = path.stat()
    file_id = path.stem
    primary, subclass = _classify_path(path, data_root)

    rec = FileRecord(
        file_id=file_id,
        file_path=str(path.relative_to(data_root)),
        primary_class=primary,
        subclass=subclass,
        n_pixels=0,
        grid_nx=0,
        grid_ny=0,
        header_numx=None,
        header_numy=None,
        xsize=None,
        ysize=None,
        laser=None,
        exposure_ms=None,
        acquisition_date=None,
        ac_calibration_date=None,
        wn_start=float("nan"),
        wn_end=float("nan"),
        is_complete_scan=True,
        file_sha256=_sha256_of(path),
        file_mtime=stat.st_mtime,
        file_size_bytes=stat.st_size,
    )

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # pragma: no cover
        rec.fatal_errors.append(f"read_error:{e}")
        return rec

    lines = text.splitlines()
    if not lines:
        rec.fatal_errors.append("empty_file")
        return rec

    header, body_start = _parse_header_lines(lines)

    # Pull select header fields
    def _hopt(k: str) -> str | None:
        v = header.get(k)
        return v if v else None

    def _hint(k: str) -> int | None:
        v = _hopt(k)
        try:
            return int(float(v)) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _hfloat(k: str) -> float | None:
        v = _hopt(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    rec.header_numx = _hint("NUMX")
    rec.header_numy = _hint("NUMY")
    rec.xsize = _hfloat("XSIZE")
    rec.ysize = _hfloat("YSIZE")
    rec.laser = _hopt("Laser")
    rec.exposure_ms = _hfloat("ExposureTime")
    rec.acquisition_date = _hopt("Aquired") or _hopt("Date")
    ac = _hopt("AC")
    if ac:
        # e.g. "Calibrated 05.03.2026 07:48:36"
        m = re.search(r"(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2})", ac)
        rec.ac_calibration_date = m.group(1) if m else ac

    if not header.get("NUMX"):
        rec.warnings.append("missing_numx_header")
    if not header.get("NUMY"):
        rec.warnings.append("missing_numy_header")

    # First non-header line is the wavenumber row
    if body_start >= len(lines):
        rec.fatal_errors.append("no_data_rows")
        return rec

    wn_row = lines[body_start].split("\t")
    # Layout: ["", "", wn0, wn1, ..., wn2047] possibly trailing ""
    wn_values_raw = [t for t in wn_row[2:] if t.strip() != ""]
    try:
        wn = np.array([_to_float_safe(t) for t in wn_values_raw], dtype=np.float64)
    except ValueError as e:
        rec.fatal_errors.append(f"wn_parse_error:{e}")
        return rec

    if wn.size != N_BINS:
        rec.fatal_errors.append(f"wn_axis_size:{wn.size}!={N_BINS}")
        return rec
    if not np.all(np.diff(wn) > 0):
        rec.fatal_errors.append("wn_not_monotonic")
        return rec

    rec.wn_start = float(wn[0])
    rec.wn_end = float(wn[-1])

    # Pixel rows
    pixel_lines = lines[body_start + 1 :]
    xs: list[float] = []
    ys: list[float] = []
    raw_specs: list[np.ndarray] = []

    for ln in pixel_lines:
        if not ln.strip():
            continue
        toks = ln.split("\t")
        if len(toks) < 3:
            continue
        try:
            xs.append(float(_strip_commas(toks[0])))
            ys.append(float(_strip_commas(toks[1])))
        except ValueError:
            rec.warnings.append("bad_coord_row")
            continue
        # Intensities: pad/truncate to N_BINS just in case
        ints_tokens = toks[2 : 2 + N_BINS]
        if len(ints_tokens) < N_BINS:
            # Truncated row -> drop it; record warning
            xs.pop()
            ys.pop()
            rec.warnings.append("truncated_pixel_row")
            continue
        arr = np.fromiter(
            (_to_float_safe(t) for t in ints_tokens),
            dtype=np.float64,
            count=N_BINS,
        )
        if not np.all(np.isfinite(arr)):
            rec.warnings.append("nan_in_raw_spectrum")
            xs.pop()
            ys.pop()
            continue
        raw_specs.append(arr)

    if not raw_specs:
        rec.fatal_errors.append("no_valid_pixel_rows")
        return rec

    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    raw_mat = np.asarray(raw_specs, dtype=np.float32)  # (n, 2048)

    # Derive grid dims from coordinate uniqueness (the headers lie for early batch)
    rec.grid_nx = int(np.unique(x_arr).size)
    rec.grid_ny = int(np.unique(y_arr).size)
    rec.n_pixels = int(raw_mat.shape[0])

    # is_complete_scan: compare pixel count to grid_nx * grid_ny (the headers
    # are unreliable for early-batch files, so we trust the coordinate grid).
    expected_full = rec.grid_nx * rec.grid_ny
    if rec.n_pixels < expected_full:
        rec.is_complete_scan = False
        rec.warnings.append(
            f"partial_scan:{rec.n_pixels}/{expected_full}"
        )

    # Interpolate to canonical wn axis
    interp = np.empty((rec.n_pixels, canonical_wn.size), dtype=np.float32)
    for i in range(rec.n_pixels):
        interp[i] = np.interp(canonical_wn, wn, raw_mat[i]).astype(np.float32)

    if not np.all(np.isfinite(interp)):
        rec.fatal_errors.append("nan_after_interp")
        return rec

    # Stratified-but-simple pixel cap: uniform-random subsample without replacement.
    # The grid is dense, so a random subsample preserves the spatial distribution
    # well enough for class-level statistics. (More elaborate stratification by
    # spatial bin is overkill for this stage.)
    if pixel_cap is not None and rec.n_pixels > pixel_cap:
        idx = rng.choice(rec.n_pixels, size=pixel_cap, replace=False)
        idx.sort()
        x_arr = x_arr[idx]
        y_arr = y_arr[idx]
        interp = interp[idx]
        rec.warnings.append(f"pixel_capped:{rec.n_pixels}->{pixel_cap}")
        rec.n_pixels = pixel_cap

    rec.x_um = x_arr
    rec.y_um = y_arr
    rec.intensities = interp
    return rec


def parse_dataset(
    data_root: Path,
    *,
    pixel_cap: int | None = 200,
    seed: int = 0,
    progress: bool = True,
) -> list[FileRecord]:
    """Parse every file under data_root. Returns one FileRecord per file
    (including ones with fatal_errors)."""
    files = discover_files(data_root)
    rng = np.random.default_rng(seed)
    records: list[FileRecord] = []
    iterator: Iterable[Path] = files
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(files, desc="parse", unit="file")
        except ImportError:
            pass
    for path in iterator:
        records.append(
            parse_file(path, data_root, pixel_cap=pixel_cap, rng=rng)
        )
    return records


# ---------- cache serialization ----------


def record_to_metadata_row(rec: FileRecord) -> dict:
    d = asdict(rec)
    # drop transient arrays
    for k in ("x_um", "y_um", "intensities"):
        d.pop(k, None)
    d["fatal_errors"] = json.dumps(d["fatal_errors"])
    d["warnings"] = json.dumps(d["warnings"])
    return d


def build_long_form_spectra(records: list[FileRecord]) -> tuple[
    list[dict], np.ndarray
]:
    """Return (rows-without-intensities, stacked-intensity-matrix).

    Long-form: one dict per pixel-spectrum. Intensity payload is stored
    separately as a contiguous (N, 2048) float32 array for fast torch consumption.
    The two are row-aligned.
    """
    rows: list[dict] = []
    blocks: list[np.ndarray] = []
    for rec in records:
        if not rec.is_valid or rec.intensities is None:
            continue
        for i in range(rec.n_pixels):
            rows.append(
                {
                    "file_id": rec.file_id,
                    "primary_class": rec.primary_class,
                    "subclass": rec.subclass,
                    "pixel_idx": i,
                    "x_um": float(rec.x_um[i]),
                    "y_um": float(rec.y_um[i]),
                }
            )
        blocks.append(rec.intensities)
    if blocks:
        mat = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    else:
        mat = np.zeros((0, N_BINS), dtype=np.float32)
    return rows, mat


def write_cache(
    records: list[FileRecord],
    cache_dir: Path,
    canonical_wn: np.ndarray = CANONICAL_WN,
) -> dict:
    """Write parquet + npy + axis + build log. Returns a small summary dict."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Metadata table
    meta_rows = [record_to_metadata_row(r) for r in records]
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(cache_dir / "metadata.parquet", compression="snappy")

    # Spectra long-form + companion array
    spec_rows, mat = build_long_form_spectra(records)
    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_parquet(cache_dir / "spectra.parquet", compression="snappy")
    np.save(cache_dir / "spectra_array.npy", mat)
    np.save(cache_dir / "wavenumber_axis.npy", canonical_wn.astype(np.float32))

    # Build log (JSONL)
    log_path = cache_dir / "build.log"
    with log_path.open("w") as f:
        for r in records:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "file_id": r.file_id,
                "file_path": r.file_path,
                "primary_class": r.primary_class,
                "subclass": r.subclass,
                "n_pixels": r.n_pixels,
                "grid": [r.grid_nx, r.grid_ny],
                "header_grid": [r.header_numx, r.header_numy],
                "is_complete_scan": r.is_complete_scan,
                "wn_range": [r.wn_start, r.wn_end],
                "fatal_errors": r.fatal_errors,
                "warnings": r.warnings,
                "file_sha256": r.file_sha256,
            }
            f.write(json.dumps(entry) + "\n")

    n_fatal = sum(1 for r in records if not r.is_valid)
    n_warn = sum(1 for r in records if r.warnings and r.is_valid)
    return {
        "n_files": len(records),
        "n_fatal": n_fatal,
        "n_warned": n_warn,
        "n_spectra": int(mat.shape[0]),
        "cache_dir": str(cache_dir),
    }


def load_cache(cache_dir: Path) -> tuple["pd.DataFrame", np.ndarray, np.ndarray, "pd.DataFrame"]:  # type: ignore[name-defined]
    """Return (spectra_df, intensities_array, wn_axis, metadata_df)."""
    import pandas as pd

    spec_df = pd.read_parquet(cache_dir / "spectra.parquet")
    mat = np.load(cache_dir / "spectra_array.npy")
    wn = np.load(cache_dir / "wavenumber_axis.npy")
    meta_df = pd.read_parquet(cache_dir / "metadata.parquet")
    return spec_df, mat, wn, meta_df
