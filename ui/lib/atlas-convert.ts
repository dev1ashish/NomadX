/**
 * In-browser port of `../convert_to_atlas.py`.
 *
 * Reshapes a raw XploRA Raman export into the canonical Atlas file layout the
 * Modal `/predict` endpoints can parse. `atlas/io.py` hardcodes 2048
 * intensities per pixel row interpolated onto `linspace(76, 3499, 2048)`, so a
 * file with a different point count (e.g. 953 points over 502-2699 cm-1 from a
 * 600 gr/mm grating) is rejected outright — every pixel row fails the
 * `len < 2048` guard and the endpoint 500s.
 *
 * This resamples each spectrum onto the canonical axis so the file parses. It
 * does NOT invent data where the instrument recorded none: any wavenumber the
 * source didn't cover is edge-clamped flat, exactly as `np.interp` does.
 * Predictions on a partially-covered file are OUT OF DISTRIBUTION — see the
 * coverage numbers in `ConversionStats` and the gate in `ConvertPanel`.
 *
 * Two differences from the CLI script, both forced by real data in
 * `all-txt-data/`:
 *
 *  1. **Streaming.** Sources run up to 441 MB / 109,071 pixel rows. The file is
 *     read in slices rather than decoded whole, and non-retained rows are never
 *     split into cells.
 *
 *  2. **Pixel subsampling.** Emitting every row would produce a 1.5 GB upload
 *     for the largest file — and both Modal endpoints immediately discard all
 *     but a random 200 rows (`PIXEL_CAP`, `modal_app.py`), as did training
 *     (`parse_file(pixel_cap=200)` in `atlas/io.py`). So we reservoir-sample to
 *     the same 200 during the streaming pass. The subset differs from the
 *     endpoint's own seed-42 pick, but the distribution is identical and the
 *     result is deterministic across runs. Files at or under the cap keep every
 *     row and are byte-identical to `convert_to_atlas.py` output.
 */

/** Must match `atlas/io.py` exactly. */
export const N_BINS = 2048;
const WN_MIN = 76.0;
const WN_MAX = 3499.0;

/** Matches `PIXEL_CAP` in `inference_api/modal_app.py`. */
export const PIXEL_CAP = 200;

/** Regions the model crops to (`atlas/preprocess.py`). */
export const FINGERPRINT: readonly [number, number] = [400.0, 1800.0];
export const CH_STRETCH: readonly [number, number] = [2800.0, 3050.0];

/** `np.linspace(76.0, 3499.0, 2048)` — endpoint inclusive. */
export const CANONICAL_WN: Float64Array = (() => {
  const out = new Float64Array(N_BINS);
  const step = (WN_MAX - WN_MIN) / (N_BINS - 1);
  for (let i = 0; i < N_BINS; i++) out[i] = WN_MIN + i * step;
  out[N_BINS - 1] = WN_MAX;
  return out;
})();

/**
 * `map`      — the standard export: a wavenumber row, then one row per pixel.
 * `spectrum` — a two-column `wavenumber \t intensity` list; a single averaged
 *              or background spectrum rather than a spatial map.
 */
export type SourceLayout = "map" | "spectrum";

export interface ConversionStats {
  layout: SourceLayout;
  /** Native wavenumber points found in the source. */
  nNative: number;
  /** [min, max] of the native wavenumber axis, cm^-1. */
  nativeRange: [number, number];
  /** Data rows found in the source, before the cap. */
  nPixelsTotal: number;
  /** Pixel rows actually written (<= PIXEL_CAP). */
  nPixelsKept: number;
  /** True when rows were dropped to reach the cap. */
  subsampled: boolean;
  /** Rows dropped as malformed / short / non-finite, among those parsed. */
  nSkipped: number;
  /** Percentage of 400-1800 cm^-1 the instrument actually measured. */
  fingerprintCov: number;
  /** Percentage of 2800-3050 cm^-1 the instrument actually measured. */
  chCov: number;
  /** Size of the source file in bytes. */
  inputBytes: number;
  /** Size of the converted file in bytes. */
  outputBytes: number;
  /** Suggested filename, matching the CLI's `<stem>__atlas2048.txt`. */
  outputName: string;
}

export interface ConversionResult {
  /** Ready to hand straight to `predict()` / `predictPlsda()`. */
  file: File;
  stats: ConversionStats;
}

export class ConversionError extends Error {}

/** Deterministic PRNG so the same file always yields the same subsample. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `1,034.00` -> 1034.0; empty/whitespace -> NaN. XploRA uses commas as
 *  thousands separators inside intensity cells. */
function toFloat(tok: string): number {
  const cleaned = tok.replace(/,/g, "").trim();
  if (cleaned === "") return NaN;
  const v = Number(cleaned);
  return Number.isNaN(v) ? NaN : v;
}

/**
 * Precompute, once, where each canonical wavenumber falls in the native axis.
 * Every pixel row shares the same axis, so the segment lookup is hoisted out of
 * the per-row loop: 2048 searches total instead of 2048 x n_pixels.
 *
 * Encodes `np.interp` edge behaviour — outside the native range the value is
 * clamped to the nearest endpoint rather than extrapolated.
 */
function buildInterpPlan(xp: Float64Array): {
  idx: Int32Array;
  w: Float64Array;
} {
  const n = xp.length;
  const idx = new Int32Array(N_BINS);
  const w = new Float64Array(N_BINS);
  let j = 0;
  for (let k = 0; k < N_BINS; k++) {
    const x = CANONICAL_WN[k];
    if (x <= xp[0]) {
      idx[k] = 0;
      w[k] = 0; // -> fp[0]
      continue;
    }
    if (x >= xp[n - 1]) {
      idx[k] = n - 2;
      w[k] = 1; // -> fp[n-1]
      continue;
    }
    // CANONICAL_WN ascends, so j only ever moves forward.
    while (j < n - 2 && xp[j + 1] < x) j++;
    const span = xp[j + 1] - xp[j];
    idx[k] = j;
    w[k] = span > 0 ? (x - xp[j]) / span : 0;
  }
  return { idx, w };
}

/**
 * Encode to latin-1. The source is latin-1 and `atlas/io.py` reads it back as
 * latin-1, so the round trip has to preserve the header bytes (degree signs,
 * micron signs). Only headers contain non-ASCII; numeric data is pure ASCII.
 */
function latin1Bytes(text: string): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(new ArrayBuffer(text.length));
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    out[i] = c > 255 ? 0x3f /* '?' */ : c;
  }
  return out;
}

/** Blank-line test that doesn't allocate. Pixel rows are ~4 KB and there can
 *  be 100k of them, so `line.trim() === ""` per row is real cost. */
function isBlank(line: string): boolean {
  for (let i = 0; i < line.length; i++) {
    const c = line.charCodeAt(i);
    if (c !== 32 && c !== 9 && c !== 13 && c !== 10) return false;
  }
  return true;
}

/** Fraction of [a, b] covered by the measured range [lo, hi], as a percentage. */
function coverage(a: number, b: number, lo: number, hi: number): number {
  const overlap = Math.max(0, Math.min(b, hi) - Math.max(a, lo));
  return (100 * overlap) / (b - a);
}

/** Ascending permutation of `wn`, plus the sorted axis itself. */
function sortAxis(wn: Float64Array): {
  order: Int32Array;
  sorted: Float64Array;
} {
  const order = Int32Array.from(
    Array.from(wn.keys()).sort((a, b) => wn[a] - wn[b]),
  );
  const sorted = new Float64Array(wn.length);
  for (let k = 0; k < order.length; k++) sorted[k] = wn[order[k]];
  return { order, sorted };
}

export interface ConvertOptions {
  /** Called with 0..1 as source bytes are consumed. */
  onProgress?: (fraction: number) => void;
  /** Max pixel rows to emit. Defaults to the endpoint's own cap. */
  pixelCap?: number;
  /** Bytes per read slice. */
  chunkBytes?: number;
}

/** Streaming line reader over a File, decoded as latin-1. */
async function* readLines(
  file: File,
  chunkBytes: number,
  onProgress?: (fraction: number) => void,
): AsyncGenerator<string> {
  const decoder = new TextDecoder("iso-8859-1");
  let buffer = "";
  for (let offset = 0; offset < file.size; offset += chunkBytes) {
    const end = Math.min(offset + chunkBytes, file.size);
    const buf = await file.slice(offset, end).arrayBuffer();
    buffer += decoder.decode(buf, { stream: end < file.size });

    let start = 0;
    for (;;) {
      const nl = buffer.indexOf("\n", start);
      if (nl === -1) break;
      let line = buffer.slice(start, nl);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      yield line;
      start = nl + 1;
    }
    buffer = buffer.slice(start);
    onProgress?.(end / file.size);
  }
  if (buffer.length > 0) {
    yield buffer.endsWith("\r") ? buffer.slice(0, -1) : buffer;
  }
}

/**
 * Convert a raw XploRA export to the canonical Atlas 2048-bin layout.
 *
 * Streams the source once. Header lines pass through verbatim; the wavenumber
 * axis is read from the first non-`#` line; pixel rows are reservoir-sampled to
 * `pixelCap` so only the retained rows are ever split into cells.
 */
export async function convertToAtlas(
  file: File,
  opts: ConvertOptions = {},
): Promise<ConversionResult> {
  const {
    onProgress,
    pixelCap = PIXEL_CAP,
    chunkBytes = 8 * 1024 * 1024,
  } = opts;

  const header: string[] = [];
  let nativeWn: Float64Array | null = null;
  let layout: SourceLayout = "map";

  // Reservoir of retained pixel rows, paired with their source index.
  const resLines: string[] = [];
  const resIndex: number[] = [];
  const rand = mulberry32(0x5eed);
  let nRows = 0;

  // `spectrum` layout accumulates (wavenumber, intensity) pairs instead.
  const specWn: number[] = [];
  const specVal: number[] = [];

  let sawWnRow = false;

  for await (const line of readLines(file, chunkBytes, onProgress)) {
    if (!sawWnRow) {
      if (line.startsWith("#")) {
        header.push(line);
        continue;
      }
      sawWnRow = true;

      const cells = line.split("\t");
      const wnValues: number[] = [];
      for (let k = 2; k < cells.length; k++) {
        if (cells[k].trim() === "") continue;
        wnValues.push(Number(cells[k].replace(/,/g, "")));
      }

      if (wnValues.length >= 2 && wnValues.every(Number.isFinite)) {
        layout = "map";
        nativeWn = Float64Array.from(wnValues);
        continue;
      }

      // Not a map header row. A two-column `wavenumber \t intensity` list is a
      // single averaged / background spectrum — handle it as a 1-pixel map.
      const a = toFloat(cells[0] ?? "");
      const b = toFloat(cells[1] ?? "");
      if (cells.length === 2 && Number.isFinite(a) && Number.isFinite(b)) {
        layout = "spectrum";
        specWn.push(a);
        specVal.push(b);
        continue;
      }

      throw new ConversionError(
        "unrecognised layout — expected either a tab-delimited wavenumber row " +
          "(two empty lead cells, then the axis) or a two-column " +
          "`wavenumber <tab> intensity` spectrum. This does not look like a " +
          "Raman export.",
      );
    }

    if (layout === "spectrum") {
      if (isBlank(line)) continue;
      const cells = line.split("\t");
      if (cells.length < 2) continue;
      const a = toFloat(cells[0]);
      const b = toFloat(cells[1]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      specWn.push(a);
      specVal.push(b);
      continue;
    }

    // Map layout: reservoir-sample without splitting non-retained rows.
    if (isBlank(line)) continue;
    if (nRows < pixelCap) {
      resLines.push(line);
      resIndex.push(nRows);
    } else {
      const j = Math.floor(rand() * (nRows + 1));
      if (j < pixelCap) {
        resLines[j] = line;
        resIndex[j] = nRows;
      }
    }
    nRows++;
  }

  if (!sawWnRow) {
    throw new ConversionError("file is empty");
  }

  if (layout === "spectrum") {
    return finishSpectrum(file, header, specWn, specVal);
  }

  if (!nativeWn) {
    throw new ConversionError("could not read a wavenumber axis");
  }

  // Restore source order — the endpoint's own cap sorts its indices too, and
  // spatial order keeps the emitted map coherent.
  const perm = Array.from(resLines.keys()).sort(
    (a, b) => resIndex[a] - resIndex[b],
  );

  const nNative = nativeWn.length;
  const { order, sorted } = sortAxis(nativeWn);
  const { idx, w } = buildInterpPlan(sorted);

  const ints = new Float64Array(nNative);
  const cells = new Array<string>(N_BINS);
  const rows: string[] = [];
  let nSkipped = 0;

  for (const p of perm) {
    const toks = resLines[p].split("\t");
    if (toks.length < 3) {
      nSkipped++;
      continue;
    }
    // Native col0 = Y (um), col1 = X (um). The Atlas parser reads toks[0] as
    // x_um, so emit X first to keep coordinates semantically correct.
    const c0 = Number(toks[0].replace(/,/g, ""));
    const c1 = Number(toks[1].replace(/,/g, ""));
    if (!Number.isFinite(c0) || !Number.isFinite(c1)) {
      nSkipped++;
      continue;
    }
    if (toks.length - 2 < nNative) {
      nSkipped++;
      continue;
    }

    let ok = true;
    for (let k = 0; k < nNative; k++) {
      const v = toFloat(toks[2 + k]);
      if (!Number.isFinite(v)) {
        ok = false;
        break;
      }
      ints[k] = v;
    }
    if (!ok) {
      nSkipped++;
      continue;
    }

    for (let k = 0; k < N_BINS; k++) {
      const j = idx[k];
      const f0 = ints[order[j]];
      const f1 = ints[order[j + 1]];
      cells[k] = (f0 + w[k] * (f1 - f0)).toFixed(1);
    }
    rows.push(`${c1.toFixed(3)}\t${c0.toFixed(3)}\t${cells.join("\t")}\n`);
  }

  if (rows.length === 0) {
    throw new ConversionError(
      "no valid pixel rows parsed — every row was short, malformed, or " +
        "contained non-numeric intensities",
    );
  }

  return assemble(file, header, nativeWn, rows, {
    layout: "map",
    // Every non-blank data row seen in the source. `nSkipped` counts malformed
    // rows found among the retained sample only, so it is not subtracted here.
    nPixelsTotal: nRows,
    nPixelsKept: rows.length,
    subsampled: nRows > pixelCap,
    nSkipped,
  });
}

/** A two-column spectrum becomes a single synthetic pixel at the origin. */
function finishSpectrum(
  file: File,
  header: string[],
  wn: number[],
  vals: number[],
): ConversionResult {
  if (wn.length < 2) {
    throw new ConversionError(
      "spectrum has fewer than two points — nothing to resample",
    );
  }
  const nativeWn = Float64Array.from(wn);
  const { order, sorted } = sortAxis(nativeWn);
  const { idx, w } = buildInterpPlan(sorted);

  const cells = new Array<string>(N_BINS);
  for (let k = 0; k < N_BINS; k++) {
    const j = idx[k];
    const f0 = vals[order[j]];
    const f1 = vals[order[j + 1]];
    cells[k] = (f0 + w[k] * (f1 - f0)).toFixed(1);
  }
  const rows = [`0.000\t0.000\t${cells.join("\t")}\n`];

  return assemble(file, header, nativeWn, rows, {
    layout: "spectrum",
    nPixelsTotal: 1,
    nPixelsKept: 1,
    subsampled: false,
    nSkipped: 0,
  });
}

/** Write the canonical file and compute the coverage report. */
function assemble(
  file: File,
  header: string[],
  nativeWn: Float64Array,
  rows: string[],
  meta: Pick<
    ConversionStats,
    "layout" | "nPixelsTotal" | "nPixelsKept" | "subsampled" | "nSkipped"
  >,
): ConversionResult {
  const nNative = nativeWn.length;
  let lo = Infinity;
  let hi = -Infinity;
  for (let k = 0; k < nNative; k++) {
    if (nativeWn[k] < lo) lo = nativeWn[k];
    if (nativeWn[k] > hi) hi = nativeWn[k];
  }

  const chunks: string[] = [];
  for (const h of header) chunks.push(h + "\n");
  chunks.push(
    `#Converted=resampled ${nNative}pts [${lo.toFixed(0)}-${hi.toFixed(0)} cm-1] ` +
      `-> ${N_BINS}pts [76-3499 cm-1] canonical Atlas axis\n`,
  );
  chunks.push("#ConvertedBy=atlas-convert.ts (in-browser)\n");
  if (meta.subsampled) {
    chunks.push(
      `#Subsampled=${meta.nPixelsKept} of ${meta.nPixelsTotal} pixel rows ` +
        `(endpoint caps at ${PIXEL_CAP})\n`,
    );
  }
  if (meta.layout === "spectrum") {
    chunks.push(
      "#SourceLayout=single two-column spectrum, emitted as one pixel row\n",
    );
  }

  const wnCells = new Array<string>(N_BINS);
  for (let k = 0; k < N_BINS; k++) wnCells[k] = CANONICAL_WN[k].toFixed(3);
  chunks.push("\t\t" + wnCells.join("\t") + "\n");
  for (const r of rows) chunks.push(r);

  const stem = file.name.replace(/\.[^./\\]+$/, "");
  const outputName = `${stem}__atlas2048.txt`;
  const bytes = latin1Bytes(chunks.join(""));

  return {
    file: new File([bytes], outputName, { type: "text/plain" }),
    stats: {
      ...meta,
      nNative,
      nativeRange: [lo, hi],
      fingerprintCov: coverage(FINGERPRINT[0], FINGERPRINT[1], lo, hi),
      chCov: coverage(CH_STRETCH[0], CH_STRETCH[1], lo, hi),
      inputBytes: file.size,
      outputBytes: bytes.byteLength,
      outputName,
    },
  };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
