/**
 * Modal prediction clients.
 * POSTs `multipart/form-data` to the two prediction endpoints deployed from
 * `inference_api/modal_app.py`:
 *   - /predict       → Stage 15F LogReg-L2 on 35 MI features
 *   - /predict_plsda → PLS-DA on raw 987-bin spectrum (project headline)
 *
 * The Live tab fans out to both in parallel.
 *
 * IMPORTANT: env reads must be *literal* `process.env.NEXT_PUBLIC_X` so
 * Next.js can inline them at build time. Dynamic accesses like
 * `process.env[key]` are NOT replaced and resolve to `undefined` in the
 * browser bundle.
 */
import type { PredictionResponse } from "./types";

async function postFile(url: string, file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Modal ${url} failed (${res.status}): ${body}`);
  }
  return (await res.json()) as PredictionResponse;
}

export async function predict(file: File): Promise<PredictionResponse> {
  const url = process.env.NEXT_PUBLIC_MODAL_PREDICT_URL;
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_MODAL_PREDICT_URL is not set. " +
        "Run `modal deploy inference_api/modal_app.py` then add the printed " +
        "/predict (LogReg) URL to `ui/.env.local` (see `ui/.env.example`).",
    );
  }
  return postFile(url, file);
}

export async function predictPlsda(file: File): Promise<PredictionResponse> {
  const url = process.env.NEXT_PUBLIC_MODAL_PREDICT_PLSDA_URL;
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_MODAL_PREDICT_PLSDA_URL is not set. " +
        "Run `modal deploy inference_api/modal_app.py` then add the printed " +
        "/predict_plsda (PLS-DA) URL to `ui/.env.local` (see `ui/.env.example`).",
    );
  }
  return postFile(url, file);
}
