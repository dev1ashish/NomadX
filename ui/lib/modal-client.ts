/**
 * Modal `/predict` client.
 * POSTs `multipart/form-data` to NEXT_PUBLIC_MODAL_PREDICT_URL.
 * Endpoint is deployed from `inference_api/modal_app.py` (plan §W9).
 */
import type { PredictionResponse } from "./types";

export async function predict(file: File): Promise<PredictionResponse> {
  const url = process.env.NEXT_PUBLIC_MODAL_PREDICT_URL;
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_MODAL_PREDICT_URL is not set. " +
        "Run `modal deploy inference_api/modal_app.py` then add the printed URL " +
        "to `ui/.env.local` (see `ui/.env.example`).",
    );
  }

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Modal /predict failed (${res.status}): ${body}`);
  }
  return (await res.json()) as PredictionResponse;
}
