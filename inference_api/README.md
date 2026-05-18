# atlas-inference (Modal)

Modal serverless endpoint for `atlas.inference.predict_from_xls`.

This is the Python half of the Atlas Raman UI (the other half is `ui/`, a
Next.js app deployed on Vercel). The browser POSTs an Atlas `.xls` upload
directly to the Modal URL; Vercel is not in the request path.

## What's in the image

- `atlas/` package (copied via `add_local_dir`)
- `artifacts/` directory (Stage 15F joblibs + JSON, ~2 MB; baked into image)
- `numpy`, `pandas`, `scikit-learn`, `joblib`, `pybaselines`, `PyWavelets`,
  `scipy`, `fastapi[standard]`

No S3, no separate package upload. `modal deploy` ships everything.

## One-time setup

```bash
cd inference_api
uv venv && source .venv/bin/activate
uv pip install modal
modal token new      # opens browser, one-time auth
```

The heavy deps (numpy, sklearn, pybaselines, etc.) live inside the Modal
image declaration -- not in `pyproject.toml` -- because they don't need to
be installed locally. Your laptop only needs the `modal` CLI.

## Deploy

```bash
modal deploy modal_app.py
```

Modal prints two URLs on success:

```
predict   -> https://<user>--atlas-inference-predict.modal.run
healthz   -> https://<user>--atlas-inference-healthz.modal.run
```

First deploy: image build is ~60-90 s (pip install + atlas/artifacts upload).
Subsequent deploys: ~10-20 s (layer cache hits, only the code layer rebuilds).

## Smoke test

```bash
# health check (no payload, no cold start cost beyond container boot)
curl https://<user>--atlas-inference-healthz.modal.run
# -> {"ok": true, "service": "atlas-inference"}

# real prediction (multipart file upload)
curl -X POST -F "file=@../Atlas Data/STEC/O157H7/R357_xxx.xls" \
     https://<user>--atlas-inference-predict.modal.run
# -> {"class": "STEC", "probabilities": {...}, "spectrum_mean": [...],
#     "wn": [...], "feature_values": {...}}
```

Note on the `file: bytes` signature: FastAPI accepts the multipart body
into a `bytes` parameter when the route has a single body-typed arg. If
the first deploy returns a 422 on the `-F file=@...` curl above, swap
the predict signature to:

```python
from fastapi import File
def predict(file: bytes = File(...)):
```

and redeploy. (This is the canonical FastAPI multipart form; the bare
`bytes` form works in current FastAPI versions but is the more brittle
of the two.)

## Wiring into the frontend

After `modal deploy` prints the predict URL, paste it into `ui/.env.local`:

```bash
NEXT_PUBLIC_MODAL_PREDICT_URL=https://<user>--atlas-inference-predict.modal.run
```

The Next.js Live Inference tab (Tab 6) POSTs `multipart/form-data` to that
URL directly from the browser (CORS-allowed; not routed through Vercel,
so Vercel's 10 s hobby-tier function timeout doesn't apply).

For production, set the same env var in Vercel:

```bash
cd ../ui
pnpm dlx vercel env add NEXT_PUBLIC_MODAL_PREDICT_URL production
```

## Cost expectation

- Modal free tier: $30/mo compute credit.
- Single inference: ~5 s on 2 CPU / 2 GB.
- Expected review traffic: low double digits per day.
- Net cost: effectively $0.

## Latency

| Scenario | Time |
|---|---|
| Cold start (container boot + first inference) | 3-5 s |
| Warm inference (container hot) | ~3 s |
| Scale-to-zero window | 300 s of idle |

`min_containers=0` + `scaledown_window=300` = the container sleeps after
5 min of no traffic, then cold-starts on the next request.

## Re-deploy after changes

| What changed | What to do |
|---|---|
| `atlas/` code | `modal deploy modal_app.py` (rebuilds the atlas layer, ~20 s) |
| `artifacts/` (e.g., new model) | `modal deploy modal_app.py` (rebuilds artifacts layer) |
| `modal_app.py` only | `modal deploy modal_app.py` (just the code layer, ~10 s) |
| `pip_install` list | `modal deploy modal_app.py` (full image rebuild, ~60-90 s) |

## Files

```
inference_api/
|-- modal_app.py      # the endpoint
|-- pyproject.toml    # local-only deps (just `modal`)
`-- README.md         # this file
```

## Reference

Modal Web Functions guide: https://modal.com/docs/guide/webhooks
Modal `fastapi_endpoint` API reference: https://modal.com/docs/reference/modal.fastapi_endpoint
