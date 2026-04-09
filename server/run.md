Run inference service (downloads/warm-loads models on startup):

`python server/inference_service.py --host 0.0.0.0 --port 8000`

Expose with a public URL (requires `cloudflared` installed):

`python server/inference_service.py --host 0.0.0.0 --port 8000 --expose-web`

Optional flags:

- `--cache-dir /path/to/hf_cache`
- `--skip-warmup` (load models lazily on first request)
