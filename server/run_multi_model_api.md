Multi-model Foley API (perception/planner/execution/verification)

Install runtime dependencies in your venv:

`pip install -r server/requirements_multi_model_api.txt`

Add your Hugging Face token to `.env` in the repo root:

`HF_TOKEN=hf_xxxxxxxxxxxxxxxxx`

Optional for bigger quantized models (e.g., AWQ variants):

`pip install -U autoawq`

Fallback if gated models still fail:

`huggingface-cli login`

Run the service (downloads + warm-loads models):

`bash server/run_multi_model_api.sh`

Switch perception model to 72B-AWQ:

`PERCEPTION_MODEL=Qwen/Qwen2-VL-72B-Instruct-AWQ bash server/run_multi_model_api.sh`

API endpoints:

- `GET /health`
- `POST /warmup`
- `POST /perception`
- `POST /planner`
- `POST /execution`
- `POST /verification`

Quick request examples:

`curl -X POST http://127.0.0.1:8000/planner -H 'content-type: application/json' -d '{"vlm_log":"[0.0s] footsteps on gravel"}'`

`curl -X POST http://127.0.0.1:8000/execution -H 'content-type: application/json' -d '{"prompt":"crunchy footsteps on gravel", "duration": 3.0}'`
