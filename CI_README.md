# AnomalyGPT CI/CD Starter Kit

This folder contains a minimal, production-friendly CI/CD setup tailored to a GroundingDINO + MMDetection + Python project.

## What's inside

- `.github/workflows/ci.yml`: GitHub Actions workflow
  - **Lint & unit (CPU)** on every push/PR
  - **GPU smoke** job on a **self-hosted runner** (`[self-hosted, gpu]` labels)
  - **Docker build & publish** on tags (to `ghcr.io/<owner>/<repo>`)

- `.pre-commit-config.yaml`: Code quality hooks (ruff, black, etc.)
- `tests/test_smoke.py`: A minimal smoke test that can optionally run your demo script if assets and a checkpoint are available.
- `assets/samples/sample.png`: Tiny test image used by the smoke test.

## Quick start

1. Copy these files into your repo (preserving paths).
2. Install pre-commit locally:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
3. Set repository **Secrets** if needed (Settings → Secrets and variables → Actions):
   - `HF_TOKEN` (Hugging Face)
   - `SMOKE_CKPT` (path/URL to a small checkpoint accessible to the GPU runner)
   - Any other tokens (e.g., `WANDB_API_KEY`)

## GPU self-hosted runner (outline)

- Prepare a GPU server with NVIDIA drivers, Docker, and `nvidia-container-toolkit`.
- Register the runner (repo → Settings → Actions → Runners → New self-hosted runner).
- Add labels: `gpu` (and optionally `cuda12`).
- Run the service; confirm `nvidia-smi` works inside jobs.

## Optional: datasets & artifacts

- For large datasets, prefer **DVC** or a private object storage (S3/GCS/MinIO).
- Keep tiny **mini-sets** in-repo (`assets/mini/`) for fast CI sanity checks.

## Release flow

- Tag a commit: `git tag v0.1.0 && git push --tags`
- The `docker_build_and_publish` job builds & pushes `ghcr.io/<owner>/<repo>:v0.1.0` and `:latest`.

Customize as needed—this is intentionally minimal to get you moving quickly.
