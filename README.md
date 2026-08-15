# CassavaCare AI

Mobile-first web app that classifies cassava leaf disease (Bacterial Blight / Mosaic Disease / Healthy) for smallholder farmers, built with Flask + MobileNetV2.

## Overview

Cassava is a staple crop for millions of smallholder farmers, and leaf diseases like Cassava Bacterial Blight (CBB) and Cassava Mosaic Disease (CMD) can devastate yields if not caught early. CassavaCare AI lets a farmer photograph a leaf with their phone and get an immediate, plain-language diagnosis. The app runs entirely on a local machine — no cloud, no internet required beyond the local network — so it works in field conditions where connectivity is unreliable.

## Features

- **3-class MobileNetV2 classification** — Cassava Bacterial Blight / Cassava Mosaic Disease / Healthy
- **Confidence threshold** — low-confidence predictions return an "uncertain — consult an extension officer" state instead of a possibly wrong diagnosis
- **Mobile-first, accessible frontend** — drag-and-drop upload, large touch targets, reduced-motion support, works on a phone browser
- **Security hardening** — upload validation (type, size, corruption), rate limiting, security headers, CSRF header check, debug mode gated behind an environment variable
- **`/health` endpoint** — JSON status for load balancers and uptime checks
- **QR-code LAN access** — `generate_qr_code.py` prints a QR code so phones on the same Wi-Fi can open the app instantly

## Project structure

```
cassava_project/
├── app.py                  # Flask app: upload page, /predict JSON API, /health
├── model_training.py       # MobileNetV2 transfer learning (seeds + class weights)
├── evaluate_model.py       # Produces the evaluation report (report.txt)
├── generate_qr_code.py     # Local-network QR code generator for phone access
├── labels.json             # Class names, descriptions, management advice
├── cassava_model.h5        # Trained model (used by the app)
├── requirements.txt        # Pinned runtime dependencies
├── requirements-dev.txt    # Pinned dev dependencies (pytest, scikit-learn)
├── .env.example            # Environment variable template (copy to .env)
├── dataset/                # train/ (181 imgs) + validation/ (47 imgs), 3 classes
├── templates/              # HTML templates (index.html)
├── static/                 # CSS + JS frontend
└── tests/                  # pytest suite for the /predict pipeline
```

## Requirements

- **Python 3.12**
- Dependencies are pinned to exact versions in `requirements.txt` (dev/test extras in `requirements-dev.txt`), so installs are reproducible.

## Setup

```bash
# 1. Create and activate the virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
copy .env.example .env        # Windows
cp .env.example .env          # Linux / macOS

# 4. Run the app
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Environment variables

All variables are optional — the app falls back to safe defaults. They are read from a `.env` file (via `python-dotenv`) or the process environment.

| Variable | Default | Purpose |
|---|---|---|
| `FLASK_DEBUG` | `False` | Enables Werkzeug's interactive debugger. **Keep OFF in production** — it allows remote code execution when the app is reachable from the network. |
| `FLASK_HOST` | `127.0.0.1` | Interface to bind to. `0.0.0.0` is required for phone/QR-code access from the same Wi-Fi. |
| `FLASK_PORT` | `5000` | Port to bind to. |
| `CONFIDENCE_THRESHOLD` | `0.65` | Minimum model confidence (0.0–1.0) for a definitive diagnosis. Below this, the app returns an "uncertain" result. |

For the QR-code phone flow, set `FLASK_HOST=0.0.0.0` in `.env` and run `python generate_qr_code.py` to get a scannable code for your LAN IP.

## Running tests

```bash
pytest tests/ -v                    # full suite (includes the real-model test)
pytest tests/ -m "not slow"         # skip the real-model test (faster)
```

The suite covers the `/predict` pipeline: happy path, uncertainty threshold, missing/corrupt/oversized uploads, CSRF header enforcement, JSON error handling, and model-load degradation.

## Model performance (honest)

Evaluated on the 47-image validation set with `evaluate_model.py` (see `report.txt`):

| Metric | Value |
|---|---|
| **Validation accuracy** | **0.9574 (45/47)** |
| CBB — precision / recall / F1 | 0.90 / 0.90 / 0.90 |
| CMD — precision / recall / F1 | 0.94 / 0.94 / 0.94 |
| Healthy — precision / recall / F1 | 1.00 / 1.00 / 1.00 |

This is a large improvement over the baseline model (**0.6809 accuracy**; CBB recall 0.40, CMD recall 0.50) — the class-weighted retrain lifted the recall on both diseases dramatically. The baseline artifacts are kept in `cassava_model_baseline.h5` and `report_baseline.txt` for comparison.

**Caveats:** the validation set is only 47 images, so these numbers are indicative, not proof of field generalization — the model is still trained on just 181 images. Low-confidence predictions are routed to the "uncertain" state by the confidence threshold rather than presented as authoritative advice.

## Training & evaluation

- **Retrain:** `python model_training.py` — MobileNetV2 transfer learning with a fixed seed (reproducible) and inverse-frequency class weights (balanced training). The best checkpoint is saved to `cassava_model.h5`.
- **Evaluate:** `python evaluate_model.py` — runs the validation set through the same preprocessing as training and writes accuracy, a per-class classification report, and a confusion matrix to `report.txt`.

## Security

- Debug mode off by default, gated behind `FLASK_DEBUG` (Werkzeug debugger is a remote code execution risk)
- Binds to `127.0.0.1` by default (no silent LAN exposure)
- Upload validation: `secure_filename`, extension whitelist, size limit, corruption check, temp-file cleanup
- Rate limiting on all routes (per-worker in-memory)
- CSRF mitigation on `/predict` via the `X-Requested-With` custom header
- Security headers on every response (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `X-XSS-Protection`)
- JSON error responses (no stack traces leaked) and graceful degradation if the model or labels fail to load

## Future work

- Collect more disease images (especially Bacterial Blight) and retrain to validate the improved recall on a larger, more diverse dataset
- Expand the dataset and run k-fold cross-validation to confirm the 95.7% result generalizes beyond the 47-image held-out set
- Add a Content-Security-Policy once the frontend is refactored (currently deferred because the app loads Google Fonts from external domains)
- Shared rate-limit store (e.g., Redis) for multi-worker deployments
- Dockerfile for containerized deployment
