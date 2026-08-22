# CassavaCare AI — Program Details

A comprehensive guide to every file in the project: what it does, how it works,
and how the pieces connect.

**Author:** Fusheini Abdul-Mumin
**Repository:** [github.com/fusheini1/CassavaCare-AI](https://github.com/fusheini1/CassavaCare-AI)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [File Structure](#3-file-structure)
4. [Core Application Files](#4-core-application-files)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [Frontend (HTML / CSS / JS)](#6-frontend)
7. [Tests](#7-tests)
8. [Dataset](#8-dataset)
9. [Configuration & Environment](#9-configuration--environment)
10. [Documentation & Assets](#10-documentation--assets)
11. [API Reference](#11-api-reference)
12. [Model Architecture](#12-model-architecture)

---

## 1. Project Overview

CassavaCare AI is a mobile-first web application that classifies cassava leaf
disease from a farmer's photograph. It uses a fine-tuned MobileNetV2 deep
learning model to distinguish three conditions:

- **Cassava Bacterial Blight (CBB)** — caused by *Xanthomonas axonopodis pv. manihotis*
- **Cassava Mosaic Disease (CMD)** — caused by cassava mosaic geminiviruses
- **Healthy** — no visible disease symptoms

The app runs entirely on a local machine over Wi-Fi — no cloud, no internet
required beyond the LAN. A farmer photographs a leaf, the phone sends it to the
Flask server, and gets a plain-language diagnosis with management advice, or an
honest "uncertain — consult an extension officer" result when the model lacks
confidence.

---

## 2. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Web framework | Flask | 3.1.3 | HTTP server, routing, templates |
| Rate limiting | Flask-Limiter | 4.1.1 | Per-route request throttling |
| Deep learning | TensorFlow / Keras | 2.21.0 | Model training and inference |
| Image processing | Pillow | 12.2.0 | Image loading, resizing, validation |
| Numerical | NumPy | 2.4.4 | Array operations for preprocessing |
| QR codes | qrcode | 8.2 | LAN QR code generation |
| Environment | python-dotenv | 1.0.1 | `.env` file loading |
| WSGI server | Gunicorn | 26.0.0 | Production deployment (Linux/macOS) |
| Testing | pytest | 9.1.1 | Test runner (dev dependency) |
| ML metrics | scikit-learn | 1.8.0 | Classification report, confusion matrix, CV |
| Frontend | Vanilla JS + CSS | — | No framework; Google Fonts (Inter) |

All runtime dependencies are pinned in `requirements.txt`. Dev/test extras are
in `requirements-dev.txt`.

---

## 3. File Structure

```
cassava_project/
│
├── app.py                        # Flask web application (main entry point)
├── utils.py                      # Image preprocessing helpers
├── labels.json                   # Class names, descriptions, management advice
├── cassava_model.h5              # Trained MobileNetV2 model (binary, ~11 MB)
├── cassava_model_baseline.h5     # Backup of pre-retrain baseline model
│
├── model_training.py             # Training script (MobileNetV2 transfer learning)
├── evaluate_model.py             # Evaluation script → report.txt
├── cross_validate.py             # 5-fold CV + confidence threshold calibration
├── generate_qr_code.py           # QR code generator for LAN phone access
│
├── report.txt                    # Evaluation report (accuracy, classification report, CV, calibration)
├── report_baseline.txt           # Backup of the pre-retrain evaluation report
│
├── requirements.txt              # Pinned runtime dependencies
├── requirements-dev.txt          # Pinned dev/test dependencies (pytest, scikit-learn)
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── pytest.ini                    # Pytest configuration (markers)
│
├── LICENSE                       # MIT License
├── README.md                     # Project documentation
├── DEFENSE_OUTLINE.md            # Project defense slide outline + Q&A
├── PROGRAM_DETAILS.md            # This file — detailed program documentation
│
├── templates/
│   ├── index.html                # Main upload page (Jinja2 template)
│   └── result.html               # Standalone result page (unused; UI is SPA)
│
├── static/
│   ├── css/
│   │   └── style.css             # Mobile-first responsive stylesheet
│   └── js/
│       └── script.js             # Frontend logic (upload, validation, display)
│
├── tests/
│   ├── helpers.py                # FakeModel, fake TensorFlow, image factory
│   ├── conftest.py               # Pytest fixtures (client, fake TF, globals restore)
│   └── test_predict.py           # 25 tests covering the /predict pipeline
│
├── dataset/
│   ├── train/                    # 181 training images (3 class folders)
│   │   ├── Cassava_Bacterial_Blight/   # 39 images (CBB)
│   │   ├── Cassava_Mosaic_Disease/     # 70 images (CMD)
│   │   └── Healthy/                    # 72 images
│   └── validation/               # 47 validation images (3 class folders)
│       ├── Cassava_Bacterial_Blight/   # 10 images
│       ├── Cassava_Mosaic_Disease/     # 18 images
│       └── Healthy/                    # 19 images
│
├── cassava leaf disease dataset/ # Raw source images (228 total; git-ignored)
│
└── assets/                       # Screenshots for README
    ├── home_desktop.png
    ├── home_mobile.png
    ├── result_mobile.png
    ├── model_summary.png
    └── training_cmd.png
```

---

## 4. Core Application Files

### 4.1 `app.py` — Flask Web Application

**Purpose:** The main entry point. Loads the trained model at startup, serves
the frontend, and exposes the `/predict` JSON API and `/health` endpoint.

**Key components:**

| Component | Description |
|---|---|
| `load_dotenv()` | Loads `.env` for `FLASK_DEBUG`, `FLASK_HOST`, `FLASK_PORT`, `CONFIDENCE_THRESHOLD` |
| `load_model_and_labels()` | Loads `cassava_model.h5` and `labels.json` at startup with graceful fallback on corrupt/missing files |
| `allowed_file()` | Extension whitelist: `.jpg`, `.jpeg`, `.png` only |
| `ensure_upload_folder()` | Creates `uploads/` directory if it doesn't exist |

**Routes:**

| Route | Method | Description |
|---|---|---|
| `GET /` | GET | Serves the main upload page (`index.html`) |
| `GET /health` | GET | JSON status check: `{"status": "ok", "model_loaded": bool, "labels_loaded": bool}` |
| `POST /predict` | POST | Accepts an image upload, runs inference, returns diagnosis JSON |

**`/predict` request pipeline (in order):**
1. CSRF check: `X-Requested-With: XMLHttpRequest` header required (403 if missing)
2. Model/labels availability check (503 if unavailable)
3. File presence and non-empty filename check (400)
4. Extension validation (400)
5. Save to `uploads/` with `secure_filename` + UUID prefix
6. Image corruption check via `validate_image()` (400)
7. Preprocess via `preprocess_image()` → MobileNetV2 input shape
8. `model.predict()` → argmax + softmax confidence
9. Below `CONFIDENCE_THRESHOLD`? → return `"uncertain"` severity (success: true)
10. Look up class name, description, advice from `labels.json`
11. Return JSON with severity (`healthy` / `disease` / `uncertain`)
12. **Always:** delete the uploaded file in `finally` block

**Security features:**
- `debug=False` by default (gated behind `FLASK_DEBUG`)
- Default host `127.0.0.1` (no silent LAN exposure)
- 5 MB upload size limit (`MAX_CONTENT_LENGTH`)
- `secure_filename` + UUID prefix (no directory traversal, no filename collisions)
- CSRF mitigation via custom header (blocks cross-origin form posts)
- Security headers on every response: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `X-XSS-Protection`
- JSON error handlers for 404, 413, 500 (no HTML stack traces)
- Graceful degradation: corrupt model or labels → 503 on `/predict`, UI still serves

**Entry point:**
```python
if __name__ == "__main__":
    app.run(debug=debug, host=host, port=port)
```

---

### 4.2 `utils.py` — Image Preprocessing Utilities

**Purpose:** Provides the preprocessing pipeline that converts a raw image file
into the NumPy array shape expected by MobileNetV2.

**Functions:**

**`preprocess_image(image_path)`**
1. Opens the image with Pillow
2. Converts to RGB (handles grayscale, RGBA, palette inputs)
3. Resizes to 224×224 pixels
4. Converts to `float32` NumPy array
5. Adds a batch dimension: `(224, 224, 3)` → `(1, 224, 224, 3)`
6. Applies `preprocess_input()` — scales pixels from `[0, 255]` to `[-1, 1]`

This must exactly match the preprocessing used during training for accurate
predictions.

**`validate_image(image_path)`**
Opens the image and calls `img.verify()` — a lightweight corruption check that
does not load the full pixel data. Returns `True` / `False`.

---

### 4.3 `labels.json` — Class Metadata

**Purpose:** Maps class indices (strings `"0"`, `"1"`, `"2"`) to human-readable
names, descriptions, and management advice. Loaded once at startup by `app.py`.

**Structure:**
```json
{
    "_comment": "Class indices are alphabetical by folder name...",
    "0": { "name": "Cassava Bacterial Blight (CBB)", "description": "...", "advice": "..." },
    "1": { "name": "Cassava Mosaic Disease (CMD)",   "description": "...", "advice": "..." },
    "2": { "name": "Healthy",                        "description": "...", "advice": "..." }
}
```

**Class index assignment:** alphabetical by folder name:
- `Cassava_Bacterial_Blight` → 0
- `Cassava_Mosaic_Disease` → 1
- `Healthy` → 2

Keys starting with `_` are stripped when loaded (used as comments only).

---

### 4.4 `generate_qr_code.py` — LAN QR Code Generator

**Purpose:** Interactive CLI tool that discovers the machine's LAN IP addresses,
lets the user select one, and generates a QR code image (`cassava_app_qr.png`)
linking to `http://<ip>:5000`. Farmers on the same Wi-Fi scan the QR code with
their phone to open the app instantly.

**Flow:**
1. `get_all_ip_addresses()` — discovers all non-loopback IPv4 addresses via `socket.getaddrinfo`
2. Presents a numbered list; user picks by number or types an IP directly
3. Creates QR code with `qrcode.QRCode` (version 1, error correction L, box size 10, border 4)
4. Saves `cassava_app_qr.png` and opens it via `os.startfile()` (Windows)

---

## 5. Machine Learning Pipeline

### 5.1 `model_training.py` — Model Training Script

**Purpose:** Trains a MobileNetV2-based CNN for cassava leaf disease
classification using transfer learning. Produces `cassava_model.h5`.

**Reproducibility:** `SEED = 42` set across `random`, `numpy`, and `tensorflow`
at the top of the file.

**Key configuration:**
- `NUM_CLASSES = 3`
- `IMG_SIZE = (224, 224)`
- `BATCH_SIZE = 32`, `EPOCHS = 50`, `LEARNING_RATE = 0.0001`
- Dataset paths: `dataset/train/` and `dataset/validation/`

**Data augmentation** (training only):
| Parameter | Value |
|---|---|
| `rotation_range` | 30° |
| `width_shift_range` | 20% |
| `height_shift_range` | 20% |
| `shear_range` | 0.2 |
| `zoom_range` | 20% |
| `horizontal_flip` | True |
| `vertical_flip` | False |
| `fill_mode` | nearest |

**Class weighting:** Inverse-frequency weights computed from the training
generator to compensate for class imbalance:
```python
counts = np.bincount(train_generator.classes)
class_weight = {i: total / (n_classes * c) for i, c in enumerate(counts)}
```
Observed weights: `{CBB: 1.55, CMD: 0.86, Healthy: 0.84}`.

**Functions:**
- `create_data_generators()` — builds augmented train and plain validation generators
- `build_model()` — constructs the MobileNetV2 + custom head architecture
- `get_callbacks()` — returns `EarlyStopping` (patience 5, restore best) and `ModelCheckpoint` (save best by val_accuracy)
- `train()` — orchestrates the full pipeline

---

### 5.2 `evaluate_model.py` — Model Evaluation

**Purpose:** Loads the trained model, runs predictions on the validation set,
and produces `report.txt` with:
- Overall accuracy
- Per-class classification report (precision / recall / F1)
- Formatted text confusion matrix

**Key details:**
- Uses the same preprocessing as training (`preprocess_input`, 224×224, no augmentation)
- Validation set is loaded via `ImageDataGenerator.flow_from_directory(shuffle=False)`
- Class names come from `labels.json` (index-aligned)
- Confusion matrix is rendered as a text table (no matplotlib dependency)
- Prints a summary to stdout and writes the full report to `report.txt`

**Current results** (retrained model, 0.9574 accuracy, 47 samples):
```
Overall accuracy: 0.9574 (45/47)
CBB:  precision 0.90, recall 0.90, F1 0.90  (10 images)
CMD:  precision 0.94, recall 0.94, F1 0.94  (18 images)
Healthy: precision 1.00, recall 1.00, F1 1.00 (19 images)
```

---

### 5.3 `cross_validate.py` — Cross-Validation + Threshold Calibration

**Purpose:** Two analyses that guard against overfitting and tune the
confidence threshold.

**Part 1 — 5-fold cross-validation:**
- Pools all 228 images (train + validation) into one dataset
- Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Each fold trains a fresh MobileNetV2 (same architecture, augmentation, class
  weights, seeds) for a reduced schedule of 5 epochs
- Reports per-fold accuracy, mean ± std, and per-class recall
- Compares against the single-split 0.9574 figure and flags optimism

**Part 2 — Threshold calibration:**
- Collects per-sample (confidence, prediction, truth) from all 5 folds
- Sweeps thresholds: `[0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]`
- For each: rejection rate, confident accuracy, confident count,
  uncertain-would-be-wrong
- Builds a reliability table: mean predicted confidence vs observed accuracy
  per bin
- Generates a data-driven interpretation of whether the current threshold
  (0.65) is well-calibrated

**Current CV results:** 0.8727 ± 0.0484 (folds: 0.848 / 0.891 / 0.913 /
0.800 / 0.911). The 0.9574 single-split figure was flagged as +8.5 points
optimistic.

**Threshold finding:** the model is underconfident — at 0.65, confident
predictions are right ~98% of the time, but the threshold rejects 58.8% of
samples (most of which are correct).

**Report output:** both sections are appended to `report.txt` (replacing
previous CV/calibration sections if they exist).

---

### 5.4 `report.txt` — Evaluation Report

**Purpose:** The single source of truth for model performance. Contains three
sections:
1. **Evaluation** (from `evaluate_model.py`) — single-split accuracy, classification report, confusion matrix
2. **Cross-Validation** (from `cross_validate.py`) — 5-fold CV accuracy and interpretation
3. **Threshold Calibration** (from `cross_validate.py`) — sweep table, reliability bins, verdict

### 5.5 `cassava_model.h5` — Trained Model

**Purpose:** The Keras HDF5 model file (~11 MB) loaded by `app.py` at startup
and by `evaluate_model.py` / `cross_validate.py` for evaluation. Produced by
`model_training.py`.

### 5.6 `cassava_model_baseline.h5` — Baseline Model Backup

**Purpose:** Backup of the original pre-retrained model (68.09% accuracy,
CBB recall 0.40). Kept for comparison. Referenced in `report_baseline.txt`.

---

## 6. Frontend

### 6.1 `templates/index.html` — Main Upload Page

**Purpose:** The single-page app served by `GET /`. Jinja2 template that
provides:
- Header with logo and tagline
- Drag-and-drop / tap-to-select upload zone
- Image preview and file info bar
- Loading spinner section
- Results section (populated dynamically by `script.js`)
- Error section with retry button
- Footer with disclaimer ("consult an extension officer")

**Key design choices:**
- No framework — vanilla HTML/CSS/JS for minimal bundle size
- Google Fonts (Inter) loaded from CDN
- `accept=".jpg,.jpeg,.png"` on the file input
- All results rendered client-side via `innerHTML` (with XSS escaping in JS)

### 6.2 `templates/result.html` — Standalone Result Page

**Purpose:** A separate result page template. Currently unused — the app
operates as a single-page app where results are rendered by JavaScript in
`index.html`. Kept for potential future use.

### 6.3 `static/css/style.css` — Responsive Stylesheet

**Purpose:** Mobile-first CSS with custom properties (design tokens), high
contrast for outdoor sunlight, and large touch targets.

**Design tokens (`:root`):**
- Primary: `#16a34a` (green — healthy)
- Danger: `#dc2626` (red — disease)
- Amber: `#d97706` (amber — uncertain)
- Neutrals: light backgrounds, clean borders

**Key features:**
- CSS custom properties for consistent theming
- Responsive layout (mobile-first, breakpoints for tablet/desktop)
- `.hidden` utility class for show/hide toggling
- `.result-card.healthy` (green border), `.result-card.disease` (red), `.result-card.uncertain` (amber)
- Animated confidence bar (width transition)
- Spinner animation for loading state
- Reduced-motion media query support
- `font-family: 'Inter'` throughout

### 6.4 `static/js/script.js` — Frontend Logic

**Purpose:** Handles the entire client-side workflow: file selection, preview,
upload, inference, and result rendering.

**Key functions:**

| Function | Purpose |
|---|---|
| `handleFileSelection(file)` | Validates type/size, shows preview, enables submit |
| `resetUpload()` | Clears form, hides results, returns to initial state |
| `submitImage()` | Async POST to `/predict` with `FormData` + CSRF header |
| `displayResults(data)` | Renders diagnosis card (icon, confidence bar, description, advice) |
| `showError(message)` | Shows error card with retry button |
| `escapeHtml(text)` | XSS prevention for dynamic content rendering |

**CSRF header:** `submitImage()` sends `X-Requested-With: XMLHttpRequest` in
the fetch headers — required by the server's CSRF check on `/predict`.

**Client-side validation:** mirrors the server: extension check (`.jpg`,
`.jpeg`, `.png`) and 5 MB size limit, with user-friendly error messages.

---

## 7. Tests

### 7.1 `tests/helpers.py` — Shared Test Utilities

**Purpose:** Provides the infrastructure for a fast test suite that runs
**without loading TensorFlow** (~8 seconds total for 25 tests).

**Key components:**

**`FakeModel`** — A minimal stand-in for the Keras model:
```python
class FakeModel:
    probs = [0.9, 0.05, 0.05]  # default softmax output
    def predict(self, x, verbose=0):
        return np.array([self.probs], dtype=float)
```
Individual tests override `app.model.probs` to simulate different confidence
levels (e.g., `[0.50, 0.30, 0.20]` for an uncertain case).

**`install_fake_tensorflow()`** — Replaces the entire `tensorflow` package
tree in `sys.modules` with lightweight fakes *before* `app.py` is imported.
This is necessary because:
- Flask-Limiter reads its enabled flag at construction time (can't be changed via `app.config`)
- Keras 3 uses lazy import proxies that defeat naive monkeypatching of
  `tensorflow.keras.models.load_model`

The fakes provide:
- `tensorflow.keras.models.load_model()` → returns `FakeModel()`
- `tensorflow.keras.applications.mobilenet_v2.preprocess_input()` → identity (pass-through)

**`make_image_bytes(fmt, size, color)`** — Generates a valid in-memory image
for upload tests, so tests don't depend on the dataset folder.

### 7.2 `tests/conftest.py` — Pytest Fixtures

**Purpose:** Runs `install_fake_tensorflow()` at module level, then provides:

**`client` fixture** — Flask test client with `TESTING=True` and rate limiting
disabled (`app.limiter.enabled = False` — the only reliable way since
`RATELIMIT_ENABLED` is only read at construction time).

**`restore_app_globals` fixture (autouse)** — Saves and restores `app.model`
and `app.labels` after each test, plus resets `FakeModel.probs` to defaults.
This lets 503-path tests safely set `app.model = None` without affecting other tests.

### 7.3 `tests/test_predict.py` — Test Suite (25 tests)

**Purpose:** Covers the entire `/predict` pipeline and edge cases.

**Test categories:**

| Category | Tests |
|---|---|
| Happy path | Valid image → diagnosis, correct severity, confidence, temp-file cleanup |
| Uncertainty | 50% and 64% confidence → "Uncertain" / "uncertain"; 66% → definitive (threshold boundary) |
| CSRF enforcement | Missing header → 403; header present → accepted |
| Validation errors | Missing file, empty filename, bad extension, corrupt image → 400 |
| Size limit | 6 MB upload → 413 |
| Degradation | Model `None` → 503; labels `None` → 503; unknown route → JSON 404 |
| Real model (slow) | Loads `cassava_model.h5`, checks output width matches `labels.json`, runs one genuine prediction |
| JSON 500 handler | Unhandled exception → JSON 500 (not HTML) |
| `/health` endpoint | Returns 200 with `status: ok`; reports model_loaded correctly |
| Security headers | Present on `/`, `/predict`, and `/health` responses |

**Running:**
```bash
venv\Scripts\python.exe -m pytest -v            # full suite (~9s)
venv\Scripts\python.exe -m pytest -m "not slow"  # skip real-model test
```

---

## 8. Dataset

### 8.1 Working Dataset (`dataset/`)

| Split | CBB | CMD | Healthy | Total |
|---|---|---|---|---|
| `dataset/train/` | 39 | 70 | 72 | **181** |
| `dataset/validation/` | 10 | 18 | 19 | **47** |
| **Total** | **49** | **88** | **91** | **228** |

**Class imbalance:** CBB (Bacterial Blight) has only 49 images in the full
pool — the minority class, addressed by inverse-frequency class weighting
during training.

**Folder naming:**
- `Cassava_Bacterial_Blight/` → class index 0
- `Cassava_Mosaic_Disease/` → class index 1
- `Healthy/` → class index 2

Indices are assigned alphabetically by folder name and must stay in sync with
`labels.json`.

### 8.2 Raw Source Data (`cassava leaf disease dataset/`)

The original, unsplit image collection (228 JPEGs in 3 folders by disease
name). This is fully represented by the `dataset/` split and is git-ignored
to avoid duplication. It exists only for provenance — you can regenerate the
split from it.

### 8.3 Baseline Comparison Files

- `cassava_model_baseline.h5` — the pre-retrained model (68.09% accuracy)
- `report_baseline.txt` — its evaluation report

Kept for before/after comparison against the current model.

---

## 9. Configuration & Environment

### 9.1 `.env.example` — Environment Variable Template

Copy to `.env` and adjust. All variables are optional with safe defaults.

| Variable | Default | Purpose |
|---|---|---|
| `FLASK_DEBUG` | `False` | Enables Werkzeug debugger. **Keep OFF in production.** |
| `FLASK_HOST` | `127.0.0.1` | Bind address. Set `0.0.0.0` for phone/QR access. |
| `FLASK_PORT` | `5000` | Server port. |
| `CONFIDENCE_THRESHOLD` | `0.65` | Minimum softmax confidence for a definitive diagnosis. |

### 9.2 `requirements.txt` — Pinned Runtime Dependencies

Every dependency is pinned to an exact version (e.g., `Flask==3.1.3`) so
installs are reproducible. Includes: Flask, Werkzeug, Flask-Limiter,
python-dotenv, tensorflow, Pillow, numpy, qrcode, gunicorn.

### 9.3 `requirements-dev.txt` — Pinned Dev Dependencies

Inherits from `requirements.txt` and adds: `pytest==9.1.1`,
`scikit-learn==1.8.0`.

### 9.4 `pytest.ini` — Pytest Configuration

Registers the `slow` marker used to tag the real-model test:
```ini
[pytest]
markers =
    slow: tests that load the real TensorFlow model (slow)
```

### 9.5 `.gitignore` — Git Ignore Rules

Covers: `venv/`, `__pycache__/`, `uploads/`, `.env` (with `!.env.example`
exception), `.pytest_cache/`, coverage artifacts, `.freebuff/`, generated QR
image, `cassava leaf disease dataset/` (raw originals), IDE/OS cruft. The
trained model (`cassava_model.h5`) is *not* ignored — it's a tracked deliverable.

---

## 10. Documentation & Assets

| File | Purpose |
|---|---|
| `README.md` | Full project documentation: overview, features, setup, env vars, performance, training, security, roadmap |
| `LICENSE` | MIT License (Copyright 2026 Fusheini Abdul-Mumin) |
| `DEFENSE_OUTLINE.md` | 10-slide defense outline with speaker notes, visual suggestions, and 18 Q&A entries |
| `PROGRAM_DETAILS.md` | This file — detailed file-by-file program documentation |

**`assets/` directory** (5 PNG screenshots referenced by README):
- `home_desktop.png` — desktop upload page
- `home_mobile.png` — mobile upload page
- `result_mobile.png` — prediction result on phone
- `model_summary.png` — Keras model architecture summary
- `training_cmd.png` — training output in terminal

---

## 11. API Reference

### `GET /`

Returns the main upload page (HTML).

### `GET /health`

**Response (200):**
```json
{
    "status": "ok",
    "model_loaded": true,
    "labels_loaded": true
}
```

### `POST /predict`

**Request:** `multipart/form-data` with field `file` (image) and header
`X-Requested-With: XMLHttpRequest`.

**Response — successful diagnosis (200):**
```json
{
    "success": true,
    "prediction": "Cassava Bacterial Blight (CBB)",
    "confidence": 90.12,
    "description": "Cassava Bacterial Blight is caused by...",
    "advice": "ACTION REQUIRED: 1) Remove and burn...",
    "severity": "disease"
}
```

**Response — uncertain (200, below threshold):**
```json
{
    "success": true,
    "prediction": "Uncertain",
    "confidence": 58.34,
    "description": "The AI model could not confidently identify...",
    "advice": "Please take another clear photo... or consult...",
    "severity": "uncertain"
}
```

**Response — error (400/403/413/500/503):**
```json
{
    "success": false,
    "error": "Human-readable error message"
}
```

---

## 12. Model Architecture

**Base:** MobileNetV2 (ImageNet-pretrained), all layers frozen.

**Head:**
```
MobileNetV2 output (batch, 7, 7, 1280)
  → GlobalAveragePooling2D         → (batch, 1280)
  → Dense(128, ReLU)               → (batch, 128)
  → Dropout(0.5)                   → (batch, 128)
  → Dense(3, Softmax)              → (batch, 3)
```

**Total parameters:** 2,422,341 (mostly frozen in the MobileNetV2 base).

**Training setup:**
- Optimizer: Adam (lr = 0.0001)
- Loss: categorical crossentropy
- Augmentation: rotation, shift, shear, zoom, horizontal flip
- Callbacks: EarlyStopping (patience 5), ModelCheckpoint (best val_accuracy)
- Class weights: `{CBB: 1.55, CMD: 0.86, Healthy: 0.84}`
- Seed: 42 (random, numpy, tensorflow)

**Why MobileNetV2:** lightweight (2.4M params vs. 20M+ for larger models),
pretrained ImageNet features transfer well with tiny datasets, runs on a
modest CPU-only machine. A larger model would overfit at 181 training images.
