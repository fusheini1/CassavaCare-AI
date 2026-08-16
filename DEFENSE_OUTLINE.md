# CassavaCare AI — Project Defense Outline

A slide outline for the project defense, with talking points and anticipated
professor follow-up questions.

---

## Slide 1 — Title

**CassavaCare AI: Mobile-First Cassava Leaf Disease Diagnosis**

- Name, course, date
- One-line thesis: *"A Flask + MobileNetV2 web app that classifies cassava leaf disease for smallholder farmers — and the evaluation discipline that kept its claims honest."*

## Slide 2 — The system in 30 seconds

- Phone camera → upload → MobileNetV2 → diagnosis / "uncertain, consult an extension officer"
- 3 classes (Bacterial Blight / Mosaic / Healthy), runs fully offline on the LAN
- Security-hardened: debug gated, rate-limited, CSRF header check, upload validation
- **Talking point:** designed for field conditions — no cloud, no internet

## Slide 3 — Challenge 1: Data scarcity & imbalance

- 181 training images total; Bacterial Blight = only 49 images
- First model: 68% accuracy, **Blight recall 0.40** — the disease that matters most was the one it missed
- Fix: inverse-frequency class weighting → recall 0.40 → 0.90
- **Talking point:** weighting compensates for imbalance but can't create information — data is the real ceiling

## Slide 4 — Challenge 2: The generalization trap

- Retrained model claimed **95.7% validation accuracy** — impressive and *wrong to trust*
- The 47-image split was an easy one; 5-fold CV over all 228 images → **87.3% ± 4.8%**, worst fold 80%
- **Talking point:** single splits flatter small models; cross-validation is what kept the report honest. Numbers on the slide: 95.7% vs 87.3% side by side.

## Slide 5 — Challenge 3: Confidence calibration (the counterintuitive one)

- 65% threshold rejected **58.8%** of predictions — most of which were correct
- Model is *underconfident*: claims of 65% were right ~98% of the time
- Trade-off: lower threshold ≈ 2× more answers but lower disease recall on answered samples
- **Talking point:** the threshold is a product decision, not just statistics

## Slide 6 — Challenge 4: Engineering discipline

- Debug mode was a remote-code-execution risk; no version control, no tests, unpinned deps
- Test suite surfaced real gotchas: Flask-Limiter ignores its flag after init; Keras 3 lazy imports defeat monkeypatching
- TensorFlow on Windows: slow cold imports that look like hangs, CPU-only
- **Talking point:** the "boring" work (tests, .gitignore, pins) was most of the hardening value

## Slide 7 — Results summary (before/after table)

| Metric | Baseline | Retrained | CV estimate |
|---|---|---|---|
| Accuracy | 0.6809 | 0.9574 | **0.8727 ± 0.048** |
| CBB recall | 0.40 | 0.90 | 0.66 |
| CMD recall | 0.50 | 0.94 | 0.88 |
| Healthy recall | 1.00 | 1.00 | 0.98 |

## Slide 8 — Limitations & future work

- 228 images, ~1 in 8 predictions wrong at true accuracy → uncertainty state is load-bearing
- CV used 5-epoch models → lower-bound estimate
- Future: more data (Blight first), field validation, shared rate-limit store, Docker
- **Talking point:** name the limitations *before* the professor does

## Slide 9 — The one lesson

> *"The hardest part was refusing to believe the model's confident numbers until cross-validation and calibration agreed."*

- Close with impact: a working, honest, deployable system — and the process that verified it

---

## Likely professor follow-up questions

1. **"Why is 87% still acceptable for a farm advisory tool?"**
   → It isn't, blindly — that's why the app has an "uncertain" state. The threshold catches low-confidence cases; when the app does answer, precision is ~99%. No tool should replace an extension officer; this one triages.

2. **"Did you tune the confidence threshold on the test set — is that leakage?"**
   → The sweep used the pooled CV held-out predictions (every image evaluated exactly once), so no sample influenced both the threshold choice and its own evaluation. Caveat: CV used short-trained models, so exact numbers are directional.

3. **"How do you know the class weighting isn't just masking the imbalance?"**
   → Good question. Weighting changed *training loss*, not evaluation — the CV numbers (with per-fold weights) still show CBB recall 0.66, the weakest class. It genuinely helped vs. baseline but didn't close the gap; the fix is data, not weights.

4. **"Why 5 epochs in CV instead of the full 50?"**
   → Computational budget — 5 folds × full training ≈ 90 min on CPU. I documented that CV is a lower-bound estimate. The fold-to-fold variance (0.80–0.91) is real signal regardless.

5. **"What would you do differently?"**
   → Collect more data first, even before modeling; start with version control and tests from day one; and validate the confidence calibration story on the production (50-epoch) model rather than only the CV models.

6. **"Is MobileNetV2 the right choice, or just convenient?"**
   → Right for the constraint: frozen ImageNet features work with tiny datasets, and the model is small enough to run on a modest machine. A larger/specialized model would likely overfit at 181 images.
