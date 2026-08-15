"""
cross_validate.py - 5-Fold Stratified Cross-Validation + Threshold Calibration
==============================================================================
Two jobs:

1. GENERALIZATION ESTIMATE
   The single train/validation split jumped from 0.6809 (baseline) to 0.9574
   after retraining with seeds + class weighting on a TINY dataset (181 train
   / 47 validation). That could be real, or an optimistic split. K-fold CV
   evaluates every sample exactly once, giving a more honest estimate of
   out-of-sample accuracy.
     - Pool ALL images (train + validation) into one dataset.
     - StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
     - Per fold: train a FRESH MobileNetV2 (identical architecture and
       preprocessing to model_training.py, incl. augmentation and per-class
       weights) for a REDUCED schedule of 5 epochs.
     - Evaluate on the held-out fold; report accuracy and per-class recall.
     - Report mean +/- std of accuracy across folds vs. the single split.

2. CONFIDENCE-THRESHOLD CALIBRATION
   The app answers "uncertain - consult an extension officer" when the
   predicted class's softmax confidence is below CONFIDENCE_THRESHOLD (0.65
   by default). Is 0.65 the right number? Using the pooled held-out
   predictions (every image evaluated exactly once), we compute for a sweep
   of thresholds: rejection rate, accuracy on confident samples only,
   per-class recall on confident samples, and how often the rejected
   ("uncertain") samples would have been misclassified. A reliability table
   (mean predicted confidence vs. observed accuracy per bin) checks whether
   the model's confidence is calibrated at all.

Results are appended to report.txt and printed to stdout.

Usage:
    python cross_validate.py
"""

import contextlib
import io
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Reuse the EXACT model architecture and hyperparameters from the training
# script so the cross-validation reflects the real pipeline.
from model_training import BATCH_SIZE, IMG_SIZE, build_model

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
N_SPLITS = 5
FOLD_EPOCHS = 5  # reduced schedule to keep runtime reasonable
NUM_CLASSES = 3

TRAIN_DIR = os.path.join("dataset", "train")
VALIDATION_DIR = os.path.join("dataset", "validation")
LABELS_PATH = "labels.json"
REPORT_PATH = "report.txt"

# The single-split figure from evaluate_model.py (current report.txt) that we
# compare the CV estimate against.
SINGLE_SPLIT_ACC = 0.9574

# Threshold sweep for the calibration analysis.
THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
CURRENT_THRESHOLD = 0.65  # matches the app's default CONFIDENCE_THRESHOLD

# Augmentation settings MUST match model_training.py's create_data_generators().
AUG_KWARGS = dict(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest",
)

SEP = "=" * 76


# =============================================================================
# DATA POOLING
# =============================================================================

def collect_pool():
    """Combine train + validation into one list of (path, class_index).

    Class indices are assigned alphabetically by folder name, exactly as
    flow_from_directory does in model_training.py (CBB=0, CMD=1, Healthy=2).
    """
    class_names = sorted(
        set(os.listdir(TRAIN_DIR)) | set(os.listdir(VALIDATION_DIR))
    )
    paths, labels = [], []
    for root in (TRAIN_DIR, VALIDATION_DIR):
        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path) or folder not in class_names:
                continue
            idx = class_names.index(folder)
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(folder_path, fname))
                    labels.append(idx)
    return paths, np.asarray(labels), class_names


def load_image(path):
    """Load an image as a float32 0-255 RGB array resized to IMG_SIZE.

    Matches what flow_from_directory feeds to the preprocessing function in
    model_training.py (PIL decode -> 224x224 -> preprocess_input).
    """
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32)


# =============================================================================
# THRESHOLD / CALIBRATION ANALYSIS
# =============================================================================

def analyze_thresholds(records):
    """Compute the sweep table + current-threshold breakdown.

    records: list of (confidence, predicted_idx, true_idx) - one per
    held-out image, pooled across all folds (every image exactly once).
    """
    confs = np.array([r[0] for r in records])
    preds = np.array([r[1] for r in records])
    trues = np.array([r[2] for r in records])
    total = len(records)

    rows = []
    for t in THRESHOLDS:
        conf_mask = confs >= t
        rej_mask = ~conf_mask
        n_conf = int(conf_mask.sum())
        n_rej = int(rej_mask.sum())
        rej_rate = rej_mask.mean()
        if n_conf > 0:
            conf_acc = accuracy_score(trues[conf_mask], preds[conf_mask])
        else:
            conf_acc = float("nan")
        # Of the rejected ("uncertain") samples, how many would have been
        # misclassified if the model had answered? High = threshold is
        # catching the hard cases.
        if n_rej > 0:
            rej_wrong = (trues[rej_mask] != preds[rej_mask]).mean()
        else:
            rej_wrong = float("nan")
        rows.append((t, rej_rate, conf_acc, n_conf, rej_wrong))

    # Full breakdown at the app's current threshold.
    conf_mask = confs >= CURRENT_THRESHOLD
    rej_mask = ~conf_mask
    n_conf = int(conf_mask.sum())
    n_rej = int(rej_mask.sum())
    rej_wrong = (trues[rej_mask] != preds[rej_mask]).mean() if n_rej else float("nan")
    rej_right = (trues[rej_mask] == preds[rej_mask]).sum() if n_rej else 0
    conf_acc = accuracy_score(trues[conf_mask], preds[conf_mask]) if n_conf else float("nan")
    conf_recalls = (
        recall_score(
            trues[conf_mask], preds[conf_mask],
            labels=list(range(NUM_CLASSES)), average=None, zero_division=0,
        ).tolist()
        if n_conf else [float("nan")] * NUM_CLASSES
    )
    breakdown = dict(
        n_total=total, n_conf=n_conf, n_rej=n_rej,
        rej_rate=n_rej / total,
        conf_acc=conf_acc, conf_recalls=conf_recalls,
        rej_wrong=rej_wrong, rej_right=rej_right,
    )
    return rows, breakdown, confs, preds, trues


def reliability_bins(confs, preds, trues):
    """Mean predicted confidence vs. observed accuracy per confidence bin."""
    bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    rows = []
    for lo, hi in bins:
        m = (confs >= lo) & (confs < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append((f"{lo:.2f}-{min(hi, 1.0):.2f}", n,
                     confs[m].mean(), accuracy_score(trues[m], preds[m])))
    return rows


def fmt_table(header, rows, num_cols=None):
    """Render a fixed-width text table. num_cols = columns to right-align/format."""
    widths = [len(str(h)) for h in header]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    lines = []
    lines.append("  " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)))
    lines.append("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        cells = []
        for i, cell in enumerate(r):
            s = str(cell)
            cells.append(s.rjust(widths[i]) if num_cols and i >= num_cols else s.ljust(widths[i]))
        lines.append("  " + " | ".join(cells))
    return "\n".join(lines)


# =============================================================================
# REPORT HELPERS
# =============================================================================

def strip_section(content, marker):
    """Remove a section (and its '====' header) from report.txt, keeping others."""
    idx = content.find(marker)
    if idx == -1:
        return content
    start = content.rfind(SEP, 0, idx)
    if start == -1:
        start = idx
    end = content.find(SEP, idx)
    if end == -1:
        return content[:start].rstrip() + "\n\n"
    return content[:start] + content[end:]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  CassavaCare AI - 5-Fold Cross-Validation + Threshold Calibration")
    print("=" * 60)

    # --- Reproducibility ---
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # --- Pool the data ---
    paths, labels, class_names = collect_pool()
    if len(paths) == 0:
        print("\n[ERROR] No images found under dataset/train or dataset/validation.")
        return 1
    print(f"\n[INFO] Pooled dataset: {len(paths)} images")
    print(f"[INFO] Class names (alphabetical): {class_names}")
    print(f"[INFO] Class distribution: {dict(zip(class_names, np.bincount(labels).tolist()))}")

    # Label display names from labels.json (consistent with evaluate_model.py)
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels_json = json.load(f)
        label_names = []
        for i in range(NUM_CLASSES):
            entry = labels_json.get(str(i))
            label_names.append(entry.get("name", class_names[i]) if entry else class_names[i])
    except Exception:
        label_names = class_names
        print(f"\n[WARNING] Could not read {LABELS_PATH}; using folder names in report.")

    # --- Stratified K-Fold ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_accs, fold_recalls, per_fold_lines = [], [], []
    records = []  # pooled per-sample (confidence, predicted, true) across folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), start=1):
        # Reset seeds so every fold's training is deterministic and comparable.
        random.seed(SEED + fold)
        np.random.seed(SEED + fold)
        tf.random.set_seed(SEED + fold)

        print(f"\n--- Fold {fold}/{N_SPLITS} "
              f"(train: {len(train_idx)}, held-out: {len(val_idx)}) ---", flush=True)

        # Load fold arrays (float32 0-255, resized). Held-out set is small, so
        # keeping all images in memory (~35 MB total) is fine.
        X_train = np.stack([load_image(paths[i]) for i in train_idx])
        y_train = to_categorical(labels[train_idx], num_classes=NUM_CLASSES)
        X_val = np.stack([load_image(paths[i]) for i in val_idx])
        y_val = to_categorical(labels[val_idx], num_classes=NUM_CLASSES)

        # Same preprocessing as model_training.py: augmented train, plain val.
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input, **AUG_KWARGS
        )
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_flow = train_datagen.flow(
            X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED + fold
        )
        val_flow = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

        # Per-fold class weights, same formula as model_training.py.
        counts = np.bincount(np.argmax(y_train, axis=1))
        class_weight = {
            i: len(y_train) / (NUM_CLASSES * c) for i, c in enumerate(counts)
        }
        print(f"[INFO] Fold class distribution: {dict(zip(range(NUM_CLASSES), counts.tolist()))}")
        print(f"[INFO] Fold class weights: {class_weight}")

        # Fresh model per fold, same architecture as model_training.build_model().
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = build_model()
        print(f"[INFO] Model built ({len(buf.getvalue().splitlines())} summary lines suppressed)")

        # Reduced training schedule (5 epochs, no early stopping).
        model.fit(
            train_flow,
            epochs=FOLD_EPOCHS,
            validation_data=val_flow,
            class_weight=class_weight,
            verbose=1,
        )

        # Evaluate on the held-out fold; record per-sample confidence too.
        preds = model.predict(val_flow, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y_val, axis=1)
        confs = preds[np.arange(len(preds)), y_pred]
        for c, p, t in zip(confs.tolist(), y_pred.tolist(), y_true.tolist()):
            records.append((c, p, t))

        acc = accuracy_score(y_true, y_pred)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

        fold_accs.append(acc)
        fold_recalls.append(recalls)
        per_fold_lines.append(
            f"  Fold {fold}: accuracy {acc:.4f} ({int(np.sum(y_true == y_pred))}/"
            f"{len(y_true)}), recall "
            + " / ".join(f"{name} {r:.4f}" for name, r in zip(label_names, recalls))
        )
        print(per_fold_lines[-1], flush=True)

    # --- Aggregate CV ---
    cv_mean = float(np.mean(fold_accs))
    cv_std = float(np.std(fold_accs, ddof=1))
    mean_recalls = np.mean(fold_recalls, axis=0)

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  5-fold CV accuracy: {cv_mean:.4f} +/- {cv_std:.4f} (mean +/- std, ddof=1)")
    for name, r in zip(label_names, mean_recalls):
        print(f"  Mean recall {name}: {r:.4f}")
    print(f"\n  Single split (evaluate_model.py): {SINGLE_SPLIT_ACC:.4f}")

    # --- Honest CV interpretation ---
    diff = SINGLE_SPLIT_ACC - cv_mean
    if diff >= 0.05:
        cv_note = (
            f"The single-split validation accuracy ({SINGLE_SPLIT_ACC:.4f}) is "
            f"NOTABLY higher than the CV mean ({cv_mean:.4f}, difference "
            f"{diff:+.4f}). The 0.9574 figure was optimistic: with only 47 held-out "
            f"images it did not generalize. The 5-fold CV mean of {cv_mean:.4f} +/- "
            f"{cv_std:.4f} is the more trustworthy estimate of out-of-sample accuracy."
        )
    elif diff >= 0.02:
        cv_note = (
            f"The single-split figure ({SINGLE_SPLIT_ACC:.4f}) runs somewhat above "
            f"the CV mean ({cv_mean:.4f}, difference {diff:+.4f}). The 0.9574 number "
            f"is mildly optimistic; the CV mean of {cv_mean:.4f} +/- {cv_std:.4f} is "
            f"the more conservative, more honest estimate."
        )
    else:
        cv_note = (
            f"The single-split figure ({SINGLE_SPLIT_ACC:.4f}) is within "
            f"{abs(diff):.4f} of the CV mean ({cv_mean:.4f}). The 0.9574 result "
            f"largely generalizes; both point to the same overall picture."
        )
    print(f"\n[INFO] {cv_note}")

    # --- Threshold calibration analysis ---
    rows, breakdown, confs, preds, trues = analyze_thresholds(records)
    rel_rows = reliability_bins(confs, preds, trues)
    overall_acc = accuracy_score(trues, preds)
    mean_conf = float(confs.mean())

    print("\n" + "=" * 60)
    print("  THRESHOLD CALIBRATION (pooled held-out predictions)")
    print("=" * 60)
    print(f"  Pooled samples: {len(records)} (every image evaluated exactly once)")
    print(f"  Overall accuracy: {overall_acc:.4f} | Mean confidence: {mean_conf:.4f}")
    print("\n" + fmt_table(
        ["threshold", "rejection_rate", "confident_accuracy", "confident_count",
         "uncertain_would_be_wrong"],
        [[f"{t:.2f}", f"{r:.4f}", f"{a:.4f}", str(n), f"{w:.4f}"]
         for t, r, a, n, w in rows],
        num_cols=1,
    ))
    print("\nReliability by bin (mean predicted confidence vs observed accuracy):")
    print(fmt_table(
        ["bin", "n", "mean_confidence", "observed_accuracy"],
        [[b, str(n), f"{mc:.4f}", f"{oa:.4f}"] for b, n, mc, oa in rel_rows],
        num_cols=1,
    ))

    # --- Current-threshold verdict (data-driven, honest) ---
    rej_rate = breakdown["rej_rate"]
    conf_acc = breakdown["conf_acc"]
    rej_wrong = breakdown["rej_wrong"]
    n_rej, n_conf = breakdown["n_rej"], breakdown["n_conf"]
    if np.isnan(conf_acc) or n_conf == 0:
        calib_note = (
            f"At {CURRENT_THRESHOLD:.2f} no held-out sample was confident enough to "
            f"answer - the threshold rejects everything. It is far too high."
        )
    elif conf_acc < CURRENT_THRESHOLD - 0.08:
        calib_note = (
            f"At {CURRENT_THRESHOLD:.2f}, confident predictions are right only "
            f"{conf_acc:.3f} of the time - BELOW the confidence the model claims. "
            f"The model is overconfident at this margin, so {CURRENT_THRESHOLD:.2f} "
            f"is too low a bar: a higher threshold (e.g. 0.75-0.80) is warranted."
        )
    elif conf_acc < CURRENT_THRESHOLD - 0.03:
        calib_note = (
            f"At {CURRENT_THRESHOLD:.2f}, confident predictions are right "
            f"{conf_acc:.3f} of the time - slightly below the claimed confidence. "
            f"The margin is roughly but not perfectly calibrated; keeping "
            f"{CURRENT_THRESHOLD:.2f} is defensible, raising it a touch (0.70) is "
            f"slightly safer."
        )
    elif conf_acc <= CURRENT_THRESHOLD + 0.08:
        calib_note = (
            f"At {CURRENT_THRESHOLD:.2f}, confident predictions are right "
            f"{conf_acc:.3f} of the time - matching (or exceeding) the claimed "
            f"confidence. The threshold is well-calibrated at the margin: the model "
            f"is right at least as often as its confidence suggests, so "
            f"{CURRENT_THRESHOLD:.2f} is a sound choice."
        )
    else:
        calib_note = (
            f"At {CURRENT_THRESHOLD:.2f}, confident predictions are right "
            f"{conf_acc:.3f} of the time - well ABOVE the claimed confidence. The "
            f"model is underconfident at this margin; {CURRENT_THRESHOLD:.2f} could "
            f"be lowered to answer more cases without sacrificing precision."
        )
    print(f"\n[INFO] {calib_note}")

    # --- Append CV + threshold sections to report.txt ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    cv_lines = []
    cv_lines.append(SEP)
    cv_lines.append("  Cross-Validation (5-fold, stratified, random_state=42)")
    cv_lines.append(f"  Generated: {now} by cross_validate.py")
    cv_lines.append(SEP)
    cv_lines.append("")
    cv_lines.append(f"Pooled dataset: {len(paths)} images (train {sum(1 for p in paths if TRAIN_DIR in p)} "
                    f"+ validation {sum(1 for p in paths if VALIDATION_DIR in p)})")
    cv_lines.append(f"Class distribution: {dict(zip(label_names, np.bincount(labels).tolist()))}")
    cv_lines.append(f"Schedule: {FOLD_EPOCHS} epochs per fold (reduced vs. the "
                    f"{50}-epoch full training), seeds fixed, class-weighted")
    cv_lines.append("")
    cv_lines.append("Per-fold results (accuracy on held-out fold):")
    cv_lines.extend(per_fold_lines)
    cv_lines.append("")
    cv_lines.append(f"5-fold CV accuracy: {cv_mean:.4f} +/- {cv_std:.4f} "
                    f"(mean +/- std, ddof=1)")
    cv_lines.append(f"Mean per-class recall: "
                    + ", ".join(f"{name} {r:.4f}" for name, r in zip(label_names, mean_recalls)))
    cv_lines.append("")
    cv_lines.append(f"Comparison with single train/validation split (evaluate_model.py): "
                    f"{SINGLE_SPLIT_ACC:.4f} vs. CV mean {cv_mean:.4f}")
    cv_lines.append("")
    cv_lines.append("Interpretation:")
    cv_lines.append(f"  {cv_note}")
    cv_lines.append("")
    cv_lines.append("Caveats: each fold trains for only 5 epochs (vs. up to 50 in the")
    cv_lines.append("full pipeline), so the CV figures are a lower-bound estimate; a")
    cv_lines.append("full-budget retrain per fold would likely score somewhat higher.")
    cv_lines.append("The dataset is small (228 images total), so all figures carry wide")
    cv_lines.append("confidence intervals.")
    cv_lines.append("")

    cal_lines = []
    cal_lines.append(SEP)
    cal_lines.append("  Confidence-Threshold Calibration (pooled 5-fold CV predictions)")
    cal_lines.append(f"  Generated: {now} by cross_validate.py")
    cal_lines.append(SEP)
    cal_lines.append("")
    cal_lines.append(f"Pooled held-out predictions: {len(records)} "
                     f"(every image evaluated exactly once by the 5 folds)")
    cal_lines.append(f"Overall held-out accuracy: {overall_acc:.4f} | "
                     f"Mean predicted confidence: {mean_conf:.4f}")
    cal_lines.append("")
    cal_lines.append("Threshold sweep (samples with confidence >= threshold are answered;")
    cal_lines.append("below are routed to the 'uncertain' state):")
    cal_lines.append(fmt_table(
        ["threshold", "rejection_rate", "confident_accuracy", "confident_count",
         "uncertain_would_be_wrong"],
        [[f"{t:.2f}", f"{r:.4f}", f"{a:.4f}", str(n), f"{w:.4f}"]
         for t, r, a, n, w in rows],
        num_cols=1,
    ))
    cal_lines.append("")
    cal_lines.append(f"Current threshold ({CURRENT_THRESHOLD:.2f}) full breakdown:")
    cal_lines.append(f"  Samples answered (confident): {n_conf}/{breakdown['n_total']} "
                     f"({conf_acc:.4f} accuracy)")
    cal_lines.append(f"  Samples rejected (uncertain): {n_rej}/{breakdown['n_total']} "
                     f"({rej_rate:.4f} rejection rate)")
    cal_lines.append(f"  Per-class recall on confident samples only: "
                     + ", ".join(
                         f"{name} {r:.4f}" for name, r in zip(label_names, breakdown["conf_recalls"])))
    cal_lines.append(f"  Uncertain-would-be-wrong: {rej_wrong:.4f} "
                     f"(fraction of rejected samples that were misclassified)")
    cal_lines.append(f"  Rejected-but-correct (cost of uncertainty): "
                     f"{breakdown['rej_right']}/{n_rej} "
                     f"({(1 - rej_wrong) * 100:.1f}%)")
    cal_lines.append("")
    cal_lines.append("Reliability by confidence bin (mean predicted confidence vs.")
    cal_lines.append("observed accuracy):")
    cal_lines.append(fmt_table(
        ["bin", "n", "mean_confidence", "observed_accuracy"],
        [[b, str(n), f"{mc:.4f}", f"{oa:.4f}"] for b, n, mc, oa in rel_rows],
        num_cols=1,
    ))
    cal_lines.append("")
    cal_lines.append("Interpretation:")
    cal_lines.append(f"  {calib_note}")
    cal_lines.append("  Caveat: with only 228 held-out samples, these rates carry wide")
    cal_lines.append("  confidence intervals; treat them as directional, not exact.")
    cal_lines.append("")

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    content = strip_section(content, "Cross-Validation (5-fold")
    content = strip_section(content, "Confidence-Threshold Calibration")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content + "\n".join(cv_lines) + "\n".join(cal_lines))
    print(f"\n[INFO] CV + threshold-calibration sections written to: {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
