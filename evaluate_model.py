"""
evaluate_model.py - Model Evaluation Report
===========================================
Loads the trained cassava model and evaluates it on the held-out
validation dataset, producing genuine evaluation artifacts:

  - overall validation accuracy
  - per-class precision / recall / F1 (classification report)
  - a formatted text confusion matrix (no matplotlib needed)

Everything is written to report.txt and summarized on stdout.

The validation preprocessing EXACTLY matches training (MobileNetV2
preprocess_input, 224x224, no augmentation, shuffle=False).

Usage:
    python evaluate_model.py
"""

import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "cassava_model.h5"
LABELS_PATH = "labels.json"
VALIDATION_DIR = os.path.join("dataset", "validation")
REPORT_PATH = "report.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# =============================================================================
# HELPERS
# =============================================================================

def load_labels(path):
    """Load labels.json and strip comment keys (keys starting with '_')."""
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return {k: v for k, v in labels.items() if not k.startswith("_")}


def format_confusion_matrix(cm, class_names):
    """Render a confusion matrix as a readable text table.

    Rows = actual class, columns = predicted class. Both axes carry a
    Total column/row so row/column sums are easy to verify.
    """
    labels = [f"[{i}] {name}" for i, name in enumerate(class_names)]
    col_w = max(max(len(label) for label in labels), len("Total")) + 1
    row_total = cm.sum(axis=1)
    col_total = cm.sum(axis=0)

    def row(cells):
        return "".join(f"{c:>{col_w}}" for c in cells)

    lines = []
    lines.append("Confusion Matrix (rows = ACTUAL, columns = PREDICTED)")
    lines.append("")
    lines.append(" " * (col_w + 1) + row(labels + ["Total"]))
    for i, label in enumerate(labels):
        lines.append(f"{label:<{col_w}} " + row(list(cm[i]) + [int(row_total[i])]))
    lines.append(f"{'Total':<{col_w}} " + row(list(col_total) + [int(cm.sum())]))
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  CassavaCare AI - Model Evaluation")
    print("=" * 60)

    # --- Pre-flight checks ---
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model file not found: {MODEL_PATH}")
        print("Please train the model first by running: python model_training.py")
        return 1

    if not os.path.isdir(VALIDATION_DIR):
        print(f"\n[ERROR] Validation directory not found: {VALIDATION_DIR}")
        return 1

    labels = load_labels(LABELS_PATH)
    n_classes = len(labels)

    # --- Load model ---
    print(f"\n[INFO] Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    if model.output_shape[-1] != n_classes:
        print(
            f"\n[WARNING] Model outputs {model.output_shape[-1]} classes but "
            f"labels.json defines {n_classes}. Metrics may be misaligned."
        )

    # --- Load validation set (same preprocessing as training, no augmentation) ---
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    val_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    # Class names in index order (folder order is alphabetical, matching
    # labels.json); fall back to folder names if a label is missing.
    class_names = []
    for folder, idx in sorted(val_generator.class_indices.items(), key=lambda kv: kv[1]):
        label = labels.get(str(idx))
        class_names.append(label.get("name", folder) if label else folder)

    # --- Predict ---
    print("\n[INFO] Running predictions on the validation set...")
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    # --- Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    correct = int(np.sum(y_true == y_pred))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # --- Build the report text ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_lines = []
    report_lines.append("=" * 76)
    report_lines.append("  CassavaCare AI - Model Evaluation Report")
    report_lines.append(f"  Generated: {now} by evaluate_model.py")
    report_lines.append("=" * 76)
    report_lines.append("")
    report_lines.append(f"Validation samples: {len(y_true)}")
    report_lines.append(f"Class distribution: {dict(zip(class_names, np.bincount(y_true).tolist()))}")
    report_lines.append("")
    report_lines.append(f"Overall accuracy: {accuracy:.4f} ({correct}/{len(y_true)})")
    report_lines.append("")
    report_lines.append("Classification report (per class):")
    report_lines.append(report)
    report_lines.append("")
    report_lines.append(format_confusion_matrix(cm, class_names))
    report_lines.append("")

    report_text = "\n".join(report_lines)

    # --- Write report.txt ---
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[INFO] Report written to: {REPORT_PATH}")

    # --- Summary to stdout ---
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Validation samples: {len(y_true)}")
    print(f"  Overall accuracy:   {accuracy:.4f} ({correct}/{len(y_true)})")
    print("\n" + format_confusion_matrix(cm, class_names))
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
