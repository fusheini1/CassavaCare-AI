# CassavaCare AI
# Author: Fusheini Abdul-Mumin <abdulmuminfusheini@gmail.com>
"""
model_training.py - Cassava Disease Classification Model Training Script
==========================================================================
This script trains a MobileNetV2-based CNN for cassava leaf disease classification
using transfer learning. MobileNetV2 is chosen for its lightweight architecture,
making it ideal for deployment on resource-constrained devices.

DATASET SETUP:
    Organize your dataset in the following directory structure:

    dataset/
    ├── train/
    │   ├── Healthy/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   ├── Cassava_Mosaic_Disease/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   └── Cassava_Bacterial_Blight/
    │       ├── img001.jpg
    │       └── ...
    └── validation/
        ├── Healthy/
        ├── Cassava_Mosaic_Disease/
        └── Cassava_Bacterial_Blight/

    This project ships a 3-class model (Bacterial Blight, Mosaic Disease,
    Healthy). Class indices are assigned ALPHABETICALLY by folder name:
    Cassava_Bacterial_Blight=0, Cassava_Mosaic_Disease=1, Healthy=2.
    Keep NUM_CLASSES below and labels.json in sync with the folder layout.

Usage:
    python model_training.py
"""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Fix every random seed so training runs are reproducible across machines.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# CONFIGURATION - Modify these settings as needed
# =============================================================================

# Number of disease classes
# Set to 3 to match the current dataset (Blight, Mosaic, Healthy)
NUM_CLASSES = 3

# Image dimensions expected by MobileNetV2
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Training hyperparameters
BATCH_SIZE = 32        # Reduce to 16 if running out of GPU memory
EPOCHS = 50            # EarlyStopping will prevent unnecessary training
LEARNING_RATE = 0.0001 # Low learning rate for fine-tuning pre-trained weights

# Dataset paths - UPDATE THESE to point to your actual dataset location
TRAIN_DIR = os.path.join("dataset", "train")
VALIDATION_DIR = os.path.join("dataset", "validation")

# Output model filename
MODEL_SAVE_PATH = "cassava_model.h5"


# =============================================================================
# DATA AUGMENTATION & GENERATORS
# =============================================================================

def create_data_generators():
    """
    Create training and validation data generators with augmentation.
    
    Data augmentation helps prevent overfitting by artificially expanding
    the training dataset through random transformations. This is especially
    important when working with limited agricultural image datasets.
    
    Returns:
        tuple: (train_generator, validation_generator)
    """

    # Training data generator WITH augmentation
    # These augmentations simulate real-world variations in how farmers
    # might photograph cassava leaves (different angles, zoom, orientations)
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=30,          # Rotate up to 30 degrees
        width_shift_range=0.2,      # Shift horizontally by up to 20%
        height_shift_range=0.2,     # Shift vertically by up to 20%
        shear_range=0.2,            # Apply shear transformation
        zoom_range=0.2,             # Zoom in/out by up to 20%
        horizontal_flip=True,       # Flip left-right (leaf orientation varies)
        vertical_flip=False,        # Don't flip vertically (unnatural for leaves)
        fill_mode="nearest",        # Fill empty pixels after transformation
    )

    # Validation data generator WITHOUT augmentation
    # Validation images should represent real conditions without modification
    validation_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )

    # Load training images from directory structure
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",   # One-hot encoded labels for multi-class
        shuffle=True,               # Shuffle training data each epoch
    )

    # Load validation images from directory structure
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,              # Don't shuffle validation data
    )

    # Print class mapping for verification
    print("\n[INFO] Class indices mapping:")
    print(train_generator.class_indices)
    print(f"[INFO] Training samples: {train_generator.samples}")
    print(f"[INFO] Validation samples: {validation_generator.samples}")
    print(f"[INFO] Number of classes: {train_generator.num_classes}\n")

    return train_generator, validation_generator


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_model():
    """
    Build a MobileNetV2-based model with custom classification head.
    
    Architecture:
        1. MobileNetV2 base (pre-trained on ImageNet) - FROZEN layers
           Acts as a powerful feature extractor for leaf visual patterns.
        2. GlobalAveragePooling2D - Reduces spatial dimensions
        3. Dense (128 units, ReLU) - Learns disease-specific features
        4. Dropout (0.5) - Prevents overfitting
        5. Dense (NUM_CLASSES, Softmax) - Output probabilities for each class
    
    Returns:
        tensorflow.keras.Model: Compiled model ready for training
    """

    # Load MobileNetV2 pre-trained on ImageNet, excluding the top classification layer
    # include_top=False removes the final Dense layer (1000 ImageNet classes)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )

    # Freeze all base model layers to preserve learned features
    # This prevents the pre-trained weights from being modified during training
    # Only our custom top layers will be trained
    base_model.trainable = False

    print(f"[INFO] MobileNetV2 base model loaded with {len(base_model.layers)} layers (all frozen)")

    # Build custom classification head on top of the base model
    x = base_model.output

    # GlobalAveragePooling reduces each feature map to a single value
    # Converts shape from (batch, 7, 7, 1280) to (batch, 1280)
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense layer to learn disease-specific feature combinations
    x = Dense(128, activation="relu", name="dense_features")(x)

    # Dropout for regularization - randomly zeros 50% of values during training
    # This forces the network to learn redundant representations, reducing overfitting
    x = Dropout(0.5, name="dropout_regularization")(x)

    # Output layer: one neuron per class with softmax activation
    # Softmax ensures outputs sum to 1.0 (probability distribution)
    predictions = Dense(NUM_CLASSES, activation="softmax", name="output_predictions")(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile with categorical crossentropy for multi-class classification
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print model summary showing trainable vs non-trainable parameters
    model.summary()

    return model


# =============================================================================
# TRAINING CALLBACKS
# =============================================================================

def get_callbacks():
    """
    Create training callbacks for early stopping and model checkpointing.
    
    Returns:
        list: List of Keras callbacks
    """

    # Early Stopping: Stop training if validation loss doesn't improve
    # patience=5 means wait 5 epochs without improvement before stopping
    # restore_best_weights=True ensures the best model is kept, not the last one
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=1,
        restore_best_weights=True,
    )

    # Model Checkpoint: Save the best model during training
    # Only saves when validation accuracy improves (save_best_only=True)
    model_checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    return [early_stopping, model_checkpoint]


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train():
    """
    Execute the complete training pipeline:
    1. Create data generators with augmentation
    2. Build the MobileNetV2-based model
    3. Train with callbacks
    4. Save the final model
    """

    print("=" * 60)
    print("  Cassava Disease Classification - Model Training")
    print("=" * 60)

    # Verify dataset directories exist
    if not os.path.isdir(TRAIN_DIR):
        print(f"\n[ERROR] Training directory not found: {TRAIN_DIR}")
        print("Please create the dataset directory structure as described in the docstring.")
        print("See the top of this file for the required folder layout.")
        return

    if not os.path.isdir(VALIDATION_DIR):
        print(f"\n[ERROR] Validation directory not found: {VALIDATION_DIR}")
        print("Please create the dataset directory structure as described in the docstring.")
        return

    # Step 1: Create data generators
    print("\n[Step 1/3] Creating data generators with augmentation...")
    train_generator, validation_generator = create_data_generators()

    # Compute per-class weights to compensate for class imbalance: the
    # minority class (Cassava Bacterial Blight) is weighted higher so each
    # of its samples contributes proportionally more to the loss.
    counts = np.bincount(train_generator.classes)
    total = len(train_generator.classes)
    n_classes = train_generator.num_classes
    class_weight = {i: total / (n_classes * c) for i, c in enumerate(counts)}
    print(f"[INFO] Training class distribution: {dict(zip(range(n_classes), counts))}")
    print(f"[INFO] Per-class weights: {class_weight}")

    # Step 2: Build the model
    print("\n[Step 2/3] Building MobileNetV2 model...")
    model = build_model()

    # Step 3: Train the model
    print("\n[Step 3/3] Training the model...")
    print(f"  - Epochs: {EPOCHS} (with EarlyStopping)")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Output: {MODEL_SAVE_PATH}\n")

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=get_callbacks(),
        class_weight=class_weight,
        verbose=1,
    )

    # Print final results
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
