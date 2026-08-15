"""
utils.py - Image Preprocessing Utilities
=========================================
Helper functions for loading and preprocessing cassava leaf images
before feeding them into the MobileNetV2 model for inference.

The preprocessing pipeline must match exactly what was used during training
to ensure consistent and accurate predictions.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Target image dimensions expected by MobileNetV2
IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    """
    Load an image from disk, resize it, and preprocess it for MobileNetV2 inference.

    Steps:
        1. Open the image file using Pillow.
        2. Convert to RGB (handles grayscale or RGBA uploads).
        3. Resize to 224x224 pixels (MobileNetV2 input size).
        4. Convert to a NumPy array of float32 values.
        5. Add a batch dimension (model expects shape: [1, 224, 224, 3]).
        6. Apply MobileNetV2-specific preprocessing (scales pixels to [-1, 1]).

    Args:
        image_path (str): Absolute or relative path to the image file.

    Returns:
        np.ndarray: Preprocessed image array with shape (1, 224, 224, 3),
                     ready to be passed to model.predict().

    Raises:
        FileNotFoundError: If the image file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.
    """
    # Load the image from the file path
    img = Image.open(image_path)

    # Convert to RGB in case the image is grayscale (L), RGBA, or palette (P)
    # MobileNetV2 requires 3-channel (RGB) input
    img = img.convert("RGB")

    # Resize to the target dimensions expected by the model
    img = img.resize(IMG_SIZE)

    # Convert the PIL Image to a NumPy array (shape: 224, 224, 3, dtype: uint8)
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    # The model expects a batch of images, even if we're predicting on a single image
    img_array = np.expand_dims(img_array, axis=0)

    # Apply MobileNetV2 preprocessing: scales pixel values from [0, 255] to [-1, 1]
    # This MUST match the preprocessing used during training to get accurate results
    img_array = preprocess_input(img_array)

    return img_array


def validate_image(image_path):
    """
    Validate that a file is a readable, non-corrupt image.

    Args:
        image_path (str): Path to the image file to validate.

    Returns:
        bool: True if image is valid, False otherwise.
    """
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it's a valid image (does not load full data)
        return True
    except Exception:
        return False
