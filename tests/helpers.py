# CassavaCare AI
# Author: Fusheini Abdul-Mumin <abdulmuminfusheini@gmail.com>
"""
helpers.py - Shared test utilities
==================================
- FakeModel: a deterministic stand-in for the trained Keras model so tests
  run fast and don't require the 11 MB model file.
- install_fake_tensorflow: fakes the tensorflow package tree in sys.modules
  BEFORE app.py is imported, so the app imports and runs without loading
  TensorFlow at all in the unit tests.
- make_image_bytes: generates a valid in-memory image so tests are
  self-contained (no dependency on the dataset folder).
"""

import io
import os
import sys
import types

import numpy as np
from PIL import Image

# Make the project root importable regardless of how pytest is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class FakeModel:
    """Minimal stand-in for the trained Keras model.

    ``probs`` is a class attribute; individual tests override it on the
    instance (``app.model.probs = [...]``) to simulate confidence levels.
    """

    probs = [0.9, 0.05, 0.05]

    def predict(self, x, verbose=0):
        return np.array([self.probs], dtype=float)


FAKE_TF_MODULES = {}


def install_fake_tensorflow():
    """Replace the tensorflow package tree in sys.modules with fakes.

    app.py / utils.py import `tensorflow.keras.models.load_model` and
    `tensorflow.keras.applications.mobilenet_v2.preprocess_input`. We fake
    these at the sys.modules level so the app imports and runs without
    loading TensorFlow (~10s) in the unit tests. The optional real-model
    test swaps these fakes out for the real package temporarily.
    """
    global FAKE_TF_MODULES

    tf_mod = types.ModuleType("tensorflow")
    keras_mod = types.ModuleType("tensorflow.keras")
    models_mod = types.ModuleType("tensorflow.keras.models")
    apps_mod = types.ModuleType("tensorflow.keras.applications")
    mobilenet_mod = types.ModuleType("tensorflow.keras.applications.mobilenet_v2")

    models_mod.load_model = lambda *args, **kwargs: FakeModel()
    mobilenet_mod.preprocess_input = lambda x: x  # identity is fine for tests

    keras_mod.models = models_mod
    keras_mod.applications = apps_mod
    apps_mod.mobilenet_v2 = mobilenet_mod
    tf_mod.keras = keras_mod

    FAKE_TF_MODULES = {
        "tensorflow": tf_mod,
        "tensorflow.keras": keras_mod,
        "tensorflow.keras.models": models_mod,
        "tensorflow.keras.applications": apps_mod,
        "tensorflow.keras.applications.mobilenet_v2": mobilenet_mod,
    }
    sys.modules.update(FAKE_TF_MODULES)


def make_image_bytes(fmt="JPEG", size=(96, 96), color=(20, 180, 20)):
    """Return the bytes of a small valid image (JPEG by default)."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()
