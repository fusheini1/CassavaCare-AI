# CassavaCare AI
# Author: Fusheini Abdul-Mumin <abdulmuminfusheini@gmail.com>
"""
conftest.py - pytest fixtures for the CassavaCare AI test suite
================================================================
Loads the Flask app with a FAKE model (load_model is patched before
app.py is imported) so the suite runs fast and does not require the
trained model file. Tests that need the real model load it explicitly.
"""

import pytest

from helpers import FakeModel, install_fake_tensorflow

# Fake the tensorflow package BEFORE app.py is imported, so the suite runs
# without loading the real (slow) TensorFlow or needing the model file.
install_fake_tensorflow()

import app as app_module  # noqa: E402


@pytest.fixture
def client():
    """Flask test client with testing mode on and the rate limiter disabled
    (the suite makes far more than the 10-per-minute /predict limit).

    Note: RATELIMIT_ENABLED is only read when the Limiter is constructed, so
    we disable enforcement directly on the limiter instance."""
    app_module.app.config["TESTING"] = True
    app_module.limiter.enabled = False
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def restore_app_globals():
    """Restore module-level model/labels after each test (503-path tests set
    them to None) and reset the fake model's probabilities."""
    original_model = app_module.model
    original_labels = app_module.labels
    yield
    app_module.model = original_model
    app_module.labels = original_labels
    if isinstance(app_module.model, FakeModel):
        app_module.model.probs = list(FakeModel.probs)
