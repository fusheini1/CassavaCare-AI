"""
app.py - Flask Application for Cassava Disease Diagnosis
==========================================================
Main entry point for the web application. Handles image uploads,
runs inference using the trained MobileNetV2 model, and returns
diagnosis results with management advice.

Usage:
    Development:  python app.py
    Production:   gunicorn -w 4 -b 0.0.0.0:5000 app:app  (Linux/macOS)

Security Features:
    - Filename sanitization to prevent directory traversal attacks
    - File extension validation (only jpg, jpeg, png allowed)
    - File size limit (5MB maximum)
    - Uploaded images are deleted immediately after processing
    - Error handling for corrupt/invalid images
    - CSRF mitigation: /predict requires the X-Requested-With custom header
"""

import os
import json
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from utils import preprocess_image, validate_image


# Load environment variables from a .env file if one exists (see .env.example).
# This must run before the Flask app is created so that any config read at
# import time picks up the values. Never commit your .env file.
load_dotenv()

# Minimum confidence (0.0-1.0) required to return a definitive diagnosis.
# Below this threshold the API returns an "uncertain" result instead of a
# potentially wrong disease diagnosis. Configurable via CONFIDENCE_THRESHOLD.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))


# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)

# Rate Limiting configuration (Memory-based for simplicity)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload size
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

# Allowed file extensions for upload validation
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# =============================================================================
# MODEL & LABELS LOADING (done once at startup to save memory)
# =============================================================================

# Path to the trained model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cassava_model.h5")

# Path to the labels mapping file
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json")


def load_model_and_labels():
    """
    Load the trained Keras model and class labels at application startup.
    This function is called once to avoid reloading on every request,
    which would be very slow and memory-intensive.

    Returns:
        tuple: (model, labels_dict) or (None, None) if files are missing
    """
    # Load the trained model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[INFO] Loading model from: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            print("[INFO] Model loaded successfully!")
        except Exception as e:
            # A corrupt/truncated model file must not crash startup; the app
            # will serve a clear 503 from /predict instead.
            print(f"[WARNING] Failed to load model from {MODEL_PATH}: {e}")
            print("[WARNING] The model file may be corrupt. Re-train it by running: python model_training.py")
            model = None
    else:
        print(f"[WARNING] Model file not found at: {MODEL_PATH}")
        print("[WARNING] Please train the model first by running: python model_training.py")
        model = None

    # Load the class labels and advice
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
            # Remove comment keys (keys starting with '_')
            labels = {k: v for k, v in labels.items() if not k.startswith("_")}
            print(f"[INFO] Loaded {len(labels)} class labels from: {LABELS_PATH}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARNING] Failed to load labels file {LABELS_PATH}: {e}")
            labels = None
    else:
        print(f"[WARNING] Labels file not found at: {LABELS_PATH}")
        print("[WARNING] The app will respond with errors until labels.json is restored.")
        labels = None

    return model, labels


# Load model and labels at startup (global variables for efficiency)
model, labels = load_model_and_labels()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): Name of the uploaded file.

    Returns:
        bool: True if the extension is allowed, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_upload_folder():
    """Create the upload folder if it doesn't exist."""
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """
    Lightweight health check for load balancers / uptime monitors.

    Always returns 200 while the process is up. `model_loaded` and
    `labels_loaded` are informational - the app still serves the UI and
    returns a clean 503 from /predict if the model is unavailable.
    """
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "labels_loaded": labels is not None,
    })


@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict():
    """
    Handle image upload and return disease prediction.

    Expects a POST request with a file field named 'file' and the
    X-Requested-With: XMLHttpRequest header (CSRF mitigation).

    Returns:
        JSON response with:
        - success (bool): Whether prediction was successful
        - prediction (str): Disease name
        - confidence (float): Confidence percentage (0-100)
        - description (str): Disease description
        - advice (str): Management advice
        - severity (str): "healthy", "disease", or "uncertain" (low confidence)
        - error (str): Error message if prediction failed
    """

    # --- Validation Checks ---

    # Lightweight CSRF mitigation: require a custom header that a
    # cross-origin browser request cannot set without CORS approval.
    # Same-origin requests from our frontend (script.js) always send it.
    # Upgrade to real CSRF tokens (Flask-WTF) if sessions/auth are added.
    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        return jsonify({
            "success": False,
            "error": "Missing required request header"
        }), 403  # Forbidden

    # Check if the model is loaded
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please train the model first by running: python model_training.py"
        }), 503  # Service Unavailable

    if labels is None:
        return jsonify({
            "success": False,
            "error": "Labels file not found. Please ensure labels.json exists."
        }), 503

    # Check if a file was included in the request
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded. Please select an image of a cassava leaf."
        }), 400  # Bad Request

    file = request.files["file"]

    # Check if the user actually selected a file
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose an image to upload."
        }), 400

    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Only JPG, JPEG, and PNG images are allowed."
        }), 400

    # --- File Saving & Processing ---
    filepath = None  # Track filepath for cleanup

    try:
        # Ensure upload directory exists
        ensure_upload_folder()

        # Sanitize the filename to prevent directory traversal attacks
        # Also add a UUID prefix to prevent filename collisions
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        # Save the uploaded file temporarily
        file.save(filepath)

        # Validate the image is not corrupt
        if not validate_image(filepath):
            return jsonify({
                "success": False,
                "error": "The uploaded file appears to be corrupt or is not a valid image. Please try another image."
            }), 400

        # --- Run Inference ---

        # Preprocess the image to match MobileNetV2 input requirements
        processed_image = preprocess_image(filepath)

        # Run prediction through the model
        predictions = model.predict(processed_image, verbose=0)

        # Get the predicted class index and confidence score
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence_fraction = float(np.max(predictions[0]))
        confidence = confidence_fraction * 100  # Convert to percentage

        # Low-confidence guard: never present a definitive diagnosis below the
        # confidence threshold. Confident wrong advice to a farmer is worse than
        # admitting uncertainty.
        if confidence_fraction < CONFIDENCE_THRESHOLD:
            return jsonify({
                "success": True,
                "prediction": "Uncertain",
                "confidence": round(confidence, 2),
                "description": (
                    "The AI model could not confidently identify the condition of "
                    "this leaf (confidence " + str(round(confidence, 1)) + "% is below the "
                    "required threshold)."
                ),
                "advice": (
                    "Please take another clear photo in natural daylight and try again, "
                    "or consult your local agricultural extension officer for an official "
                    "diagnosis."
                ),
                "severity": "uncertain",
            })

        # Look up the class label and advice
        class_key = str(predicted_class_index)
        if class_key in labels:
            class_info = labels[class_key]
            prediction_name = class_info["name"]
            description = class_info["description"]
            advice = class_info["advice"]
        else:
            prediction_name = f"Unknown Class ({predicted_class_index})"
            description = "No description available."
            advice = "Please consult your local agricultural extension officer."

        # Determine severity level for frontend styling
        # Check by label name for robustness across different class counts
        if prediction_name.lower().startswith("healthy"):
            severity = "healthy"
        else:
            severity = "disease"

        # Return the prediction results
        return jsonify({
            "success": True,
            "prediction": prediction_name,
            "confidence": round(confidence, 2),
            "description": description,
            "advice": advice,
            "severity": severity,
        })

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An error occurred while processing the image. Please try again with a different image."
        }), 500  # Internal Server Error

    finally:
        # SECURITY: Always delete the uploaded image after processing
        # This prevents accumulation of user images on the server
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"[WARNING] Could not delete temporary file: {filepath} - {e}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeding the 5MB limit."""
    return jsonify({
        "success": False,
        "error": "File is too large. Maximum allowed size is 5MB. Please compress or resize your image."
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "The requested page was not found."
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle unexpected internal errors with JSON (never HTML stack traces)."""
    return jsonify({
        "success": False,
        "error": "An unexpected internal server error occurred. Please try again."
    }), 500


# =============================================================================
# SECURITY HEADERS
# =============================================================================

@app.after_request
def set_security_headers(response):
    """
    Apply standard security headers to every response.

    Note: Content-Security-Policy is intentionally NOT set here - the
    frontend loads Google Fonts from external domains, and a restrictive
    CSP would break it. Revisit if the app is ever fully self-hosted.
    """
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configuration from environment variables (with safe defaults).
    #
    # SECURITY: debug mode is OFF by default. Enabling it turns on the
    # Werkzeug debugger, which allows arbitrary remote code execution if
    # the app is reachable from the network. Only set FLASK_DEBUG=1 in
    # local development, and keep FLASK_HOST=127.0.0.1 while you do.
    #
    # For production, use: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    # ------------------------------------------------------------------
    debug = os.getenv("FLASK_DEBUG", "False").strip().lower() in ("1", "true", "yes", "on")
    host = os.getenv("FLASK_HOST", "127.0.0.1").strip()
    try:
        port = int(os.getenv("FLASK_PORT", "5000"))
    except ValueError:
        port = 5000

    print("\n" + "=" * 60)
    print("  Cassava Disease Diagnosis - Web Application")
    print("=" * 60)
    print(f"  Model Path:  {MODEL_PATH}")
    print(f"  Labels Path: {LABELS_PATH}")
    print(f"  Upload Dir:  {app.config['UPLOAD_FOLDER']}")
    print(f"  Max Upload:  5 MB")
    print(f"  Debug Mode:  {debug}")
    print("=" * 60)
    print(f"\n  Starting Flask server on {host}:{port}...")
    if debug:
        print("  [SECURITY WARNING] Debug mode is ON - remote code execution risk.")
        print("  Do NOT expose this server to a network. For production, set FLASK_DEBUG=False.")
    print(f"  Open your browser to: http://{host}:{port}\n")

    app.run(debug=debug, host=host, port=port)
