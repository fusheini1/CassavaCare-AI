/**
 * script.js - CassavaCare AI Frontend Logic
 * ============================================
 * Handles:
 * 1. Image file selection and preview (tap/click + drag & drop)
 * 2. Client-side file validation (type, size)
 * 3. Asynchronous image upload to Flask /predict endpoint
 * 4. Loading spinner display during inference
 * 5. Dynamic rendering of diagnosis results
 * 6. Error handling and user-friendly messages
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
    maxFileSize: 5 * 1024 * 1024,   // 5MB in bytes
    allowedTypes: ["image/jpeg", "image/jpg", "image/png"],
    allowedExtensions: [".jpg", ".jpeg", ".png"],
    predictEndpoint: "/predict",
};


// =============================================================================
// DOM ELEMENTS
// =============================================================================

/** @type {HTMLFormElement} */
const uploadForm = document.getElementById("upload-form");

/** @type {HTMLInputElement} */
const fileInput = document.getElementById("file-input");

/** @type {HTMLDivElement} */
const dropZone = document.getElementById("drop-zone");

/** @type {HTMLDivElement} */
const dropZoneContent = document.getElementById("drop-zone-content");

/** @type {HTMLImageElement} */
const imagePreview = document.getElementById("image-preview");

/** @type {HTMLDivElement} */
const fileInfo = document.getElementById("file-info");

/** @type {HTMLSpanElement} */
const fileName = document.getElementById("file-name");

/** @type {HTMLButtonElement} */
const clearBtn = document.getElementById("clear-btn");

/** @type {HTMLButtonElement} */
const submitBtn = document.getElementById("submit-btn");

/** @type {HTMLElement} */
const uploadSection = document.getElementById("upload-section");

/** @type {HTMLElement} */
const loadingSection = document.getElementById("loading-section");

/** @type {HTMLElement} */
const resultsSection = document.getElementById("results-section");

/** @type {HTMLElement} */
const errorSection = document.getElementById("error-section");

/** @type {HTMLParagraphElement} */
const errorMessage = document.getElementById("error-message");


// =============================================================================
// EVENT LISTENERS
// =============================================================================

// --- File Input (tap to select) ---
dropZone.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelection(file);
    }
});

// --- Drag and Drop ---
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");

    const file = e.dataTransfer.files[0];
    if (file) {
        handleFileSelection(file);
    }
});

// --- Clear Button ---
clearBtn.addEventListener("click", (e) => {
    e.stopPropagation(); // Prevent triggering the drop zone click
    resetUpload();
});

// --- Form Submission ---
uploadForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitImage();
});


// =============================================================================
// FILE HANDLING
// =============================================================================

/**
 * Handle file selection from either input or drag & drop.
 * Validates the file and shows a preview if valid.
 *
 * @param {File} file - The selected image file 
 */
function handleFileSelection(file) {
    // Validate file type
    if (!CONFIG.allowedTypes.includes(file.type)) {
        showError("Invalid file type. Please upload a JPG, JPEG, or PNG image.");
        return;
    }

    // Validate file size
    if (file.size > CONFIG.maxFileSize) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        showError(`File is too large (${sizeMB}MB). Maximum allowed size is 5MB. Please compress or resize your image.`);
        return;
    }

    // Set the file to the input (needed for FormData submission)
    // This is necessary when file comes from drag & drop
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove("hidden");
        dropZoneContent.classList.add("hidden");
    };
    reader.readAsDataURL(file);

    // Show file info bar
    fileName.textContent = file.name;
    fileInfo.classList.remove("hidden");

    // Enable the submit button
    submitBtn.disabled = false;

    // Hide any previous error
    errorSection.classList.add("hidden");
}


/**
 * Reset the upload form to its initial state.
 * Called when user clicks Clear or Try Again.
 */
function resetUpload() {
    // Clear file input
    fileInput.value = "";

    // Hide preview, show drop zone content
    imagePreview.classList.add("hidden");
    imagePreview.src = "";
    dropZoneContent.classList.remove("hidden");

    // Hide file info
    fileInfo.classList.add("hidden");
    fileName.textContent = "";

    // Disable submit button
    submitBtn.disabled = true;

    // Show upload section, hide other sections
    uploadSection.classList.remove("hidden");
    loadingSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
    errorSection.classList.add("hidden");
}

// Make resetUpload accessible globally (used by the inline onclick in index.html)
window.resetUpload = resetUpload;


// =============================================================================
// IMAGE UPLOAD & PREDICTION
// =============================================================================

/**
 * Submit the selected image to the Flask /predict endpoint.
 * Sends the image via FormData (multipart/form-data) and handles the response.
 */
async function submitImage() {
    const file = fileInput.files[0];
    if (!file) {
        showError("No file selected. Please choose an image first.");
        return;
    }

    // Show loading state
    uploadSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");

    // Prepare form data
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Send the image to the backend
        const response = await fetch(CONFIG.predictEndpoint, {
            method: "POST",
            headers: {
                // CSRF mitigation: the server rejects requests without this
                // custom header (cross-origin requests can't set it).
                "X-Requested-With": "XMLHttpRequest",
            },
            body: formData,
            // Note: Do NOT set Content-Type header manually — browser sets it
            // with the correct multipart boundary for FormData.
        });

        // Parse the JSON response
        const data = await response.json();

        // Hide loading spinner
        loadingSection.classList.add("hidden");

        if (data.success) {
            // Prediction was successful — show results
            displayResults(data);
        } else {
            // Server returned an error
            showError(data.error || "An unknown error occurred during prediction.");
        }
    } catch (error) {
        // Network or parsing error
        console.error("Prediction request failed:", error);
        loadingSection.classList.add("hidden");
        showError(
            "Could not connect to the server. Please check your internet connection and try again."
        );
    }
}


// =============================================================================
// RESULTS DISPLAY
// =============================================================================

/**
 * Render the diagnosis results dynamically in the results section.
 *
 * @param {Object} data - Response data from the /predict endpoint
 * @param {string} data.prediction - Disease name
 * @param {number} data.confidence - Confidence percentage (0-100)
 * @param {string} data.description - Disease description
 * @param {string} data.advice - Management advice
 * @param {string} data.severity - "healthy", "disease", or "uncertain" (low confidence)
 */
function displayResults(data) {
    let statusIcon, severityClass;
    if (data.severity === "healthy") {
        statusIcon = "✅";
        severityClass = "healthy";
    } else if (data.severity === "uncertain") {
        statusIcon = "⚠️";
        severityClass = "uncertain";
    } else {
        statusIcon = "🔴";
        severityClass = "disease";
    }

    resultsSection.innerHTML = `
        <div class="card result-card ${severityClass}">
            <div class="result-header">
                <div class="result-icon">${statusIcon}</div>
                <h2 class="result-title">${escapeHtml(data.prediction)}</h2>
            </div>

            <!-- Confidence Score -->
            <div class="confidence-section">
                <div class="confidence-label">Confidence Level</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar" style="width: 0%" id="confidence-bar-fill">
                        <span class="confidence-value">${data.confidence}%</span>
                    </div>
                </div>
            </div>

            <!-- Description -->
            <div class="info-block">
                <h3>📋 Description</h3>
                <p>${escapeHtml(data.description)}</p>
            </div>

            <!-- Advice -->
            <div class="info-block advice-block">
                <h3>💊 Management Advice</h3>
                <p>${escapeHtml(data.advice)}</p>
            </div>
        </div>

        <!-- Action Buttons -->
        <div class="result-actions">
            <button type="button" class="btn-primary" onclick="resetUpload()" id="new-diagnosis-btn">
                <span class="btn-text">📷 New Diagnosis</span>
            </button>
        </div>
    `;

    // Show the results section
    resultsSection.classList.remove("hidden");

    // Animate the confidence bar (slight delay for visual effect)
    requestAnimationFrame(() => {
        setTimeout(() => {
            const bar = document.getElementById("confidence-bar-fill");
            if (bar) {
                bar.style.width = `${data.confidence}%`;
            }
        }, 100);
    });

    // Scroll to the results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}


// =============================================================================
// ERROR HANDLING
// =============================================================================

/**
 * Display an error message to the user.
 *
 * @param {string} message - Human-readable error message
 */
function showError(message) {
    // Hide loading and results
    loadingSection.classList.add("hidden");
    resultsSection.classList.add("hidden");

    // Show the upload section (so user can try again)
    uploadSection.classList.remove("hidden");

    // Show error section with the message
    errorMessage.textContent = message;
    errorSection.classList.remove("hidden");

    // Scroll to the error
    errorSection.scrollIntoView({ behavior: "smooth", block: "center" });
}


// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Escape HTML characters to prevent XSS when rendering dynamic content.
 *
 * @param {string} text - Raw text to escape
 * @returns {string} HTML-safe text
 */
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
