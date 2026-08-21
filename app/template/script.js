const API_BASE = ""; // same origin — FastAPI serves the frontend too

const dropzone = document.getElementById("dropzone");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const predictBtnText = document.getElementById("predictBtnText");
const predictSpinner = document.getElementById("predictSpinner");
const resetBtn = document.getElementById("resetBtn");
const errorBanner = document.getElementById("errorBanner");

const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultCard = document.getElementById("resultCard");
const verdictIcon = document.getElementById("verdictIcon");
const verdictLabel = document.getElementById("verdictLabel");
const verdictConfidence = document.getElementById("verdictConfidence");
const probBars = document.getElementById("probBars");

const modelDot = document.getElementById("modelDot");
const modelBadgeText = document.getElementById("modelBadgeText");

let selectedFile = null;

// ---------- Model status check ----------
async function checkModelStatus() {
    try {
        const res = await fetch(`${API_BASE}/model-info`);
        if (!res.ok) throw new Error("not ok");
        const data = await res.json();
        modelDot.classList.add("online");
        modelBadgeText.textContent = `${data.model_type} · ${data.version} · ROC-AUC ${data.primary_metric_value.toFixed(3)}`;
    } catch (err) {
        modelDot.classList.add("offline");
        modelBadgeText.textContent = "Model unavailable";
    }
}
checkModelStatus();

// ---------- Upload handling ----------
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        showError("Please upload an image file (JPG, PNG, JPEG).");
        return;
    }
    hideError();
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        dropzoneEmpty.hidden = true;
    };
    reader.readAsDataURL(file);

    predictBtn.disabled = false;
    resetBtn.hidden = false;
    resetResultUI();
}

resetBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    previewImage.hidden = true;
    previewImage.src = "";
    dropzoneEmpty.hidden = false;
    predictBtn.disabled = true;
    resetBtn.hidden = true;
    hideError();
    resetResultUI();
});

// ---------- Prediction ----------
predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    setLoading(true);
    hideError();

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Request failed (${res.status})`);
        }

        const data = await res.json();
        renderResult(data);

    } catch (err) {
        showError(err.message || "Something went wrong while analyzing the image.");
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    predictSpinner.hidden = !isLoading;
    predictBtnText.textContent = isLoading ? "Analyzing…" : "Analyze X-Ray";
}

function renderResult(data) {
    resultPlaceholder.hidden = true;
    resultCard.hidden = false;

    const isPneumonia = data.prediction.toUpperCase() === "PNEUMONIA";

    verdictIcon.className = "verdict-icon " + (isPneumonia ? "pneumonia" : "normal");
    verdictIcon.textContent = isPneumonia ? "⚠" : "✓";
    verdictLabel.textContent = isPneumonia ? "Pneumonia Detected" : "Normal";
    verdictConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence · model ${data.model_version}`;

    probBars.innerHTML = "";
    const order = ["PNEUMONIA", "NORMAL"];
    order.forEach((cls) => {
        if (!(cls in data.probabilities)) return;
        const value = data.probabilities[cls];
        const pct = (value * 100).toFixed(1);
        const cssClass = cls === "PNEUMONIA" ? "pneumonia" : "normal";

        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <div class="prob-row-top">
                <span>${cls}</span>
                <span>${pct}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill ${cssClass}" style="width: 0%"></div>
            </div>
        `;
        probBars.appendChild(row);

        // animate width after insertion
        requestAnimationFrame(() => {
            row.querySelector(".prob-fill").style.width = `${pct}%`;
        });
    });
}

function resetResultUI() {
    resultCard.hidden = true;
    resultPlaceholder.hidden = false;
}

function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
}

function hideError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
}
