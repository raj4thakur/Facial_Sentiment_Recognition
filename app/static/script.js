const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const statusText = document.getElementById("statusText");
const statusDot = document.getElementById("statusDot");
const resultsDiv = document.getElementById("results");

let stream = null;
let detectionTimer = null;
let requestInProgress = false;

const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d");

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;

        await video.play();

        resizeOverlay();

        startButton.disabled = true;
        stopButton.disabled = false;

        statusText.textContent = "Camera running";
        statusDot.classList.add("active");

        // Run detection approximately every 400 ms.
        detectionTimer = setInterval(sendFrameForPrediction, 400);

    } catch (error) {
        console.error(error);

        statusText.textContent =
            "Camera permission was denied or the camera is unavailable.";

        alert(
            "Could not access the camera. Please allow camera permission and try again."
        );
    }
}

function stopCamera() {
    if (detectionTimer) {
        clearInterval(detectionTimer);
        detectionTimer = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;

    startButton.disabled = false;
    stopButton.disabled = true;

    statusText.textContent = "Camera stopped";
    statusDot.classList.remove("active");

    clearOverlay();

    resultsDiv.innerHTML =
        '<p class="muted">Start the camera to begin detection.</p>';
}

async function sendFrameForPrediction() {
    if (!stream || video.readyState < 2 || requestInProgress) {
        return;
    }

    requestInProgress = true;

    try {
        const width = video.videoWidth;
        const height = video.videoHeight;

        if (!width || !height) {
            return;
        }

        captureCanvas.width = width;
        captureCanvas.height = height;

        captureContext.drawImage(video, 0, 0, width, height);

        const blob = await new Promise(resolve =>
            captureCanvas.toBlob(resolve, "image/jpeg", 0.75)
        );

        if (!blob) {
            return;
        }

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText);
        }

        const data = await response.json();

        drawResults(data);
    } catch (error) {
        console.error("Prediction error:", error);
        statusText.textContent = "Prediction error";
    } finally {
        requestInProgress = false;
    }
}

function resizeOverlay() {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
}

function clearOverlay() {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
}

function drawResults(data) {
    resizeOverlay();

    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);

    resultsDiv.innerHTML = "";

    if (!data.faces || data.faces.length === 0) {
        resultsDiv.innerHTML =
            '<p class="muted">No face detected.</p>';
        return;
    }

    const scaleX = canvas.width / data.image_width;
    const scaleY = canvas.height / data.image_height;

    data.faces.forEach((face, index) => {
        const x = face.box.x * scaleX;
        const y = face.box.y * scaleY;
        const width = face.box.width * scaleX;
        const height = face.box.height * scaleY;

        context.strokeStyle = "#00ff00";
        context.lineWidth = 3;
        context.strokeRect(x, y, width, height);

        const text = `${face.emotion} ${(face.confidence * 100).toFixed(1)}%`;

        context.font = "bold 18px Arial";

        const textWidth = context.measureText(text).width;
        const textHeight = 24;

        context.fillStyle = "rgba(0, 0, 0, 0.7)";
        context.fillRect(
            x,
            Math.max(0, y - textHeight),
            textWidth + 12,
            textHeight
        );

        context.fillStyle = "#ffffff";
        context.fillText(
            text,
            x + 6,
            Math.max(18, y - 6)
        );

        const item = document.createElement("div");
        item.className = "emotion";

        item.innerHTML = `
            <strong>Face ${index + 1}</strong>
            <span>
                ${escapeHtml(face.emotion)}
                (${(face.confidence * 100).toFixed(1)}%)
            </span>
        `;

        resultsDiv.appendChild(item);
    });
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

window.addEventListener("resize", resizeOverlay);
