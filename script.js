// Initialize Lucide Icons
lucide.createIcons();

// Live clock update
function updateClock() {
    const clockElement = document.getElementById('clock');
    const now = new Date();
    
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    
    clockElement.textContent = `${hours}:${minutes}:${seconds}`;
}

// Update clock every second
setInterval(updateClock, 1000);
updateClock();

// Glitch effect on title
const title = document.querySelector('.title-glitch');
setInterval(() => {
    if (Math.random() > 0.95) {
        title.style.textShadow = `
            ${(Math.random() - 0.5) * 10}px ${(Math.random() - 0.5) * 10}px 0 rgba(0, 240, 255, 0.8),
            ${(Math.random() - 0.5) * 10}px ${(Math.random() - 0.5) * 10}px 0 rgba(255, 75, 75, 0.8)
        `;
        setTimeout(() => {
            title.style.textShadow = '0 0 10px var(--cyan), 0 0 20px var(--cyan)';
        }, 50 + Math.random() * 100);
    }
}, 2000);

// Animate timeline progress
const progress = document.querySelector('.timeline-progress');
let progressPos = 45;
let direction = 1;
setInterval(() => {
    progressPos += 0.5 * direction;
    if (progressPos > 55) direction = -1;
    if (progressPos < 35) direction = 1;
    progress.style.left = `${progressPos}%`;
}, 100);

// Video and Webcam logic
const btnWebcamOn = document.getElementById('btn-webcam-on');
const btnWebcamOff = document.getElementById('btn-webcam-off');
const btnSampleVideo = document.getElementById('btn-sample-video');
const videoUpload = document.getElementById('video-upload');
const mainVideo = document.getElementById('main-video');
const videoContainer = document.getElementById('video-container');
const overlayCanvas = document.getElementById('overlay-canvas');
const overlayCtx = overlayCanvas.getContext('2d');

// Off-screen canvas used to grab frames to send to the backend
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

let currentStream = null;
let detectionRunning = false;

function resizeOverlayToVideo() {
    if (!mainVideo.videoWidth || !mainVideo.videoHeight) return;
    const rect = videoContainer.getBoundingClientRect();

    overlayCanvas.width = rect.width;
    overlayCanvas.height = rect.height;
}

window.addEventListener('resize', resizeOverlayToVideo);
mainVideo.addEventListener('loadedmetadata', () => {
    resizeOverlayToVideo();
});

function clearDetections() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function drawDetections(boxes) {
    clearDetections();

    if (!boxes || !boxes.length) return;
    const width = overlayCanvas.width;
    const height = overlayCanvas.height;

    overlayCtx.lineWidth = 2;
    overlayCtx.font = '12px "Share Tech Mono", monospace';

    boxes.forEach(box => {
        const x1 = box.x1 * width;
        const y1 = box.y1 * height;
        const x2 = box.x2 * width;
        const y2 = box.y2 * height;

        const w = x2 - x1;
        const h = y2 - y1;

        overlayCtx.strokeStyle = 'rgba(0, 240, 255, 0.9)';
        overlayCtx.fillStyle = 'rgba(0, 240, 255, 0.15)';

        overlayCtx.beginPath();
        overlayCtx.rect(x1, y1, w, h);
        overlayCtx.stroke();
        overlayCtx.fill();

        // Label
        const label = `${box.label || 'OBJ'} ${Math.round((box.conf || 0) * 100)}%`;
        const paddingX = 4;
        const textWidth = overlayCtx.measureText(label).width;

        const labelX = x1;
        const labelY = Math.max(0, y1 - 16);

        const rectX = labelX - paddingX;
        const rectWidth = textWidth + paddingX * 2;
        const centerX = rectX + rectWidth / 2;

        // Draw label background and text with an additional horizontal flip
        // around its center to counteract the CSS mirror on the canvas.
        overlayCtx.save();
        overlayCtx.translate(centerX, 0);
        overlayCtx.scale(-1, 1);
        overlayCtx.translate(-centerX, 0);

        overlayCtx.fillStyle = 'rgba(0, 240, 255, 0.9)';
        overlayCtx.fillRect(rectX, labelY - 10, rectWidth, 14);
        overlayCtx.fillStyle = '#000';
        overlayCtx.fillText(label, labelX, labelY);

        overlayCtx.restore();
    });
}

async function detectionLoop() {
    if (!detectionRunning) return;

    try {
        if (mainVideo.readyState >= 2 && mainVideo.videoWidth && mainVideo.videoHeight) {
            const vw = mainVideo.videoWidth;
            const vh = mainVideo.videoHeight;

            captureCanvas.width = vw;
            captureCanvas.height = vh;
            captureCtx.drawImage(mainVideo, 0, 0, vw, vh);

            const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.6);

            const response = await fetch('http://127.0.0.1:5000/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: dataUrl })
            });

            if (response.ok) {
                const data = await response.json();
                drawDetections(data.boxes || []);
            }
        }
    } catch (err) {
        console.error('Detection error:', err);
    }

    if (detectionRunning) {
        setTimeout(detectionLoop, 250); // ~4 detections per second
    }
}

function startDetection() {
    if (detectionRunning) return;
    detectionRunning = true;
    detectionLoop();
}

function stopDetection() {
    detectionRunning = false;
    clearDetections();
}

// Stop current video/stream
function stopMedia() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    stopDetection();
    mainVideo.pause();
    mainVideo.srcObject = null;
    mainVideo.src = "";
    mainVideo.style.display = 'none';
    videoContainer.style.backgroundImage = "";
}

btnWebcamOn.addEventListener('click', async () => {
    try {
        stopMedia();
        currentStream = await navigator.mediaDevices.getUserMedia({ video: true });
        mainVideo.srcObject = currentStream;
        mainVideo.style.display = 'block';
        videoContainer.style.backgroundImage = 'none';
        mainVideo.play();
        resizeOverlayToVideo();
        startDetection();
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please allow permission.");
    }
});

btnWebcamOff.addEventListener('click', () => {
    stopMedia();
    videoContainer.style.backgroundImage = "url('assets/bg.png')";
});

btnSampleVideo.addEventListener('click', () => {
    videoUpload.click();
});

videoUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        stopMedia();
        const fileURL = URL.createObjectURL(file);
        mainVideo.src = fileURL;
        mainVideo.loop = true;
        mainVideo.style.display = 'block';
        videoContainer.style.backgroundImage = 'none';
        mainVideo.play();
        resizeOverlayToVideo();
        startDetection();
    }
});
