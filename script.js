// Initialize Lucide Icons
lucide.createIcons();

// Default: Restricted = Right side of line, and load saved clips on Saved Clips page
document.addEventListener('DOMContentLoaded', () => {
    const br = document.getElementById('btn-restricted-right');
    if (br) br.classList.add('active');

    const savedClipsGrid = document.getElementById('saved-clips-grid');
    if (savedClipsGrid) {
        loadSavedClips(savedClipsGrid);
    }
});

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
const intrusionAlert = document.getElementById('intrusion-alert');
const btnBoundaryStart = document.getElementById('btn-boundary-start');
const btnBoundaryFinish = document.getElementById('btn-boundary-finish');
const btnBoundaryClear = document.getElementById('btn-boundary-clear');
const btnRestrictedLeft = document.getElementById('btn-restricted-left');
const btnRestrictedRight = document.getElementById('btn-restricted-right');

// Off-screen canvas used to grab frames to send to the backend
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

let currentStream = null;
let detectionRunning = false;
let boundaryPoints = [];  // normalized {x, y} 0-1
let isDrawingBoundary = false;
let hasIntrusion = false;
let restrictedSide = 1;  // 1 = right of line (px > boundaryX), 0 = left
const sessionId = 'sess-' + Math.random().toString(36).slice(2) + Date.now();
let alarmAudio = null;

function resizeOverlayToVideo() {
    const rect = videoContainer.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    if (w === overlayCanvas.width && h === overlayCanvas.height) return;
    overlayCanvas.width = w;
    overlayCanvas.height = h;
}

window.addEventListener('resize', resizeOverlayToVideo);
mainVideo.addEventListener('loadedmetadata', resizeOverlayToVideo);
resizeOverlayToVideo();

function clearDetections() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function getCanvasCoordsFromClick(clientX, clientY) {
    const rect = overlayCanvas.getBoundingClientRect();
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    const offsetX = clientX - rect.left;
    const canvasX = w - offsetX;
    const canvasY = clientY - rect.top;
    return { x: canvasX / w, y: canvasY / h };
}

function getBoundaryXAtY(normY) {
    if (boundaryPoints.length < 2) return null;
    const h = overlayCanvas.height;
    const py = normY * h;
    for (let i = 0; i < boundaryPoints.length - 1; i++) {
        const p1 = boundaryPoints[i];
        const p2 = boundaryPoints[i + 1];
        const y1 = p1.y * h, y2 = p2.y * h;
        if ((py >= y1 && py <= y2) || (py >= y2 && py <= y1)) {
            const t = (py - y1) / (y2 - y1 || 0.001);
            return (p1.x + t * (p2.x - p1.x)) * overlayCanvas.width;
        }
    }
    return null;
}

function isInRestrictedZone(normX, normY) {
    const boundaryX = getBoundaryXAtY(normY);
    if (boundaryX === null) return false;
    const px = normX * overlayCanvas.width;
    return px > boundaryX;
}

function drawBoundary() {
    if (boundaryPoints.length === 0) return;
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    if (boundaryPoints.length >= 2) {
        overlayCtx.strokeStyle = 'rgba(255, 75, 75, 0.95)';
        overlayCtx.lineWidth = 3;
        overlayCtx.beginPath();
        overlayCtx.moveTo(boundaryPoints[0].x * w, boundaryPoints[0].y * h);
        for (let i = 1; i < boundaryPoints.length; i++) {
            overlayCtx.lineTo(boundaryPoints[i].x * w, boundaryPoints[i].y * h);
        }
        overlayCtx.stroke();
    }
    boundaryPoints.forEach((p, i) => {
        overlayCtx.fillStyle = i === 0 ? 'rgba(255, 75, 75, 0.9)' : 'rgba(255, 215, 0, 0.9)';
        overlayCtx.beginPath();
        overlayCtx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2);
        overlayCtx.fill();
    });
}

function drawDetections(boxes) {
    clearDetections();
    drawBoundary();

    if (!boxes || !boxes.length) {
        hasIntrusion = false;
        return;
    }
    if (overlayCanvas.width === 0 || overlayCanvas.height === 0) {
        resizeOverlayToVideo();
        if (overlayCanvas.width === 0 || overlayCanvas.height === 0) return;
    }
    const width = overlayCanvas.width;
    const height = overlayCanvas.height;

    overlayCtx.lineWidth = 2;
    overlayCtx.font = '12px "Share Tech Mono", monospace';

    hasIntrusion = false;
    boxes.forEach(box => {
        const x1 = box.x1 * width;
        const y1 = box.y1 * height;
        const x2 = box.x2 * width;
        const y2 = box.y2 * height;
        const cx = (box.x1 + box.x2) / 2;
        const cy = (box.y1 + box.y2) / 2;
        const inRestricted = boundaryPoints.length >= 2
            ? (box.in_restricted_zone !== undefined ? box.in_restricted_zone : isInRestrictedZone(cx, cy))
            : false;
        if (inRestricted) hasIntrusion = true;

        const w = x2 - x1;
        const h = y2 - y1;

        const isBreach = box.breach === true;
        const isAuth = box.authorized === true;
        const isUnauth = box.authorized === false;
        const status = box.status || 'Person';

        // Restricted area = green, Unrestricted area = red (when boundary is set)
        let strokeColor = 'rgba(0, 240, 255, 0.9)';
        if (boundaryPoints.length >= 2) {
            strokeColor = inRestricted ? 'rgba(0, 255, 100, 0.95)' : 'rgba(255, 75, 75, 0.95)';
        }
        if (isBreach) strokeColor = 'rgba(255, 75, 75, 0.95)';
        else if (isUnauth) strokeColor = 'rgba(255, 75, 75, 0.95)';
        else if (isAuth && boundaryPoints.length < 2) strokeColor = 'rgba(0, 255, 100, 0.95)';

        overlayCtx.strokeStyle = strokeColor;
        overlayCtx.fillStyle = strokeColor.replace('0.95)', '0.2)').replace('0.9)', '0.15)');

        overlayCtx.beginPath();
        overlayCtx.rect(x1, y1, w, h);
        overlayCtx.stroke();
        overlayCtx.fill();

        // Label: Person | BREACH! / AUTHORIZED / UNAUTHORIZED / Walking / Standing
        let label = box.label || 'Person';
        if (box.status && box.label === 'Person') label = `Person | ${status}`;
        else if (box.label !== 'Person') label = `Person | ${box.label}`;
        const paddingX = 4;
        const textWidth = overlayCtx.measureText(label).width;

        const labelX = x1;
        const labelY = Math.max(0, y1 - 16);

        const rectX = labelX - paddingX;
        const rectWidth = textWidth + paddingX * 2;
        const centerX = rectX + rectWidth / 2;

        overlayCtx.save();
        overlayCtx.translate(centerX, 0);
        overlayCtx.scale(-1, 1);
        overlayCtx.translate(-centerX, 0);

        overlayCtx.fillStyle = strokeColor;
        overlayCtx.fillRect(rectX, labelY - 10, rectWidth, 14);
        overlayCtx.fillStyle = '#000';
        overlayCtx.fillText(label, labelX, labelY);

        overlayCtx.restore();
    });
}

function playBreachAlarm() {
    try {
        if (!alarmAudio) {
            alarmAudio = new Audio('/audio/alert');
        }
        alarmAudio.currentTime = 0;
        alarmAudio.play().catch(() => {});
    } catch (e) {}
}

async function saveBreachFrame(imageDataUrl) {
    try {
        await fetch('/save_clip', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageDataUrl,
                session_id: sessionId,
            }),
        });
    } catch (e) {
        console.error('Error saving breach frame:', e);
    }
}

async function loadSavedClips(gridEl) {
    try {
        const resp = await fetch('/saved_clips');
        if (!resp.ok) {
            gridEl.innerHTML = '<div class="small-text cyan">Unable to load saved clips.</div>';
            return;
        }
        const data = await resp.json();
        const clips = data.clips || [];
        if (!clips.length) {
            gridEl.innerHTML = '<div class="small-text cyan">No saved clips yet.</div>';
            return;
        }
        gridEl.innerHTML = '';
        clips.forEach((url) => {
            const item = document.createElement('div');
            item.className = 'saved-clip-item';
            const img = document.createElement('img');
            img.src = url;
            img.className = 'saved-clip-thumb';
            img.alt = 'Saved clip';
            const meta = document.createElement('div');
            meta.className = 'saved-clip-meta';
            meta.textContent = url.split('/').pop();
            item.appendChild(img);
            item.appendChild(meta);
            gridEl.appendChild(item);
        });
    } catch (e) {
        console.error('Error loading saved clips:', e);
        gridEl.innerHTML = '<div class="small-text cyan">Error loading saved clips.</div>';
    }
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

            const jpegQuality = boundaryPoints.length >= 2 ? 0.5 : 0.6;
            const dataUrl = captureCanvas.toDataURL('image/jpeg', jpegQuality);

            const payload = {
                image: dataUrl,
                session_id: sessionId
            };
            if (boundaryPoints.length >= 2) {
                payload.boundary = boundaryPoints.slice(0, 2).map(p => [p.x, p.y]);
                payload.restricted_side = restrictedSide;
            }

            const response = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const data = await response.json();
                resizeOverlayToVideo();
                drawDetections(data.boxes || []);
                if (data.breaches && data.breaches.length > 0) {
                    playBreachAlarm();
                    saveBreachFrame(dataUrl);
                    intrusionAlert.textContent = 'Intruder Detected';
                    intrusionAlert.classList.add('visible');
                    clearTimeout(window._intrusionAlertHide);
                    window._intrusionAlertHide = setTimeout(() => intrusionAlert.classList.remove('visible'), 5000);
                }
            } else {
                const errText = await response.text();
                console.error('Detect API error:', response.status, errText);
            }
        }
    } catch (err) {
        console.error('Detection error:', err);
    }

    if (detectionRunning) {
        // Faster polling when boundary is set for quicker breach alarm (~10–12 FPS)
        const intervalMs = boundaryPoints.length >= 2 ? 80 : 120;
        setTimeout(detectionLoop, intervalMs);
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
        mainVideo.addEventListener('loadedmetadata', function onMeta() {
            mainVideo.removeEventListener('loadedmetadata', onMeta);
            resizeOverlayToVideo();
            setTimeout(resizeOverlayToVideo, 100);
        }, { once: true });
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

// Boundary drawing
btnBoundaryStart.addEventListener('click', () => {
    isDrawingBoundary = true;
    overlayCanvas.classList.add('drawing');
    btnBoundaryStart.classList.add('active');
    btnBoundaryFinish.classList.remove('active');
    const rect = videoContainer.getBoundingClientRect();
    if (rect.width && rect.height) {
        overlayCanvas.width = rect.width;
        overlayCanvas.height = rect.height;
        clearDetections();
        drawBoundary();
    }
});

btnBoundaryFinish.addEventListener('click', () => {
    isDrawingBoundary = false;
    overlayCanvas.classList.remove('drawing');
    btnBoundaryStart.classList.remove('active');
    btnBoundaryFinish.classList.add('active');
});

btnBoundaryClear.addEventListener('click', () => {
    boundaryPoints = [];
    isDrawingBoundary = false;
    overlayCanvas.classList.remove('drawing');
    btnBoundaryStart.classList.remove('active');
    btnBoundaryFinish.classList.remove('active');
    intrusionAlert.classList.remove('visible');
    if (window._intrusionAlertHide) clearTimeout(window._intrusionAlertHide);
    hasIntrusion = false;
    clearDetections();
    drawBoundary();
});

btnRestrictedLeft.addEventListener('click', () => {
    restrictedSide = 0;
    btnRestrictedLeft.classList.add('active');
    if (btnRestrictedRight) btnRestrictedRight.classList.remove('active');
});
btnRestrictedRight.addEventListener('click', () => {
    restrictedSide = 1;
    btnRestrictedRight.classList.add('active');
    if (btnRestrictedLeft) btnRestrictedLeft.classList.remove('active');
});

overlayCanvas.addEventListener('click', (e) => {
    if (!isDrawingBoundary) return;
    const { x, y } = getCanvasCoordsFromClick(e.clientX, e.clientY);
    boundaryPoints.push({ x, y });
    clearDetections();
    drawBoundary();
});
