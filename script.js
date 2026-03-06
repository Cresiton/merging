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

// Subtle bounding box animation (simulating AI tracking)
const boundingBoxes = document.querySelectorAll('.bounding-box');
setInterval(() => {
    boundingBoxes.forEach(box => {
        // Only slightly jitter position for realistic AI tracking effect
        const currentTop = parseFloat(box.style.top);
        const currentLeft = parseFloat(box.style.left);
        
        const jitterY = (Math.random() - 0.5) * 0.5;
        const jitterX = (Math.random() - 0.5) * 0.5;
        
        box.style.top = `${currentTop + jitterY}%`;
        box.style.left = `${currentLeft + jitterX}%`;
    });
}, 500);

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

let currentStream = null;

// Stop current video/stream
function stopMedia() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
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
        mainVideo.style.display = 'block';
        videoContainer.style.backgroundImage = 'none';
        mainVideo.play();
    }
});
