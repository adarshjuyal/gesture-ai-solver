const toggleBtn = document.getElementById("toggle-button");
const outputText = document.getElementById("output-text");
let isRunning = false;

toggleBtn.addEventListener("click", () => {
    isRunning = !isRunning;
    toggleBtn.textContent = isRunning ? "Stop" : "Start";

    if (isRunning) {
        outputText.textContent = "Detecting gesture...";
        startWebcam();
    } else {
        stopWebcam();
        outputText.textContent = "Stopped.";
    }
});

function startWebcam() {
    const video = document.getElementById("webcam");
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((err) => {
            console.error("Error accessing webcam: ", err);
            outputText.textContent = "Webcam access failed.";
        });
}

function stopWebcam() {
    const video = document.getElementById("webcam");
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        video.srcObject = null;
    }
}
