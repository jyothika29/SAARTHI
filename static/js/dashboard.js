(() => {
    const root = document.documentElement;
    const themeSelect = document.getElementById("themeSelect");
    const storageKey = "saarthi-theme";

    // ---------------- THEME SETTINGS ----------------

    const validThemes = ["light", "dark", "system"];

    function applyTheme(theme) {
        if (!validThemes.includes(theme)) {
            theme = "system";
        }

        root.dataset.theme = theme;

        if (themeSelect) {
            themeSelect.value = theme;
        }

        try {
            localStorage.setItem(storageKey, theme);
        } catch (error) {
            console.warn("Theme preference could not be saved.");
        }
    }

    let savedTheme = "system";

    try {
        const storedTheme = localStorage.getItem(storageKey);

        if (validThemes.includes(storedTheme)) {
            savedTheme = storedTheme;
        }
    } catch (error) {
        console.warn("Saved theme preference is unavailable.");
    }

    applyTheme(savedTheme);

    if (themeSelect) {
        themeSelect.addEventListener("change", (event) => {
            applyTheme(event.target.value);
        });
    }

    // ---------------- LIVE CAMERA ----------------

    const startButton = document.getElementById("startCamera");
    const stopButton = document.getElementById("stopCamera");
    const cameraFeed = document.getElementById("cameraFeed");
    const cameraPlaceholder =
        document.getElementById("cameraPlaceholder");
    const cameraMessage = document.getElementById("cameraMessage");

    if (
        startButton &&
        stopButton &&
        cameraFeed &&
        cameraPlaceholder &&
        cameraMessage
    ) {
        startButton.addEventListener("click", async () => {
            startButton.disabled = true;
            cameraMessage.textContent = "Starting camera...";

            try {
                const response = await fetch("/api/camera/start", {
                    method: "POST"
                });

                const result = await response.json();

                if (!response.ok || !result.success) {
                    throw new Error(
                        result.message || "Could not start the camera."
                    );
                }

                cameraFeed.src = "/video_feed?t=" + Date.now();
                cameraFeed.hidden = false;
                cameraPlaceholder.hidden = true;

                cameraMessage.textContent =
                    "Camera monitoring started.";

                stopButton.disabled = false;

            } catch (error) {
                cameraMessage.textContent =
                    error.message || "Camera could not start.";

                startButton.disabled = false;

                console.error("Camera start error:", error);
            }
        });

        stopButton.addEventListener("click", async () => {
            stopButton.disabled = true;
            cameraMessage.textContent = "Stopping camera...";

            try {
                const response = await fetch("/api/camera/stop", {
                    method: "POST"
                });

                const responseText = await response.text();

let result;

try {
    result = JSON.parse(responseText);
} catch (error) {
    throw new Error(
        "Server returned HTML instead of JSON. Check the Flask terminal."
    );
}

                if (!response.ok) {
                    throw new Error(
                        result.message || "Could not stop the camera."
                    );
                }

                cameraFeed.removeAttribute("src");
                cameraFeed.hidden = true;
                cameraPlaceholder.hidden = false;

                cameraMessage.textContent =
                    result.message || "Camera monitoring stopped.";

                startButton.disabled = false;

            } catch (error) {
                cameraMessage.textContent =
                    error.message || "Could not stop the camera.";

                stopButton.disabled = false;

                console.error("Camera stop error:", error);
            }
        });
    }

    // ---------------- VIDEO UPLOAD AND AI RESULTS ----------------

    const videoUploadForm =
        document.getElementById("videoUploadForm");

    const videoFileInput =
        document.getElementById("videoFile");

    const uploadedVideoContainer =
        document.getElementById("uploadedVideoContainer");

    const uploadedVideoPreview =
        document.getElementById("uploadedVideoPreview");

    const videoAnalysisMessage =
        document.getElementById("videoAnalysisMessage");

    const uploadBadge =
        document.getElementById("uploadBadge");

    const analyseVideoButton =
        document.getElementById("analyseVideoButton");

    const videoDetectionResult =
        document.getElementById("videoDetectionResult");

    const videoAlertStatus =
        document.getElementById("videoAlertStatus");

    const videoAlertMessage =
        document.getElementById("videoAlertMessage");

    if (videoUploadForm) {

        // Preview the selected video.
        videoFileInput.addEventListener("change", () => {
            const file = videoFileInput.files[0];

            if (!file) {
                return;
            }

            uploadedVideoPreview.src =
                URL.createObjectURL(file);

            uploadedVideoContainer.hidden = false;

            videoAnalysisMessage.textContent =
                "Video selected. Click Analyse Video to start AI analysis.";

            uploadBadge.textContent = "Selected";
            videoDetectionResult.textContent = "Not analysed";
            videoAlertStatus.textContent = "Not evaluated";
            videoAlertMessage.hidden = true;
        });

        // Upload the video and receive the model results.
        videoUploadForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            const file = videoFileInput.files[0];

            if (!file) {
                videoAnalysisMessage.textContent =
                    "Please select a video first.";
                return;
            }

            const formData = new FormData();
            formData.append("video", file);

            analyseVideoButton.disabled = true;
            uploadBadge.textContent = "Analysing";
            videoAnalysisMessage.textContent =
                "Uploading video and running AI analysis. Please wait...";

            videoDetectionResult.textContent = "Processing...";
            videoAlertStatus.textContent = "Pending";
            videoAlertMessage.hidden = true;

            try {
                const response = await fetch("/api/video/upload", {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();

                if (!response.ok || !result.success) {
                    throw new Error(
                        result.message || "Video analysis failed."
                    );
                }

                // Analysis completed.
                uploadBadge.textContent = "Analysis Complete";

                videoAnalysisMessage.textContent =
                    result.message ||
                    "Video analysis completed.";

                // Display the experimental model score.
                if (
                    typeof result.experimental_score === "number"
                ) {
                    const scorePercent =
                        result.experimental_score * 100;

                    videoDetectionResult.textContent =
                        "Experimental model score: " +
                        scorePercent.toFixed(1) +
                        "%";

                } else {
                    videoDetectionResult.textContent =
                        "Model score unavailable";
                }

                // Display the model's uncertainty output.
                if (typeof result.uncertainty === "number") {
                    videoAlertStatus.textContent =
                        "Uncertainty output: " +
                        (result.uncertainty * 100).toFixed(1) +
                        "%";
                } else {
                    videoAlertStatus.textContent =
                        "Uncertainty unavailable";
                }

                // Show an explanation instead of falsely claiming
                // that the model has confirmed violence.
                videoAlertMessage.textContent =
                    "Experimental AI screening only. Review the video " +
                    "and verify the model result before making a safety decision.";

                videoAlertMessage.hidden = false;

                console.log("SAARTHI AI analysis:", result);

            } catch (error) {
                uploadBadge.textContent = "Error";

                videoAnalysisMessage.textContent =
                    error.message ||
                    "Video analysis failed. Check the server terminal.";

                videoDetectionResult.textContent =
                    "Analysis failed";

                videoAlertStatus.textContent =
                    "No result available";

                videoAlertMessage.textContent =
                    "The video could not be analysed. Check the Flask terminal.";

                videoAlertMessage.hidden = false;

                console.error("SAARTHI video analysis error:", error);

            } finally {
                analyseVideoButton.disabled = false;
            }
        });
    }
    // ---------------- ATTENDANCE DASHBOARD ----------------

const attendanceCount = document.getElementById("attendanceCount");
const attendanceMessage = document.getElementById("attendanceMessage");
const attendanceBadge = document.getElementById("attendanceBadge");

async function loadAttendance() {
    if (!attendanceCount || !attendanceMessage) {
        return;
    }

    try {
        const response = await fetch("/api/attendance/today", {
            cache: "no-store"
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Could not load attendance.");
        }

        attendanceCount.textContent = String(data.count);
        if (attendanceBadge) {
    attendanceBadge.textContent = "Connected";
    attendanceBadge.className = "badge success";
}

        attendanceMessage.textContent =
            `${data.records.length} attendance records today`;

    } catch (error) {
        if (attendanceBadge) {
    attendanceBadge.textContent = "Unavailable";
    attendanceBadge.className = "badge neutral";
}
        attendanceCount.textContent = "—";
        attendanceMessage.textContent = "Attendance unavailable";

        console.error("Attendance loading error:", error);
    }
}

loadAttendance();
setInterval(loadAttendance, 10000);
    // ---------------- RECENT ACTIVITY ----------------

    const activityList = document.getElementById("activityList");
    const activityEmptyState =
        document.getElementById("activityEmptyState");
    const eventCount = document.getElementById("eventCount");

    async function loadRecentActivity() {
        if (!activityList || !activityEmptyState) {
            return;
        }

        try {
            const response = await fetch("/api/alerts/recent", {
                cache: "no-store"
            });

            if (!response.ok) {
                throw new Error("Could not load recent alerts.");
            }

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.message || "Alert loading failed.");
            }

            const events = Array.isArray(data.events) ? data.events : [];

            // Remove previous entries before refreshing.
            activityList.replaceChildren();

            activityEmptyState.hidden = events.length !== 0;

            if (events.length === 0) {
                if (eventCount) {
                    eventCount.textContent = "0";
                }
                return;
            }

            events.forEach((event) => {
                const item = document.createElement("div");
                item.className = "activity-item";

                const title = document.createElement("strong");
                title.textContent =
                    event.event || "Unknown person detected";

                const details = document.createElement("p");
                details.textContent =
                    [event.date, event.time].filter(Boolean).join(" ");

                item.appendChild(title);
                item.appendChild(details);
                activityList.appendChild(item);
            });

            if (eventCount) {
                eventCount.textContent = String(events.length);
            }

        } catch (error) {
            console.error("Recent activity error:", error);
        }
    }

    // Load immediately and refresh every 5 seconds.
    loadRecentActivity();
    setInterval(loadRecentActivity, 5000);
})();