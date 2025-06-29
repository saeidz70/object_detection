document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const imageInput = document.getElementById('image');
    const thresholdInput = document.getElementById("threshold");
    const thresholdValue = document.getElementById("threshold-value");

    // Update slider value
    if (thresholdInput && thresholdValue) {
        thresholdInput.addEventListener("input", () => {
            thresholdValue.textContent = thresholdInput.value;
        });
    }

    // Drop area logic
    if (dropArea && imageInput) {
        dropArea.addEventListener("click", () => imageInput.click());

        dropArea.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropArea.style.borderColor = "#009688";
        });

        dropArea.addEventListener("dragleave", () => {
            dropArea.style.borderColor = "#bbb";
        });

        dropArea.addEventListener("drop", (e) => {
            e.preventDefault();
            dropArea.style.borderColor = "#bbb";

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                imageInput.files = files;
            }
        });
    }
});
