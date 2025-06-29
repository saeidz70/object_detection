document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const imageInput = document.getElementById('image');
    const thresholdInput = document.getElementById("threshold");
    const thresholdValue = document.getElementById("threshold-value");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("image-preview");
    const form = document.getElementById("upload-form");

    // Update threshold display
    if (thresholdInput && thresholdValue) {
        thresholdInput.addEventListener("input", () => {
            thresholdValue.textContent = thresholdInput.value;
        });
    }

    // Drag & drop logic
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
                showPreview(files[0]);
            }
        });

        imageInput.addEventListener("change", () => {
            if (imageInput.files.length > 0) {
                showPreview(imageInput.files[0]);
            }
        });
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = () => {
            previewImage.src = reader.result;
            previewContainer.style.display = "block";
        };
        reader.readAsDataURL(file);
    }

    // Auto-reset form after submission if result is shown
    if (form && document.querySelector(".result")) {
        form.reset();
        previewContainer.style.display = "none";
        thresholdValue.textContent = "50"; // reset slider label
    }
});
