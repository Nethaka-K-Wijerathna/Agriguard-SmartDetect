const imageInput = document.getElementById("imageInput");
const uploadBtn = document.getElementById("uploadBtn");
const resultsArea = document.getElementById("resultsArea");
const originalImage = document.getElementById("originalImage");
const resultImage = document.getElementById("resultImage");
const loading = document.getElementById("loading");

// 1. When a user selects a file, enable the upload button and show preview
imageInput.addEventListener("change", function () {
  if (this.files && this.files[0]) {
    uploadBtn.disabled = false;

    // Show a preview of the selected image immediately
    const reader = new FileReader();
    reader.onload = function (e) {
      originalImage.src = e.target.result;
      resultsArea.classList.remove("hidden");
      resultImage.src = ""; // Clear previous result
    };
    reader.readAsDataURL(this.files[0]);
  }
});

// 2. Handle the upload button click
uploadBtn.addEventListener("click", function () {
  const file = imageInput.files[0];
  if (!file) return;

  // Prepare form data to send to backend
  const formData = new FormData();
  formData.append("image", file);

  // Show UI states
  loading.classList.remove("hidden");
  uploadBtn.disabled = true;
  resultImage.style.opacity = "0.3"; // Dim result area while loading

  // Send POST request to Flask API
  fetch("/detect", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Success:", data);
      // Update the result image source with the URL provided by Flask
      resultImage.src = data.result_image_url;
      resultImage.style.opacity = "1";
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("An error occurred during detection.");
    })
    .finally(() => {
      // Hide loading state regardless of success/failure
      loading.classList.add("hidden");
      uploadBtn.disabled = false;
    });
});
