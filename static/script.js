const imageInput = document.getElementById("imageInput");
const uploadBtn = document.getElementById("uploadBtn");
const resultsArea = document.getElementById("resultsArea");
const originalImage = document.getElementById("originalImage");
const resultImage = document.getElementById("resultImage");
const loading = document.getElementById("loading");
const overlay = document.getElementById("overlay");
const fileMeta = document.getElementById("fileMeta");
const resultInfo = document.getElementById("resultInfo");

// 1. When a user selects a file, enable the upload button and show preview
imageInput.addEventListener("change", function () {
  if (this.files && this.files[0]) {
    uploadBtn.disabled = false;
    // update filename meta
    fileMeta.textContent = this.files[0].name;

    // Show a preview of the selected image immediately
    const reader = new FileReader();
    reader.onload = function (e) {
      originalImage.src = e.target.result;
      resultsArea.classList.remove("hidden");
      resultImage.src = ""; // Clear previous result
      resultInfo.textContent = "";
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
  overlay.classList.add("active");
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
      if (data && data.result_image_url) {
        resultImage.src = data.result_image_url;
        resultImage.style.opacity = "1";
        resultInfo.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
      } else {
        throw new Error('No result url returned');
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("An error occurred during detection. Check server logs.");
    })
    .finally(() => {
      // Hide loading state regardless of success/failure
      overlay.classList.remove("active");
      uploadBtn.disabled = false;
    });
});
