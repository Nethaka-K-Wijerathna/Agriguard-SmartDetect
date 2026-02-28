// element references
const imageInput = document.getElementById("imageInput");
const uploadBtn = document.getElementById("uploadBtn");
const resultsArea = document.getElementById("resultsArea");
const originalImage = document.getElementById("originalImage");
const resultImage = document.getElementById("resultImage");
const resultInfo = document.getElementById("resultInfo");
// live elements
const liveStream = document.getElementById("liveStream");
const liveVideo = document.getElementById("liveVideo");
const captureCanvas = document.getElementById("captureCanvas");
const toggleStream = document.getElementById("toggleStream");
const streamType = document.getElementById("streamType");
const detList = document.getElementById("detList");
const suggestionEl = document.getElementById("suggestion");
const detTime = document.getElementById("detTime");

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
      resultInfo.textContent = "";
    };
    reader.readAsDataURL(this.files[0]);
  }
});

// 2. Handle the upload button click
uploadBtn.addEventListener("click", function () {
  const file = imageInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("image", file);
  uploadBtn.disabled = true;
  resultImage.style.opacity = "0.3";

  fetch("/detect", { method: "POST", body: formData })
    .then((r) => r.json())
    .then((data) => {
      if (data && data.result_image_url) {
        resultImage.src = data.result_image_url;
        resultImage.style.opacity = "1";
        resultInfo.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
      }
    })
    .catch((e) => {
      console.error(e);
      alert("Detection error");
    })
    .finally(() => {
      uploadBtn.disabled = false;
    });
});

// live feed / mobile camera support
let usingMobileCamera = false;
let analyzeInterval = null;
let pollInterval = null;

function updateDetectionsUI(json){
  if(!json) return;
  const detections = json.detections || {};
  const entries = Object.entries(detections);
  if(entries.length === 0){
    detList.innerHTML = '<span class="meta">No detections.</span>';
    suggestionEl.textContent = '—';
  } else {
    entries.sort((a,b)=>b[1]-a[1]);
    detList.innerHTML = entries.map(e=>`${e[0]}: ${e[1]}`).join('<br>');
    suggestionEl.textContent = json.suggestion || 'Consult Agri Officer';
  }
  detTime.textContent = 'Updated: ' + (json.timestamp?new Date(json.timestamp*1000).toLocaleTimeString():new Date().toLocaleTimeString());
}

function captureAndSendFrame(){
  if(!usingMobileCamera) return;
  captureCanvas.width = liveVideo.videoWidth;
  captureCanvas.height = liveVideo.videoHeight;
  const ctx = captureCanvas.getContext('2d');
  ctx.drawImage(liveVideo,0,0);
  captureCanvas.toBlob(blob=>{
    if(!blob) return;
    console.log('sending frame to analyze');
    const form = new FormData();
    form.append('frame', blob, 'frame.jpg');
    fetch('/analyze',{method:'POST',body:form})
      .then(r=>r.json())
      .then(json=>{console.log('analyze response',json); updateDetectionsUI(json)})
      .catch(err=>{console.error('analyze error',err)});
  },'image/jpeg',0.7);
}

function startMobileCamera(){
  if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia) return;
  clearInterval(pollInterval);
  navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}})
    .then(stream=>{
      usingMobileCamera=true;
      liveVideo.srcObject=stream;
      liveVideo.style.display='block';
      liveStream.style.display='none';
      toggleStream.style.display='none';
      streamType.textContent='camera';
      liveVideo.onloadedmetadata = () => {
        captureAndSendFrame();
        analyzeInterval=setInterval(captureAndSendFrame,1000);
      };
    })
    .catch(e=>console.warn('camera access failed',e));
}

// default polling for desktop
pollInterval=setInterval(()=>{
  fetch('/detections').then(r=>r.json()).then(updateDetectionsUI).catch(()=>{});
},800);

// initialize stream
if(/Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent)){
  startMobileCamera();
  streamType.textContent='camera';
} else {
  streamType.textContent='MJPEG';
}

let streamPaused=false;
toggleStream.addEventListener('click',function(){
  streamPaused=!streamPaused;
  if(streamPaused){
    if(usingMobileCamera) liveVideo.pause(); else liveStream.src='';
    this.textContent='Resume Stream';
  } else {
    if(usingMobileCamera) liveVideo.play(); else liveStream.src='/video_feed';
    this.textContent='Pause Stream';
  }
});
