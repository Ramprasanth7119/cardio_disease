// =============================
// Sample Patients (Placeholder Images)
// =============================
let samplePatients = [
  {label:"Young Adult", age:28, gender:1, height:175, weight:68, ap_hi:118, ap_lo:76, cholesterol:1, gluc:1, smoke:0, alco:0, active:1, image:"static/images/img-1.jpg"},
  {label:"Middle-aged", age:52, gender:1, height:170, weight:88, ap_hi:150, ap_lo:95, cholesterol:3, gluc:2, smoke:1, alco:1, active:0, image:"static/images/img-2.jpg"},
  {label:"Elderly", age:67, gender:2, height:160, weight:72, ap_hi:165, ap_lo:100, cholesterol:3, gluc:3, smoke:0, alco:0, active:0, image:"static/images/img-3.webp"}
];

// =============================
// Load Sample Patient Data
// =============================
function loadSampleData() {
  const sample = samplePatients[Math.floor(Math.random()*samplePatients.length)];

  // Fill form fields
  age.value=sample.age; gender.value=sample.gender; height.value=sample.height;
  weight.value=sample.weight; ap_hi.value=sample.ap_hi; ap_lo.value=sample.ap_lo;
  cholesterol.value=sample.cholesterol; gluc.value=sample.gluc;
  smoke.value=sample.smoke; alco.value=sample.alco; active.value=sample.active;

  // Show placeholder image
  const originalImg = document.getElementById("originalImg");
  originalImg.src = sample.image;

  // Hide Grad-CAM and feature maps until prediction
  document.getElementById("gradcamWrapper").classList.add("hidden");
  document.getElementById("featureMapsContainer").innerHTML = "";

  alert(`Loaded Sample: ${sample.label}`);
}

// =============================
// Predict Function (Text + Image)
// =============================
function predict() {
  const textData = {
    age:Number(age.value), gender:Number(gender.value), height:Number(height.value),
    weight:Number(weight.value), ap_hi:Number(ap_hi.value), ap_lo:Number(ap_lo.value),
    cholesterol:Number(cholesterol.value), gluc:Number(gluc.value),
    smoke:Number(smoke.value), alco:Number(alco.value), active:Number(active.value)
  };

  // ---- TEXT RISK PREDICTION ----
  fetch("http://127.0.0.1:5000/predict/text", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(textData)
  })
  .then(res=>res.json())
  .then(result=>{
    const box=document.getElementById("resultBox");
    box.classList.remove("hidden");

    // Risk Text
    document.getElementById("riskText").innerHTML=result.prediction=="High Risk"
      ? '<span class="high">High Cardiovascular Risk</span>'
      : '<span class="low">Low Cardiovascular Risk</span>';

    // Probability
    document.getElementById("probText").innerHTML=`Estimated Risk Probability: <b>${result.probability}</b>`;

    // SHAP / Risk Factors
    const up=document.getElementById("riskUp"); up.innerHTML="";
    const down=document.getElementById("riskDown"); down.innerHTML="";
    result.risk_increasing_factors.forEach(f=>up.innerHTML+=`<li>${f||"N/A"}</li>`);
    result.risk_decreasing_factors.forEach(f=>down.innerHTML+=`<li>${f||"N/A"}</li>`);
  })
  .catch(()=>alert("Clinical backend not running"));

  // ---- IMAGE PREDICTION ----
  const imgFile = document.getElementById("imageInput").files[0];
  let imageToSend = imgFile || samplePatients[0].image; // fallback to first sample
  sendImageToBackend(imageToSend);
}

function sendImageToBackend(file, filename) {
  const formData = new FormData();

  // If the file is a URL string, fetch the image as blob first
  if (typeof file === "string") {
    fetch(file)
      .then(res => {
        if (!res.ok) throw new Error("Cannot fetch image");
        return res.blob();
      })
      .then(blob => {
        formData.append("image", blob, filename || "sample.jpg");
        sendToServer(formData);
        // Preview
        document.getElementById("originalImg").src = URL.createObjectURL(blob);
      })
      .catch(err => console.error("Image fetch error:", err));
  } else {
    // It's already a File object
    formData.append("image", file, filename || "image.jpg");
    sendToServer(formData);
    // Preview
    document.getElementById("originalImg").src = URL.createObjectURL(file);
  }
}


function sendToServer(formData){
  // Send to Grad-CAM endpoint
  fetch("http://127.0.0.1:5000/predict/image", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(imgRes => {
    const gradcamImg = document.getElementById("gradcamImg");
    gradcamImg.src = "data:image/png;base64," + imgRes.gradcam_image;
    document.getElementById("gradcamWrapper").classList.remove("hidden");
  })
  .catch(err => console.error("Grad-CAM error:", err));

  // Send to Feature Maps endpoint
  fetch("http://127.0.0.1:5000/feature_maps", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    const container = document.getElementById("featureMapsContainer");
    container.innerHTML = "";
    data.feature_maps.forEach(b64 => {
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + b64;
      img.classList.add("feature-map");
      container.appendChild(img);
    });
  })
  .catch(err => console.error("Feature maps error:", err));
}

