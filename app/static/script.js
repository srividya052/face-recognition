// Interactive Frontend Logic for Face Recognition Demo
document.addEventListener("DOMContentLoaded", () => {
  // --- Tab Navigation ---
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // --- 1:1 Verification Dropzones ---
  let file1 = null;
  let file2 = null;

  const setupDropzone = (dropAreaId, fileInputId, previewId, placeholderId, onFileLoaded) => {
    const dropArea = document.getElementById(dropAreaId);
    const fileInput = document.getElementById(fileInputId);
    const preview = document.getElementById(previewId);
    const placeholder = document.getElementById(placeholderId);

    const handleFile = (file) => {
      if (!file || !file.type.startsWith("image/")) return;
      onFileLoaded(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove("hidden");
        placeholder.classList.add("hidden");
      };
      reader.readAsDataURL(file);
      checkVerifyReady();
    };

    dropArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
      dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (e) => {
      e.preventDefault();
      dropArea.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length) {
        handleFile(e.target.files[0]);
      }
    });
  };

  const btnVerify = document.getElementById("btn-verify");
  const checkVerifyReady = () => {
    btnVerify.disabled = !(file1 && file2);
  };

  setupDropzone("drop-area-1", "file-input-1", "preview-1", "placeholder-1", (f) => file1 = f);
  setupDropzone("drop-area-2", "file-input-2", "preview-2", "placeholder-2", (f) => file2 = f);

  // --- Load Demo Faces ---
  const btnSample = document.getElementById("btn-sample-pair");
  btnSample.addEventListener("click", async () => {
    btnSample.disabled = true;
    btnSample.textContent = "Loading Demo...";
    try {
      const res = await fetch("/api/gallery");
      const data = await res.json();
      if (data.identities && data.identities.length >= 2) {
        const item1 = data.identities[0];
        const item2 = data.identities[1];

        // Fetch thumbnails as blobs
        const blob1 = await (await fetch(item1.thumb)).blob();
        const blob2 = await (await fetch(item2.thumb)).blob();

        file1 = new File([blob1], "sample1.jpg", { type: "image/jpeg" });
        file2 = new File([blob2], "sample2.jpg", { type: "image/jpeg" });

        document.getElementById("preview-1").src = item1.thumb;
        document.getElementById("preview-1").classList.remove("hidden");
        document.getElementById("placeholder-1").classList.add("hidden");

        document.getElementById("preview-2").src = item2.thumb;
        document.getElementById("preview-2").classList.remove("hidden");
        document.getElementById("placeholder-2").classList.add("hidden");

        checkVerifyReady();
      }
    } catch (e) {
      console.error("Failed to load demo faces:", e);
    } finally {
      btnSample.disabled = false;
      btnSample.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg> Load Demo Faces`;
    }
  });

  // --- Verify Form Submission ---
  const verifyForm = document.getElementById("verify-form");
  const verifyResults = document.getElementById("verify-results");
  const resultBanner = document.getElementById("result-banner");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictDesc = document.getElementById("verdict-desc");
  const confidenceVal = document.getElementById("confidence-val");
  const statDist = document.getElementById("stat-dist");
  const statCos = document.getElementById("stat-cos");
  const statLatency = document.getElementById("stat-latency");
  const statDetection = document.getElementById("stat-detection");
  const cropImg1 = document.getElementById("crop-img-1");
  const cropImg2 = document.getElementById("crop-img-2");
  const distanceFill = document.getElementById("distance-fill");

  verifyForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!file1 || !file2) return;

    btnVerify.disabled = true;
    btnVerify.innerHTML = `<svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 10 10"/></svg> Processing...`;

    const formData = new FormData();
    formData.append("image1", file1);
    formData.append("image2", file2);

    try {
      const res = await fetch("/api/verify", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        verifyResults.classList.remove("hidden");

        const isMatch = data.is_match;
        resultBanner.className = `result-banner ${isMatch ? "match" : "mismatch"}`;
        verdictIcon.innerHTML = isMatch ? "&#10003;" : "&#10007;";
        verdictTitle.textContent = isMatch ? "SAME PERSON" : "DIFFERENT PEOPLE";
        verdictDesc.textContent = isMatch 
          ? `Euclidean distance (${data.distance}) is below the threshold (${data.threshold}).`
          : `Euclidean distance (${data.distance}) exceeds the decision threshold (${data.threshold}).`;

        confidenceVal.textContent = `${data.confidence_pct}%`;
        statDist.textContent = data.distance;
        statCos.textContent = data.cosine_similarity;
        statLatency.textContent = `${data.latency_ms} ms`;
        statDetection.textContent = (data.face1_detected && data.face2_detected) ? "100% MTCNN" : "Center Fallback";

        cropImg1.src = data.face1_crop;
        cropImg2.src = data.face2_crop;

        // Position distance fill on a 0.0 - 2.0 scale (0% - 100%)
        const fillPct = Math.min(100, (data.distance / 2.0) * 100);
        distanceFill.style.width = `${fillPct}%`;

        verifyResults.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else {
        alert(data.error || "Verification failed");
      }
    } catch (err) {
      console.error(err);
      alert("Inference request failed. Please check backend connection.");
    } finally {
      btnVerify.disabled = false;
      btnVerify.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 14 14"/></svg> <span>Compare Faces</span>`;
    }
  });

  // --- 1:N Identification ---
  let queryFile = null;
  const btnRunIdentify = document.getElementById("btn-run-identify");
  const identifyResultsBox = document.getElementById("identify-results-box");
  const matchesList = document.getElementById("matches-list");

  setupDropzone("drop-area-query", "file-input-query", "preview-query", "placeholder-query", (f) => {
    queryFile = f;
    btnRunIdentify.disabled = !queryFile;
  });

  btnRunIdentify.addEventListener("click", async () => {
    if (!queryFile) return;

    btnRunIdentify.disabled = true;
    btnRunIdentify.textContent = "Searching...";

    const formData = new FormData();
    formData.append("image", queryFile);

    try {
      const res = await fetch("/api/identify", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        identifyResultsBox.classList.remove("hidden");
        matchesList.innerHTML = "";

        if (data.found_face === false) {
          const noFaceBanner = document.createElement("div");
          noFaceBanner.className = "identify-status-banner banner-warning";
          noFaceBanner.innerHTML = `
            <div class="status-icon">⚠️</div>
            <div>
              <div class="status-title">No Face Detected</div>
              <div class="status-subtitle">MTCNN could not detect any face in this image. Please upload a clear photo of a person.</div>
            </div>
          `;
          matchesList.appendChild(noFaceBanner);
          return;
        }

        const best = data.best_match;
        const isKnown = best && best.is_match;

        const statusBanner = document.createElement("div");
        statusBanner.className = `identify-status-banner ${isKnown ? "banner-success" : "banner-unknown"}`;
        statusBanner.innerHTML = `
          <div class="status-icon">${isKnown ? "✅" : "❓"}</div>
          <div>
            <div class="status-title">${isKnown ? `Identified: ${best.name}` : "Unknown Person (Stranger)"}</div>
            <div class="status-subtitle">${isKnown 
              ? `Match found in gallery! Euclidean distance: ${best.distance} (Threshold: &lt; ${data.threshold})`
              : `Not enrolled in gallery. Closest distance: ${best ? best.distance : 'N/A'} (Threshold: &ge; ${data.threshold})`}</div>
          </div>
        `;
        matchesList.appendChild(statusBanner);

        const listTitle = document.createElement("div");
        listTitle.className = "candidates-header";
        listTitle.textContent = isKnown ? "Ranked Gallery Matches:" : "Closest Enrolled Candidates (All Rejected):";
        matchesList.appendChild(listTitle);

        if (data.top_matches && data.top_matches.length > 0) {
          data.top_matches.forEach((m, idx) => {
            const card = document.createElement("div");
            card.className = `match-card ${idx === 0 && m.is_match ? "is-top-match" : ""}`;
            card.innerHTML = `
              <div class="match-info">
                <img src="${m.thumb}" class="match-thumb" alt="${m.name}">
                <div>
                  <div class="match-name">${m.name} ${m.is_match ? '<span class="match-pill">MATCH</span>' : '<span class="nomatch-pill">NO MATCH</span>'}</div>
                  <div class="match-details">Dist: ${m.distance} &bull; Cos: ${m.similarity}</div>
                </div>
              </div>
              <div class="match-badge" style="color: ${m.is_match ? '#34d399' : '#f87171'}">
                ${m.confidence_pct}%
              </div>
            `;
            matchesList.appendChild(card);
          });
        } else {
          matchesList.innerHTML += `<p class="card-desc">No gallery entries available.</p>`;
        }
      }
    } catch (e) {
      console.error(e);
      alert("Identification failed.");
    } finally {
      btnRunIdentify.disabled = false;
      btnRunIdentify.textContent = "Search Gallery";
    }
  });

  // --- Quick 1-Click Assignment Tests ---
  document.querySelectorAll(".quick-test-pills .pill-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const filename = btn.dataset.img;
      try {
        btn.disabled = true;
        const res = await fetch(`/test_images/${filename}`);
        const blob = await res.blob();
        queryFile = new File([blob], filename, { type: blob.type || "image/jpeg" });

        const previewQuery = document.getElementById("preview-query");
        const placeholderQuery = document.getElementById("placeholder-query");
        previewQuery.src = `/test_images/${filename}`;
        previewQuery.classList.remove("hidden");
        placeholderQuery.classList.add("hidden");
        btnRunIdentify.disabled = false;

        // Automatically trigger search
        btnRunIdentify.click();
      } catch (err) {
        console.error("Failed to load sample test image:", err);
      } finally {
        btn.disabled = false;
      }
    });
  });

  // --- Gallery Display & Enrollment ---
  const galleryContainer = document.getElementById("gallery-container");
  const btnOpenEnroll = document.getElementById("btn-open-enroll");
  const enrollFormCard = document.getElementById("enroll-form-card");
  const btnCancelEnroll = document.getElementById("btn-cancel-enroll");
  const enrollForm = document.getElementById("enroll-form");

  const loadGallery = async () => {
    try {
      const res = await fetch("/api/gallery");
      const data = await res.json();
      galleryContainer.innerHTML = "";
      if (data.identities) {
        data.identities.forEach(item => {
          const div = document.createElement("div");
          div.className = "gallery-item";
          div.innerHTML = `
            <img src="${item.thumb}" class="gallery-thumb" alt="${item.name}">
            <span class="gallery-name" title="${item.name}">${item.name}</span>
          `;
          galleryContainer.appendChild(div);
        });
      }
    } catch (e) {
      console.error(e);
    }
  };

  loadGallery();

  btnOpenEnroll.addEventListener("click", () => {
    enrollFormCard.classList.toggle("hidden");
  });

  btnCancelEnroll.addEventListener("click", () => {
    enrollFormCard.classList.add("hidden");
  });

  enrollForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const nameInput = document.getElementById("enroll-name");
    const fileInput = document.getElementById("enroll-file");

    if (!nameInput.value.trim() || !fileInput.files.length) return;

    const formData = new FormData();
    formData.append("name", nameInput.value.trim());
    formData.append("image", fileInput.files[0]);

    try {
      const res = await fetch("/api/enroll", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        alert(data.message);
        nameInput.value = "";
        fileInput.value = "";
        enrollFormCard.classList.add("hidden");
        loadGallery();
      }
    } catch (err) {
      console.error(err);
      alert("Failed to enroll identity.");
    }
  });

  // --- Live Camera Modal ---
  const modal = document.getElementById("camera-modal");
  const video = document.getElementById("webcam-video");
  const btnCloseCam = document.getElementById("btn-close-cam");
  const btnCancelCam = document.getElementById("btn-cancel-cam");
  const btnSnapCam = document.getElementById("btn-snap-cam");

  let activeCamTarget = null;
  let mediaStream = null;

  const startCamera = async (target) => {
    activeCamTarget = target;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = mediaStream;
      modal.showModal();
    } catch (err) {
      alert("Camera access denied or unavailable.");
    }
  };

  const stopCamera = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    modal.close();
  };

  document.getElementById("cam-btn-1")?.addEventListener("click", () => startCamera(1));
  document.getElementById("cam-btn-2")?.addEventListener("click", () => startCamera(2));
  document.getElementById("btn-cam-query")?.addEventListener("click", () => startCamera("query"));

  btnCloseCam.addEventListener("click", stopCamera);
  btnCancelCam.addEventListener("click", stopCamera);

  btnSnapCam.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const snapFile = new File([blob], `snap_${Date.now()}.jpg`, { type: "image/jpeg" });
      const dataUrl = canvas.toDataURL("image/jpeg");

      if (activeCamTarget === 1) {
        file1 = snapFile;
        document.getElementById("preview-1").src = dataUrl;
        document.getElementById("preview-1").classList.remove("hidden");
        document.getElementById("placeholder-1").classList.add("hidden");
      } else if (activeCamTarget === 2) {
        file2 = snapFile;
        document.getElementById("preview-2").src = dataUrl;
        document.getElementById("preview-2").classList.remove("hidden");
        document.getElementById("placeholder-2").classList.add("hidden");
      } else if (activeCamTarget === "query") {
        queryFile = snapFile;
        document.getElementById("preview-query").src = dataUrl;
        document.getElementById("preview-query").classList.remove("hidden");
        document.getElementById("placeholder-query").classList.add("hidden");
        btnRunIdentify.disabled = false;
      }
      checkVerifyReady();
      stopCamera();
    }, "image/jpeg", 0.9);
  });

  // --- Dynamic Metrics Loading ---
  const loadMetrics = async () => {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      if (data.current_metrics) {
        const m = data.current_metrics;
        document.getElementById("kpi-accuracy").textContent = `${(m.accuracy_mean * 100).toFixed(2)}%`;
        document.getElementById("kpi-auc").textContent = m.AUC.toFixed(4);
        document.getElementById("kpi-far").textContent = `${(m.FAR_mean * 100).toFixed(2)}%`;
        document.getElementById("kpi-frr").textContent = `${(m.FRR_mean * 100).toFixed(2)}%`;
        document.getElementById("kpi-threshold").textContent = m.threshold_mean.toFixed(2);
      }
    } catch (e) {
      console.error("Failed to load metrics:", e);
    }
  };

  loadMetrics();
});
