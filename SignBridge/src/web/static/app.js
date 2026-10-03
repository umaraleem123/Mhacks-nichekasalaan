"use strict";

const CAPTURE_WIDTH = 640;
const JPEG_QUALITY = 0.75;
const STABLE_FRAMES = 6;
const UNKNOWN = "unknown";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);

const $ = (id) => document.getElementById(id);
const el = {
  video: $("video"), overlay: $("overlay"), placeholder: $("placeholder"),
  startBtn: $("start-btn"), stopBtn: $("stop-btn"), cameraSelect: $("camera-select"),
  cameraError: $("camera-error"), hud: $("hud"), hudFps: $("hud-fps"), hudHand: $("hud-hand"),
  banner: $("sign-banner"), bannerSign: $("banner-sign"), bigSign: $("big-sign"),
  meterFill: $("meter-fill"), meterMark: $("meter-mark"), confidenceText: $("confidence-text"),
  latency: $("latency"), pill: $("model-pill"), pillText: $("model-pill-text"),
  noModel: $("no-model"), reloadBtn: $("reload-btn"), chips: $("chips"), classCount: $("class-count"),
  transcript: $("transcript"), clearBtn: $("clear-btn"), speakToggle: $("speak-toggle"),
  threshold: $("threshold"), thresholdText: $("threshold-text"),
};

const state = {
  stream: null,
  running: false,
  modelLoaded: false,
  classes: [],
  frameTimes: [],
  candidate: null,
  candidateCount: 0,
  lastCommitted: null,
  words: [],
};

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");
const overlayCtx = el.overlay.getContext("2d");

// ---------- Model status ----------

function applyStatus(status) {
  state.modelLoaded = status.model_loaded;
  state.classes = status.classes;

  el.pill.className = "pill " + (status.model_loaded ? "pill-good" : "pill-warn");
  el.pillText.textContent = status.model_loaded
    ? `Model ready · ${status.classes.length} signs`
    : "Tracking only · no model";
  el.noModel.hidden = status.model_loaded;

  el.chips.replaceChildren(...status.classes.map((name) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.dataset.sign = name;
    chip.textContent = name;
    return chip;
  }));
  el.classCount.textContent = status.model_loaded ? `${status.classes.length} trained` : "planned";

  if (!el.threshold.dataset.ready) {
    el.threshold.value = status.default_threshold;
    el.threshold.dataset.ready = "1";
    updateThresholdLabel();
  }
}

async function loadStatus(path = "/api/status", method = "GET") {
  try {
    const response = await fetch(path, { method });
    applyStatus(await response.json());
  } catch {
    el.pill.className = "pill pill-bad";
    el.pillText.textContent = "Server unreachable";
  }
}

function updateThresholdLabel() {
  const value = Number(el.threshold.value);
  el.thresholdText.textContent = `${Math.round(value * 100)}%`;
  el.meterMark.style.left = `${value * 100}%`;
}

// ---------- Camera ----------

async function listCameras(activeId) {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  el.cameraSelect.replaceChildren(...devices.map((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    option.selected = device.deviceId === activeId;
    return option;
  }));
  el.cameraSelect.disabled = devices.length < 2;
  return devices;
}

async function startCamera(deviceId) {
  el.cameraError.textContent = "";
  stopCamera();
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (deviceId) video.deviceId = { exact: deviceId };

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  } catch (error) {
    el.cameraError.textContent = error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow it in the address bar and try again."
      : `Could not open the camera: ${error.message}`;
    return;
  }

  const activeId = state.stream.getVideoTracks()[0].getSettings().deviceId;
  const devices = await listCameras(activeId);

  // Same preference as the desktop app: use a Logitech Brio when one is connected.
  if (!deviceId) {
    const brio = devices.find((d) => /brio/i.test(d.label));
    if (brio && brio.deviceId !== activeId) return startCamera(brio.deviceId);
  }

  el.video.srcObject = state.stream;
  await el.video.play();

  el.placeholder.hidden = true;
  el.hud.hidden = false;
  el.banner.hidden = false;
  el.stopBtn.disabled = false;
  state.running = true;
  state.frameTimes = [];
  loop();
}

function stopCamera() {
  state.running = false;
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  el.video.srcObject = null;
  overlayCtx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  el.placeholder.hidden = false;
  el.hud.hidden = true;
  el.banner.hidden = true;
  el.stopBtn.disabled = true;
  showPrediction({ hands: [], sign: UNKNOWN, confidence: 0 }, true);
}

// ---------- Frame loop ----------

function grabFrame() {
  const scale = Math.min(1, CAPTURE_WIDTH / el.video.videoWidth);
  captureCanvas.width = Math.round(el.video.videoWidth * scale);
  captureCanvas.height = Math.round(el.video.videoHeight * scale);
  captureCtx.drawImage(el.video, 0, 0, captureCanvas.width, captureCanvas.height);
  return new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
}

async function loop() {
  while (state.running) {
    const started = performance.now();
    try {
      const blob = await grabFrame();
      const response = await fetch(`/api/frame?threshold=${el.threshold.value}`, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      const result = await response.json();
      if (!state.running) break;
      if (!response.ok) throw new Error(result.error || response.statusText);

      if (result.model_loaded !== state.modelLoaded) loadStatus();
      drawHands(result.hands);
      showPrediction(result);
      el.latency.textContent = `${Math.round(performance.now() - started)} ms`;
    } catch (error) {
      el.pill.className = "pill pill-bad";
      el.pillText.textContent = "Server error";
      console.error(error);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    tickFps();
  }
}

function tickFps() {
  const now = performance.now();
  state.frameTimes.push(now);
  while (state.frameTimes.length && now - state.frameTimes[0] > 1000) state.frameTimes.shift();
  el.hudFps.textContent = `${state.frameTimes.length} FPS`;
}

// ---------- Rendering ----------

function drawHands(hands) {
  // The stream can renegotiate its resolution after it starts; the overlay
  // must match it exactly or object-fit crops the two layers differently.
  if (el.overlay.width !== el.video.videoWidth || el.overlay.height !== el.video.videoHeight) {
    el.overlay.width = el.video.videoWidth;
    el.overlay.height = el.video.videoHeight;
  }
  const { width, height } = el.overlay;
  overlayCtx.clearRect(0, 0, width, height);
  const lineWidth = Math.max(2, width / 260);

  for (const hand of hands) {
    const pts = hand.points.map(([x, y]) => [x * width, y * height]);

    const gradient = overlayCtx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#7c8cff");
    gradient.addColorStop(1, "#3dd6c6");
    overlayCtx.strokeStyle = gradient;
    overlayCtx.lineWidth = lineWidth;
    overlayCtx.lineCap = "round";
    overlayCtx.shadowColor = "rgba(124, 140, 255, 0.8)";
    overlayCtx.shadowBlur = 12;
    overlayCtx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      overlayCtx.moveTo(...pts[a]);
      overlayCtx.lineTo(...pts[b]);
    }
    overlayCtx.stroke();

    overlayCtx.shadowBlur = 0;
    pts.forEach(([x, y], index) => {
      const radius = lineWidth * (FINGERTIPS.has(index) ? 2.2 : 1.5);
      overlayCtx.beginPath();
      overlayCtx.arc(x, y, radius, 0, Math.PI * 2);
      overlayCtx.fillStyle = FINGERTIPS.has(index) ? "#3dd6c6" : "#ffffff";
      overlayCtx.fill();
    });
  }
}

function showPrediction({ hands, sign, confidence }, idle = false) {
  const hasHand = hands.length > 0;
  const recognized = hasHand && sign !== UNKNOWN;

  let label;
  if (idle) label = "Waiting";
  else if (!hasHand) label = "No hand";
  else if (!state.modelLoaded) label = "Tracking";
  else label = recognized ? sign : "Unknown";

  el.bigSign.textContent = label;
  el.bigSign.className = "big-sign" + (recognized ? " recognized" : !hasHand ? " idle" : "");
  el.bannerSign.textContent = label;
  el.banner.classList.toggle("recognized", recognized);

  const shown = hasHand && state.modelLoaded ? confidence : 0;
  el.meterFill.style.width = `${Math.round(shown * 100)}%`;
  el.confidenceText.textContent = `${Math.round(shown * 100)}%`;

  el.hudHand.textContent = hasHand
    ? hands.map((h) => `${h.label} hand`).join(" · ")
    : "No hand";

  for (const chip of el.chips.children) {
    chip.classList.toggle("active", recognized && chip.dataset.sign === sign);
  }

  trackTranscript(recognized ? sign : null);
}

// ---------- Transcript ----------

function trackTranscript(sign) {
  if (sign === state.candidate) {
    state.candidateCount += 1;
  } else {
    state.candidate = sign;
    state.candidateCount = 1;
  }

  // A sign is committed once it holds steady; lowering the hand re-arms it.
  if (sign === null && state.candidateCount >= STABLE_FRAMES) state.lastCommitted = null;
  if (sign && state.candidateCount === STABLE_FRAMES && sign !== state.lastCommitted) {
    state.lastCommitted = sign;
    addWord(sign);
  }
}

function addWord(sign) {
  if (!state.words.length) el.transcript.replaceChildren();
  state.words.push(sign);
  const word = document.createElement("span");
  word.className = "word";
  word.textContent = sign;
  el.transcript.appendChild(word);
  el.transcript.scrollTop = el.transcript.scrollHeight;

  if (el.speakToggle.checked && "speechSynthesis" in window) {
    speechSynthesis.speak(new SpeechSynthesisUtterance(sign));
  }
}

function clearTranscript() {
  state.words = [];
  state.lastCommitted = null;
  const empty = document.createElement("span");
  empty.className = "muted";
  empty.textContent = "Recognized signs will appear here.";
  el.transcript.replaceChildren(empty);
}

// ---------- Wiring ----------

el.startBtn.addEventListener("click", () => startCamera());
el.stopBtn.addEventListener("click", stopCamera);
el.cameraSelect.addEventListener("change", () => startCamera(el.cameraSelect.value));
el.reloadBtn.addEventListener("click", () => loadStatus("/api/reload", "POST"));
el.clearBtn.addEventListener("click", clearTranscript);
el.threshold.addEventListener("input", updateThresholdLabel);

if (!navigator.mediaDevices?.getUserMedia) {
  el.startBtn.disabled = true;
  el.cameraError.textContent = "This browser cannot access a webcam. Open the page at http://127.0.0.1:5000.";
}

loadStatus();
