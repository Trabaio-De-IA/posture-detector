import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// Default thresholds — used as fallback when config sliders are absent
const DEFAULTS = {
  neckThreshold:  30.0,
  torsoThreshold: 12,
  headFwd:        0.7,
  headDown:       0.3,
  badFrames:      15,
  alertCooldown:  30,   // seconds
};

const BAD_SHOULDER_TILT_THRESHOLD = 8.0;
const TARGET_INTERVAL_MS = 1000 / 12; // 12 FPS cap

const ALERT_MESSAGES = [
  "Por favor, corrija sua postura! Você está curvado.",
  "Atenção! Sua postura está incorreta. Sente-se ereto.",
  "Lembrete de postura: endireite as costas e levante a cabeça.",
  "Cuide da sua saúde! Corrija sua postura agora.",
];

const SKELETON_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [7, 11],  [8, 12],
];

// DOM
const videoEl       = document.getElementById("webcam");
const canvasEl      = document.getElementById("output-canvas");
const ctx           = canvasEl.getContext("2d");
const badgeEl       = document.getElementById("posture-badge");
const badgeTitleEl  = document.getElementById("badge-title");
const badgeSubEl    = document.getElementById("badge-sub");
const neckEl        = document.getElementById("neck-angle");
const torsoEl       = document.getElementById("torso-angle");
const alertInfoEl   = document.getElementById("alert-info");
const statusMsgEl   = document.getElementById("status-msg");
const loadingEl     = document.getElementById("loading-overlay");
const sessionScoreEl = document.getElementById("session-score");

// Session data for k-means
const sessionData  = [];
const SAMPLE_EVERY = 12;
const KMEANS_K     = 3;
const KMEANS_MIN_PTS = 15;
let frameCount    = 0;
let lastKmeansTime = 0;

// Detection state
let badCount      = 0;
let lastAlertTime = 0;
let alertMsgIdx   = 0;
let lastFrameTime = 0;

// ── Helper: read config slider value with fallback ────────────────────────────
function cfg(id, fallback) {
  const el = document.getElementById(id);
  return el ? +el.value : fallback;
}

// ── K-Means (pure JS, 2D) ─────────────────────────────────────────────────────
function kmeans(points, k, maxIter = 60) {
  const step = Math.floor(points.length / k);
  let centroids = Array.from({ length: k }, (_, i) => [...points[i * step]]);
  let assignments = new Array(points.length).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    for (let i = 0; i < points.length; i++) {
      let best = 0, bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        const d = Math.hypot(
          points[i][0] - centroids[c][0],
          points[i][1] - centroids[c][1],
        );
        if (d < bestDist) { bestDist = d; best = c; }
      }
      if (assignments[i] !== best) { assignments[i] = best; changed = true; }
    }
    if (!changed) break;

    const sums = Array.from({ length: k }, () => [0, 0, 0]);
    for (let i = 0; i < points.length; i++) {
      sums[assignments[i]][0] += points[i][0];
      sums[assignments[i]][1] += points[i][1];
      sums[assignments[i]][2]++;
    }
    for (let c = 0; c < k; c++) {
      const n = sums[c][2] || 1;
      centroids[c] = [sums[c][0] / n, sums[c][1] / n];
    }
  }

  const counts = new Array(k).fill(0);
  for (const a of assignments) counts[a]++;
  return { centroids, assignments, counts };
}

function clusterLabel(centroid) {
  const [neck, torso] = centroid;
  if (neck < 20 && torso < 5)                        return "Ereta";
  if (neck > DEFAULTS.neckThreshold)                 return "Pescoço Inclinado";
  if (torso > DEFAULTS.torsoThreshold)               return "Tronco Curvado";
  return "Intermediária";
}

function runKmeans() {
  if (sessionData.length < KMEANS_MIN_PTS) return null;
  return kmeans(sessionData, KMEANS_K);
}

// ── Geometry helpers ──────────────────────────────────────────────────────────
function verticalAngle(base, tip) {
  return Math.atan2(Math.abs(tip.x - base.x), Math.abs(tip.y - base.y)) * (180 / Math.PI);
}

function horizontalAngle(leftPoint, rightPoint) {
  return Math.atan2(
    Math.abs(leftPoint.y - rightPoint.y),
    Math.abs(leftPoint.x - rightPoint.x),
  ) * (180 / Math.PI);
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

// ── Audio ─────────────────────────────────────────────────────────────────────
function speak(msg) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(msg);
  utter.lang = "pt-BR";
  utter.rate = 0.9;
  window.speechSynthesis.speak(utter);
}

// ── Render helpers ────────────────────────────────────────────────────────────
function drawSkeleton(landmarks, w, h, isBad) {
  ctx.strokeStyle = isBad ? "#ff1744" : "#00d4ff";
  ctx.lineWidth   = 2;
  for (const [a, b] of SKELETON_CONNECTIONS) {
    const la = landmarks[a], lb = landmarks[b];
    if (!la || !lb) continue;
    ctx.beginPath();
    ctx.moveTo(la.x * w, la.y * h);
    ctx.lineTo(lb.x * w, lb.y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// ── UI updaters ───────────────────────────────────────────────────────────────
const BADGE_LABELS = {
  good:      ["Boa Postura",   "Coluna alinhada — continue assim"],
  warn:      ["Atenção",       "Você está quase saindo do alinhamento"],
  bad:       ["Postura Ruim",  "Reajuste-se: pescoço e tronco inclinados"],
  analyzing: ["Analisando…",   "Posicione-se em frente à câmera"],
};

function updateBadge(state) {
  if (badgeEl)      badgeEl.dataset.state = state === "analyzing" ? "warn" : state;
  const labels = BADGE_LABELS[state] || BADGE_LABELS.warn;
  if (badgeTitleEl) badgeTitleEl.textContent = labels[0];
  if (badgeSubEl)   badgeSubEl.textContent   = labels[1];
}

function updateGauge(barId, ratio) {
  const bar = document.getElementById(barId);
  if (!bar) return;
  bar.style.width = Math.min(100, Math.max(0, ratio * 100)) + "%";
}

function updateMetricState(boxId, ratio) {
  const box = document.getElementById(boxId);
  if (!box) return;
  box.dataset.state = ratio < 0.6 ? "good" : ratio < 0.85 ? "warn" : "bad";
}

function updateKmeansPanel(result) {
  const textEl = document.getElementById("kmeans-result");
  const vizEl  = document.getElementById("kmeans-viz");

  if (!result) {
    if (textEl) textEl.textContent = `Coletando dados… (${sessionData.length}/${KMEANS_MIN_PTS})`;
    return;
  }

  // Classify each cluster into good / warn / bad
  const classified = result.centroids.map((c, i) => {
    const label   = clusterLabel(c);
    const pct     = Math.round((result.counts[i] / sessionData.length) * 100);
    const quality = label === "Ereta" ? "good" : label === "Intermediária" ? "warn" : "bad";
    return { quality, label, pct, count: result.counts[i] };
  });

  // Accumulate percentages per quality bucket
  const qData = {
    good: { pct: 0, label: "Ergonômica" },
    warn: { pct: 0, label: "Inclinação leve" },
    bad:  { pct: 0, label: "Curvatura acentuada" },
  };
  for (const c of classified) {
    qData[c.quality].pct   += c.pct;
    qData[c.quality].label  = c.label;
  }

  // Resize bubbles proportionally to frequency
  const BASE = { good: 96, warn: 74, bad: 56 };
  if (vizEl) {
    ["good", "warn", "bad"].forEach(q => {
      const bubble = vizEl.querySelector(`.bubble[data-q="${q}"]`);
      if (!bubble) return;
      const scale = 0.4 + (qData[q].pct / 100) * 0.9;
      const size  = Math.round(BASE[q] * scale);
      bubble.style.width  = size + "px";
      bubble.style.height = size + "px";
      const pctEl = bubble.querySelector(".pct");
      if (pctEl) pctEl.textContent = qData[q].pct + "%";
    });
  }

  // Text summary
  const dominant = classified.reduce((a, b) => a.count > b.count ? a : b);
  if (textEl) {
    textEl.innerHTML = `Cluster dominante: <b>${dominant.label}</b> (${Math.round(dominant.count / sessionData.length * 100)}% da sessão · ${sessionData.length} amostras).`;
  }
}

// ── Main detection loop ───────────────────────────────────────────────────────
function detectLoop(poseLandmarker, timestamp) {
  requestAnimationFrame((ts) => detectLoop(poseLandmarker, ts));

  if (videoEl.readyState < 2) return;
  if (timestamp - lastFrameTime < TARGET_INTERVAL_MS) return;
  lastFrameTime = timestamp;
  frameCount++;

  // Read dynamic thresholds from config sliders
  const badNeckThr      = cfg("cfg-neck-threshold",  DEFAULTS.neckThreshold);
  const badTorsoThr     = cfg("cfg-torso-threshold", DEFAULTS.torsoThreshold);
  const badHeadFwd      = cfg("cfg-head-fwd",        DEFAULTS.headFwd);
  const badHeadDown     = cfg("cfg-head-down",       DEFAULTS.headDown);
  const consecBadFrames = cfg("cfg-bad-frames",      DEFAULTS.badFrames);
  const alertCooldownMs = cfg("cfg-alert-cooldown",  DEFAULTS.alertCooldown) * 1000;

  const w = videoEl.videoWidth  || 640;
  const h = videoEl.videoHeight || 480;
  canvasEl.width  = w;
  canvasEl.height = h;
  ctx.drawImage(videoEl, 0, 0, w, h);

  const result = poseLandmarker.detectForVideo(videoEl, timestamp);

  if (!result.landmarks || result.landmarks.length === 0) {
    ctx.font      = "16px monospace";
    ctx.fillStyle = "#888";
    ctx.fillText("Nenhuma pessoa detectada", 12, 28);
    updateBadge("analyzing");
    neckEl.textContent  = "--";
    torsoEl.textContent = "--";
    const headFwdEl  = document.getElementById("head-fwd");
    const headDownEl = document.getElementById("head-down");
    if (headFwdEl)  headFwdEl.textContent  = "--";
    if (headDownEl) headDownEl.textContent = "--";
    updateGauge("neck-bar",     0);
    updateGauge("torso-bar",    0);
    updateGauge("head-fwd-bar", 0);
    updateGauge("head-down-bar",0);
    badCount = 0;
    return;
  }

  const lm = result.landmarks[0];

  const shoulderMid = midpoint(lm[11], lm[12]);
  const hipMid      = midpoint(lm[23], lm[24]);
  const earMid      = midpoint(lm[7],  lm[8]);

  // Shoulder tilt
  const shoulderTiltAngle = horizontalAngle(lm[11], lm[12]);
  let shoulderTiltSide = "Alinhado";
  if (shoulderTiltAngle > BAD_SHOULDER_TILT_THRESHOLD) {
    shoulderTiltSide = lm[11].y > lm[12].y
      ? "Ombro esquerdo mais baixo"
      : "Ombro direito mais baixo";
  }

  // Lateral angles
  const neckAngle  = verticalAngle(shoulderMid, earMid);
  const torsoAngle = verticalAngle(hipMid, shoulderMid);

  // Forward head (Z depth)
  const shoulderWidth = Math.max(
    Math.hypot(lm[11].x - lm[12].x, lm[11].y - lm[12].y), 0.01,
  );
  const headZ     = (lm[7].z + lm[8].z) / 2;
  const shoulderZ = (lm[11].z + lm[12].z) / 2;
  const headFwd   = (shoulderZ - headZ) / shoulderWidth;

  // Head pitch (down)
  const eyeMid  = midpoint(lm[2], lm[5]);
  const headDown = (lm[0].y - eyeMid.y) / shoulderWidth;

  // Update numeric displays
  neckEl.textContent  = neckAngle.toFixed(1);
  torsoEl.textContent = torsoAngle.toFixed(1);
  const headFwdEl  = document.getElementById("head-fwd");
  const headDownEl = document.getElementById("head-down");
  if (headFwdEl)  headFwdEl.textContent  = headFwd.toFixed(2);
  if (headDownEl) headDownEl.textContent = headDown.toFixed(2);

  // Compute ratios (1.0 = at threshold)
  const neckRatio  = neckAngle  / badNeckThr;
  const torsoRatio = torsoAngle / badTorsoThr;
  const fwdRatio   = headFwd    / badHeadFwd;
  const downRatio  = headDown   / badHeadDown;

  // Update gauges and card states
  updateGauge("neck-bar",      neckRatio);
  updateGauge("torso-bar",     torsoRatio);
  updateGauge("head-fwd-bar",  fwdRatio);
  updateGauge("head-down-bar", downRatio);
  updateMetricState("m-neck",  neckRatio);
  updateMetricState("m-torso", torsoRatio);
  updateMetricState("m-fwd",   fwdRatio);
  updateMetricState("m-down",  downRatio);

  // Posture decision
  const isShoulderTiltBad = shoulderTiltAngle > BAD_SHOULDER_TILT_THRESHOLD;
  const isBad =
    neckAngle  > badNeckThr   ||
    torsoAngle > badTorsoThr  ||
    headFwd    > badHeadFwd   ||
    headDown   > badHeadDown  ||
    isShoulderTiltBad;
  badCount = isBad ? badCount + 1 : 0;

  const worstRatio    = Math.max(neckRatio, torsoRatio, fwdRatio, downRatio);
  const isConfirmedBad = badCount >= consecBadFrames;
  const badgeState    = isConfirmedBad ? "bad" : worstRatio > 0.7 ? "warn" : "good";

  drawSkeleton(lm, w, h, isConfirmedBad);
  updateBadge(badgeState);

  // Session score and chart
  const score = Math.max(0, Math.min(100, Math.round(100 - worstRatio * 60)));
  if (sessionScoreEl) {
    sessionScoreEl.textContent = score + "%";
    sessionScoreEl.style.color =
      score >= 66 ? "var(--good)" : score >= 40 ? "var(--warn)" : "var(--bad)";
  }
  if (window.PostureChart) window.PostureChart.push(score);

  // Shoulder tilt canvas overlay
  if (isShoulderTiltBad) {
    ctx.font      = "16px monospace";
    ctx.fillStyle = "#ff1744";
    ctx.fillText(`Ombros inclinados: ${shoulderTiltAngle.toFixed(1)}°`, 12, 52);
    ctx.fillText(shoulderTiltSide, 12, 74);
  }

  // Voice alert
  if (isConfirmedBad) {
    const now = Date.now();
    if (now - lastAlertTime > alertCooldownMs) {
      lastAlertTime = now;
      const msg = isShoulderTiltBad
        ? `Atenção! ${shoulderTiltSide}. Alinhe os ombros.`
        : ALERT_MESSAGES[alertMsgIdx++ % ALERT_MESSAGES.length];
      speak(msg);
      if (alertInfoEl) {
        alertInfoEl.classList.add("active");
        setTimeout(() => alertInfoEl.classList.remove("active"), 5000);
      }
    }
  }

  // K-Means data collection
  if (frameCount % SAMPLE_EVERY === 0) {
    sessionData.push([neckAngle, torsoAngle]);
  }

  const nowMs = Date.now();
  if (nowMs - lastKmeansTime > 5000) {
    lastKmeansTime = nowMs;
    updateKmeansPanel(runKmeans());
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
function setStatus(msg) {
  if (statusMsgEl) statusMsgEl.textContent = msg;
}

async function init() {
  setStatus("Carregando modelo MediaPipe...");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );

  const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  setStatus("Acessando câmera...");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
  });
  videoEl.srcObject = stream;
  await new Promise((resolve) => { videoEl.onloadedmetadata = resolve; });
  await videoEl.play();

  if (loadingEl) loadingEl.classList.add("hidden");

  updateKmeansPanel(null);

  requestAnimationFrame((ts) => detectLoop(poseLandmarker, ts));
}

init().catch((err) => {
  console.error(err);
  setStatus("Erro: " + err.message);
});
