import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// ── Thresholds para postura sentada ──────────────────────────────────────────

// XY lateral: detecta inclinação para o lado
const BAD_NECK_THRESHOLD = 30.0; // orelha→ombro vs. vertical > 20°
const BAD_TORSO_THRESHOLD = 12; // quadril→ombro vs. vertical > 6.5°
// Ombros desalinhados: detecta se um ombro está mais alto que o outro
// Mede o ângulo da linha entre ombro esquerdo e direito em relação à horizontal
const BAD_SHOULDER_TILT_THRESHOLD = 8.0; // > 8° indica inclinação lateral dos ombros

// Z profundidade normalizada: detecta curvatura para frente
// Quanto menor o valor, mais sensível fica para cabeça projetada à frente.
const BAD_HEAD_FWD = 0.7; // cabeça > 55% da largura dos ombros à frente dos ombros

// Cabeça baixa: mede quanto o nariz está abaixo dos olhos
// Unidade: fração da largura dos ombros
// Olhando reto para o monitor: ≈ 0.15–0.25
// Cabeça inclinada demais pra baixo: > 0.30
const BAD_HEAD_DOWN = 0.3;

// Quantidade de frames ruins seguidos antes de confirmar postura ruim
// Com 12 FPS, 10 frames ≈ 0.83 segundo
const CONSECUTIVE_BAD_FRAMES = 15;

const ALERT_COOLDOWN_MS = 30_000;
const TARGET_INTERVAL_MS = 1000 / 12; // 12 FPS cap

const ALERT_MESSAGES = [
  "Por favor, corrija sua postura! Você está curvado.",
  "Atenção! Sua postura está incorreta. Sente-se ereto.",
  "Lembrete de postura: endireite as costas e levante a cabeça.",
  "Cuide da sua saúde! Corrija sua postura agora.",
];

const SKELETON_CONNECTIONS = [
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [7, 11],
  [8, 12], // orelha → ombro (linha de referência postural)
];

// DOM
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("output-canvas");
const ctx = canvasEl.getContext("2d");
const badgeEl = document.getElementById("posture-badge");
const neckEl = document.getElementById("neck-angle");
const torsoEl = document.getElementById("torso-angle");
const alertInfoEl = document.getElementById("alert-info");
const statusMsgEl = document.getElementById("status-msg");
const loadingEl = document.getElementById("loading-overlay");

// Session data for k-means (unsupervised)
const sessionData = []; // [[neckAngle, torsoAngle], ...]
const SAMPLE_EVERY = 12; // collect one sample per ~1 second at 12 FPS
const KMEANS_K = 3;
const KMEANS_MIN_PTS = 15; // need at least 5 pts per cluster before running
let frameCount = 0;
let lastKmeansTime = 0;

// Detection state
let badCount = 0;
let lastAlertTime = 0;
let alertMsgIdx = 0;
let lastFrameTime = 0;

// ─── K-Means (pure JS, 2D) ──────────────────────────────────────────────────

function kmeans(points, k, maxIter = 60) {
  // Initialise centroids with k-means++ style: spread first picks
  const step = Math.floor(points.length / k);
  let centroids = Array.from({ length: k }, (_, i) => [...points[i * step]]);

  let assignments = new Array(points.length).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign
    let changed = false;
    for (let i = 0; i < points.length; i++) {
      let best = 0,
        bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        const d = Math.hypot(
          points[i][0] - centroids[c][0],
          points[i][1] - centroids[c][1],
        );
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      if (assignments[i] !== best) {
        assignments[i] = best;
        changed = true;
      }
    }
    if (!changed) break;

    // Recompute centroids
    const sums = Array.from({ length: k }, () => [0, 0, 0]); // [sx, sy, count]
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

  // Count members per cluster
  const counts = new Array(k).fill(0);
  for (const a of assignments) counts[a]++;

  return { centroids, assignments, counts };
}

function clusterLabel(centroid) {
  const [neck, torso] = centroid;
  if (neck < 20 && torso < 5) return "Ereta";
  if (neck > BAD_NECK_THRESHOLD) return "Pescoço Inclinado";
  if (torso > BAD_TORSO_THRESHOLD) return "Tronco Curvado";
  return "Intermediária";
}

function runKmeans() {
  if (sessionData.length < KMEANS_MIN_PTS) return null;
  return kmeans(sessionData, KMEANS_K);
}

// ─── Render helpers ─────────────────────────────────────────────────────────
function verticalAngle(base, tip) {
  return (
    Math.atan2(Math.abs(tip.x - base.x), Math.abs(tip.y - base.y)) *
    (180 / Math.PI)
  );
}

function horizontalAngle(leftPoint, rightPoint) {
  return (
    Math.atan2(
      Math.abs(leftPoint.y - rightPoint.y),
      Math.abs(leftPoint.x - rightPoint.x),
    ) *
    (180 / Math.PI)
  );
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

function speak(msg) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(msg);
  utter.lang = "pt-BR";
  utter.rate = 0.9;
  window.speechSynthesis.speak(utter);
}

function drawSkeleton(landmarks, w, h, isBad) {
  ctx.strokeStyle = isBad ? "#ff1744" : "#00d4ff";
  ctx.lineWidth = 2;
  for (const [a, b] of SKELETON_CONNECTIONS) {
    const la = landmarks[a],
      lb = landmarks[b];
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

function updateBadge(status) {
  badgeEl.className = "badge";
  if (status === "Boa") {
    badgeEl.classList.add("badge-good");
    badgeEl.textContent = "Boa";
  } else if (status === "Ruim") {
    badgeEl.classList.add("badge-bad");
    badgeEl.textContent = "Ruim";
  } else {
    badgeEl.classList.add("badge-analyzing");
    badgeEl.textContent = status;
  }
}

function updateKmeansPanel(result) {
  const el = document.getElementById("kmeans-result");
  if (!el) return;

  if (!result) {
    el.innerHTML = `<span class="kmeans-waiting">Coletando dados… (${sessionData.length}/${KMEANS_MIN_PTS})</span>`;
    return;
  }

  const clusterColors = ["#00d4ff", "#7b5ea7", "#00e676"];
  const rows = result.centroids
    .map((c, i) => {
      const label = clusterLabel(c);
      const pct = Math.round((result.counts[i] / sessionData.length) * 100);
      return `
      <div class="kmeans-cluster">
        <span class="kmeans-dot" style="background:${clusterColors[i]}"></span>
        <span class="kmeans-label">${label}</span>
        <span class="kmeans-bar-wrap">
          <span class="kmeans-bar" style="width:${pct}%;background:${clusterColors[i]}"></span>
        </span>
        <span class="kmeans-pct">${pct}%</span>
      </div>`;
    })
    .join("");

  el.innerHTML = `
    <div class="kmeans-header">K-Means K=3 &mdash; ${sessionData.length} amostras</div>
    ${rows}
  `;
}

// ─── Main detection loop ─────────────────────────────────────────────────────

function detectLoop(poseLandmarker, timestamp) {
  requestAnimationFrame((ts) => detectLoop(poseLandmarker, ts));

  if (videoEl.readyState < 2) return;
  if (timestamp - lastFrameTime < TARGET_INTERVAL_MS) return;
  lastFrameTime = timestamp;
  frameCount++;

  const w = videoEl.videoWidth || 640;
  const h = videoEl.videoHeight || 480;
  canvasEl.width = w;
  canvasEl.height = h;
  ctx.drawImage(videoEl, 0, 0, w, h);

  const result = poseLandmarker.detectForVideo(videoEl, timestamp);

  if (!result.landmarks || result.landmarks.length === 0) {
    ctx.font = "16px monospace";
    ctx.fillStyle = "#888";
    ctx.fillText("Nenhuma pessoa detectada", 12, 28);
    updateBadge("Analisando...");
    neckEl.textContent = "--";
    torsoEl.textContent = "--";

    const headFwdEl = document.getElementById("head-fwd");
    const headDownEl = document.getElementById("head-down");
    const shoulderTiltEl = document.getElementById("shoulder-tilt");

    if (headFwdEl) headFwdEl.textContent = "--";
    if (headDownEl) headDownEl.textContent = "--";
    if (shoulderTiltEl) shoulderTiltEl.textContent = "--";

    badCount = 0;
    return;
  }

  const lm = result.landmarks[0];

  const shoulderMid = midpoint(lm[11], lm[12]);
  const hipMid = midpoint(lm[23], lm[24]);
  const earMid = midpoint(lm[7], lm[8]);

  // ── Métrica extra: inclinação dos ombros ────────────────────────────────
  // lm[11] = ombro esquerdo
  // lm[12] = ombro direito
  // Como o eixo Y da tela cresce para baixo:
  // se lm[11].y > lm[12].y, o ombro esquerdo está mais baixo.
  const shoulderTiltAngle = horizontalAngle(lm[11], lm[12]);

  let shoulderTiltSide = "Alinhado";

  if (shoulderTiltAngle > BAD_SHOULDER_TILT_THRESHOLD) {
    if (lm[11].y > lm[12].y) {
      shoulderTiltSide = "Ombro esquerdo mais baixo";
    } else {
      shoulderTiltSide = "Ombro direito mais baixo";
    }
  }

  // ── Métricas XY: inclinação lateral ─────────────────────────────────────
  const neckAngle = verticalAngle(shoulderMid, earMid);
  const torsoAngle = verticalAngle(hipMid, shoulderMid);

  // ── Métrica Z: cabeça projetada para frente ─────────────────────────────
  const shoulderWidth = Math.max(
    Math.hypot(lm[11].x - lm[12].x, lm[11].y - lm[12].y),
    0.01,
  );

  const headZ = (lm[7].z + lm[8].z) / 2;
  const shoulderZ = (lm[11].z + lm[12].z) / 2;
  const headFwd = (shoulderZ - headZ) / shoulderWidth;

  // ── Métrica pitch: cabeça inclinada para baixo ──────────────────────────
  // Usa nariz (0) vs. olho esquerdo (2) e direito (5).
  // Olhando reto: nariz pouco abaixo dos olhos → headDown ≈ 0.15–0.25
  // Cabeça muito baixa: headDown > 0.40
  const eyeMid = midpoint(lm[2], lm[5]);
  const headDown = (lm[0].y - eyeMid.y) / shoulderWidth;

  neckEl.textContent = neckAngle.toFixed(1);
  torsoEl.textContent = torsoAngle.toFixed(1);

  const headFwdEl = document.getElementById("head-fwd");
  const headDownEl = document.getElementById("head-down");
  const shoulderTiltEl = document.getElementById("shoulder-tilt");

  if (headFwdEl) headFwdEl.textContent = headFwd.toFixed(2);
  if (headDownEl) headDownEl.textContent = headDown.toFixed(2);

  if (shoulderTiltEl) {
    shoulderTiltEl.textContent = `${shoulderTiltAngle.toFixed(1)}° - ${shoulderTiltSide}`;
  }

  // ── Decisão de postura ruim ─────────────────────────────────────────────
  const isShoulderTiltBad = shoulderTiltAngle > BAD_SHOULDER_TILT_THRESHOLD;

  const isBad =
    neckAngle > BAD_NECK_THRESHOLD ||
    torsoAngle > BAD_TORSO_THRESHOLD ||
    headFwd > BAD_HEAD_FWD ||
    headDown > BAD_HEAD_DOWN ||
    isShoulderTiltBad;

  badCount = isBad ? badCount + 1 : 0;

  const status = badCount >= CONSECUTIVE_BAD_FRAMES ? "Ruim" : "Boa";

  drawSkeleton(lm, w, h, status === "Ruim");
  updateBadge(status);

  // Mostra uma mensagem no canvas quando os ombros estiverem desalinhados
  if (isShoulderTiltBad) {
    ctx.font = "16px monospace";
    ctx.fillStyle = "#ff1744";
    ctx.fillText(`Ombros inclinados: ${shoulderTiltAngle.toFixed(1)}°`, 12, 52);
    ctx.fillText(shoulderTiltSide, 12, 74);
  }

  // --- Voice alert ---
  if (status === "Ruim") {
    const now = Date.now();

    if (now - lastAlertTime > ALERT_COOLDOWN_MS) {
      lastAlertTime = now;

      let msg;

      if (isShoulderTiltBad) {
        msg = `Atenção! ${shoulderTiltSide}. Alinhe os ombros.`;
      } else {
        msg = ALERT_MESSAGES[alertMsgIdx % ALERT_MESSAGES.length];
        alertMsgIdx++;
      }

      speak(msg);

      if (alertInfoEl) {
        alertInfoEl.style.display = "block";
        setTimeout(() => {
          alertInfoEl.style.display = "none";
        }, 5000);
      }
    }
  }

  // --- K-Means data collection (unsupervised) ---
  // Mantemos apenas neckAngle e torsoAngle para não quebrar o K-Means atual,
  // que está estruturado para trabalhar com pontos 2D.
  if (frameCount % SAMPLE_EVERY === 0) {
    sessionData.push([neckAngle, torsoAngle]);
  }

  // Re-run k-means every 5 seconds
  const nowMs = Date.now();

  if (nowMs - lastKmeansTime > 5000) {
    lastKmeansTime = nowMs;
    const kResult = runKmeans();
    updateKmeansPanel(kResult);
  }
}

// ─── Init ────────────────────────────────────────────────────────────────────

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
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
  });
  videoEl.srcObject = stream;
  await new Promise((resolve) => {
    videoEl.onloadedmetadata = resolve;
  });
  await videoEl.play();

  if (loadingEl) loadingEl.style.display = "none";

  // Initial k-means panel state
  updateKmeansPanel(null);

  requestAnimationFrame((ts) => detectLoop(poseLandmarker, ts));
}

init().catch((err) => {
  console.error(err);
  setStatus("Erro: " + err.message);
});
