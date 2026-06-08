/* ==========================================================================
   ui.js — Config panel + PostureChart
   Loaded only on the monitor page (index.html).
   The detector.js reads config values via document.getElementById('cfg-...').value
   ========================================================================== */

// ── Config panel: toggle + slider <-> number sync + restore defaults ─────────
(function () {
  const toggle = document.getElementById('cfg-toggle');
  const panel  = document.getElementById('cfg-panel');
  if (!toggle || !panel) return;

  toggle.addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    toggle.setAttribute('aria-expanded', open);
  });

  const ranges   = [...document.querySelectorAll('.cfg input[type=range]')];
  const defaults = {};

  function paintRange(r) {
    const min = +r.min, max = +r.max;
    const p   = ((+r.value - min) / (max - min)) * 100;
    r.style.setProperty('--p', p + '%');
  }

  ranges.forEach(r => {
    defaults[r.id] = r.value;
    const num = document.querySelector(`.cfg__num[data-for="${r.id}"]`);
    paintRange(r);
    r.addEventListener('input', () => {
      if (num) num.value = r.value;
      paintRange(r);
      updateLimitLabels();
    });
    if (num) {
      num.addEventListener('input', () => {
        const v = Math.min(+num.max, Math.max(+num.min, +num.value || +num.min));
        r.value = v;
        paintRange(r);
        updateLimitLabels();
      });
      num.addEventListener('blur', () => { num.value = r.value; });
    }
  });

  document.getElementById('cfg-reset').addEventListener('click', () => {
    ranges.forEach(r => {
      r.value = defaults[r.id];
      const num = document.querySelector(`.cfg__num[data-for="${r.id}"]`);
      if (num) num.value = defaults[r.id];
      paintRange(r);
    });
    updateLimitLabels();
  });

  // Update the metric card limit chips when thresholds change
  function updateLimitLabels() {
    const neck  = document.getElementById('cfg-neck-threshold');
    const torso = document.getElementById('cfg-torso-threshold');
    const fwd   = document.getElementById('cfg-head-fwd');
    const down  = document.getElementById('cfg-head-down');
    const ln = document.getElementById('lim-neck');
    const lt = document.getElementById('lim-torso');
    const lf = document.getElementById('lim-fwd');
    const ld = document.getElementById('lim-down');
    if (neck  && ln) ln.textContent = `lim ${neck.value}°`;
    if (torso && lt) lt.textContent = `lim ${torso.value}°`;
    if (fwd   && lf) lf.textContent = `lim ${fwd.value}`;
    if (down  && ld) ld.textContent = `lim ${down.value}`;
  }
})();

// ── PostureChart: session history area chart (last 60 data points) ───────────
const PostureChart = (function () {
  const cv  = document.getElementById('session-chart');
  if (!cv) return { push: () => {}, draw: () => {} };
  const ctx = cv.getContext('2d');
  const N   = 60;
  const data = new Array(N).fill(null); // null = no data yet

  function size() {
    const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    cv.width  = r.width  * dpr;
    cv.height = r.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function color(v) {
    return v >= 66 ? '#2ee6a6' : v >= 40 ? '#ffc34d' : '#ff5c7a';
  }

  function hexToRgba(hex, a) {
    hex = (hex || '').trim().replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const n = parseInt(hex || '21d4fd', 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function draw() {
    const w = cv.clientWidth, h = cv.clientHeight, pad = 6;
    ctx.clearRect(0, 0, w, h);

    const filled = data.filter(v => v !== null);
    if (filled.length === 0) return;

    const x = i => pad + (i / (N - 1)) * (w - pad * 2);
    const y = v => h - pad - (v / 100) * (h - pad * 2);

    // threshold guide lines
    ctx.strokeStyle = 'rgba(255,255,255,.06)';
    ctx.lineWidth   = 1;
    [40, 66].forEach(t => {
      ctx.beginPath();
      ctx.moveTo(pad, y(t));
      ctx.lineTo(w - pad, y(t));
      ctx.stroke();
    });

    // area fill
    const ac   = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, hexToRgba(ac, 0.26));
    grad.addColorStop(1, hexToRgba(ac, 0));

    // build path only for non-null points
    const points = data.map((v, i) => v !== null ? { x: x(i), y: y(v), v } : null);
    const first  = points.find(p => p !== null);
    if (!first) return;

    ctx.beginPath();
    ctx.moveTo(first.x, h - pad);
    points.forEach(p => { if (p) ctx.lineTo(p.x, p.y); });
    const last = [...points].reverse().find(p => p !== null);
    ctx.lineTo(last.x, h - pad);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // colored line segments
    ctx.lineWidth = 2;
    ctx.lineJoin  = 'round';
    let prev = null;
    points.forEach(p => {
      if (!p) { prev = null; return; }
      if (prev) {
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(p.x, p.y);
        ctx.strokeStyle = color((prev.v + p.v) / 2);
        ctx.stroke();
      }
      prev = p;
    });

    // leading dot
    if (last) {
      ctx.beginPath();
      ctx.arc(last.x, last.y, 3.5, 0, 7);
      ctx.fillStyle = color(last.v);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(last.x, last.y, 6.5, 0, 7);
      ctx.strokeStyle = color(last.v);
      ctx.globalAlpha = 0.35;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  function push(v) {
    data.push(Math.max(0, Math.min(100, v)));
    if (data.length > N) data.shift();
    draw();
  }

  window.addEventListener('resize', () => { size(); draw(); });
  size();
  draw();

  return { push, draw };
})();

window.PostureChart = PostureChart;
