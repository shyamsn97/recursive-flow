// Boids simulation (plain JS) — constant speed, colorful, fast, no UI.
// Exposes window.Boids = { start(), stop(), reset() } and autostarts on DOMContentLoaded.
(function () {
  'use strict';

  // Simulation constants (tweak internally, no UI)
  const NUM_BOIDS = 400;         // "hundreds" of boids
  const SPEED = 170;             // px/s — constant speed for all boids
  const NEIGHBOR_RADIUS = 48;    // px
  const SEPARATION_RADIUS = 20;  // px
  const ALIGN_WEIGHT = 1.0;
  const COHERE_WEIGHT = 0.8;
  const SEPARATE_WEIGHT = 1.7;
  const MAX_FORCE = 240;         // px/s^2 — clamp steering acceleration
  const TRAILS = false;          // solid render; set true for slight trails if desired

  // Rendering / world
  let canvas = null;
  let ctx = null;
  let cssW = 0, cssH = 0; // CSS pixels
  let dpr = 1;

  // Simulation state
  let N = NUM_BOIDS;
  let pos = null;   // Float32Array [x0,y0,x1,y1,...] in CSS px
  let vel = null;   // Float32Array [vx0,vy0,...] in CSS px/s
  let hue = null;   // Float32Array [h0,h1,...] hue 0..360
  let running = false;
  let rafId = 0;
  let lastT = 0;

  // Spatial hash grid for neighbor queries
  const cellSize = NEIGHBOR_RADIUS;
  let grid = null; // Map key -> array of boid indices

  function initCanvas() {
    canvas = document.getElementById('boids-canvas');
    ctx = canvas.getContext('2d', { alpha: false });
    resize();
    window.addEventListener('resize', resize);
  }

  function resize() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    cssW = Math.max(1, Math.floor(window.innerWidth || 1));
    cssH = Math.max(1, Math.floor(window.innerHeight || 1));
    canvas.width  = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    // Render in CSS pixel space for consistent SPEED in CSS px/sec:
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // No need to reinit boids; they wrap on new bounds
  }

  function reset() {
    N = NUM_BOIDS;
    pos = new Float32Array(N * 2);
    vel = new Float32Array(N * 2);
    hue = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const x = Math.random() * cssW;
      const y = Math.random() * cssH;
      const th = Math.random() * Math.PI * 2;
      const vx = Math.cos(th) * SPEED;
      const vy = Math.sin(th) * SPEED;
      pos[2*i] = x; pos[2*i+1] = y;
      vel[2*i] = vx; vel[2*i+1] = vy;
      hue[i] = Math.floor(Math.random() * 360);
    }
    if (!grid) grid = new Map();
  }

  function buildGrid() {
    grid.clear();
    const cols = Math.floor(cssW / cellSize) + 2;
    const rows = Math.floor(cssH / cellSize) + 2;
    const toKey = (ix, iy) => (ix << 16) ^ iy; // fast-ish key
    for (let i = 0; i < N; i++) {
      const x = pos[2*i], y = pos[2*i + 1];
      const ix = Math.floor(x / cellSize);
      const iy = Math.floor(y / cellSize);
      const key = toKey(ix, iy);
      let bucket = grid.get(key);
      if (!bucket) { bucket = []; grid.set(key, bucket); }
      bucket.push(i);
    }
  }

  function neighborsOf(i, outIdx) {
    // Gather indices in 3x3 neighboring cells
    const x = pos[2*i], y = pos[2*i + 1];
    const ix = Math.floor(x / cellSize);
    const iy = Math.floor(y / cellSize);
    const toKey = (ix, iy) => (ix << 16) ^ iy;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const key = toKey(ix + dx, iy + dy);
        const bucket = grid.get(key);
        if (bucket) {
          for (let k = 0; k < bucket.length; k++) outIdx.push(bucket[k]);
        }
      }
    }
  }

  function steer(dt) {
    // For each boid, compute steering acceleration (alignment, cohesion, separation).
    const maxForce = MAX_FORCE;
    const r2 = NEIGHBOR_RADIUS * NEIGHBOR_RADIUS;
    const sepR2 = SEPARATION_RADIUS * SEPARATION_RADIUS;
    const halfW = cssW * 0.5, halfH = cssH * 0.5;

    const nbrIdx = [];
    for (let i = 0; i < N; i++) {
      nbrIdx.length = 0;
      neighborsOf(i, nbrIdx);

      const xi = pos[2*i], yi = pos[2*i+1];
      const vxi = vel[2*i], vyi = vel[2*i+1];

      let count = 0;
      let sumVx = 0, sumVy = 0;
      let sumCx = 0, sumCy = 0;
      let sepX = 0, sepY = 0;

      for (let n = 0; n < nbrIdx.length; n++) {
        const j = nbrIdx[n];
        if (j === i) continue;
        let dx = pos[2*j]   - xi;
        let dy = pos[2*j+1] - yi;
        // Toroidal shortest distance
        if (dx >  halfW) dx -= cssW;
        if (dx < -halfW) dx += cssW;
        if (dy >  halfH) dy -= cssH;
        if (dy < -halfH) dy += cssH;

        const d2 = dx*dx + dy*dy;
        if (d2 <= r2) {
          count++;
          sumVx += vel[2*j];
          sumVy += vel[2*j+1];
          sumCx += dx;
          sumCy += dy;
          if (d2 > 0 && d2 <= sepR2) {
            // Flee inverse to distance
            const invd = 1 / Math.sqrt(d2);
            sepX -= dx * invd;
            sepY -= dy * invd;
          }
        }
      }

      // Accumulate weighted steering
      let ax = 0, ay = 0;
      if (count > 0) {
        // Alignment: steer towards average heading
        let avx = sumVx / count, avy = sumVy / count;
        const avLen = Math.hypot(avx, avy) || 1;
        avx = (avx / avLen) * SPEED;
        avy = (avy / avLen) * SPEED;
        let steerAx = avx - vxi, steerAy = avy - vyi;

        // Cohesion: steer towards center of mass
        let cx = (sumCx / count), cy = (sumCy / count);
        const cLen = Math.hypot(cx, cy) || 1;
        cx = (cx / cLen) * SPEED;
        cy = (cy / cLen) * SPEED;
        let steerCx = cx - vxi, steerCy = cy - vyi;

        // Separation: already points away; normalize
        let sx = sepX, sy = sepY;
        const sLen = Math.hypot(sx, sy) || 1;
        sx = (sx / sLen) * SPEED;
        sy = (sy / sLen) * SPEED;
        let steerSx = sx - vxi, steerSy = sy - vyi;

        ax += ALIGN_WEIGHT   * steerAx;
        ay += ALIGN_WEIGHT   * steerAy;
        ax += COHERE_WEIGHT  * steerCx;
        ay += COHERE_WEIGHT  * steerCy;
        ax += SEPARATE_WEIGHT* steerSx;
        ay += SEPARATE_WEIGHT* steerSy;
      }

      // Limit acceleration
      const aMag = Math.hypot(ax, ay);
      if (aMag > maxForce) {
        const s = maxForce / aMag;
        ax *= s; ay *= s;
      }

      // Integrate velocity (then normalize to constant SPEED)
      let nvx = vxi + ax * dt;
      let nvy = vyi + ay * dt;
      const vMag = Math.hypot(nvx, nvy) || 1;
      nvx = (nvx / vMag) * SPEED; // constant speed
      nvy = (nvy / vMag) * SPEED;

      vel[2*i] = nvx;
      vel[2*i+1] = nvy;
    }
  }

  function integrate(dt) {
    for (let i = 0; i < N; i++) {
      let x = pos[2*i]   + vel[2*i]   * dt;
      let y = pos[2*i+1] + vel[2*i+1] * dt;
      // Toroidal wrap
      if (x < 0) x += cssW; else if (x >= cssW) x -= cssW;
      if (y < 0) y += cssH; else if (y >= cssH) y -= cssH;
      pos[2*i] = x; pos[2*i+1] = y;
    }
  }

  function render() {
    if (!TRAILS) {
      ctx.clearRect(0, 0, cssW, cssH);
    } else {
      ctx.fillStyle = "rgba(0,0,0,0.15)";
      ctx.fillRect(0, 0, cssW, cssH);
    }
    const tipLen = 8;
    const wing = 3.2;
    for (let i = 0; i < N; i++) {
      const x = pos[2*i], y = pos[2*i+1];
      const vx = vel[2*i], vy = vel[2*i+1];
      const ang = Math.atan2(vy, vx);
      const cos = Math.cos, sin = Math.sin;

      const x1 = x + cos(ang) * tipLen;
      const y1 = y + sin(ang) * tipLen;
      const x2 = x + cos(ang + 2.5) * wing;
      const y2 = y + sin(ang + 2.5) * wing;
      const x3 = x + cos(ang - 2.5) * wing;
      const y3 = y + sin(ang - 2.5) * wing;

      ctx.fillStyle = 'hsl(' + hue[i] + ',90%,60%)';
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x3, y3);
      ctx.closePath();
      ctx.fill();
    }
  }

  function frame(t) {
    if (!running) return;
    if (!lastT) lastT = t;
    let dt = (t - lastT) / 1000;
    if (dt > 0.05) dt = 0.05; // clamp for stability on tab switches
    lastT = t;

    buildGrid();
    steer(dt);
    integrate(dt);
    render();

    rafId = requestAnimationFrame(frame);
  }

  function start() {
    if (running) return;
    if (!canvas) initCanvas();
    if (!pos) reset();
    running = true;
    lastT = 0;
    rafId = requestAnimationFrame(frame);
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
  }

  function apiReset() {
    reset();
  }

  window.Boids = { start, stop, reset: apiReset };

  window.addEventListener('DOMContentLoaded', () => {
    start();
  });
})();
