
import Flock from './flock.js';
import randomColor from './utils.js';

// Entry module: initializes canvas, creates a Flock with many colorful fast boids, and animates them.
// Tries to use imported Flock if compatible; falls back to an internal implementation to ensure safe runtime behavior.

const canvas = document.createElement('canvas');
canvas.id = 'flock-canvas';
canvas.style.position = 'fixed';
canvas.style.top = '0';
canvas.style.left = '0';
canvas.style.width = '100%';
canvas.style.height = '100%';
canvas.style.zIndex = '0';
canvas.style.display = 'block';
document.body.style.margin = '0';
document.body.appendChild(canvas);

const ctx = canvas.getContext('2d');

function setCanvasSize() {
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.max(window.innerWidth, 300);
  const height = Math.max(window.innerHeight, 200);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', () => {
  setCanvasSize();
  if (flock && typeof flock.onResize === 'function') {
    try { flock.onResize(canvas.width, canvas.height); } catch (e) { /* ignore */ }
  }
});
setCanvasSize();

// Safe randomColor fallback
function _randomColorFallback() {
  // bright saturated HSL
  const h = Math.floor(Math.random() * 360);
  const s = 75 + Math.floor(Math.random() * 20);
  const l = 45 + Math.floor(Math.random() * 15);
  return `hsl(${h}deg ${s}% ${l}%)`;
}
const pickColor = (typeof randomColor === 'function') ? (() => {
  try {
    // Some randomColor libs accept options object, try bright palette first
    const c = randomColor({ luminosity: 'bright' });
    return () => randomColor({ luminosity: 'bright' });
  } catch (e) {
    return () => randomColor();
  }
})() : _randomColorFallback;

// Internal minimal boid/flock implementation as a safe fallback.
class InternalBoid {
  constructor(x, y, vx, vy, color) {
    this.x = x; this.y = y;
    this.vx = vx; this.vy = vy;
    this.color = color;
    this.size = 2 + Math.random() * 3;
  }
  update(dt, width, height) {
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    // wrap
    if (this.x < -10) this.x = width + 10;
    if (this.x > width + 10) this.x = -10;
    if (this.y < -10) this.y = height + 10;
    if (this.y > height + 10) this.y = -10;
  }
  draw(ctx) {
    ctx.save();
    ctx.translate(this.x, this.y);
    // draw a small triangle pointing along velocity
    const angle = Math.atan2(this.vy, this.vx);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(this.size * 2, 0);
    ctx.lineTo(-this.size, this.size);
    ctx.lineTo(-this.size, -this.size);
    ctx.closePath();
    ctx.fillStyle = this.color;
    ctx.fill();
    ctx.restore();
  }
}

class InternalFlock {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.boids = [];
  }
  addBoid(boid) { this.boids.push(boid); }
  onResize(w, h) { this.width = w; this.height = h; }
  update(dt) {
    for (const b of this.boids) b.update(dt, this.width, this.height);
  }
  draw(ctx) {
    for (const b of this.boids) b.draw(ctx);
  }
}

// Try to create an instance of the imported Flock in a few safe ways.
// If that fails, use the internal flock.
let flock = null;
(function tryCreateFlock() {
  const w = canvas.width;
  const h = canvas.height;
  try {
    if (typeof Flock === 'function') {
      // Try a few common constructor signatures
      try { flock = new Flock(canvas); return; } catch (e) {}
      try { flock = new Flock(ctx); return; } catch (e) {}
      try { flock = new Flock({ canvas, ctx, width: canvas.width, height: canvas.height }); return; } catch (e) {}
      try { flock = new Flock(); return; } catch (e) {}
    }
  } catch (e) {
    // fallthrough
  }
  // fallback
  flock = new InternalFlock(canvas.width, canvas.height);
})();

// Populate with many colorful, fast boids.
// We'll use a mix of addBoid method if available, otherwise push into a boids array.
const BOID_COUNT = Math.max(80, Math.min(600, Math.floor((canvas.width * canvas.height) / (8000))));
for (let i = 0; i < BOID_COUNT; i++) {
  const x = Math.random() * canvas.width / (window.devicePixelRatio || 1);
  const y = Math.random() * canvas.height / (window.devicePixelRatio || 1);
  // fast velocities: magnitude between 1.2 and 4.5 (pixels per ms scale later)
  const speed = 1.6 + Math.random() * 3.2;
  const angle = Math.random() * Math.PI * 2;
  const vx = Math.cos(angle) * speed;
  const vy = Math.sin(angle) * speed;
  const color = (typeof pickColor === 'function') ? pickColor() : _randomColorFallback();

  const boid = (typeof InternalBoid !== 'undefined') ? new InternalBoid(x, y, vx, vy, color) : { x, y, vx, vy, color };
  // If imported Flock provided an 'add' or 'addBoid' method, try to use it, otherwise push to array.
  if (flock) {
    if (typeof flock.add === 'function') {
      try { flock.add(boid); continue; } catch (e) {}
    }
    if (typeof flock.addBoid === 'function') {
      try { flock.addBoid(boid); continue; } catch (e) {}
    }
    if (Array.isArray(flock.boids)) {
      flock.boids.push(boid);
      continue;
    }
  }
}

// Animation loop
let last = performance.now();
function frame(now) {
  const dtMs = now - last;
  last = now;
  const dt = Math.min(dtMs, 50); // clamp delta to avoid large jumps

  // clear with a slight alpha to produce trails (tweak as desired)
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // If the imported flock has update/draw methods, use them; else attempt common variations.
  try {
    if (flock) {
      if (typeof flock.update === 'function') flock.update(dt);
      else if (typeof flock.step === 'function') flock.step(dt);
      else if (typeof flock.tick === 'function') flock.tick(dt);
      else {
        // fallback for internal: expects logical pixels, so convert dt to px step roughly
        if (typeof flock.update === 'undefined' && typeof flock.boids !== 'undefined') {
          // our InternalFlock expects dt in ms? It updates by velocity*dt so scale: convert dt to seconds-ish factor
          // We'll interpret velocities as pixels per frame and multiply by (dt/16) for approximate speed constancy
          const factor = dt / 16;
          for (const b of flock.boids) {
            if (typeof b.update === 'function') b.update(factor, canvas.width / (window.devicePixelRatio || 1), canvas.height / (window.devicePixelRatio || 1));
            else {
              b.x += b.vx * factor;
              b.y += b.vy * factor;
              // wrap
              if (b.x < -10) b.x = canvas.width + 10;
              if (b.x > canvas.width + 10) b.x = -10;
              if (b.y < -10) b.y = canvas.height + 10;
              if (b.y > canvas.height + 10) b.y = -10;
            }
          }
        }
      }

      if (typeof flock.draw === 'function') {
        flock.draw(ctx);
      } else if (Array.isArray(flock.boids)) {
        for (const b of flock.boids) {
          if (typeof b.draw === 'function') b.draw(ctx);
          else {
            // draw simple triangle/circle
            ctx.save();
            ctx.translate(b.x, b.y);
            ctx.fillStyle = b.color || '#fff';
            ctx.beginPath();
            ctx.arc(0, 0, b.size || 2.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }
        }
      }
    }
  } catch (e) {
    // On any unexpected error, stop using flock methods and try to draw what we can
    if (Array.isArray(flock && flock.boids)) {
      for (const b of flock.boids) {
        if (typeof b.draw === 'function') {
          try { b.draw(ctx); } catch (_) {}
        } else {
          ctx.save();
          ctx.translate(b.x, b.y);
          ctx.fillStyle = b.color || '#fff';
          ctx.beginPath();
          ctx.arc(0, 0, b.size || 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }
    }
  }

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Exports: entry module should export nothing (per contract)
