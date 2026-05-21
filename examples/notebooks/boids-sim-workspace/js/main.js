
// js/main.js
// Full-window high-speed Boids animation with resize handling

(function () {
  const CONFIG = {
    BOID_COUNT: 300,
    MAX_SPEED: 180,         // pixels per second
    MAX_FORCE: 50,          // steering force
    NEIGHBOR_DIST: 50,
    DESIRED_SEPARATION: 20,
    SPEED_MULTIPLIER: 4,    // visual speed multiplier
    PHYSICS_STEPS: 3        // physics sub-steps per frame for stability at high speed
  };

  // Create or find canvas
  let canvas = document.querySelector('canvas#boids') || document.createElement('canvas');
  if (!canvas.parentNode) {
    canvas.id = 'boids';
    document.body.style.margin = '0';
    document.body.style.overflow = 'hidden';
    document.body.appendChild(canvas);
  }

  const ctx = canvas.getContext('2d', { alpha: false });

  function resizeCanvas() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = window.innerWidth;
    const height = window.innerHeight;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // scale drawing to CSS pixels
    // clamp boids inside new bounds
    for (let b of boids) {
      b.pos.x = Math.max(0, Math.min(b.pos.x, width));
      b.pos.y = Math.max(0, Math.min(b.pos.y, height));
    }
  }

  window.addEventListener('resize', () => {
    resizeCanvas();
  });

  // Simple 2D vector helpers
  function v(x = 0, y = 0) { return { x: x, y: y }; }
  function add(a, b) { a.x += b.x; a.y += b.y; return a; }
  function sub(a, b) { return v(a.x - b.x, a.y - b.y); }
  function mul(a, s) { a.x *= s; a.y *= s; return a; }
  function div(a, s) { if (s !== 0) { a.x /= s; a.y /= s; } return a; }
  function mag(a) { return Math.sqrt(a.x * a.x + a.y * a.y); }
  function setMag(a, m) { let k = mag(a) || 1; a.x = (a.x / k) * m; a.y = (a.y / k) * m; return a; }
  function limit(a, max) { let m = mag(a); if (m > max) setMag(a, max); return a; }

  class Boid {
    constructor(width, height) {
      this.pos = v(Math.random() * width, Math.random() * height);
      const angle = Math.random() * Math.PI * 2;
      this.vel = v(Math.cos(angle), Math.sin(angle));
      setMag(this.vel, Math.random() * CONFIG.MAX_SPEED * 0.5 + CONFIG.MAX_SPEED * 0.2);
      this.acc = v(0, 0);
    }

    applyForce(f) { add(this.acc, f); }

    // Basic flocking: separation, alignment, cohesion
    flock(others) {
      let sep = v(0, 0);
      let ali = v(0, 0);
      let coh = v(0, 0);
      let countSep = 0, countAli = 0, countCoh = 0;
      for (let other of others) {
        if (other === this) continue;
        let d = Math.hypot(other.pos.x - this.pos.x, other.pos.y - this.pos.y);
        if (d < CONFIG.DESIRED_SEPARATION && d > 0) {
          let diff = sub(v(this.pos.x, this.pos.y), v(other.pos.x, other.pos.y));
          div(diff, d); // weight by distance
          add(sep, diff);
          countSep++;
        }
        if (d < CONFIG.NEIGHBOR_DIST) {
          add(ali, other.vel);
          add(coh, other.pos);
          countAli++;
          countCoh++;
        }
      }

      if (countSep > 0) { div(sep, countSep); setMag(sep, CONFIG.MAX_SPEED); sub(sep, this.vel); limit(sep, CONFIG.MAX_FORCE); }
      if (countAli > 0) { div(ali, countAli); setMag(ali, CONFIG.MAX_SPEED); sub(ali, this.vel); limit(ali, CONFIG.MAX_FORCE); }
      if (countCoh > 0) { div(coh, countCoh); coh = sub(coh, this.pos); setMag(coh, CONFIG.MAX_SPEED); sub(coh, this.vel); limit(coh, CONFIG.MAX_FORCE); }

      // Tweak weights to create lively motion
      sep && this.applyForce(mul(sep, 1.5));
      ali && this.applyForce(mul(ali, 1.0));
      coh && this.applyForce(mul(coh, 1.0));
    }

    update(dt, width, height) {
      // Integrate velocity & position
      add(this.vel, mul(v(this.acc.x, this.acc.y), dt));
      limit(this.vel, CONFIG.MAX_SPEED * CONFIG.SPEED_MULTIPLIER);
      add(this.pos, mul(v(this.vel.x, this.vel.y), dt));
      // Wrap around edges
      if (this.pos.x < 0) this.pos.x += width;
      if (this.pos.x > width) this.pos.x -= width;
      if (this.pos.y < 0) this.pos.y += height;
      if (this.pos.y > height) this.pos.y -= height;
      // reset accel
      this.acc.x = 0; this.acc.y = 0;
    }

    draw(ctx) {
      // draw a rotated triangle pointing along velocity
      const angle = Math.atan2(this.vel.y, this.vel.x);
      ctx.save();
      ctx.translate(this.pos.x, this.pos.y);
      ctx.rotate(angle);
      ctx.fillStyle = 'rgba(20, 160, 220, 0.95)';
      ctx.beginPath();
      ctx.moveTo(8, 0);
      ctx.lineTo(-6, 4);
      ctx.lineTo(-6, -4);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  // Setup boids
  let boids = [];
  function initBoids() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    boids = [];
    for (let i = 0; i < CONFIG.BOID_COUNT; i++) {
      boids.push(new Boid(w, h));
    }
  }

  // Animation loop
  let lastTime = performance.now();
  function frame(now) {
    let dt = (now - lastTime) / 1000; // seconds
    if (dt > 0.1) dt = 0.1; // clamp large gaps
    lastTime = now;

    // physics steps to improve stability at high speed
    const stepDt = dt / CONFIG.PHYSICS_STEPS;
    for (let s = 0; s < CONFIG.PHYSICS_STEPS; s++) {
      // compute flocking forces
      for (let b of boids) b.flock(boids);
      // update positions
      for (let b of boids) b.update(stepDt, canvas.width / (window.devicePixelRatio || 1), canvas.height / (window.devicePixelRatio || 1));
    }

    // draw
    ctx.fillStyle = '#0b0b0c';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Optionally draw faint trails by using globalCompositeOperation or alpha; keep crisp for speed
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    for (let b of boids) b.draw(ctx);
    ctx.restore();

    requestAnimationFrame(frame);
  }

  // Kick things off after DOM ready
  function start() {
    resizeCanvas();
    initBoids();
    lastTime = performance.now();
    requestAnimationFrame(frame);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

  // Expose a small API for tweaking from console if desired
  window.BOIDS_APP = {
    config: CONFIG,
    boids,
    resize: resizeCanvas,
    reinit: () => { initBoids(); }
  };
})();
