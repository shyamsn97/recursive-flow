
import { Boids } from './boids.js';
import { render } from './renderer.js';

window.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('boids');
  const ctx = canvas.getContext('2d', { alpha: false });

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.floor(window.innerWidth);
    const cssH = Math.floor(window.innerHeight);
    const pxW = Math.max(1, Math.floor(cssW * dpr));
    const pxH = Math.max(1, Math.floor(cssH * dpr));
    if (canvas.width !== pxW) canvas.width = pxW;
    if (canvas.height !== pxH) canvas.height = pxH;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    // Reset any scaling/transforms; renderer draws in device pixels
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  resizeCanvas();

  const N = 3000;
  let sim = new Boids(canvas.width, canvas.height, N);

  let last = performance.now();
  function loop(now) {
    let dt = (now - last) / 1000;
    last = now;
    if (!Number.isFinite(dt) || dt <= 0) dt = 0.016;
    if (dt > 0.05) dt = 0.05;

    sim.update(dt);
    render(ctx, sim);

    requestAnimationFrame(loop);
  }

  // Kick off with a calibrated timestamp
  requestAnimationFrame((t) => {
    last = t;
    requestAnimationFrame(loop);
  });

  window.addEventListener('resize', () => {
    resizeCanvas();
    sim.resize(canvas.width, canvas.height);
  }, { passive: true });
});
