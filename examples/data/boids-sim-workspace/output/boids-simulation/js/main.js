// main.js — entrypoint for Boids Simulation
// - Imports Simulation from ./sim.js
// - Selects <canvas id="boids">, sizes it to the window (handles DPR), creates Simulation(ctx, canvas)
// - Starts the animation loop and handles window resizes
//
// This file intentionally keeps interactions simple and robust so it works with the provided sim/boid/utils modules.

import { Simulation } from './sim.js';

const canvas = document.querySelector('#boids');
if (!canvas) {
  throw new Error('Canvas element #boids not found');
}

const ctx = canvas.getContext('2d', { alpha: false });

// Handle high-DPI displays and size the canvas to fill the window
function resizeCanvas() {
  const DPR = Math.max(window.devicePixelRatio || 1, 1);
  // Use CSS viewport sizing for layout; set actual pixel buffer according to DPR
  const cssWidth = window.innerWidth;
  const cssHeight = window.innerHeight;
  canvas.style.width = cssWidth + 'px';
  canvas.style.height = cssHeight + 'px';
  // Set the internal resolution
  canvas.width = Math.round(cssWidth * DPR);
  canvas.height = Math.round(cssHeight * DPR);
  // Reset transform and scale to account for DPR so drawing uses CSS pixels
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(DPR, DPR);
  // If the Simulation instance exposes a resize method or reads canvas dimensions, it will pick up the change automatically.
  if (simulation && typeof simulation.onResize === 'function') {
    try { simulation.onResize(cssWidth, cssHeight); } catch (e) { /* ignore */ }
  }
}

// Create the Simulation: contract expects constructor(ctx, canvas)
const simulation = new Simulation(ctx, canvas);

// Initial resize before starting loop
resizeCanvas();
window.addEventListener('resize', () => {
  // Throttle resize slightly via requestAnimationFrame
  requestAnimationFrame(resizeCanvas);
});

// Basic animation loop with delta-time (seconds)
let lastTime = performance.now();
let paused = false;

function step(now) {
  if (!lastTime) lastTime = now;
  const dt = Math.min(0.1, (now - lastTime) / 1000); // clamp large dt
  lastTime = now;
  if (!paused) {
    if (typeof simulation.update === 'function') {
      // Some Simulation implementations take (dt) or no args; try to call with dt when accepted.
      try {
        simulation.update(dt);
      } catch (e) {
        // Fallback: call without args
        try { simulation.update(); } catch (e2) { console.error('simulation.update error', e, e2); }
      }
    }
    if (typeof simulation.draw === 'function') {
      try {
        simulation.draw();
      } catch (e) {
        console.error('simulation.draw error', e);
      }
    }
  } else {
    // still draw paused frame (optional)
    if (typeof simulation.draw === 'function') {
      try { simulation.draw(); } catch (e) { console.error('simulation.draw error', e); }
    }
  }
  requestAnimationFrame(step);
}
requestAnimationFrame(step);

// Simple controls:
// - Space: toggle pause
// - Click/tap on canvas: add a boid at pointer if simulation exposes addBoid(x,y) or createBoid
window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Space') {
    paused = !paused;
    ev.preventDefault();
  }
});

canvas.addEventListener('pointerdown', (ev) => {
  // Compute CSS-space coordinates (account for canvas CSS size)
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  // Try common hook names; ignore failures
  const tryAdd = [
    () => simulation.addBoid && simulation.addBoid(x, y),
    () => simulation.spawn && simulation.spawn(x, y),
    () => simulation.createBoid && simulation.createBoid(x, y),
  ];
  for (const fn of tryAdd) {
    try { const r = fn(); if (r !== undefined) break; } catch (_) { /* ignore */ }
  }
});

// Expose simulation for debugging from the browser console
window.__BOIDS_SIMULATION__ = simulation;
