import { Simulation } from './simulation.js';
import { Renderer } from './render.js';

const canvas = document.getElementById('canvas');

const COUNT = 400;
const sim = new Simulation(canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight, COUNT);
const renderer = new Renderer(canvas, sim);

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.033, (now - last) / 1000); // cap dt for stability
  last = now;
  sim.update(dt);
  renderer.renderFrame();
  requestAnimationFrame(frame);
}

renderer.resize();
requestAnimationFrame(frame);

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    last = performance.now();
  }
});
