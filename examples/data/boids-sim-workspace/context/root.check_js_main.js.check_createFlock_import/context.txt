// js/main.js
import Boid from './boid.js';
import { createFlock } from './flock.js';

const canvas = document.getElementById('flockCanvas');
const ctx = canvas.getContext('2d');

let width = 0, height = 0;
function resize(){
  const dpr = window.devicePixelRatio || 1;
  width = Math.floor(window.innerWidth);
  height = Math.floor(window.innerHeight);
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener('resize', resize);

let boids = [];
let countControl = null;
let speedControl = null;
let trail = false;
let last = performance.now();

function init(){
  resize();
  countControl = document.getElementById('count');
  speedControl = document.getElementById('speed');
  const toggleTrailBtn = document.getElementById('toggleTrail');
  toggleTrailBtn.addEventListener('click', ()=>{ trail = !trail; });

  const initialCount = parseInt(countControl.value,10) || 180;
  boids = createFlock(initialCount, width, height);

  countControl.addEventListener('input', ()=>{
    const n = parseInt(countControl.value,10);
    if (n > boids.length){
      const add = n - boids.length;
      for (let i=0;i<add;i++){
        boids.push(new Boid(Math.random()*width, Math.random()*height, boids.length));
      }
    } else {
      boids.length = n;
    }
  });

  requestAnimationFrame(loop);
}

function loop(t){
  const dt = Math.min(50, t - last);
  last = t;

  // background / trail
  if (!trail){
    ctx.clearRect(0,0,width,height);
    // subtle vignette background
    ctx.fillStyle = 'rgba(10,12,24,0.08)';
    ctx.fillRect(0,0,width,height);
  } else {
    ctx.fillStyle = 'rgba(11,16,32,0.08)';
    ctx.fillRect(0,0,width,height);
  }

  // parameters
  const baseSpeed = parseFloat(speedControl.value) || 4;

  for (let i=0;i<boids.length;i++){
    const b = boids[i];
    b.maxSpeed = baseSpeed + (Math.sin((i/boids.length)*Math.PI*2 + t*0.001) * 0.8);
    b.flock(boids, {align:1.0, cohesion:0.9, separation:1.7});
    b.update();
    b.edges(width, height);

    // color hue varies with index and velocity
    const speed = Math.hypot(b.vel.x, b.vel.y);
    const hue = (i * 360 / Math.max(1, boids.length) + (speed - baseSpeed) * 40 + t*0.01) % 360;
    b.draw(ctx, Math.floor(hue), 6 + Math.min(6, speed));
  }

  requestAnimationFrame(loop);
}

document.addEventListener('DOMContentLoaded', init);
