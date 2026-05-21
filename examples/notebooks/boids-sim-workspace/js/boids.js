// js/boids.js
// Boid class implementing basic flocking (separation, alignment, cohesion)
// Exports: Boid (ES module)
// Usage:
//   import { Boid } from './js/boids.js';
//   const b = new Boid([x,y], [vx,vy]);
//   b.update(dt, boids, {width, height});
//   b.draw(ctx, time);

function add(a, b){ return [a[0]+b[0], a[1]+b[1]]; }
function sub(a, b){ return [a[0]-b[0], a[1]-b[1]]; }
function mul(a, s){ return [a[0]*s, a[1]*s]; }
function div(a, s){ return [a[0]/s, a[1]/s]; }
function len(a){ return Math.hypot(a[0], a[1]); }
function norm(a){ const L = len(a)||1; return [a[0]/L, a[1]/L]; }
function limit(a, max){ const L = len(a); if(L > max){ return mul(a, max / L); } return a; }
function distance(a,b){ return Math.hypot(a[0]-b[0], a[1]-b[1]); }

export class Boid {
  constructor(position = [0,0], velocity = [0,0], opts = {}) {
    this.pos = [position[0], position[1]];
    this.vel = [velocity[0] || (Math.random()*2-1), velocity[1] || (Math.random()*2-1)];
    // Options / tunables
    this.maxSpeed = opts.maxSpeed || 120; // px/s
    this.maxForce = opts.maxForce || 150; // steering force magnitude
    this.size = opts.size || 6;
    this.perception = opts.perception || 60;
    this.separationDist = opts.separationDist || 20;
    this.wrap = opts.wrap === undefined ? true : !!opts.wrap; // wrap around edges
    // Optional hue seed to vary individual colors
    this.hueSeed = opts.hueSeed || (Math.random() * 360);
  }

  // Basic flocking update. dt in seconds. boids is array of nearby boids (all boids recommended).
  update(dt, boids = [], bounds = {width: 800, height: 600}) {
    // Compute steering components
    const perceived = { align: [0,0], coh: [0,0], sep: [0,0] };
    let total = 0;
    for (const other of boids) {
      if (other === this) continue;
      const d = distance(this.pos, other.pos);
      if (d < this.perception) {
        // alignment: steer toward average heading
        perceived.align = add(perceived.align, other.vel);
        // cohesion: steer toward average position
        perceived.coh = add(perceived.coh, other.pos);
        // separation: steer away from close neighbors
        if (d < this.separationDist && d > 0) {
          const diff = div(sub(this.pos, other.pos), d); // inversely weighted
          perceived.sep = add(perceived.sep, diff);
        }
        total++;
      }
    }

    let steerAlign = [0,0], steerCoh = [0,0], steerSep = [0,0];
    if (total > 0) {
      // alignment
      perceived.align = div(perceived.align, total);
      perceived.align = norm(perceived.align);
      perceived.align = mul(perceived.align, this.maxSpeed);
      steerAlign = sub(perceived.align, this.vel);
      steerAlign = limit(steerAlign, this.maxForce);

      // cohesion
      perceived.coh = div(perceived.coh, total);
      const desired = sub(perceived.coh, this.pos);
      steerCoh = norm(desired);
      steerCoh = mul(steerCoh, this.maxSpeed);
      steerCoh = sub(steerCoh, this.vel);
      steerCoh = limit(steerCoh, this.maxForce);

      // separation
      perceived.sep = div(perceived.sep, total);
      steerSep = perceived.sep;
      steerSep = norm(steerSep);
      steerSep = mul(steerSep, this.maxSpeed);
      steerSep = sub(steerSep, this.vel);
      steerSep = limit(steerSep, this.maxForce);
    }

    // Weights for behaviors
    const wAlign = 1.0;
    const wCoh = 0.8;
    const wSep = 1.6;

    // Apply steering (accelerations). Using simple Euler integration.
    let accel = [0,0];
    accel = add(accel, mul(steerAlign, wAlign));
    accel = add(accel, mul(steerCoh, wCoh));
    accel = add(accel, mul(steerSep, wSep));
    // Scale acceleration by dt to convert force-like numbers into velocity change
    const dv = mul(accel, dt);
    this.vel = add(this.vel, dv);
    // Limit speed
    this.vel = limit(this.vel, this.maxSpeed);
    // Integrate position
    this.pos = add(this.pos, mul(this.vel, dt));

    // Edge handling: wrap or reflect
    if (this.wrap) {
      const w = bounds.width, h = bounds.height;
      if (this.pos[0] < 0) this.pos[0] += w;
      if (this.pos[0] >= w) this.pos[0] -= w;
      if (this.pos[1] < 0) this.pos[1] += h;
      if (this.pos[1] >= h) this.pos[1] -= h;
    } else {
      // simple reflection
      const [x,y] = this.pos;
      if (x < 0 || x > bounds.width) { this.vel[0] *= -1; this.pos[0] = Math.max(0, Math.min(bounds.width, x)); }
      if (y < 0 || y > bounds.height) { this.vel[1] *= -1; this.pos[1] = Math.max(0, Math.min(bounds.height, y)); }
    }
  }

  // Draw the boid as a rotated triangle. ctx is a CanvasRenderingContext2D.
  // time is seconds elapsed (used for hue cycling).
  draw(ctx, time = 0) {
    const speed = len(this.vel);
    // Hue cycles by time and is modulated by speed and a per-boid seed
    const timeHue = (this.hueSeed + time * 30) % 360; // time contributes strongly
    const speedHue = (speed / this.maxSpeed) * 60; // small shift based on speed
    const hue = (timeHue + speedHue) % 360;

    ctx.save();
    ctx.translate(this.pos[0], this.pos[1]);
    const angle = Math.atan2(this.vel[1], this.vel[0]);
    ctx.rotate(angle);

    // Styling
    ctx.fillStyle = `hsl(${hue.toFixed(1)}, 80%, 50%)`;
    ctx.strokeStyle = `hsla(${hue.toFixed(1)}, 80%, 20%, 0.9)`;
    ctx.lineWidth = 1;

    // Draw triangle pointing to +X
    const s = this.size;
    ctx.beginPath();
    ctx.moveTo(s * 1.3, 0);
    ctx.lineTo(-s * 0.8, s * 0.7);
    ctx.lineTo(-s * 0.8, -s * 0.7);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Optional subtle velocity trail (small)
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.moveTo(-s * 0.8, 0);
    ctx.lineTo(-s * 0.8 - this.vel[0] * 0.02, -this.vel[1] * 0.02);
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    ctx.restore();
  }

  // Convenience: create a random-position boid
  static random(bounds = {width:800, height:600}, opts = {}) {
    const x = Math.random() * bounds.width;
    const y = Math.random() * bounds.height;
    const angle = Math.random() * Math.PI * 2;
    const speed = (opts.initSpeed !== undefined) ? opts.initSpeed : (Math.random() * 40 + 20);
    const vx = Math.cos(angle) * speed;
    const vy = Math.sin(angle) * speed;
    return new Boid([x,y], [vx,vy], opts);
  }
}
