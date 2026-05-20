
/**
 * js/flock.js
 *
 * Simple, dependency-free flocking (boids) module.
 *
 * Usage:
 *   const Flock = require('./js/flock')   // CommonJS
 *   // or in browser: import Flock from './js/flock.js'
 *
 *   const flock = Flock.createFlock({ count: 100, width: 800, height: 600 });
 *   // call flock.update(dt) each frame (dt in seconds)
 *
 * API:
 *   createFlock(options) -> flock instance
 *   flock.update(dt) -> advances the simulation
 *   flock.add(boid) -> add a boid { x, y, vx, vy }
 *   flock.remove(index) -> remove boid by index
 *   flock.reset() -> clears and recreates initial boids
 *
 * Options:
 *   count: initial number of boids (default 50)
 *   width, height: bounds for wrap-around (default 800x600)
 *   maxSpeed: maximum boid speed (default 120)
 *   maxForce: maximum steering force (default 200)
 *   perception: { align, cohesion, separation } perception radii
 *   strengths: { align, cohesion, separation } steering strengths
 */

function clampMagnitude(vx, vy, max) {
  const mag = Math.hypot(vx, vy);
  if (mag === 0) return [0, 0];
  if (mag <= max) return [vx, vy];
  const s = max / mag;
  return [vx * s, vy * s];
}

function limit(vx, vy, max) {
  return clampMagnitude(vx, vy, max);
}

function vecAdd(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
function vecSub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
function vecScale(a, s) { return [a[0] * s, a[1] * s]; }

/**
 * Create a flock instance
 */
function createFlock(options) {
  const opts = Object.assign({
    count: 50,
    width: 800,
    height: 600,
    maxSpeed: 120,   // units per second
    maxForce: 200,   // steering (units per second^2)
    perception: { align: 50, cohesion: 50, separation: 25 },
    strengths: { align: 1.0, cohesion: 1.0, separation: 1.5 },
    wrap: true
  }, options || {});

  // Internal boid structure: { x, y, vx, vy }
  let boids = [];

  function init() {
    boids = [];
    for (let i = 0; i < opts.count; i++) {
      const x = Math.random() * opts.width;
      const y = Math.random() * opts.height;
      // random velocity
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * (opts.maxSpeed * 0.5) + (opts.maxSpeed * 0.1);
      boids.push({ x: x, y: y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed });
    }
  }

  function neighbors(i, radius) {
    const b = boids[i];
    const list = [];
    const r2 = radius * radius;
    for (let j = 0; j < boids.length; j++) {
      if (j === i) continue;
      const dx = boids[j].x - b.x;
      const dy = boids[j].y - b.y;
      const dist2 = dx * dx + dy * dy;
      if (dist2 <= r2) list.push({ boid: boids[j], dx: dx, dy: dy, dist2: dist2 });
    }
    return list;
  }

  function steerTowards(currentVx, currentVy, desiredVx, desiredVy) {
    // desired velocity minus current velocity => desired steering
    let sx = desiredVx - currentVx;
    let sy = desiredVy - currentVy;
    // limit to maxForce
    [sx, sy] = limit(sx, sy, opts.maxForce);
    return [sx, sy];
  }

  function applyBehavior(i, dt) {
    const b = boids[i];

    // Alignment
    const alignNeighbors = neighbors(i, opts.perception.align);
    let alignForce = [0, 0];
    if (alignNeighbors.length > 0) {
      let avgVx = 0, avgVy = 0;
      for (const n of alignNeighbors) { avgVx += n.boid.vx; avgVy += n.boid.vy; }
      avgVx /= alignNeighbors.length; avgVy /= alignNeighbors.length;
      // normalize desired to max speed
      const [dvx, dvy] = limit(avgVx, avgVy, opts.maxSpeed);
      alignForce = steerTowards(b.vx, b.vy, dvx, dvy);
      alignForce = vecScale(alignForce, opts.strengths.align);
    }

    // Cohesion
    const cohNeighbors = neighbors(i, opts.perception.cohesion);
    let cohesionForce = [0, 0];
    if (cohNeighbors.length > 0) {
      let cx = 0, cy = 0;
      for (const n of cohNeighbors) { cx += n.boid.x; cy += n.boid.y; }
      cx /= cohNeighbors.length; cy /= cohNeighbors.length;
      // desired velocity is towards center
      const desired = vecSub([cx, cy], [b.x, b.y]);
      // scale to max speed
      const [dvx, dvy] = limit(desired[0], desired[1], opts.maxSpeed);
      cohesionForce = steerTowards(b.vx, b.vy, dvx, dvy);
      cohesionForce = vecScale(cohesionForce, opts.strengths.cohesion);
    }

    // Separation
    const sepNeighbors = neighbors(i, opts.perception.separation);
    let separationForce = [0, 0];
    if (sepNeighbors.length > 0) {
      let sx = 0, sy = 0;
      for (const n of sepNeighbors) {
        // push away inversely proportional to distance (approx)
        const invDist = 1 / (Math.sqrt(n.dist2) + 1e-6);
        sx += -n.dx * invDist;
        sy += -n.dy * invDist;
      }
      sx /= sepNeighbors.length; sy /= sepNeighbors.length;
      const [dvx, dvy] = limit(sx, sy, opts.maxSpeed);
      separationForce = steerTowards(b.vx, b.vy, dvx, dvy);
      separationForce = vecScale(separationForce, opts.strengths.separation);
    }

    // Sum forces (they are accelerations), apply dt
    let ax = alignForce[0] + cohesionForce[0] + separationForce[0];
    let ay = alignForce[1] + cohesionForce[1] + separationForce[1];

    // integrate velocity
    b.vx += ax * dt;
    b.vy += ay * dt;

    // limit speed
    [b.vx, b.vy] = limit(b.vx, b.vy, opts.maxSpeed);
  }

  function wrapPosition(b) {
    if (!opts.wrap) return;
    if (b.x < 0) b.x += opts.width;
    else if (b.x >= opts.width) b.x -= opts.width;
    if (b.y < 0) b.y += opts.height;
    else if (b.y >= opts.height) b.y -= opts.height;
  }

  function update(dt) {
    // update each boid: compute acceleration from neighbors based on current state
    // We compute steering based on current positions/velocities, then integrate.
    for (let i = 0; i < boids.length; i++) {
      applyBehavior(i, dt);
    }
    // integrate positions
    for (let i = 0; i < boids.length; i++) {
      const b = boids[i];
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      wrapPosition(b);
    }
  }

  function add(boid) {
    // boid expected to have { x, y, vx, vy } (vx/vy optional)
    const b = { x: boid.x || 0, y: boid.y || 0, vx: boid.vx || 0, vy: boid.vy || 0 };
    boids.push(b);
    return boids.length - 1;
  }

  function remove(index) {
    if (index < 0 || index >= boids.length) return false;
    boids.splice(index, 1);
    return true;
  }

  function reset() {
    init();
  }

  function getBoids() {
    // return shallow copy to avoid external mutation
    return boids.map(b => ({ x: b.x, y: b.y, vx: b.vx, vy: b.vy }));
  }

  // initialize
  init();

  return {
    update,
    add,
    remove,
    reset,
    getBoids,
    options: opts
  };
}

// Export: support both CommonJS and ES modules
const Flock = { createFlock };
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Flock;
} else if (typeof define === 'function' && define.amd) {
  define([], function () { return Flock; });
} else {
  // attach to window if present
  if (typeof window !== 'undefined') {
    window.Flock = Flock;
  }
  // also as default export name for ESM consumers who may load as a script type=module
  try { export default Flock; } catch (e) { /* ignore in non-module contexts */ }
}
