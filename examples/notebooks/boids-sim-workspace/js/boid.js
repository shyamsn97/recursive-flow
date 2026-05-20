
// js/boid.js
// Boid class implementing basic flocking behaviors: alignment, cohesion, separation.
// Designed to be framework-agnostic. Assumes a vector-like object with x, y and basic operations.
// If using p5.js, supply p5.Vector instances; otherwise, the class will use a minimal internal Vec helper.

class Vec {
  // Minimal vector helper for plain JS usage (if p5.Vector is not provided)
  constructor(x = 0, y = 0) {
    this.x = x;
    this.y = y;
  }
  add(v) { this.x += v.x; this.y += v.y; return this; }
  sub(v) { this.x -= v.x; this.y -= v.y; return this; }
  mult(n) { this.x *= n; this.y *= n; return this; }
  div(n) { if (n !== 0) { this.x /= n; this.y /= n; } return this; }
  mag() { return Math.hypot(this.x, this.y); }
  setMag(n) { const m = this.mag(); if (m !== 0) this.mult(n / m); return this; }
  normalize() { const m = this.mag(); if (m !== 0) this.div(m); return this; }
  limit(max) { if (this.mag() > max) this.setMag(max); return this; }
  copy() { return new Vec(this.x, this.y); }
  static sub(a, b) { return new Vec(a.x - b.x, a.y - b.y); }
  static add(a, b) { return new Vec(a.x + b.x, a.y + b.y); }
  static dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
}

// Boid class
class Boid {
  constructor(x, y, opts = {}) {
    // Position, velocity, acceleration (use plain objects with x,y or Vec/p5.Vector)
    this.pos = opts.pos || new Vec(x || 0, y || 0);
    this.vel = opts.vel || new Vec((Math.random() * 2 - 1), (Math.random() * 2 - 1));
    this.acc = opts.acc || new Vec(0, 0);

    // Tuning parameters (can be overridden per-instance via opts)
    this.maxSpeed = opts.maxSpeed || 4;
    this.maxForce = opts.maxForce || 0.1;

    // Perception radii for behaviors
    this.perception = {
      align: opts.alignPerception || 50,
      cohesion: opts.cohesionPerception || 50,
      separation: opts.separationPerception || 25
    };

    // Size for drawing or separation weighting
    this.size = opts.size || 6;
  }

  // Utility to ensure compatibility with p5.Vector or our Vec
  static _ensureVec(v) {
    if (!v) return new Vec(0,0);
    if (typeof v.x === 'number' && typeof v.y === 'number') return v;
    // If a framework uses arrays, convert (not expected but be permissive)
    if (Array.isArray(v) && v.length >= 2) return new Vec(v[0], v[1]);
    return new Vec(0,0);
  }

  applyForce(force) {
    const f = Boid._ensureVec(force);
    this.acc.add(f);
  }

  // Wrap around edges (assumes width & height provided)
  edges(width, height) {
    if (this.pos.x > width) this.pos.x = 0;
    if (this.pos.x < 0) this.pos.x = width;
    if (this.pos.y > height) this.pos.y = 0;
    if (this.pos.y < 0) this.pos.y = height;
  }

  // Alignment: steer toward average heading of local flockmates
  align(boids) {
    const steering = new Vec(0,0);
    let total = 0;
    for (const other of boids) {
      const d = Boid._distance(this.pos, other.pos);
      if (other !== this && d < this.perception.align) {
        steering.add(other.vel);
        total++;
      }
    }
    if (total > 0) {
      steering.div(total);
      steering.setMag(this.maxSpeed);
      steering.sub(this.vel);
      steering.limit(this.maxForce);
    }
    return steering;
  }

  // Cohesion: steer toward average position of local flockmates
  cohesion(boids) {
    const steering = new Vec(0,0);
    let total = 0;
    for (const other of boids) {
      const d = Boid._distance(this.pos, other.pos);
      if (other !== this && d < this.perception.cohesion) {
        steering.add(other.pos);
        total++;
      }
    }
    if (total > 0) {
      steering.div(total);
      steering.sub(this.pos);
      steering.setMag(this.maxSpeed);
      steering.sub(this.vel);
      steering.limit(this.maxForce);
    }
    return steering;
  }

  // Separation: steer away from close neighbors
  separation(boids) {
    const steering = new Vec(0,0);
    let total = 0;
    for (const other of boids) {
      const d = Boid._distance(this.pos, other.pos);
      if (other !== this && d < this.perception.separation) {
        const diff = Vec.sub(this.pos, other.pos);
        if (d !== 0) diff.div(d); // weight by distance
        steering.add(diff);
        total++;
      }
    }
    if (total > 0) {
      steering.div(total);
      steering.setMag(this.maxSpeed);
      steering.sub(this.vel);
      steering.limit(this.maxForce);
    }
    return steering;
  }

  // Combine flocking behaviors with optional weights
  flock(boids, weights = { align: 1.0, cohesion: 1.0, separation: 1.5 }) {
    const alignment = this.align(boids);
    const cohesion = this.cohesion(boids);
    const separation = this.separation(boids);

    alignment.mult(weights.align || 1.0);
    cohesion.mult(weights.cohesion || 1.0);
    separation.mult(weights.separation || 1.0);

    this.applyForce(alignment);
    this.applyForce(cohesion);
    this.applyForce(separation);
  }

  update() {
    this.vel.add(this.acc);
    this.vel.limit(this.maxSpeed);
    this.pos.add(this.vel);
    // reset acceleration
    this.acc.mult(0);
  }

  // Simple draw helper: returns a small descriptor that a caller can use to render.
  // If using p5.js, you may ignore this and draw using the boid's pos/vel directly.
  getRenderData() {
    return {
      x: this.pos.x,
      y: this.pos.y,
      angle: Math.atan2(this.vel.y, this.vel.x),
      size: this.size
    };
  }

  // Static helper for distance: supports Vec-like objects
  static _distance(a, b) {
    const ax = a.x, ay = a.y, bx = b.x, by = b.y;
    return Math.hypot(ax - bx, ay - by);
  }
}

// Export default for module usage
export default Boid;
