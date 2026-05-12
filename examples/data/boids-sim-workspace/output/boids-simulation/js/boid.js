// Auto-generated boid.js implementing the Boid class per contract.
// Exports: Boid class and tuning constants.

export const MAX_SPEED = 2.0;
export const MAX_FORCE = 0.05;
export const VISION_RADIUS = 60;
export const SEPARATION_RADIUS = 24;
export const ALIGN_WEIGHT = 1.0;
export const COHESION_WEIGHT = 1.0;
export const SEPARATION_WEIGHT = 1.5;

function limit(vec, maxMag) {
  const m2 = vec.x * vec.x + vec.y * vec.y;
  if (m2 > maxMag * maxMag) {
    const m = Math.sqrt(m2) || 1e-8;
    vec.x = (vec.x / m) * maxMag;
    vec.y = (vec.y / m) * maxMag;
  }
  return vec;
}

function mag(vec) {
  return Math.sqrt(vec.x * vec.x + vec.y * vec.y);
}

function normalize(vec) {
  const m = mag(vec) || 1e-8;
  return { x: vec.x / m, y: vec.y / m };
}

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}

function addInPlace(a, b) {
  a.x += b.x; a.y += b.y; return a;
}

function multInPlace(a, s) {
  a.x *= s; a.y *= s; return a;
}

function heading(vec) {
  return Math.atan2(vec.y, vec.x);
}

export class Boid {
  // constructor(x, y, vx, vy, hue)
  constructor(x, y, vx, vy, hue) {
    this.pos = { x: x || 0, y: y || 0 };
    this.vel = { x: vx ?? (Math.random() * 2 - 1), y: vy ?? (Math.random() * 2 - 1) };
    // Start with limited speed
    limit(this.vel, MAX_SPEED);
    this.acc = { x: 0, y: 0 };
    this.hue = hue ?? Math.floor(Math.random() * 360);
    this.size = 4; // rendering size
  }

  // update(dt, boids, width, height)
  update(dt, boids, width, height) {
    // Accumulate forces
    let align = { x: 0, y: 0 };
    let cohesion = { x: 0, y: 0 };
    let separation = { x: 0, y: 0 };
    let total = 0;

    for (let other of boids || []) {
      if (other === this) continue;
      const diff = sub(other.pos, this.pos);
      const d2 = diff.x * diff.x + diff.y * diff.y;
      const vr = VISION_RADIUS;
      const vr2 = vr * vr;
      if (d2 < 1e-12 || d2 > vr2) continue;
      total += 1;

      // alignment: steer toward average velocity
      align.x += other.vel.x;
      align.y += other.vel.y;

      // cohesion: steer toward average position
      cohesion.x += other.pos.x;
      cohesion.y += other.pos.y;

      // separation: avoid crowding; stronger within separation radius
      const sr = SEPARATION_RADIUS;
      const sr2 = sr * sr;
      if (d2 < sr2) {
        const invd = 1.0 / Math.sqrt(d2);
        separation.x -= diff.x * invd;
        separation.y -= diff.y * invd;
      }
    }

    if (total > 0) {
      // Alignment
      align.x /= total; align.y /= total;
      align = normalize(align);
      multInPlace(align, MAX_SPEED);
      align.x -= this.vel.x; align.y -= this.vel.y;
      limit(align, MAX_FORCE);
      multInPlace(align, ALIGN_WEIGHT);

      // Cohesion
      cohesion.x /= total; cohesion.y /= total; // avg position
      cohesion.x -= this.pos.x; cohesion.y -= this.pos.y; // vector to center
      cohesion = normalize(cohesion);
      multInPlace(cohesion, MAX_SPEED);
      cohesion.x -= this.vel.x; cohesion.y -= this.vel.y;
      limit(cohesion, MAX_FORCE);
      multInPlace(cohesion, COHESION_WEIGHT);

      // Separation
      if (separation.x !== 0 || separation.y !== 0) {
        separation = normalize(separation);
        multInPlace(separation, MAX_SPEED);
        separation.x -= this.vel.x; separation.y -= this.vel.y;
        limit(separation, MAX_FORCE);
      }
      multInPlace(separation, SEPARATION_WEIGHT);
    }

    // Apply accelerations
    this.acc.x = 0; this.acc.y = 0;
    addInPlace(this.acc, align);
    addInPlace(this.acc, cohesion);
    addInPlace(this.acc, separation);

    // Integrate
    const dtSec = Math.max(0.0005, dt || 0.016);
    this.vel.x += this.acc.x * dtSec;
    this.vel.y += this.acc.y * dtSec;
    limit(this.vel, MAX_SPEED);
    this.pos.x += this.vel.x * dtSec * 60; // scale to ~per-frame speed
    this.pos.y += this.vel.y * dtSec * 60;

    // Screen wrapping
    if (Number.isFinite(width) && Number.isFinite(height)) {
      if (this.pos.x < 0) this.pos.x += width;
      if (this.pos.x >= width) this.pos.x -= width;
      if (this.pos.y < 0) this.pos.y += height;
      if (this.pos.y >= height) this.pos.y -= height;
    }
  }

  // draw(ctx)
  draw(ctx) {
    if (!ctx) return;
    const angle = heading(this.vel);

    ctx.save();
    ctx.translate(this.pos.x, this.pos.y);
    ctx.rotate(angle);
    ctx.fillStyle = `hsl(${this.hue}, 80%, 60%)`;
    ctx.strokeStyle = `hsl(${this.hue}, 80%, 30%)`;
    ctx.lineWidth = 1;

    const s = this.size;
    ctx.beginPath();
    // Triangle pointing along +x
    ctx.moveTo(2.0 * s, 0);
    ctx.lineTo(-1.2 * s, 0.9 * s);
    ctx.lineTo(-0.8 * s, 0);
    ctx.lineTo(-1.2 * s, -0.9 * s);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }
}
