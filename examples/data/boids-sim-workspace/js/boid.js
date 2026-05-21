import Vector from './utils.js';

/*
Boid class implementing basic steering behaviors.
Defensive: works if Vector is present (as a constructor) or falls back to plain {x,y} vectors.
No globals used; defaults provided in constructor.
*/

function isVectorInstance(v) {
  return v && typeof v === 'object' && ('x' in v) && ('y' in v);
}

function createVec(x = 0, y = 0) {
  // Prefer the imported Vector if it looks like a constructor
  try {
    if (typeof Vector === 'function') return new Vector(x, y);
  } catch (e) {
    // fall back
  }
  return { x: x, y: y };
}

function vCopy(v) {
  if (!isVectorInstance(v)) return createVec(0, 0);
  // If Vector instances have copy(), try to use it
  if (typeof v.copy === 'function') return v.copy();
  return createVec(v.x, v.y);
}

function vAdd(a, b) {
  if (typeof a.add === 'function') return a.add(b);
  return createVec(a.x + b.x, a.y + b.y);
}

function vSub(a, b) {
  if (typeof a.sub === 'function') return a.sub(b);
  return createVec(a.x - b.x, a.y - b.y);
}

function vMult(v, n) {
  if (typeof v.mult === 'function') return v.mult(n);
  return createVec(v.x * n, v.y * n);
}

function vDiv(v, n) {
  if (typeof v.div === 'function') return v.div(n);
  if (n === 0) return createVec(v.x, v.y);
  return createVec(v.x / n, v.y / n);
}

function vMag(v) {
  if (typeof v.mag === 'function') return v.mag();
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

function vNormalize(v) {
  if (typeof v.normalize === 'function') return v.normalize();
  const m = vMag(v) || 0;
  return m === 0 ? createVec(0, 0) : vDiv(v, m);
}

function vSetMag(v, m) {
  if (typeof v.setMag === 'function') return v.setMag(m);
  const n = vNormalize(v);
  return vMult(n, m);
}

function vLimit(v, max) {
  if (typeof v.limit === 'function') return v.limit(max);
  const m = vMag(v);
  if (m > max) {
    return vSetMag(v, max);
  }
  return vCopy(v);
}

function vDist(a, b) {
  if (typeof a.dist === 'function') return a.dist(b);
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export class Boid {
  constructor(x = 0, y = 0) {
    // position, velocity, acceleration are vector-like objects
    this.position = createVec(x, y);
    // random initial velocity
    this.velocity = createVec((Math.random() * 2 - 1), (Math.random() * 2 - 1));
    this.acceleration = createVec(0, 0);

    // Tunable parameters (defensive defaults)
    this.maxSpeed = 3;    // typical cruising speed
    this.maxForce = 0.05; // steering force limit

    this.perceptionAlign = 50;
    this.perceptionCohesion = 50;
    this.perceptionSeparation = 24;

    // weights for flocking
    this.alignWeight = 1.0;
    this.cohesionWeight = 1.0;
    this.separationWeight = 1.5;
  }

  edges(width = 0, height = 0) {
    // Wrap-around behavior
    if (!isVectorInstance(this.position)) return;
    if (this.position.x > width) {
      this.position.x = 0;
    } else if (this.position.x < 0) {
      this.position.x = width;
    }
    if (this.position.y > height) {
      this.position.y = 0;
    } else if (this.position.y < 0) {
      this.position.y = height;
    }
  }

  align(boids = []) {
    const steering = createVec(0, 0);
    let total = 0;
    for (const other of boids) {
      if (!other || other === this) continue;
      const d = vDist(this.position, other.position);
      if (d > 0 && d < this.perceptionAlign) {
        steering.x += other.velocity.x;
        steering.y += other.velocity.y;
        total++;
      }
    }
    if (total === 0) return createVec(0, 0);
    const avg = vDiv(steering, total);
    const desired = vSetMag(avg, this.maxSpeed);
    const steer = vSub(desired, this.velocity);
    return vLimit(steer, this.maxForce);
  }

  cohesion(boids = []) {
    const steering = createVec(0, 0);
    let total = 0;
    for (const other of boids) {
      if (!other || other === this) continue;
      const d = vDist(this.position, other.position);
      if (d > 0 && d < this.perceptionCohesion) {
        steering.x += other.position.x;
        steering.y += other.position.y;
        total++;
      }
    }
    if (total === 0) return createVec(0, 0);
    const center = vDiv(steering, total);
    const desired = vSetMag(vSub(center, this.position), this.maxSpeed);
    const steer = vSub(desired, this.velocity);
    return vLimit(steer, this.maxForce);
  }

  separation(boids = []) {
    const steering = createVec(0, 0);
    let total = 0;
    for (const other of boids) {
      if (!other || other === this) continue;
      const d = vDist(this.position, other.position);
      if (d > 0 && d < this.perceptionSeparation) {
        const diff = vDiv(vSub(this.position, other.position), d || 1);
        steering.x += diff.x;
        steering.y += diff.y;
        total++;
      }
    }
    if (total === 0) return createVec(0, 0);
    const avg = vDiv(steering, total);
    const desired = vSetMag(avg, this.maxSpeed);
    const steer = vSub(desired, this.velocity);
    return vLimit(steer, this.maxForce);
  }

  flock(boids = []) {
    // Combine steering forces with weights
    const alignment = this.align(boids);
    const cohesion = this.cohesion(boids);
    const separation = this.separation(boids);

    // weighted contribution
    const a = vMult(alignment, this.alignWeight);
    const c = vMult(cohesion, this.cohesionWeight);
    const s = vMult(separation, this.separationWeight);

    // sum into acceleration
    this.acceleration = vAdd(this.acceleration, a);
    this.acceleration = vAdd(this.acceleration, c);
    this.acceleration = vAdd(this.acceleration, s);
  }

  update() {
    // v += a
    this.velocity = vAdd(this.velocity, this.acceleration);
    // limit speed
    this.velocity = vLimit(this.velocity, this.maxSpeed);
    // pos += vel
    this.position = vAdd(this.position, this.velocity);
    // reset acceleration
    this.acceleration = createVec(0, 0);
  }

  draw(ctx) {
    if (!ctx || typeof ctx.save !== 'function') return;
    const pos = this.position;
    const vel = this.velocity;
    const angle = Math.atan2(vel.y || 0, vel.x || 0);

    ctx.save();
    ctx.translate(pos.x || 0, pos.y || 0);
    ctx.rotate(angle);

    // Draw a triangle to represent the boid
    ctx.beginPath();
    ctx.moveTo(10, 0);
    ctx.lineTo(-8, 6);
    ctx.lineTo(-8, -6);
    ctx.closePath();
    ctx.fillStyle = '#555';
    ctx.fill();
    ctx.strokeStyle = '#222';
    ctx.stroke();

    ctx.restore();
  }
}
